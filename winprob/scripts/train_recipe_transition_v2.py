"""Run the six fixed convex fits only after the complete native artifact exists."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import file_hash
from winprob.recipe_transition_data_v1 import RecipeData, PROTOCOL, PROTOCOL_SHA, array_hash, value_hash, require
from winprob.recipe_transition_v1 import fit, objective
from winprob.recipe_transition_probe_v1 import Standardizer, ARMS, game_weights
from winprob.recipe_transition_evaluation_v1 import predict_correction, compact_prediction, evaluate


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def run(source, output):
    started = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'new immutable experiment output required')
    output.mkdir(parents=True)
    code_paths = [Path(__file__), PROTOCOL, ROOT/'docs/recipe_fit_runtime_v2.md', *(ROOT/'winprob'/name for name in (
        'recipe_transition_v1.py', 'recipe_transition_probe_v1.py', 'recipe_transition_data_v1.py',
        'recipe_transition_evaluation_v1.py', 'ipo_residual.py', 'full_action_catalogue.py',
        'iql_dataset_contract.py', 'admission.py', 'preference_reference.py', 'trajectory_contract.py'))]
    code = {str(p.resolve()): file_hash(p) for p in code_paths}
    require(code[str(PROTOCOL)] == PROTOCOL_SHA, 'consumed protocol changed')
    for path in code_paths:
        copied = output/'code_snapshot'/path.resolve().relative_to(ROOT)
        copied.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, copied)
    def guard():
        require(time.monotonic()-started < 2700, 'fixed45minute consumer/fit/evaluation bound reached')
    def status(stage, **extra):
        guard(); value = {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra}
        (output/'status.json').write_text(json.dumps(value, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(value), flush=True)
    status('validating_artifact_inputs')
    data = RecipeData(source); n = len(data.entries)
    input_lock = {'state': 'inputs_frozen', 'native_manifest_sha256': data.digest,
        'feature_digests': data.feature_digests, 'states': n, 'focal_decisions': len(data.focal),
        'native_receipts_verified': data.receipts_verified, 'target_file_opened': False, 'code_sha256': code}
    previous_lock_path = ROOT/'data/runs/recipe-transition-fit-20261001-v1/experiment/inputs_lock.json'
    previous_lock = json.loads(previous_lock_path.read_text(encoding='utf-8'))
    require(previous_lock['target_file_opened'] is False and previous_lock['native_manifest_sha256'] == data.digest
            and previous_lock['feature_digests'] == data.feature_digests, 'runtime-only retry changed actual inputs')
    input_lock['previous_input_lock_sha256'] = file_hash(previous_lock_path)
    input_lock['runtime_version'] = 'recipe-fit-runtime-v2'
    write(output/'inputs_lock.json', input_lock)
    status('inputs_frozen', states=n, focal_decisions=len(data.focal), receipts=data.receipts_verified)
    scalers = {}; train_rows = {}; samples = []
    for fold in (0, 1):
        train = np.flatnonzero(data.folds != fold); hold = np.flatnonzero(data.folds == fold)
        require(not set(data.game_ids[train]) & set(data.game_ids[hold]), 'whole-game folds overlap')
        raw_inputs = data.batch(train); weights = game_weights(data.game_ids[train])
        scalers[fold] = Standardizer.fit(raw_inputs, data.game_ids[train]); train_rows[fold] = train
        for arm in ARMS:
            transformed = scalers[fold].transform(raw_inputs, arm, data.permutations[train] if arm == 'shuffled_net' else None)
            zero = np.zeros(676); dummy = np.zeros(len(train), dtype=np.int64)
            # Timing only: zero synthetic category, no true purchase access.
            objective(zero, transformed, dummy, weights)
            for _ in range(2):
                t = time.monotonic(); objective(zero, transformed, dummy, weights)
                samples.append({'fold': fold, 'arm': arm, 'seconds': time.monotonic()-t, 'rows': len(train)})
            parity = predict_correction(raw_inputs, data.permutations[train], scalers[fold], np.zeros((26, 26)), arm)
            require(np.array_equal(parity['probabilities'], raw_inputs.reference)
                    and np.array_equal(parity['candidate_order'], np.tile([0, 1, 2], (len(train), 1))), 'zero correction changed native prior/order')
            del transformed, parity
        del raw_inputs
    remaining = 2700-(time.monotonic()-started)
    # Operational estimate, not a worst-case solver proof: eight evaluations per
    # allowed iteration, 1.5margin and180seconds of extra work. Hard limit remains.
    projected = 1.5*max(r['seconds'] for r in samples)*300*8*6+180
    preflight = {'passed': projected <= .8*remaining, 'projected_seconds': projected, 'remaining_seconds': remaining,
        'objective_gradient_timings': samples, 'synthetic_target_only': True, 'real_purchase_fits': 0,
        'zero_reference_order_verified_all_training_states': True,
        'assumptions': {'max_iterations_per_fit': 300, 'fits': 6, 'function_evaluations_per_iteration_proxy': 8,
                        'margin': 1.5, 'additional_seconds': 180, 'hard_worker_seconds': 2700},
        'limits': ['Runtime projection is an estimate; solver maxls remains50 and the hard worker guard is enforced.']}
    write(output/'preflight.json', preflight)
    require(preflight['passed'], 'fixed full consumer experiment fails runtime projection')
    labels = data.load_targets(output/'inputs_lock.json')
    model_files = {}; fit_summaries = {}; status('fitting', projected_seconds=projected)
    for fold in (0, 1):
        train = train_rows[fold]; raw_inputs = data.batch(train)
        weights = game_weights(data.game_ids[train])
        for arm in ARMS:
            guard(); transformed = scalers[fold].transform(raw_inputs, arm, data.permutations[train] if arm == 'shuffled_net' else None)
            t = time.monotonic()
            coefficients, receipt = fit(transformed, labels['coarsened'][train], weights, ridge=.01, max_iter=300)
            seconds = time.monotonic()-t
            require(receipt['converged'] and receipt['gradient_max_abs'] < 1e-6, 'declared convex convergence gate failed')
            require(receipt['parameters'] == 676 and np.isfinite(coefficients).all(), 'coefficient schema failed')
            if arm == 'raw':
                require(np.array_equal(coefficients[:, 16:], np.zeros((26, 10))), 'zero input block acquired nonzero parameters')
            payload = {'schema': 'recipe-transition-correction-v1', 'arm': arm, 'evaluation_fold': fold,
                'coefficients': coefficients.tolist(), 'standardizer': scalers[fold].as_dict(),
                'native_manifest_sha256': data.digest, 'protocol_sha256': PROTOCOL_SHA,
                'train_indices_sha256': array_hash(train), 'train_game_ids_sha256': value_hash(sorted(set(data.game_ids[train]))),
                'train_game_count': len(set(data.game_ids[train])), 'train_state_count': len(train),
                'fit': receipt, 'elapsed_fit_s': seconds, 'code_sha256': code}
            name = f'correction_{arm}_fold{fold}.json'; write(output/name, payload)
            model_files[name] = file_hash(output/name); fit_summaries[f'{fold}:{arm}'] = {'seconds': seconds, **receipt}
            status('fitting', finished_models=len(model_files), fold=fold, arm=arm, seconds=seconds)
            del transformed
        del raw_inputs
    require(len(model_files) == 6, 'not all fixed models completed')
    data.revalidate()
    for raw, want in code.items():
        require(file_hash(raw) == want, 'consumed code changed during fitting')
    training_lock = {'state': 'all_models_frozen', 'models': model_files, 'feature_digests': data.feature_digests,
                     'native_manifest_sha256': data.digest, 'prediction_started': False}
    write(output/'training_lock.json', training_lock)
    query = data.focal; qfold = data.folds[query]; nq = len(query)
    predictions = {arm: {'probabilities': np.zeros((nq, 4)), 'log_probabilities': np.full((nq, 4), -np.inf),
                         'candidate_order': np.full((nq, 3), -1, dtype=np.int64)} for arm in ARMS}
    status('predicting_frozen_models', query_games=nq)
    for fold in (0, 1):
        local = np.flatnonzero(qfold == fold); rows = query[local]; inputs = data.batch(rows)
        for arm in ARMS:
            name = f'correction_{arm}_fold{fold}.json'
            require(file_hash(output/name) == model_files[name], 'checkpoint changed before prediction')
            spec = json.loads((output/name).read_text(encoding='utf-8'))
            scaler = Standardizer.from_dict(spec['standardizer']); theta = np.asarray(spec['coefficients'], dtype=np.float64)
            result = predict_correction(inputs, data.permutations[rows], scaler, theta, arm)
            for key, value in result.items():
                predictions[arm][key][local] = value
    baseline = data.inputs.reference[query]
    baseline_log = np.full_like(baseline, -np.inf); np.log(baseline, out=baseline_log, where=baseline > 0)
    predictions['m2'] = {'probabilities': baseline.copy(), 'log_probabilities': baseline_log,
                         'candidate_order': np.tile([0, 1, 2], (nq, 1))}
    arrays = {'query_indices': query, 'candidate_groups': data.group_ids[query], 'folds': qfold}
    for arm, values in predictions.items():
        for key, value in values.items():
            arrays[f'{arm}__{key}'] = value
    np.savez_compressed(output/'predictions.npz', **arrays)
    write(output/'prediction_lock.json', {'state': 'predictions_frozen_before_scoring',
        'predictions_sha256': file_hash(output/'predictions.npz'), 'training_lock_sha256': file_hash(output/'training_lock.json'),
        'query_indices_sha256': array_hash(query), 'true_targets_are_not_prediction_inputs': True})
    # Actual held-out labels enter only the scoring operation below.
    focal_labels = {key: value[query] for key, value in labels.items()}
    compact = {arm: compact_prediction(p, data.group_ids[query], focal_labels['coarsened']) for arm, p in predictions.items()}
    metrics = evaluate(focal_labels, compact, data.group_ids[query], data.game_ids[query].tolist(), qfold)
    require(metrics['games'] == 2900 and metrics['native_option_membership_unchanged'] and metrics['top3_coverage_unchanged'], 'complete primary population or policy identity failed')
    data.revalidate()
    for raw, want in code.items():
        require(file_hash(raw) == want, 'consumed code changed before publication')
    report = {'schema': 'recipe-transition-experiment-v1', 'state': 'complete', 'runtime_version': 'recipe-fit-runtime-v2', **metrics,
        'fit_summaries': fit_summaries, 'models': model_files, 'preflight': preflight,
        'native_manifest_path': str(data.directory/'manifest.json'), 'native_manifest_sha256': data.digest,
        'bindings_sha256': data.bindings, 'code_sha256': code, 'feature_digests': data.feature_digests,
        'predictions_sha256': file_hash(output/'predictions.npz'), 'training_lock_sha256': file_hash(output/'training_lock.json'),
        'prediction_lock_sha256': file_hash(output/'prediction_lock.json'), 'selected_states': n, 'selected_games': 3518,
        'unavailable_original_focal_games': 618, 'native_receipts_verified': data.receipts_verified,
        'training_performed': True, 'source_runtime_imported': False, 'test_payload_reads': False,
        'confirmation_payload_reads': False, 'outcomes_analyzed': False, 'promotion_authorized': False,
        'practical_breakthrough_established': False, 'elapsed_s': time.monotonic()-started,
        'scope': 'Exposed retrospective16.17 internal screen; frozen M2 trained16.18. No chronological transfer or confirmation claim.',
        'likelihood': 'Coarsened original native candidates plus OTHER; not full-catalogue NLL.'}
    write(output/'report.json', report)
    status('complete', metric_gate_passed=metrics['metric_gate_passed'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
