"""Fixed four-fit personal-graph comparison; no native/source runtime imports."""
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
from winprob.recipe_transition_data_v1 import array_hash, value_hash, require
from winprob.recipe_transition_probe_v1 import game_weights
from winprob.recipe_transition_v1 import objective
from winprob.recipe_transition_solver_v2 import fit
from winprob.personal_graph_data_v1 import GraphData, PROTOCOL, PROTOCOL_SHA
from winprob.personal_graph_models_v1 import ARMS, RMSScaler, predict_graph
from winprob.personal_graph_evaluation_v1 import compact_prediction, evaluate


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def run(output):
    started = time.monotonic(); output = output.resolve(); require(not output.exists(), 'immutable new personal-graph run required')
    output.mkdir(parents=True)
    names = ('personal_graph_data_v1.py', 'personal_graph_models_v1.py', 'personal_graph_evaluation_v1.py',
        'personal_kill_graph_v2.py', 'recipe_transition_data_v1.py', 'recipe_transition_v1.py', 'recipe_transition_solver_v2.py',
        'recipe_transition_probe_v1.py', 'recipe_transition_evaluation_v1.py', 'ipo_residual.py', 'full_action_catalogue.py',
        'iql_dataset_contract.py', 'admission.py', 'preference_reference.py', 'trajectory_contract.py')
    paths = [Path(__file__), PROTOCOL, *[ROOT/'winprob'/name for name in names]]
    code = {str(p.resolve()): file_hash(p) for p in paths}; require(code[str(PROTOCOL)] == PROTOCOL_SHA, 'fixed protocol changed')
    for path in paths:
        dest = output/'code_snapshot'/path.resolve().relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    def guard():
        require(time.monotonic()-started < 2700, 'fixed45minute personal-graph worker bound reached')
    def status(stage, **extra):
        guard(); info = {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra}
        (output/'status.json').write_text(json.dumps(info)+'\n', encoding='utf-8'); print(json.dumps(info), flush=True)
    status('validating_features')
    data = GraphData(); n = len(data.entries)
    lock = data.input_lock(); lock['code_sha256'] = code; write(output/'inputs_lock.json', lock)
    status('inputs_frozen', states=n, active_personal_focals=data.active_focals)
    scalers = {}; trains = {}; timings = []
    for fold in (0, 1):
        train = np.flatnonzero(data.folds != fold); trains[fold] = train
        require(not set(data.game_ids[train]) & set(data.game_ids[data.folds == fold]), 'whole-game fold overlap')
        weights = game_weights(data.game_ids[train]); candidate_scale = None
        for arm in ARMS:
            inputs = data.batch(train, arm); scaler = RMSScaler.fit(inputs, data.game_ids[train]); scalers[(fold, arm)] = scaler
            if candidate_scale is not None:
                require(np.array_equal(candidate_scale, scaler.candidate_scale), 'candidate scales differ between arms')
            candidate_scale = scaler.candidate_scale
            transformed = scaler.transform(inputs); zero = np.zeros(272); dummy = np.zeros(len(train), dtype=np.int64)
            for _ in range(2):
                objective(zero, transformed, dummy, weights)
            t = time.monotonic()
            for _ in range(20):
                objective(zero, transformed, dummy, weights)
            elapsed = time.monotonic()-t
            timings.append({'fold': fold, 'arm': arm, 'seconds': elapsed/20, 'block_seconds': elapsed, 'passes': 20, 'rows': len(train)})
            pred = predict_graph(inputs, scaler, np.zeros((17, 16)))
            require(np.array_equal(pred['probabilities'], inputs.reference)
                    and np.array_equal(pred['candidate_order'], np.tile([0, 1, 2], (len(train), 1))), 'zero-model reference parity failed')
    remaining = 2700-(time.monotonic()-started); projected = 1.5*max(r['seconds'] for r in timings)*300*8*4+180
    preflight = {'passed': projected <= .8*remaining, 'projected_seconds': projected, 'remaining_seconds': remaining,
        'objective_gradient_timings': timings, 'synthetic_category0_only': True, 'actual_purchase_fits': 0,
        'zero_coefficients_parity': True, 'scales_fitted_before_target_read': True,
        'assumptions': {'margin': 1.5, 'iterations': 300, 'evaluation_proxy': 8, 'fits': 4, 'overhead_s': 180}}
    write(output/'preflight.json', preflight); require(preflight['passed'], 'fixed graph comparison fails resource projection')
    write(output/'scales_lock.json', {'state': 'training_scales_frozen',
        'scales': {f'{fold}:{arm}': scale.as_dict() for (fold, arm), scale in scalers.items()}, 'targets_opened': False})
    labels = data.load_targets(output/'inputs_lock.json'); models = {}; receipts = {}; status('fitting', projected_seconds=projected)
    for fold in (0, 1):
        train = trains[fold]; weights = game_weights(data.game_ids[train])
        for arm in ARMS:
            guard(); scaler = scalers[(fold, arm)]; transformed = scaler.transform(data.batch(train, arm)); t = time.monotonic()
            theta, receipt = fit(transformed, labels['coarsened'][train], weights, ridge=.01, max_iter=300)
            seconds = time.monotonic()-t; qualified = bool(receipt['converged'] and receipt['gradient_max_abs'] < 1e-6)
            write(output/f'solver_{arm}_fold{fold}.json', {'fit': receipt, 'qualified': qualified, 'elapsed_fit_s': seconds,
                'heldout_predictions_computed': False, 'train_indices_sha256': array_hash(train)})
            if not qualified:
                np.save(output/f'unqualified_{arm}_fold{fold}.npy', theta, allow_pickle=False)
            require(qualified and theta.shape == (17, 16) and receipt['parameters'] == 272, 'fixed graph numerical gate failed')
            spec = {'schema': 'personal-graph-correction-v1', 'arm': arm, 'evaluation_fold': fold, 'coefficients': theta.tolist(),
                'scaler': scaler.as_dict(), 'fit': receipt, 'elapsed_fit_s': seconds, 'graph_feature_digests': data.graph_feature_digests,
                'native_manifest_sha256': data.digest, 'protocol_sha256': PROTOCOL_SHA, 'code_sha256': code,
                'train_indices_sha256': array_hash(train), 'train_game_ids_sha256': value_hash(sorted(set(data.game_ids[train]))),
                'train_states': len(train), 'train_games': len(set(data.game_ids[train]))}
            name = f'personal_graph_{arm}_fold{fold}.json'; write(output/name, spec); models[name] = file_hash(output/name)
            receipts[f'{fold}:{arm}'] = {**receipt, 'seconds': seconds}
            status('fitting', completed_models=len(models), fold=fold, arm=arm, seconds=seconds)
    data.revalidate(); require(len(models) == 4, 'incomplete fixed model set')
    write(output/'models_lock.json', {'state': 'all_models_frozen_before_prediction', 'models': models,
                                      'graph_feature_digests': data.graph_feature_digests})
    query = data.focal; nq = len(query); qfold = data.folds[query]; predictions = {
        arm: {'probabilities': np.zeros((nq, 4)), 'log_probabilities': np.full((nq, 4), -np.inf),
              'candidate_order': np.zeros((nq, 3), dtype=np.int64)} for arm in ARMS}
    status('predicting_frozen_models', games=nq)
    for fold in (0, 1):
        local = np.flatnonzero(qfold == fold); rows = query[local]
        for arm in ARMS:
            name = f'personal_graph_{arm}_fold{fold}.json'; require(file_hash(output/name) == models[name], 'saved model changed')
            spec = json.loads((output/name).read_text(encoding='utf-8'))
            pred = predict_graph(data.batch(rows, arm), RMSScaler.from_dict(spec['scaler']), np.asarray(spec['coefficients'], dtype=np.float64))
            for key, value in pred.items():
                predictions[arm][key][local] = value
    prior = data.reference[query]; lp = np.full_like(prior, -np.inf); np.log(prior, out=lp, where=prior > 0)
    predictions['m2'] = {'probabilities': prior.copy(), 'log_probabilities': lp, 'candidate_order': np.tile([0, 1, 2], (nq, 1))}
    zero_profiles = {arm: data.zero_profiles[arm][query] for arm in ARMS}
    arrays = {'query_indices': query, 'candidate_groups': data.group_ids[query], 'folds': qfold,
              **{arm+'__zero_profile': z for arm, z in zero_profiles.items()}}
    for arm, result in predictions.items():
        for key, value in result.items():
            arrays[arm+'__'+key] = value
        if arm in ARMS:
            z = zero_profiles[arm]
            require(np.array_equal(result['probabilities'][z], prior[z]) and np.array_equal(result['log_probabilities'][z], lp[z])
                    and np.array_equal(result['candidate_order'][z], np.tile([0, 1, 2], (int(z.sum()), 1))), 'zero-information baseline parity failed')
    np.savez_compressed(output/'predictions.npz', **arrays)
    write(output/'predictions_lock.json', {'state': 'predictions_frozen_before_scoring', 'sha256': file_hash(output/'predictions.npz'),
        'models_lock_sha256': file_hash(output/'models_lock.json'), 'prediction_accepts_no_targets': True})
    focal_labels = {k: v[query] for k, v in labels.items()}
    compact = {arm: compact_prediction(p, data.group_ids[query], focal_labels['coarsened']) for arm, p in predictions.items()}
    metrics = evaluate(focal_labels, compact, data.group_ids[query], data.game_ids[query].tolist(), qfold, zero_profiles)
    require(metrics['games'] == 2900, 'primary denominator changed')
    data.revalidate()
    for path, want in code.items():
        require(file_hash(path) == want, 'consumed graph code changed')
    report = {'schema': 'personal-graph-experiment-v1', 'state': 'complete', **metrics, 'models': models,
        'fit_summaries': receipts, 'preflight': preflight, 'graph_feature_digests': data.graph_feature_digests,
        'feature_availability': {k: lock[k] for k in ('active_personal_focals', 'zero_profile_counts', 'zero_profile_focals')},
        'inputs_lock_sha256': file_hash(output/'inputs_lock.json'), 'selected_targets_loaded_for_training_and_scoring': True,
        'bindings_sha256': data.bindings, 'code_sha256': code,
        'predictions_sha256': file_hash(output/'predictions.npz'), 'models_lock_sha256': file_hash(output/'models_lock.json'),
        'predictions_lock_sha256': file_hash(output/'predictions_lock.json'), 'selected_states': n, 'selected_games': 3518,
        'unavailable_original_focal_games': 618, 'training_performed': True, 'source_runtime_imported': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'outcomes_analyzed': False,
        'promotion_authorized': False, 'practical_breakthrough_established': False, 'elapsed_s': time.monotonic()-started,
        'scope': 'Exposed retrospective16.17 internal screen with frozen native16.18 M2; not forward or live confirmation.'}
    write(output/'report.json', report); status('complete', metric_gate_passed=metrics['metric_gate_passed'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
