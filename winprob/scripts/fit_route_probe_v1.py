"""Four fixed purchase-supervised fits and full-catalogue export; no scoring."""
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
from winprob.recipe_transition_data_v1 import require, array_hash, value_hash
from winprob.recipe_transition_probe_v1 import game_weights
from winprob.route_probe_v1 import ARMS, ContextScaler, objective, fit, predict, normalized_prior
from winprob.route_probe_data_v1 import RouteData, PROTOCOL, PROTOCOL_SHA, PRIOR, OLD


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def run(output):
    start = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'new immutable route fit required'); output.mkdir(parents=True)
    def guard(): require(time.monotonic()-start < 10800, 'fixed180minute route deadline reached')
    def status(stage, **extra):
        guard(); value = {'stage': stage, 'elapsed_s': time.monotonic()-start, **extra}
        (output/'status.json').write_text(json.dumps(value)+'\n', encoding='utf-8'); print(json.dumps(value), flush=True)
    paths = [Path(__file__), PROTOCOL, *[ROOT/'winprob'/name for name in (
        'route_probe_data_v1.py', 'route_probe_v1.py', 'route_capacity_v1.py', 'route_marginal_v1.py',
        'recipe_transition_data_v1.py', 'recipe_transition_probe_v1.py', 'recipe_transition_v1.py',
        'full_action_catalogue.py', 'iql_dataset_contract.py', 'admission.py',
        'preference_reference.py', 'trajectory_contract.py')]]
    code = {str(p.resolve()): file_hash(p) for p in paths}
    require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'fixed route protocol changed')
    for path in paths:
        dest = output/'code_snapshot'/path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    status('validating_admitted_inputs')
    data = RouteData(output/'input_cache', guard, status)
    lock = data.input_lock(); lock['code_sha256'] = code; write(output/'inputs_lock.json', lock)
    scalers = {}; timings = []; trains = {}
    for fold in (0, 1):
        train = np.flatnonzero(data.folds != fold); trains[fold] = train
        require(not set(data.game_ids[train]) & set(data.game_ids[data.folds == fold]), 'whole-game fold overlap')
        scale = ContextScaler.fit(data.context[train], data.game_ids[train]); scalers[fold] = scale
        x = scale.transform(data.context[train]); w = game_weights(data.game_ids[train])
        for arm in ARMS:
            guard(); m = data.marginals[arm][train]; dummy = np.broadcast_to(data.kernels[arm][0], m.shape)
            zero = np.zeros(26*258)
            for _ in range(2): objective(zero, x, m, dummy, w, ridge=.01/258)
            t = time.monotonic()
            for _ in range(5): guard(); objective(zero, x, m, dummy, w, ridge=.01/258)
            timings.append({'fold': fold, 'arm': arm, 'mean_seconds': (time.monotonic()-t)/5, 'passes': 5})
            q, neutral = predict(data.full[train[:16]], data.kernels[arm], x[:16], zero.reshape(26, 258))
            require(neutral.all() and np.array_equal(q, normalized_prior(data.full[train[:16]])), 'zero-fit prior parity failed')
            del m, dummy
    remaining = 10800-(time.monotonic()-start)
    projected = 1.5*max(v['mean_seconds'] for v in timings)*500*8*4+900
    preflight = {'passed': projected <= .8*remaining, 'projected_seconds': projected, 'remaining_seconds': remaining,
        'timings': timings, 'warmups': 2, 'dummy_group0_only': True, 'targets_opened': False,
        'actual_fits': 0, 'zero_fit_parity': True}
    write(output/'preflight.json', preflight); require(preflight['passed'], 'fixed route trial exceeds resource projection')
    write(output/'scales_lock.json', {'state': 'scales_frozen_before_targets',
        'scalers': {str(k): v.as_dict() for k, v in scalers.items()}, 'targets_opened': False})
    labels = data.load_targets(output/'inputs_lock.json'); models = {}; receipts = {}
    for fold in (0, 1):
        train = trains[fold]; x = scalers[fold].transform(data.context[train]); w = game_weights(data.game_ids[train])
        for arm in ARMS:
            status('fitting', fold=fold, arm=arm, complete_models=len(models))
            t = time.monotonic(); m = data.marginals[arm][train]; y = data.kernels[arm][labels['main'][train]]
            theta, receipt = fit(x, m, y, w, guard=guard)
            del m, y
            receipt['elapsed_fit_s'] = time.monotonic()-t
            qualified = receipt['success'] and receipt['gradient_max_abs'] < 1e-6
            write(output/f'solver_{arm}_fold{fold}.json', {'receipt': receipt, 'qualified': qualified, 'heldout_predictions_computed': False})
            if not qualified: np.save(output/f'unqualified_{arm}_fold{fold}.npy', theta, allow_pickle=False)
            require(qualified and theta.shape == (26, 258), 'fixed route optimizer did not qualify; no heldout score')
            spec = {'schema': 'route-probe-model-v1', 'arm': arm, 'evaluation_fold': fold,
                'coefficients': theta.tolist(), 'scaler': scalers[fold].as_dict(), 'fit': receipt,
                'protocol_sha256': PROTOCOL_SHA, 'prior_manifest_sha256': data.prior_sha,
                'kernel_sha256': array_hash(data.kernels[arm]), 'train_indices_sha256': array_hash(train),
                'train_game_ids_sha256': value_hash(sorted(set(data.game_ids[train]))),
                'train_states': len(train), 'train_games': len(set(data.game_ids[train])), 'code_sha256': code}
            name = f'route_{arm}_fold{fold}.json'; write(output/name, spec)
            models[name] = file_hash(output/name); receipts[f'{fold}:{arm}'] = receipt
    require(len(models) == 4, 'incomplete route model set'); data.revalidate()
    write(output/'models_lock.json', {'state': 'all_models_frozen_before_prediction', 'models': models})
    query = data.focal; nq = len(query); qfold = data.folds[query]
    distribution = output/'distribution'; distribution.mkdir()
    selection = [{**data.entries[int(index)], 'query_ordinal': ordinal} for ordinal, index in enumerate(query)]
    write(distribution/'selection.json', selection); write(distribution/'group_main.json', data.base.groups)
    arrays = {arm: np.lib.format.open_memmap(distribution/f'{arm}.npy', mode='w+', dtype=np.float64,
                                           shape=(nq, 9333)) for arm in ARMS}
    original = np.lib.format.open_memmap(distribution/'m2.npy', mode='w+', dtype=np.float32, shape=(nq, 9333))
    neutral_flags = np.zeros((2, nq), dtype=bool)
    for fold in (0, 1):
        local = np.flatnonzero(qfold == fold)
        for arm_index, arm in enumerate(ARMS):
            name = f'route_{arm}_fold{fold}.json'; require(file_hash(output/name) == models[name], 'frozen model changed')
            spec = json.loads((output/name).read_text()); scale = ContextScaler.from_dict(spec['scaler'])
            theta = np.asarray(spec['coefficients'], dtype=np.float64)
            for lo in range(0, len(local), 32):
                guard(); positions = local[lo:lo+32]; ids = query[positions]
                raw = data.full[ids]; x = scale.transform(data.context[ids])
                q, neutral = predict(raw, data.kernels[arm], x, theta)
                # Target mutation is irrelevant by construction: none is accepted.
                arrays[arm][positions] = q; neutral_flags[arm_index, positions] = neutral
                original[positions] = raw
            arrays[arm].flush(); original.flush()
            status('exporting_full_catalogue_predictions', fold=fold, arm=arm, rows=len(local))
    np.save(distribution/'neutral_flags.npy', neutral_flags, allow_pickle=False)
    del original, arrays
    files = {p.name: file_hash(p) for p in distribution.iterdir() if p.suffix in ('.npy', '.json')}
    write(output/'predictions_lock.json', {'state': 'predictions_frozen_before_decoding_and_scoring', 'files': files,
                                         'models_lock_sha256': file_hash(output/'models_lock.json')})
    data.revalidate()
    for path, sha in code.items(): require(file_hash(path) == sha, 'consumed route code changed')
    manifest = {'schema': 'route-probe-distribution-v1', 'state': 'complete', 'selection_count': nq,
        'groups': 9333, 'arms': list(ARMS), 'files': files, 'models': models,
        'model_directory': str(output), 'prior_manifest_path': str(PRIOR), 'prior_manifest_sha256': data.prior_sha,
        'reference_manifest_path': str(OLD/'manifest.json'), 'reference_manifest_sha256': data.base.digest,
        'source_context_manifest_path': str(data.base.context_directory/'manifest.json'),
        'source_context_manifest_sha256': data.prior_manifest['source_context_manifest_sha256'],
        'protocol_path': str(PROTOCOL), 'protocol_sha256': PROTOCOL_SHA,
        'inputs_lock_sha256': file_hash(output/'inputs_lock.json'), 'code_sha256': code,
        'prediction_accepts_targets': False, 'accuracy_computed': False, 'source_runtime_imported': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False,
        'neutral_decoder_rule': 'Use original float32 M2 on neutral rows to preserve native ties.'}
    write(distribution/'manifest.json', manifest)
    write(output/'report.json', {'schema': 'route-probe-fit-v1', 'state': 'complete', 'models': models,
        'fit_receipts': receipts, 'preflight': preflight, 'games': 3518, 'states': len(data.entries), 'query_games': nq,
        'distribution_manifest_sha256': file_hash(distribution/'manifest.json'), 'code_sha256': code,
        'training_performed': True, 'accuracy_computed': False, 'decoded_evaluation_completed': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'breakthrough_established': False, 'elapsed_s': time.monotonic()-start})
    status('complete_waiting_for_native_decode', query_games=nq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
