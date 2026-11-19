"""Two fixed native-bias fits; full predictions freeze before native scoring."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from winprob.native_bias_data_v1 import NativeBiasData, PROTOCOL, PROTOCOL_SHA, REFERENCE, ARMS, array_hash, opposite_fold_rows
from winprob.native_bias_calibration_v1 import BiasObjective, fit, predict, normalized_prior, require
from winprob.native_bias_gpu_v1 import configure, CudaBiasObjective
from winprob.npy_row_writer_v1 import NpyRowWriter
from winprob.iql_dataset_contract import file_hash
from winprob.iql_training import memory_snapshot


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def code_bindings():
    result = {str(PROTOCOL): file_hash(PROTOCOL)}
    for module in list(sys.modules.values()):
        raw = getattr(module, '__file__', None)
        if not raw: continue
        p = Path(raw).resolve()
        if p.is_file() and p.is_relative_to(ROOT) and p.suffix == '.py' and '.venv' not in p.parts and 'data' not in p.relative_to(ROOT).parts:
            result[str(p)] = file_hash(p)
    return result


def run(output):
    start = time.monotonic(); output = output.resolve()
    require(not output.exists() and file_hash(PROTOCOL) == PROTOCOL_SHA, 'new immutable calibration run required')
    output.mkdir(parents=True)
    def guard(): require(time.monotonic()-start < 10800, 'fixed3hour calibration deadline')
    def status(stage, **extra):
        guard(); row = {'stage': stage, 'elapsed_s': time.monotonic()-start, **extra}
        (output/'status.json').write_text(json.dumps(row)+'\n', encoding='utf-8'); print(json.dumps(row), flush=True)
    runtime = configure(); status('validating_admitted_inputs'); data = NativeBiasData(); code = code_bindings()
    for raw, sha in code.items():
        p = Path(raw); destination = output/'code_snapshot'/p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, destination)
        require(file_hash(destination) == sha, 'code snapshot differs')
    lock = data.input_lock(); lock['code_sha256'] = code; write(output/'inputs_lock.json', lock)
    np.save(output/'class_permutation.npy', data.permutation, allow_pickle=False)
    np.save(output/'basket_sizes.npy', data.sizes, allow_pickle=False)
    write(output/'control_lock.json', {'state': 'class_control_frozen_before_targets',
        'permutation_sha256': file_hash(output/'class_permutation.npy'), 'sizes_sha256': file_hash(output/'basket_sizes.npy'), 'seed': 20261004})
    timings = []; trains = {}
    for fold in (0, 1):
        rows = opposite_fold_rows(data.folds, fold); trains[fold] = rows
        sample = rows[:16]; dummy = np.zeros(9333, dtype=np.int64); dummy[0] = len(sample)
        cpu = BiasObjective(data.priors[sample], dummy)
        gpu = CudaBiasObjective(data.priors, sample, dummy, guard=guard)
        errors = []
        try:
            for b in (np.zeros(9333), np.random.default_rng(20261004+fold).normal(0, .01, 9333)):
                cv, cg = cpu(b); gv, gg = gpu(b)
                ev, eg = abs(cv-gv), float(np.max(np.abs(cg-gg)))
                require(ev <= 2e-10 and eg <= 2e-10, 'real-input dummy-target CPU/GPU parity failed')
                errors.append({'value_error': ev, 'max_gradient_error': eg})
        finally:
            gpu.close(); del gpu, cpu; gc.collect(); torch.cuda.empty_cache()
        free_before, _ = torch.cuda.mem_get_info(); reserved_before = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        dummy[0] = len(rows)
        calculation = CudaBiasObjective(data.priors, rows, dummy, guard=guard,
            status=lambda **kw: status('preflight_loading_train_priors', fold=fold, **kw))
        seconds = []
        try:
            zero = np.zeros(9333)
            for _ in range(2): calculation(zero)
            for _ in range(5):
                guard(); t = time.monotonic(); calculation(zero); seconds.append(time.monotonic()-t)
            free_after, _ = torch.cuda.mem_get_info(); peak_reserved = torch.cuda.max_memory_reserved()
            headroom = min(free_after, free_before-(peak_reserved-reserved_before))
            timings.append({'fold': fold, 'training_rows': len(rows), 'call_seconds': seconds,
                'cpu_gpu_parity': errors, 'conservative_device_headroom_bytes': headroom,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_cuda_reserved_bytes': peak_reserved})
        finally:
            calculation.close(); del calculation; gc.collect(); torch.cuda.empty_cache()
        status('preflight_fold_complete', fold=fold, max_call_seconds=max(seconds), headroom_bytes=headroom)
    remaining = 10800-(time.monotonic()-start)
    projected = 1.5*max(t for row in timings for t in row['call_seconds'])*4002+900
    preflight = {'passed': projected <= .8*remaining and min(r['conservative_device_headroom_bytes'] for r in timings) >= 1024**3,
        'timings': timings, 'projected_seconds': projected, 'remaining_seconds': remaining, 'runtime': runtime,
        'optimizer_and_recheck_evaluations_bound': 4002, 'real_purchase_targets_opened': False,
        'dummy_group0_only': True, 'warmups_per_fold': 2, 'timed_calls_per_fold': 5, 'host': memory_snapshot()}
    write(output/'preflight.json', preflight); require(preflight['passed'], 'native calibration fails resource gate')
    data.load_targets(output/'inputs_lock.json'); counts = {}; identities = {}
    for fold in (0, 1):
        counts[fold], identities[fold] = data.counts_for(fold, guard)
        np.save(output/f'train_counts_fold{fold}.npy', counts[fold], allow_pickle=False)
        write(output/f'training_identity_fold{fold}.json', identities[fold])
    data.labels = None  # Prediction never has a current-label table or label argument.
    count_files = {p.name: file_hash(p) for p in output.iterdir() if p.name.startswith(('train_counts_', 'training_identity_')) and p.suffix in ('.npy', '.json')}
    write(output/'counts_lock.json', {'state': 'opposite_fold_counts_frozen_before_fits', 'files': count_files,
        'target_ledger_sha256': data.base.context_manifest['files']['targets.jsonl'], 'actual_label_storage_released': True})
    models = {}; receipts = {}; biases = {}
    for fold in (0, 1):
        status('loading_fit_priors', fold=fold)
        calculation = CudaBiasObjective(data.priors, trains[fold], counts[fold], guard=guard)
        try:
            status('fitting', fold=fold); t = time.monotonic()
            b, receipt = fit(calculation, guard); receipt['elapsed_fit_s'] = time.monotonic()-t
        finally:
            calculation.close(); del calculation; gc.collect(); torch.cuda.empty_cache()
        write(output/f'solver_fold{fold}.json', receipt)
        if not receipt['qualified']: np.save(output/f'unqualified_bias_fold{fold}.npy', b, allow_pickle=False)
        require(receipt['qualified'], 'fixed optimizer did not qualify; no held-out predictions')
        name = f'bias_fold{fold}.npy'; np.save(output/name, b, allow_pickle=False)
        models[str(fold)] = {'file': name, 'sha256': file_hash(output/name), 'training_identity_sha256': file_hash(output/f'training_identity_fold{fold}.json'),
            'counts_sha256': file_hash(output/f'train_counts_fold{fold}.npy'), 'solver_sha256': file_hash(output/f'solver_fold{fold}.json')}
        biases[fold] = b; receipts[str(fold)] = receipt
        status('fit_complete', fold=fold, iterations=receipt['iterations'], gradient_max_abs=receipt['gradient_max_abs'])
    require(len(models) == 2 and data.labels is None, 'two fixed calibrators and no prediction labels required')
    write(output/'models_lock.json', {'state': 'both_calibrators_frozen_before_prediction', 'models': models,
        'counts_lock_sha256': file_hash(output/'counts_lock.json'), 'control_lock_sha256': file_hash(output/'control_lock.json')})
    distribution = output/'distribution'; distribution.mkdir()
    writers = {'m2': NpyRowWriter(distribution/'m2.npy', (2900, 9333), np.float32),
        **{arm: NpyRowWriter(distribution/f'{arm}.npy', (2900, 9333), np.float64) for arm in ARMS}}
    neutral = np.zeros((2, 2900), dtype=bool); qfold = data.folds[data.focals]
    for fold in (0, 1):
        positions = np.flatnonzero(qfold == fold); b = biases[fold]
        for lo in range(0, len(positions), 128):
            guard(); local = positions[lo:lo+128]; p = data.priors[data.focals[local]]
            require(data.labels is None, 'prediction acquired current labels')
            writers['m2'].write(local, p)
            for j, arm in enumerate(ARMS):
                coefficients = b if arm == 'calibrated' else b[data.permutation]
                q = predict(p, coefficients); identity = bool(np.all(coefficients == 0))
                if identity: require(np.array_equal(q, normalized_prior(p)), 'zero-bias baseline changed')
                writers[arm].write(local, q); neutral[j, local] = identity
        status('predicted_frozen_fold', fold=fold, queries=len(positions))
    for writer in writers.values(): writer.close()
    require(file_hash(distribution/'m2.npy') == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'original M2 prior did not reproduce')
    np.save(distribution/'neutral_flags.npy', neutral, allow_pickle=False)
    write(distribution/'selection.json', [{**data.entries[int(row)], 'query_ordinal': i} for i, row in enumerate(data.focals)])
    write(distribution/'group_main.json', data.base.groups)
    files = {p.name: file_hash(p) for p in distribution.iterdir() if p.suffix in ('.json', '.npy')}
    write(output/'predictions_lock.json', {'state': 'calibrated_distributions_frozen_before_native_scoring',
        'files': files, 'models_lock_sha256': file_hash(output/'models_lock.json')})
    data.revalidate()
    for path, sha in code.items(): require(file_hash(path) == sha, 'consumed calibration code changed')
    manifest = {'schema': 'native-bias-distribution-v1', 'state': 'complete', 'selection_count': 2900, 'groups': 9333,
        'arms': list(ARMS), 'files': files, 'models': models, 'experiment_directory': str(output),
        'protocol_path': str(PROTOCOL), 'protocol_sha256': PROTOCOL_SHA, 'code_sha256': code,
        'inputs_lock_sha256': file_hash(output/'inputs_lock.json'), 'prediction_lock_sha256': file_hash(output/'predictions_lock.json'),
        'reference_manifest_path': str(REFERENCE/'manifest.json'), 'reference_manifest_sha256': data.base.digest,
        'source_context_manifest_path': data.base.manifest['source_context_manifest_path'],
        'source_context_manifest_sha256': data.base.manifest['source_context_manifest_sha256'],
        'current_labels_used_in_prediction': False, 'reference_probabilities_used_at_inference': True,
        'accuracy_computed': False, 'source_runtime_imported': False, 'test_payload_reads': False, 'confirmation_payload_reads': False}
    write(distribution/'manifest.json', manifest)
    report = {'schema': 'native-bias-fit-v1', 'state': 'complete', 'models': models, 'fit_receipts': receipts,
        'selected_states': 56288, 'selected_games': 3518, 'query_games': 2900, 'parameters_per_model': 9333,
        'actual_fits': 2, 'preflight': preflight, 'training_performed': True, 'code_sha256': code,
        'distribution_manifest_sha256': file_hash(distribution/'manifest.json'), 'accuracy_computed': False,
        'heldout_targets_used_by_its_calibrator': False, 'current_labels_used_in_prediction': False,
        'source_runtime_imported': False, 'test_payload_reads': False, 'confirmation_payload_reads': False,
        'breakthrough_established': False, 'elapsed_s': time.monotonic()-start}
    write(output/'report.json', report); status('complete_waiting_for_native_scoring')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); run(p.parse_args().output)
