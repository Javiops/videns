"""Inference-only recovery of six unchanged final checkpoints with scoped metadata loading."""
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
from winprob.causal_student_v1 import CausalStudent
from winprob.causal_student_data_v1 import StudentData, ARMS, PROTOCOL, PROTOCOL_SHA, array_hash, value_hash, epoch_order, require
from winprob.student_checkpoint_v1 import load_checkpoint
from winprob.student_public_inference_v1 import focal_store
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.iql_training import tree_hash
from winprob.npy_row_writer_v1 import NpyRowWriter

AMENDMENT = ROOT/'docs/causal_student_metadata_load_amendment_v2.md'


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def run(training_directory, output):
    start = time.monotonic(); training_directory = training_directory.resolve(); output = output.resolve()
    require(training_directory.is_relative_to(ROOT/'data/runs') and not output.exists(), 'new own prediction recovery required')
    terminal = json.loads((training_directory.parent/'launcher_status.json').read_text(encoding='utf-8-sig'))
    error = (training_directory.parent/'worker.err').read_text(encoding='utf-8-sig')
    require(terminal['state'] == 'failed' and 'UnpicklingError' in error and 'numpy._core.multiarray.scalar' in error,
            'requires actual terminal metadata-load failure, not a live/unknown worker')
    require(not (training_directory/'report.json').exists(), 'completed original run must not be replaced')
    original_lock, _ = bound_json(training_directory/'inputs_lock.json', {})
    model_lock, _ = bound_json(training_directory/'models_lock.json', {})
    require(model_lock['state'] == 'all_six_final_models_frozen_before_prediction' and len(model_lock['models']) == 6
            and model_lock['updates'] == 26460, 'all fixed fits must finish before recovery')
    require(file_hash(PROTOCOL) == PROTOCOL_SHA, 'statistical protocol changed')
    for raw, sha in original_lock['code_sha256'].items():
        p = Path(raw).resolve(); require(p.is_relative_to(ROOT) and file_hash(p) == sha
            and file_hash(training_directory/'code_snapshot'/p.relative_to(ROOT)) == sha, 'original training code/snapshot changed')
    def guard(): require(time.monotonic()-start < 3600, 'fixed60minute prediction recovery bound')
    data = StudentData(); require(data.input_lock()['bindings_sha256'] == original_lock['bindings_sha256'], 'original learner inputs changed')
    models = model_lock['models']; checkpoints = {}; receipts = {}; total_updates = 0
    for fold in (0, 1):
        train = np.flatnonzero(data.folds != fold); ids = sorted(str(x) for x in set(data.game_ids[train])); expected_steps = 20*int(np.ceil(len(train)/128))
        initial = None
        for arm in ARMS:
            guard(); name = f'student_{arm}_fold{fold}.pt'; checkpoint = load_checkpoint(training_directory/name, models[name])
            require(checkpoint['arm'] == arm and checkpoint['scored_fold'] == fold and checkpoint['epochs'] == 20
                    and checkpoint['updates'] == expected_steps and checkpoint['training_game_ids'] == ids
                    and checkpoint['training_ordinals_sha256'] == array_hash(train)
                    and checkpoint['training_game_ids_sha256'] == value_hash(ids)
                    and checkpoint['protocol_sha256'] == PROTOCOL_SHA
                    and checkpoint['public_manifest_sha256'] == data.input_sha and checkpoint['teacher_manifest_sha256'] == data.teacher_sha,
                    'fixed training/fold/checkpoint identity changed')
            if initial is None: initial = checkpoint['initial_state_sha256']
            require(initial == checkpoint['initial_state_sha256'] and tree_hash(checkpoint['state_dict']) == checkpoint['final_state_sha256'], 'model tensor identity changed')
            require(len(checkpoint['curves']) == 20, 'missing actual training epochs')
            for epoch, curve in enumerate(checkpoint['curves'], 1):
                order_sha = array_hash(epoch_order(train, fold, epoch))
                require(curve['epoch'] == epoch and curve['states'] == len(train)
                        and curve['updates'] == epoch*int(np.ceil(len(train)/128))
                        and curve['order_sha256'] == order_sha == checkpoint['epoch_order_sha256'][str(epoch)], 'fixed training order/accounting changed')
            receipts[f'{fold}:{arm}'] = {'updates': expected_steps, 'epochs': 20, 'training_games': len(ids),
                'training_states': len(train), 'initial_state_sha256': initial, 'final_state_sha256': checkpoint['final_state_sha256'],
                'seconds': checkpoint['elapsed_training_s']}
            total_updates += expected_steps; checkpoints[name] = checkpoint
    require(total_updates == 26460, 'incomplete model family')
    output.mkdir(parents=True)
    for name, sha in models.items():
        shutil.copyfile(training_directory/name, output/name); require(file_hash(output/name) == sha, 'copied checkpoint differs')
    original_files = dict(models)
    for name in ('inputs_lock.json', 'inputs_lock.sha256', 'models_lock.json', 'models_lock.sha256',
                 'teacher_control_lock.json', 'teacher_control_lock.sha256', 'preflight.json', 'preflight.sha256',
                 'training.jsonl', 'teacher_permutation_fold0.npy', 'teacher_permutation_fold1.npy'):
        shutil.copyfile(training_directory/name, output/name)
        original_files[name] = file_hash(training_directory/name)
        require(file_hash(output/name) == original_files[name], 'copied training receipt differs')
    code = dict(original_lock['code_sha256'])
    for p in (Path(__file__), AMENDMENT, ROOT/'winprob/student_checkpoint_v1.py', ROOT/'winprob/student_public_inference_v1.py'):
        code[str(p.resolve())] = file_hash(p)
    for raw, sha in code.items():
        p = Path(raw); dest = output/'code_snapshot'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        source = training_directory/'code_snapshot'/p.relative_to(ROOT) if raw in original_lock['code_sha256'] else p
        shutil.copyfile(source, dest); require(file_hash(dest) == sha, 'recovery snapshot differs')
    write(output/'recovery_lineage.json', {'state': 'all_original_models_preserved_before_prediction',
        'training_directory': str(training_directory), 'original_launcher_status': terminal,
        'original_models_lock_sha256': file_hash(training_directory/'models_lock.json'),
        'original_files_sha256': original_files,
        'original_failure_sha256': file_hash(training_directory.parent/'worker.err'),
        'original_launcher_status_sha256': file_hash(training_directory.parent/'launcher_status.json'),
        'amendment_sha256': file_hash(AMENDMENT), 'metadata_only_repair': True, 'training_restarted': False})
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'original CUDA inference required')
    store, actions = focal_store(data, guard)
    dist = output/'distribution'; dist.mkdir()
    original_native = training_directory/'distribution/m2.npy'
    require(file_hash(original_native) == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'original focal reference incomplete')
    shutil.copyfile(original_native, dist/'m2.npy'); data.teacher_reader = None
    write(output/'inference_boundary.json', {'state': 'student_supervision_storage_released',
        'teacher_gpu_available': False, 'observed_labels_available': False, 'forward_accepts_only_public_inputs': True})
    writers = {arm: NpyRowWriter(dist/f'{arm}.npy', (2900, 9333), np.float64) for arm in ARMS}
    qfold = data.folds[data.focals]
    with torch.inference_mode():
        for fold in (0, 1):
            local = np.flatnonzero(qfold == fold)
            for arm in ARMS:
                name = f'student_{arm}_fold{fold}.pt'; checkpoint = checkpoints.pop(name)
                model = CausalStudent(data.num_champions, data.num_items).to('cuda').eval()
                model.load_state_dict(checkpoint['state_dict'], strict=True); del checkpoint
                for lo in range(0, len(local), 128):
                    guard(); positions = local[lo:lo+128]; board, runes = store.batch(positions)
                    require(data.teacher_gpu is None and data.labels_gpu is None, 'student inference acquired supervision')
                    p = model(board, runes, *actions, chunk=512, checkpoint_deep=False).cpu().numpy().astype(np.float64)
                    require(np.isfinite(p).all(), 'nonfinite recovered student forecast')
                    p -= p.max(1, keepdims=True); np.exp(p, out=p); p /= p.sum(1, keepdims=True)
                    writers[arm].write(positions, p)
                del model, p; gc.collect(); torch.cuda.empty_cache()
                print(json.dumps({'stage': 'predicting_unchanged_final_models', 'fold': fold, 'arm': arm, 'queries': len(local)}), flush=True)
    for writer in writers.values(): writer.close()
    np.save(dist/'neutral_flags.npy', np.zeros((3, 2900), dtype=bool), allow_pickle=False)
    write(dist/'selection.json', [{**data.entries[int(row)], 'query_ordinal': i} for i, row in enumerate(data.focals)])
    write(dist/'group_main.json', data.catalogue['group_main'])
    files = {p.name: file_hash(p) for p in dist.iterdir() if p.suffix in ('.json', '.npy')}
    write(output/'predictions_lock.json', {'state': 'all_student_distributions_frozen_before_decoding_or_scoring', 'files': files,
        'models_lock_sha256': file_hash(output/'models_lock.json'), 'inference_boundary_sha256': file_hash(output/'inference_boundary.json')})
    data.revalidate()
    for raw, sha in code.items(): require(file_hash(raw) == sha, 'consumed training/recovery code changed')
    manifest = {'schema': 'causal-student-distribution-v1', 'state': 'complete', 'selection_count': 2900, 'groups': 9333,
        'arms': list(ARMS), 'files': files, 'models': models, 'experiment_directory': str(output),
        'protocol_path': str(PROTOCOL), 'protocol_sha256': PROTOCOL_SHA, 'code_sha256': code,
        'inputs_lock_sha256': file_hash(output/'inputs_lock.json'), 'prediction_lock_sha256': file_hash(output/'predictions_lock.json'),
        'public_manifest_sha256': data.input_sha, 'teacher_manifest_sha256': data.teacher_sha,
        'reference_manifest_path': str(ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2/manifest.json'),
        'reference_manifest_sha256': 'f0bf5e91a06a04151654c2293364bfe47e46d8b9acbfe96eb96822c3020c9abe',
        'source_context_manifest_path': data.teacher_manifest['source_context_manifest_path'],
        'source_context_manifest_sha256': data.teacher_manifest['source_context_manifest_sha256'],
        'student_inference_uses_teacher_or_labels': False, 'accuracy_computed': False, 'source_runtime_imported': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'metadata_load_recovery': True,
        'recovery_lineage_sha256': file_hash(output/'recovery_lineage.json')}
    write(dist/'manifest.json', manifest)
    preflight = json.loads((output/'preflight.json').read_text(encoding='utf-8'))
    write(output/'report.json', {'schema': 'causal-student-training-v1', 'state': 'complete', 'models': models,
        'fit_receipts': receipts, 'updates': 26460, 'selected_states': 56288, 'selected_games': 3518, 'query_games': 2900,
        'parameters_per_model': 196545, 'preflight': preflight, 'teacher_shuffle': data.shuffle_facts,
        'distribution_manifest_sha256': file_hash(dist/'manifest.json'), 'code_sha256': code,
        'training_performed': True, 'student_inference_uses_teacher_or_labels': False,
        'accuracy_computed': False, 'source_runtime_imported': False, 'test_payload_reads': False,
        'confirmation_payload_reads': False, 'breakthrough_established': False,
        'training_phase_directory': str(training_directory), 'training_restarted': False, 'metadata_load_recovery': True,
        'recovery_lineage_sha256': file_hash(output/'recovery_lineage.json'),
        'total_training_seconds': sum(r['seconds'] for r in receipts.values()), 'elapsed_s': time.monotonic()-start,
        'elapsed_scope': 'Prediction-recovery worker only; training cost is reported separately.'})
    print(json.dumps({'state': 'complete', 'models': 6, 'training_restarted': False, 'prediction_seconds': time.monotonic()-start}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--training-directory', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.training_directory, a.output)
