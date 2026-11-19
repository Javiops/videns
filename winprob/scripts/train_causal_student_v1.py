"""Six fixed complete-student fits and teacher-free predictions; no scoring."""
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
from winprob.causal_student_v1 import CausalStudent, purchase_objective
from winprob.causal_student_data_v1 import StudentData, ARMS, PROTOCOL, PROTOCOL_SHA, epoch_order, array_hash, value_hash, require
from winprob.iql_dataset_contract import file_hash
from winprob.iql_training import tree_hash, memory_snapshot
from winprob.npy_row_writer_v1 import NpyRowWriter


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def code_bindings():
    result = {str(PROTOCOL): file_hash(PROTOCOL)}
    for module in list(sys.modules.values()):
        raw = getattr(module, '__file__', None)
        if not raw: continue
        p = Path(raw).resolve()
        if p.is_file() and p.is_relative_to(ROOT) and p.suffix == '.py' and '.venv' not in p.parts and 'data' not in p.relative_to(ROOT).parts:
            result[str(p)] = file_hash(p)
    return result


def forward(model, data, rows, checkpoint_deep):
    board, runes = data.public_store.batch(rows)
    return model(board, runes, *data.actions, chunk=512, checkpoint_deep=checkpoint_deep)


def make_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=3e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)


def run(output):
    start = time.monotonic(); output = output.resolve()
    require(not output.exists() and file_hash(PROTOCOL) == PROTOCOL_SHA, 'new immutable run and fixed student protocol required')
    output.mkdir(parents=True)
    def guard():
        require(time.monotonic()-start < 21600, 'fixed6hour learner deadline reached')
    def status(stage, **extra):
        guard(); value = {'stage': stage, 'elapsed_s': time.monotonic()-start, **extra}
        (output/'status.json').write_text(json.dumps(value)+'\n', encoding='utf-8'); print(json.dumps(value), flush=True)
    status('validating_admitted_inputs'); data = StudentData(); code = code_bindings()
    for raw, sha in code.items():
        p = Path(raw); dest = output/'code_snapshot'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    lock = data.input_lock(); lock['code_sha256'] = code; write(output/'inputs_lock.json', lock)
    for fold in (0, 1): np.save(output/f'teacher_permutation_fold{fold}.npy', data.permutations[fold], allow_pickle=False)
    write(output/'teacher_control_lock.json', {'state': 'teacher_controls_frozen_before_actual_labels', 'facts': data.shuffle_facts,
        'files': {str(f): file_hash(output/f'teacher_permutation_fold{f}.npy') for f in (0, 1)}})
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'idle CUDA required')
    free_before, total = torch.cuda.mem_get_info(); reserve_before = torch.cuda.memory_reserved()
    data.preload(guard, status); status('real_input_cost_pilot')
    torch.manual_seed(20261004); pilot_model = CausalStudent(data.num_champions, data.num_items).to('cuda').train()
    optimizer = make_optimizer(pilot_model); pilot_rows = np.flatnonzero(data.folds != 0)[:128]
    dummy = torch.zeros(len(pilot_rows), dtype=torch.long, device='cuda'); timings = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(6):
        guard(); torch.cuda.synchronize(); t = time.monotonic(); optimizer.zero_grad(set_to_none=True)
        logits = forward(pilot_model, data, pilot_rows, True)
        value, _ = purchase_objective(logits, dummy, data.teacher_training(pilot_rows, 0, 'distilled'))
        (.5*value).backward()
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in pilot_model.parameters()), 'nonfinite cost-pilot gradient')
        optimizer.step(); torch.cuda.synchronize(); seconds = time.monotonic()-t
        if step >= 2: timings.append(seconds)
    peak_allocated = torch.cuda.max_memory_allocated(); peak_reserved = torch.cuda.max_memory_reserved(); free_after, _ = torch.cuda.mem_get_info()
    projected = 1.5*max(timings)*26460+900; remaining = 21600-(time.monotonic()-start)
    headroom = min(free_after, free_before-(peak_reserved-reserve_before))
    preflight = {'passed': projected <= .8*remaining and headroom >= 1024**3,
        'timed_steps_s': timings, 'warmups': 2, 'synthetic_label_group0_only': True, 'real_purchase_labels_opened': False,
        'teacher_rows': pilot_rows.tolist(), 'teacher_rows_all_opposite_fold': bool(np.all(data.folds[pilot_rows] != 0)),
        'projected_seconds': projected, 'remaining_seconds': remaining, 'updates': 26460,
        'peak_cuda_allocated_bytes': peak_allocated, 'peak_cuda_reserved_bytes': peak_reserved,
        'conservative_device_headroom_bytes': headroom, 'host': memory_snapshot()}
    write(output/'preflight.json', preflight); require(preflight['passed'], 'complete six-model comparison fails resource gate')
    del pilot_model, optimizer, logits, value, dummy; gc.collect(); torch.cuda.empty_cache()
    data.load_targets(output/'inputs_lock.json')
    models = {}; receipts = {}; total_updates = 0; order_hashes = {}; model_parameters = None
    log_path = output/'training.jsonl'
    with log_path.open('x', encoding='utf-8') as training_log:
        for fold in (0, 1):
            train = np.flatnonzero(data.folds != fold); train_ids = sorted(set(data.game_ids[train])); initial_hash = None
            orders = {epoch: epoch_order(train, fold, epoch) for epoch in range(1, 21)}
            order_hashes[str(fold)] = {str(epoch): array_hash(order) for epoch, order in orders.items()}
            for arm in ARMS:
                guard(); torch.manual_seed(20261004+fold)
                model = CausalStudent(data.num_champions, data.num_items).to('cuda').train(); optimizer = make_optimizer(model)
                init = tree_hash(model.state_dict())
                if initial_hash is None: initial_hash = init
                require(init == initial_hash, 'within-fold initialization differs across arms')
                model_parameters = sum(p.numel() for p in model.parameters()); updates = 0; curves = []; arm_start = time.monotonic()
                for epoch in range(1, 21):
                    t = time.monotonic(); totals = {'optimization_loss': 0., 'hard_loss': 0., 'teacher_KL': 0.}; count = 0
                    order = orders[epoch]
                    for lo in range(0, len(order), 128):
                        guard(); rows = order[lo:lo+128]; optimizer.zero_grad(set_to_none=True)
                        logits = forward(model, data, rows, True)
                        value, pieces = purchase_objective(logits, data.targets_training(rows, fold), data.teacher_training(rows, fold, arm))
                        objective = value if arm == 'hard' else .5*value
                        scaled = objective*(len(rows)/128); scaled.backward()
                        require(bool(torch.isfinite(scaled)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'nonfinite actual training gradient')
                        optimizer.step(); updates += 1; total_updates += 1; count += len(rows)
                        totals['optimization_loss'] += float(objective.detach())*len(rows)
                        totals['hard_loss'] += float(pieces['hard'].detach())*len(rows)
                        totals['teacher_KL'] += float(pieces['teacher_KL'].detach())*len(rows)
                    require(count == len(train), 'epoch lost or added training states')
                    receipt = {'fold': fold, 'arm': arm, 'epoch': epoch, 'states': count,
                        **{key: number/count for key, number in totals.items()}, 'seconds': time.monotonic()-t,
                        'updates': updates, 'order_sha256': order_hashes[str(fold)][str(epoch)]}
                    curves.append(receipt); training_log.write(json.dumps(receipt, allow_nan=False)+'\n'); training_log.flush()
                    status('training', fold=fold, arm=arm, epoch=epoch, updates=updates, complete_models=len(models), epoch_seconds=receipt['seconds'])
                require(updates == 20*int(np.ceil(len(train)/128)), 'fixed training update count differs')
                checkpoint = {'schema': 'causal-student-checkpoint-v1', 'arm': arm, 'scored_fold': fold, 'epochs': 20,
                    'updates': updates, 'state_dict': cpu_tree(model.state_dict()), 'optimizer_state_dict': cpu_tree(optimizer.state_dict()),
                    'initial_state_sha256': init, 'final_state_sha256': tree_hash(model.state_dict()),
                    'training_game_ids': train_ids, 'training_ordinals_sha256': array_hash(train),
                    'training_game_ids_sha256': value_hash(train_ids), 'epoch_order_sha256': order_hashes[str(fold)],
                    'public_manifest_sha256': data.input_sha, 'teacher_manifest_sha256': data.teacher_sha,
                    'protocol_sha256': PROTOCOL_SHA, 'code_sha256': code, 'parameters': model_parameters,
                    'num_champions': data.num_champions, 'num_items': data.num_items,
                    'optimizer': {'name': 'Adam', 'lr': 3e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0},
                    'partial_batch_scale': 'batch_rows/128', 'teacher_in_prediction_interface': False,
                    'curves': curves, 'elapsed_training_s': time.monotonic()-arm_start}
                name = f'student_{arm}_fold{fold}.pt'; torch.save(checkpoint, output/name); models[name] = file_hash(output/name)
                receipts[f'{fold}:{arm}'] = {'updates': updates, 'epochs': 20, 'training_games': len(train_ids),
                    'training_states': len(train), 'initial_state_sha256': init, 'final_state_sha256': checkpoint['final_state_sha256'],
                    'seconds': checkpoint['elapsed_training_s']}
                del model, optimizer, checkpoint, logits, value, pieces, objective, scaled; gc.collect(); torch.cuda.empty_cache()
    require(len(models) == 6 and total_updates == 26460, 'incomplete fixed candidate family')
    data.revalidate(); write(output/'models_lock.json', {'state': 'all_six_final_models_frozen_before_prediction', 'models': models,
        'updates': total_updates, 'teacher_controls_sha256': file_hash(output/'teacher_control_lock.json')})
    dist = output/'distribution'; dist.mkdir()
    native_writer = NpyRowWriter(dist/'m2.npy', (2900, 9333), np.float32)
    for lo in range(0, len(data.focals), 128):
        positions = np.arange(lo, min(lo+128, len(data.focals)))
        native_writer.write(positions, data.teacher_reader[data.focals[positions]])
    native_writer.close()
    require(file_hash(dist/'m2.npy') == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'native focal prior differs')
    # Withhold all supervision storage from student inference.
    data.teacher_gpu = None; data.labels_gpu = None; data.labels_cpu = None; data.teacher_reader = None
    for name in ('full_action_id', 'main_group_id'):
        a = data.base.arrays.pop(name, None)
        if a is not None and getattr(a, '_mmap', None) is not None: a._mmap.close()
    gc.collect(); torch.cuda.empty_cache()
    write(output/'inference_boundary.json', {'state': 'student_supervision_storage_released',
        'teacher_gpu_available': False, 'observed_labels_available': False, 'forward_accepts_only_public_inputs': True})
    qfold = data.folds[data.focals]; predictions = {arm: NpyRowWriter(dist/f'{arm}.npy', (2900, 9333), np.float64) for arm in ARMS}
    with torch.inference_mode():
        for fold in (0, 1):
            positions = np.flatnonzero(qfold == fold)
            for arm in ARMS:
                name = f'student_{arm}_fold{fold}.pt'; require(file_hash(output/name) == models[name], 'saved model changed')
                saved = torch.load(output/name, map_location='cpu', weights_only=True)
                model = CausalStudent(data.num_champions, data.num_items).to('cuda').eval()
                model.load_state_dict(saved['state_dict'], strict=True); del saved
                for lo in range(0, len(positions), 128):
                    guard(); local = positions[lo:lo+128]; rows = data.focals[local]
                    require(data.teacher_gpu is None and data.labels_gpu is None, 'supervision reached prediction stage')
                    logits = forward(model, data, rows, False).cpu().numpy().astype(np.float64)
                    require(np.isfinite(logits).all(), 'nonfinite student prediction')
                    logits -= logits.max(1, keepdims=True); np.exp(logits, out=logits); logits /= logits.sum(1, keepdims=True)
                    predictions[arm].write(local, logits)
                status('predicting_frozen_students', fold=fold, arm=arm, queries=len(positions))
                del model, logits; gc.collect(); torch.cuda.empty_cache()
    for value in predictions.values(): value.close()
    np.save(dist/'neutral_flags.npy', np.zeros((3, 2900), dtype=bool), allow_pickle=False)
    write(dist/'selection.json', [{**data.entries[int(row)], 'query_ordinal': i} for i, row in enumerate(data.focals)])
    write(dist/'group_main.json', data.catalogue['group_main'])
    files = {p.name: file_hash(p) for p in dist.iterdir() if p.suffix in ('.npy', '.json')}
    write(output/'predictions_lock.json', {'state': 'all_student_distributions_frozen_before_decoding_or_scoring', 'files': files,
        'models_lock_sha256': file_hash(output/'models_lock.json'), 'inference_boundary_sha256': file_hash(output/'inference_boundary.json')})
    data.revalidate()
    for raw, sha in code.items(): require(file_hash(raw) == sha, 'consumed learner code changed')
    manifest = {'schema': 'causal-student-distribution-v1', 'state': 'complete', 'selection_count': 2900, 'groups': 9333,
        'arms': list(ARMS), 'files': files, 'models': models, 'experiment_directory': str(output),
        'protocol_path': str(PROTOCOL), 'protocol_sha256': PROTOCOL_SHA, 'code_sha256': code,
        'inputs_lock_sha256': file_hash(output/'inputs_lock.json'), 'prediction_lock_sha256': file_hash(output/'predictions_lock.json'),
        'public_manifest_sha256': data.input_sha, 'teacher_manifest_sha256': data.teacher_sha,
        'reference_manifest_path': str(ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition/recipe-m2-release-20261001-v2/manifest.json'),
        'reference_manifest_sha256': 'f0bf5e91a06a04151654c2293364bfe47e46d8b9acbfe96eb96822c3020c9abe',
        'source_context_manifest_path': data.teacher_manifest['source_context_manifest_path'],
        'source_context_manifest_sha256': data.teacher_manifest['source_context_manifest_sha256'],
        'student_inference_uses_teacher_or_labels': False, 'accuracy_computed': False,
        'source_runtime_imported': False, 'test_payload_reads': False, 'confirmation_payload_reads': False}
    write(dist/'manifest.json', manifest)
    write(output/'report.json', {'schema': 'causal-student-training-v1', 'state': 'complete', 'models': models,
        'fit_receipts': receipts, 'updates': total_updates, 'selected_states': 56288, 'selected_games': 3518, 'query_games': 2900,
        'parameters_per_model': model_parameters, 'preflight': preflight, 'teacher_shuffle': data.shuffle_facts,
        'distribution_manifest_sha256': file_hash(dist/'manifest.json'), 'code_sha256': code,
        'training_performed': True, 'student_inference_uses_teacher_or_labels': False,
        'accuracy_computed': False, 'source_runtime_imported': False, 'test_payload_reads': False,
        'confirmation_payload_reads': False, 'breakthrough_established': False, 'elapsed_s': time.monotonic()-start})
    status('complete_waiting_for_native_decoding', models=6, query_games=2900)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); run(p.parse_args().output)
