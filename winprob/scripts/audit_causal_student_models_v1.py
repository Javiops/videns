"""Verify six fixed training receipts and exact teacher-free focal inference."""
from __future__ import annotations
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from winprob.causal_student_data_v1 import StudentData, ARMS, PROTOCOL_SHA, epoch_order, permitted_supervision_indices, array_hash, value_hash, require
from winprob.causal_student_v1 import CausalStudent
from winprob.student_checkpoint_v1 import load_checkpoint
from winprob.student_public_inference_v1 import focal_store
from winprob.bounded_npy_rows_v1 import BoundedNpyRows
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.iql_training import tree_hash


def run(directory, output):
    started = time.monotonic(); directory = directory.resolve(); output = output.resolve()
    require(directory.is_relative_to(ROOT/'data/runs') and not output.exists(), 'new immutable student model audit required')
    audit_code = {str(p.resolve()): file_hash(p) for p in (Path(__file__),
        ROOT/'winprob/student_public_inference_v1.py', ROOT/'winprob/student_checkpoint_v1.py')}
    bindings = dict(audit_code); report, rh = bound_json(directory/'report.json', bindings)
    dist, dh = bound_json(directory/'distribution/manifest.json', bindings)
    require(report['schema'] == 'causal-student-training-v1' and report['state'] == 'complete'
            and report['updates'] == 26460 and len(report['models']) == 6 and report['accuracy_computed'] is False
            and dh == report['distribution_manifest_sha256'] and dist['models'] == report['models'], 'complete fixed six-model family required')
    recovery_verified = False
    interruption_verified = False
    if report.get('interruption_recovery'):
        lineage, lh = bound_json(directory/'recovery_lineage.json', bindings)
        restart, eh = bound_json(directory/'interrupted_epoch_replay.json', bindings)
        original = Path(lineage['original_directory']).resolve(strict=True)
        require(original.is_relative_to(ROOT/'data/runs') and original != directory
                and str(original) == report['original_training_directory']
                and lh == report['recovery_lineage_sha256'] == dist['recovery_lineage_sha256']
                and eh == report['interrupted_epoch_replay_sha256'] == dist['interrupted_epoch_replay_sha256']
                and lineage['state'] == 'five_models_preserved_one_unsaved_control_recomputed'
                and report['training_restarted'] is True and report['completed_models_retrained'] is False
                and report['restarted_arm'] == lineage['restarted_arm'] == {'fold': 1, 'arm': 'shuffled_teacher'}
                and report['new_training_updates'] == 4420 and report['discarded_updates_lower_bound'] == 221
                and len(lineage['original_models']) == 5, 'interrupted-family recovery contract changed')
        evidence_path = Path(lineage['interruption_evidence_path']).resolve(strict=True)
        require(file_hash(evidence_path) == lineage['interruption_evidence_sha256'], 'interruption evidence changed')
        bindings[str(evidence_path)] = lineage['interruption_evidence_sha256']
        evidence = json.loads(evidence_path.read_text(encoding='utf-8-sig'))
        require(evidence['state'] == 'original_process_absent' and evidence['matching_processes'] == 0
                and {r['name']: r['sha256'] for r in evidence['completed_models']} == lineage['original_models'],
                'original completed model evidence differs')
        for name, sha in lineage['original_files_sha256'].items():
            path = (original/name).resolve()
            require(path.parent == original and file_hash(path) == sha, 'original interrupted artifact changed')
            bindings[str(path)] = sha
        for name, sha in lineage['original_models'].items():
            require(report['models'][name] == sha == file_hash(directory/name), 'a completed model was retrained or changed')
        original_rows = [json.loads(line) for line in (original/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        retained_rows = [json.loads(line) for line in (directory/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(original_rows) == 101 and len(retained_rows) == 120 and retained_rows[:100] == original_rows[:100]
                and file_hash(directory/'original_training.jsonl') == file_hash(original/'training.jsonl')
                and restart['state'] == 'complete' and restart['passed'] is True and restart['exact_except_elapsed_time'] is True
                and restart['original'] == original_rows[-1] == lineage['interrupted_epoch']
                and restart['recomputed'] == retained_rows[100]
                and {k: v for k, v in restart['original'].items() if k != 'seconds'}
                    == {k: v for k, v in restart['recomputed'].items() if k != 'seconds'}, 'interrupted epoch/retained trace did not reproduce')
        interruption_verified = True
    if report.get('metadata_load_recovery'):
        lineage, lh = bound_json(directory/'recovery_lineage.json', bindings)
        original = Path(lineage['training_directory']).resolve(strict=True)
        require(original.is_relative_to(ROOT/'data/runs') and original != directory
                and str(original) == report['training_phase_directory']
                and lh == report['recovery_lineage_sha256'] == dist['recovery_lineage_sha256']
                and lineage['state'] == 'all_original_models_preserved_before_prediction'
                and lineage['metadata_only_repair'] is True and lineage['training_restarted'] is False
                and report['training_restarted'] is False and lineage['original_launcher_status']['state'] == 'failed',
                'metadata-only recovery lineage changed')
        for name, sha in lineage['original_files_sha256'].items():
            source = (original/name).resolve(); copied = (directory/name).resolve()
            require(source.parent == original and copied.parent == directory
                    and file_hash(source) == sha == file_hash(copied), 'training recovery did not preserve artifact bytes')
            bindings[str(source)] = sha; bindings[str(copied)] = sha
        for leaf, key in (('worker.err', 'original_failure_sha256'), ('launcher_status.json', 'original_launcher_status_sha256')):
            p = original.parent/leaf; require(file_hash(p) == lineage[key], 'original terminal evidence changed')
            bindings[str(p)] = lineage[key]
        require(set(report['models']).issubset(lineage['original_files_sha256'])
                and lineage['original_models_lock_sha256'] == file_hash(directory/'models_lock.json'), 'original model family incomplete')
        recovery_verified = True
    for raw, sha in report['code_sha256'].items():
        p = Path(raw).resolve(); require(p.is_relative_to(ROOT) and file_hash(p) == sha
            and file_hash(directory/'code_snapshot'/p.relative_to(ROOT)) == sha, 'training/inference code changed')
        bindings[str(p)] = sha
    model_lock, mlh = bound_json(directory/'models_lock.json', bindings)
    pred_lock, _ = bound_json(directory/'predictions_lock.json', bindings)
    boundary, bh = bound_json(directory/'inference_boundary.json', bindings)
    controls, ch = bound_json(directory/'teacher_control_lock.json', bindings)
    require(model_lock['state'] == 'all_six_final_models_frozen_before_prediction'
            and model_lock['models'] == report['models'] and model_lock['updates'] == 26460
            and model_lock['teacher_controls_sha256'] == ch and pred_lock['models_lock_sha256'] == mlh
            and pred_lock['inference_boundary_sha256'] == bh and pred_lock['files'] == dist['files']
            and boundary == {'state': 'student_supervision_storage_released', 'teacher_gpu_available': False,
                             'observed_labels_available': False, 'forward_accepts_only_public_inputs': True}, 'model/prediction freeze boundary changed')
    for name, sha in dist['files'].items():
        p = directory/'distribution'/name; require(p.resolve().parent == directory/'distribution' and file_hash(p) == sha, 'prediction bytes changed')
        bindings[str(p)] = sha
    def guard(): require(time.monotonic()-started < 7200, 'fixed120minute model replay deadline')
    data = StudentData()
    original_lock, _ = bound_json(directory/'inputs_lock.json', bindings)
    require(original_lock['public_manifest_sha256'] == data.input_sha and original_lock['teacher_manifest_sha256'] == data.teacher_sha
            and original_lock['folds_sha256'] == array_hash(data.folds) and original_lock['state_indices_sha256'] == array_hash(data.indices), 'training input identities changed')
    trace = [json.loads(line) for line in (directory/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    require(len(trace) == 120 and len({(r['fold'], r['arm'], r['epoch']) for r in trace}) == 120, 'incomplete actual training trace')
    trace_lookup = {(r['fold'], r['arm'], r['epoch']): r for r in trace}
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'original CUDA inference required for exact replay')
    store, actions = focal_store(data, guard)
    data.teacher_reader = None
    require(data.teacher_gpu is None and data.labels_gpu is None and data.labels_cpu is None, 'audit must not create inference supervision storage')
    signature = set(inspect.signature(CausalStudent.forward).parameters)
    require(not {'teacher', 'observed_main', 'labels'} & signature, 'student predictor accepts forbidden supervision')
    matrices = {a: BoundedNpyRows(directory/'distribution'/f'{a}.npy', expected_shape=(2900, 9333), expected_dtype=np.float64) for a in ARMS}
    require(file_hash(directory/'distribution/m2.npy') == '6b014af07a30e2481db20650bfd933936be873bdd0cc5c9e1111925d52b4a7b2', 'native baseline changed')
    checks = {}; replayed = 0; permission_batches = 0; forbidden_rejections = 0; qfold = data.folds[data.focals]
    for fold in (0, 1):
        train = np.flatnonzero(data.folds != fold); held = np.flatnonzero(data.folds == fold)
        train_ids = sorted(str(v) for v in set(data.game_ids[train])); expected_updates = 20*int(np.ceil(len(train)/128))
        saved_perm = np.load(directory/f'teacher_permutation_fold{fold}.npy', allow_pickle=False)
        require(np.array_equal(saved_perm, data.permutations[fold]) and file_hash(directory/f'teacher_permutation_fold{fold}.npy') == controls['files'][str(fold)], 'teacher control does not reproduce')
        for row in held:
            rejected = False
            try: permitted_supervision_indices(data.folds, np.array([row]), fold)
            except ValueError: rejected = True
            require(rejected, 'held-out training supervision was permitted'); forbidden_rejections += 1
        torch.manual_seed(20261004+fold); initial = CausalStudent(data.num_champions, data.num_items)
        initial_hash = tree_hash(initial.state_dict()); parameter_count = sum(p.numel() for p in initial.parameters()); del initial
        local = np.flatnonzero(qfold == fold)
        for arm in ARMS:
            guard(); name = f'student_{arm}_fold{fold}.pt'; expected_sha = report['models'][name]
            checkpoint = load_checkpoint(directory/name, expected_sha); bindings[str(directory/name)] = expected_sha
            require(checkpoint['arm'] == arm and checkpoint['scored_fold'] == fold and checkpoint['epochs'] == 20
                    and checkpoint['updates'] == expected_updates and checkpoint['training_game_ids'] == train_ids
                    and checkpoint['training_game_ids_sha256'] == value_hash(train_ids)
                    and checkpoint['training_ordinals_sha256'] == array_hash(train)
                    and checkpoint['initial_state_sha256'] == initial_hash
                    and checkpoint['parameters'] == parameter_count and checkpoint['protocol_sha256'] == PROTOCOL_SHA
                    and checkpoint['public_manifest_sha256'] == data.input_sha and checkpoint['teacher_manifest_sha256'] == data.teacher_sha
                    and checkpoint['partial_batch_scale'] == 'batch_rows/128' and checkpoint['teacher_in_prediction_interface'] is False,
                    'checkpoint training recipe or identity changed')
            require(tree_hash(checkpoint['state_dict']) == checkpoint['final_state_sha256'], 'final tensor hash differs')
            require(checkpoint['optimizer'] == {'name': 'Adam', 'lr': 3e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0}, 'optimizer declaration changed')
            optimizer = checkpoint['optimizer_state_dict']; require(len(optimizer['param_groups']) == 1, 'optimizer group changed')
            group = optimizer['param_groups'][0]
            require(group['lr'] == 3e-4 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == 0,
                    'saved optimizer hyperparameters differ')
            require(len(optimizer['state']) == len(group['params'])
                    and all(float(value['step']) == expected_updates for value in optimizer['state'].values()), 'saved optimizer update counts differ')
            for epoch in range(1, 21):
                order = epoch_order(train, fold, epoch); record = trace_lookup[(fold, arm, epoch)]
                require(checkpoint['curves'][epoch-1] == record and record['states'] == len(train)
                        and record['updates'] == epoch*int(np.ceil(len(train)/128))
                        and record['order_sha256'] == array_hash(order) == checkpoint['epoch_order_sha256'][str(epoch)], 'actual epoch/order trace differs')
                for lo in range(0, len(order), 128):
                    permitted_supervision_indices(data.folds, order[lo:lo+128], fold, saved_perm if arm == 'shuffled_teacher' else None)
                    permission_batches += 1
            model = CausalStudent(data.num_champions, data.num_items).to('cuda').eval()
            model.load_state_dict(checkpoint['state_dict'], strict=True); del checkpoint
            with torch.inference_mode():
                for lo in range(0, len(local), 128):
                    guard(); positions = local[lo:lo+128]; board, runes = store.batch(positions)
                    require(data.teacher_gpu is None and data.labels_gpu is None, 'supervision entered audit prediction')
                    logits = model(board, runes, *actions, chunk=512, checkpoint_deep=False).cpu().numpy().astype(np.float64)
                    logits -= logits.max(1, keepdims=True); np.exp(logits, out=logits); logits /= logits.sum(1, keepdims=True)
                    require(np.array_equal(logits, matrices[arm][positions]), 'exact original-batch student prediction differs')
                    replayed += len(positions)
            checks[name] = {'parameters': parameter_count, 'epochs': 20, 'updates': expected_updates,
                            'training_games': len(train_ids), 'inference_queries': len(local), 'exact_replay': True}
            del model, logits; gc.collect(); torch.cuda.empty_cache()
            print(json.dumps({'stage': 'replayed_student', 'fold': fold, 'arm': arm, 'queries': len(local)}), flush=True)
    require(replayed == 8700 and permission_batches == 26460 and forbidden_rejections == 56288, 'audit accounting incomplete')
    require(not np.load(directory/'distribution/neutral_flags.npy', allow_pickle=False).any(), 'unexpected native fallback request')
    data.revalidate()
    for p, sha in bindings.items(): require(file_hash(p) == sha, 'student artifact changed during audit')
    output.mkdir(parents=True)
    result = {'schema': 'causal-student-model-audit-v1', 'state': 'complete', 'passed': True,
        'student_report_sha256': rh, 'distribution_manifest_sha256': dh, 'models_verified': checks,
        'student_queries_replayed_exactly': replayed, 'baseline_queries_verified': 2900,
        'training_batches_permission_checked': permission_batches, 'heldout_supervision_requests_rejected': forbidden_rejections,
        'prediction_had_no_teacher_or_labels': True, 'current_purchase_targets_opened': False,
        'unchanged_original_training_artifacts_verified': recovery_verified,
        'five_preserved_models_and_interrupted_epoch_verified': interruption_verified,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'elapsed_s': time.monotonic()-started, 'auditor_sha256': file_hash(__file__), 'audit_code_sha256': audit_code,
        'limits': ['Verifies training code/identity/order/optimizer receipts, not a second 26460-update training run.',
                   'Exact public-only inference replay; no native decoder rerun, forward/live gain or goal-completion claim.']}
    p = output/'report.json'; p.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.with_suffix('.sha256').write_text(file_hash(p)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'student_queries_replayed_exactly', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.directory, a.output)
