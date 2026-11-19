"""Two-candidate native comparison after six-model and native audits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from winprob.causal_student_io_v1 import CompletedStudent
from winprob.causal_student_evaluation_v1 import evaluate, require
from winprob.iql_dataset_contract import bound_json, file_hash


def run(args):
    start = time.monotonic(); output = args.output.resolve(); require(not output.exists(), 'new immutable student score required')
    data = CompletedStudent(args.directory, args.decoded_manifest, args.native_manifest)
    audit, ah = bound_json(args.model_audit, {})
    require(audit['schema'] == 'causal-student-model-audit-v1' and audit['state'] == 'complete' and audit['passed'] is True
            and audit['distribution_manifest_sha256'] == data.dist_sha
            and audit['student_report_sha256'] == data.training_sha
            and audit['student_queries_replayed_exactly'] == 8700
            and audit['training_batches_permission_checked'] == 26460
            and audit['current_purchase_targets_opened'] is False
            and (not data.training.get('metadata_load_recovery') or audit['unchanged_original_training_artifacts_verified'] is True)
            and (not data.training.get('interruption_recovery') or audit['five_preserved_models_and_interrupted_epoch_verified'] is True),
            'completed six-model inference audit required')
    output.mkdir(parents=True); lock = output/'scoring_inputs_lock.json'
    lock.write_text(json.dumps(data.scoring_lock(), sort_keys=True, indent=2)+'\n', encoding='utf-8')
    labels, _ = data.load_targets(lock)
    metrics = evaluate(labels, data.scores(labels), data.base.game_ids[data.query].tolist(), data.folds, data.legality)
    require(metrics['games'] == 2900 and metrics['arms']['m2']['main_top1']['correct'] == 1483
            and metrics['arms']['m2']['main_top3']['correct'] == 1959, 'original M2/target population did not reproduce')
    data.revalidate()
    training_seconds = sum(v['seconds'] for v in data.training['fit_receipts'].values())
    report = {'schema': 'causal-student-evaluation-v1', 'state': 'complete', **metrics,
        'student_directory': str(data.directory), 'student_report_sha256': data.training_sha,
        'distribution_manifest_sha256': data.dist_sha, 'model_audit_path': str(args.model_audit.resolve()), 'model_audit_sha256': ah,
        'scoring_inputs_lock_sha256': file_hash(lock),
        'source_manifests': {world: {k: v[k] for k in ('manifest_path', 'manifest_sha256')} for world, v in data.source.items()},
        'costs': {'retained_model_training_seconds': training_seconds,
            'known_discarded_training_seconds_lower_bound': data.training.get('discarded_training_seconds_lower_bound', 0.),
            'known_discarded_updates_lower_bound': data.training.get('discarded_updates_lower_bound', 0),
            'later_discarded_partial_updates_unknown': data.training.get('later_partial_updates_unknown', False),
            'reported_worker_elapsed_s': data.training['elapsed_s'],
            'reported_worker_elapsed_scope': data.training.get('elapsed_scope', 'complete training and prediction worker'),
            'metadata_load_recovery': data.training.get('metadata_load_recovery', False),
            'interruption_recovery': data.training.get('interruption_recovery', False),
            'source_stage_elapsed_s': {world: v['manifest']['elapsed_s'] for world, v in data.source.items()},
            'incremental_purchase_costs': data.costs},
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'promotion_authorized': False, 'practical_breakthrough_established': False,
        'bindings_sha256': data.bindings, 'code_sha256': {str(p.resolve()): file_hash(p) for p in (
            Path(__file__), ROOT/'winprob/causal_student_io_v1.py', ROOT/'winprob/causal_student_evaluation_v1.py')},
        'elapsed_s': time.monotonic()-start,
        'scope': 'Exposed retrospective16.17 paired internal comparison with native16.18 M2; not forward/live confirmation.'}
    p = output/'report.json'; p.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.with_suffix('.sha256').write_text(file_hash(p)+' report.json\n', encoding='ascii')
    print(json.dumps({k: report[k] for k in ('state', 'metric_gate_passed', 'selected_candidate', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--decoded-manifest', type=Path, required=True); p.add_argument('--native-manifest', type=Path, required=True)
    p.add_argument('--model-audit', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
