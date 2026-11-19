"""Paired displayed-policy scoring after complete frozen proxy/native artifacts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from winprob.iql_dataset_contract import file_hash
from winprob.bayes_proxy_io_v1 import CompletedBayes
from winprob.bayes_proxy_evaluation_v1 import evaluate
from winprob.bayes_proxy_ratio_v1 import require


def run(args):
    start = time.monotonic(); output = args.output.resolve(); require(not output.exists(), 'new immutable scoring run required')
    data = CompletedBayes(args.directory, args.decoded_manifest, args.native_manifest)
    output.mkdir(parents=True); lock = output/'scoring_inputs_lock.json'
    lock.write_text(json.dumps(data.scoring_lock(), sort_keys=True, indent=2)+'\n', encoding='utf-8')
    labels, _ = data.load_targets(lock); scores = data.scores(labels)
    metrics = evaluate(labels, scores, data.base.game_ids[data.query].tolist(), data.folds, data.legality)
    require(metrics['games'] == 2900 and metrics['arms']['m2']['main_top1']['correct'] == 1483
            and metrics['arms']['m2']['main_top3']['correct'] == 1959, 'original focal M2 baseline did not reproduce')
    data.revalidate()
    report = {'schema': 'bayes-proxy-evaluation-v1', 'state': 'complete', **metrics,
        'forecast_directory': str(data.directory), 'forecast_report_sha256': data.forecast_sha,
        'distribution_manifest_sha256': data.dist_sha, 'scoring_inputs_lock_sha256': file_hash(lock),
        'source_manifests': {world: {key: value[key] for key in ('manifest_path', 'manifest_sha256')} for world, value in data.source.items()},
        'costs': {'forecast_worker_elapsed_s': data.forecast['elapsed_s'], 'neural_training_performed': False,
                  'source_stage_elapsed_s': {world: value['manifest']['elapsed_s'] for world, value in data.source.items()},
                  'incremental_purchase_costs': data.costs},
        'neutral_counts': data.forecast['neutral_counts'], 'label_scope': 'recipient-fold current/future labels excluded during its prediction',
        'targets_opened_after_predictions_and_native_audits': True, 'test_payload_reads': False,
        'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'promotion_authorized': False, 'practical_breakthrough_established': False,
        'bindings_sha256': data.bindings, 'code_sha256': {str(p.resolve()): file_hash(p) for p in (
            Path(__file__), ROOT/'winprob/bayes_proxy_io_v1.py', ROOT/'winprob/bayes_proxy_evaluation_v1.py')},
        'elapsed_s': time.monotonic()-start,
        'scope': 'Exposed retrospective16.17 internal OOF screen against native16.18 M2; no forward/live/causal gain claim.'}
    target = output/'report.json'; target.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    target.with_suffix('.sha256').write_text(file_hash(target)+' report.json\n', encoding='ascii')
    print(json.dumps({k: report[k] for k in ('state', 'games', 'metric_gate_passed', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--decoded-manifest', type=Path, required=True); p.add_argument('--native-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
