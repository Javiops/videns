"""Score frozen native route recommendations only after both world audits."""
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
from winprob.recipe_transition_data_v1 import require
from winprob.route_probe_io_v1 import CompletedRoute
from winprob.route_probe_evaluation_v1 import evaluate


def run(args):
    start = time.monotonic(); output = args.output.resolve(); require(not output.exists(), 'new immutable evaluation required')
    data = CompletedRoute(args.directory, args.decoded_manifest, args.native_manifest)
    labels, _ = data.load_targets(); scores = data.scores(labels)
    metrics = evaluate(labels, scores, data.base.game_ids[data.query].tolist(), data.folds, data.legality)
    require(metrics['games'] == 2900 and metrics['arms']['m2']['main_top1']['correct'] == 1483
            and metrics['arms']['m2']['main_top3']['correct'] == 1959, 'frozen M2 baseline/target population does not reproduce')
    data.revalidate(); output.mkdir(parents=True)
    costs = {'fit_worker_elapsed_s': data.fit['elapsed_s'],
        'total_optimization_s': sum(r['elapsed_fit_s'] for r in data.fit['fit_receipts'].values()),
        'source_stage_elapsed_s': {k: v['manifest']['elapsed_s'] for k, v in data.source.items()},
        'incremental_purchase_costs': data.costs,
        'distribution_array_bytes_on_disk': sum((data.dist_path.parent/name).stat().st_size for name in data.dist['files'] if name.endswith('.npy'))}
    report = {'schema': 'route-probe-evaluation-v1', 'state': 'complete', **metrics, 'costs': costs,
        'fit_directory': str(data.directory), 'fit_report_sha256': data.fit_sha,
        'distribution_manifest_sha256': data.dist_sha,
        'source_manifests': {world: {k: v[k] for k in ('manifest_path', 'manifest_sha256')} for world, v in data.source.items()},
        'bindings_sha256': data.bindings, 'code_sha256': {str(p.resolve()): file_hash(p) for p in (
            Path(__file__), ROOT/'winprob/route_probe_io_v1.py', ROOT/'winprob/route_probe_evaluation_v1.py')},
        'targets_opened_after_complete_predictions_and_native_audits': True,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'promotion_authorized': False, 'practical_breakthrough_established': False,
        'elapsed_s': time.monotonic()-start,
        'scope': 'Exposed retrospective16.17 internal comparison with frozen native16.18 M2; not forward, live or causal evidence.'}
    path = output/'report.json'; path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    path.with_suffix('.sha256').write_text(file_hash(path)+' report.json\n', encoding='ascii')
    print(json.dumps({k: report[k] for k in ('state', 'games', 'metric_gate_passed', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--decoded-manifest', type=Path, required=True); p.add_argument('--native-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); run(p.parse_args())
