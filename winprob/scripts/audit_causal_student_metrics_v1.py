"""Repeat fixed metrics and independently count native exactness after replay."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.causal_student_io_v1 import CompletedStudent
from winprob.causal_student_evaluation_v1 import evaluate, ARMS
from winprob.bayes_proxy_ratio_v1 import require


def compare(actual, expected, name='root'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'metric keys differ: '+name)
        for key in expected: compare(actual[key], expected[key], name+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'metric array differs: '+name)
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)): compare(a, e, name+'.'+str(i))
    elif isinstance(expected, float): require(np.isclose(actual, expected, atol=1e-12, rtol=1e-12), 'metric value differs: '+name)
    else: require(actual == expected, 'metric differs: '+name)


def run(args):
    start = time.monotonic(); output = args.output.resolve(); require(not output.exists(), 'new immutable metric audit required')
    report, rh = bound_json(args.report, {}); replay, ah = bound_json(args.model_audit, {})
    require(report['schema'] == 'causal-student-evaluation-v1' and report['state'] == 'complete'
            and replay['schema'] == 'causal-student-model-audit-v1' and replay['state'] == 'complete'
            and replay['passed'] is True and replay['student_queries_replayed_exactly'] == 8700
            and replay['distribution_manifest_sha256'] == report['distribution_manifest_sha256']
            and ah == report['model_audit_sha256'], 'completed forecast replay required')
    for p, sha in report['code_sha256'].items(): require(file_hash(p) == sha, 'evaluation code changed')
    src = report['source_manifests']
    data = CompletedStudent(report['student_directory'], src['16.18']['manifest_path'], src['16.17']['manifest_path'])
    require(data.training_sha == report['student_report_sha256'] == replay['student_report_sha256']
            and data.dist_sha == report['distribution_manifest_sha256'], 'scored forecast identity differs')
    for world in src: require(data.source[world]['manifest_sha256'] == src[world]['manifest_sha256'], 'source audit identity differs')
    lock = args.report.parent/'scoring_inputs_lock.json'
    require(file_hash(lock) == report['scoring_inputs_lock_sha256'], 'scoring input lock changed')
    labels, _ = data.load_targets(lock); scores = data.scores(labels)
    metrics = evaluate(labels, scores, data.base.game_ids[data.query].tolist(), data.folds, data.legality)
    for key, value in metrics.items(): compare(report[key], value, key)
    direct = {}
    for arm in ARMS:
        first = coverage = full = typed = 0
        for i, row in enumerate(data.rows['16.18']):
            options = row['options'][arm]; target = int(labels['main'][i]); hit = options[0]['group'] == target
            first += int(hit); coverage += int(any(option['group'] == target for option in options))
            full += int(hit and not bool(labels['contains_wards'][i]))
            typed += int(hit and (target != 0 or bool(labels['is_no_buy'][i])))
        counts = {'main_top1': first, 'main_top3': coverage, 'complete_top1': full, 'typed_main_top1': typed}
        require(all(value == report['arms'][arm][key]['correct'] for key, value in counts.items()), 'independent native exactness count differs')
        direct[arm] = counts
    data.revalidate(); require(file_hash(args.report) == rh and file_hash(args.model_audit) == ah, 'completed evidence changed')
    output.mkdir(parents=True)
    result = {'schema': 'causal-student-result-audit-v1', 'state': 'complete', 'passed': True,
        'evaluation_report_sha256': rh, 'model_audit_sha256': ah, 'independent_exactness_counts': direct,
        'all_metrics_intervals_and_gates_recomputed': True, 'metrics_recomputed': metrics,
        'metric_gate_passed': metrics['metric_gate_passed'], 'selected_candidate': metrics['selected_candidate'], 'test_payload_reads': False,
        'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'elapsed_s': time.monotonic()-start, 'auditor_sha256': file_hash(__file__),
        'limits': ['Exposed retrospective internal evidence, conditional on frozen models; no forward/live/causal gain or goal completion claimed.']}
    p = output/'report.json'; p.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.with_suffix('.sha256').write_text(file_hash(p)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'metric_gate_passed', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--report', type=Path, required=True)
    p.add_argument('--model-audit', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
