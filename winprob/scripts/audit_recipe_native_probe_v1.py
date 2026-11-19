"""Consume only synthetic source receipts and benchmark the own convex solver."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.recipe_transition_v1 import (
    CANDIDATE_NAMES, CONTEXT_NAMES, ChoiceInputs, fit, objective, predict, transition_features)

STATIC = ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'
STATIC_SHA = '5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def run(source, output):
    start = time.monotonic(); source = source.resolve(); output = output.resolve()
    require(source.is_relative_to(ROOT.parent/'build_tracker/data/exports/winprob-recipe-transition'), 'source artifact scope')
    require(not output.exists(), 'new immutable audit output required')
    report_path = source/'report.json'; raw = json.loads(report_path.read_text(encoding='utf-8'))
    sha = digest(report_path)
    require(sha == (source/'report.sha256').read_text().split()[0], 'source report sidecar changed')
    require(raw['schema'] == 'recipe-transition-native-probe-v2' and raw['state'] == 'complete'
            and raw['passed'] and raw['same_input_exact_replay'] and raw['deterministic_algorithms'], 'native gates failed')
    for flag in ('game_payload_reads', 'target_file_opened', 'network_reads', 'training_performed', 'consumer_runtime_imported'):
        require(raw[flag] is False, 'synthetic boundary failed: '+flag)
    for path, want in raw['code_sha256'].items():
        require(digest(path) == want, 'source code changed since probe')
    receipts_path = source/'receipts.json'
    require(digest(receipts_path) == raw['files']['receipts.json'], 'synthetic receipts changed')
    payload = json.loads(receipts_path.read_text(encoding='utf-8'))
    require(payload['schema'] == 'synthetic-native-recipe-receipts-v1' and payload['world'] == '16.18', 'wrong receipt scope')
    require(digest(STATIC) == STATIC_SHA, 'pinned own statics changed')
    static = json.loads(STATIC.read_text(encoding='utf-8'))['patches']['16.18']['items']
    for iid, item in payload['native_items'].items():
        require(item['stats'] == static[iid]['stats'], 'native/own item-stat mismatch')
    arrays = {}
    for kind in ('receipts', 'fixtures'):
        arrays[kind] = np.stack([transition_features(r['before'], r['action'], r['wallet'], r['steps'], static)
                                for r in payload[kind]])
    require(len(arrays['receipts']) == raw['receipt_count'] and len(arrays['fixtures']) == raw['fixture_count'] == 6,
            'synthetic receipt coverage changed')
    example = {name: float(value) for name, value in zip(CANDIDATE_NAMES, arrays['fixtures'][0], strict=True) if value}

    # Bounded, fully synthetic timing. Never a performance-on-purchases number.
    rng = np.random.default_rng(20261001); n = 8192
    prior = rng.uniform(.1, 1, size=(n, 4)); prior /= prior.sum(1, keepdims=True)
    h = rng.normal(size=(n, len(CONTEXT_NAMES))); h[:, 0] = 1
    f = rng.normal(size=(n, 3, len(CANDIDATE_NAMES)))
    data = ChoiceInputs(prior, h, f).validate(); known = rng.normal(scale=.01, size=data.parameter_shape)
    generated, _ = predict(data, known)
    uniforms = rng.random(n)
    labels = (uniforms[:, None] > np.cumsum(generated, axis=1)).sum(1)
    weights = np.ones(n); zero = np.zeros(np.prod(data.parameter_shape))
    objective(zero, data, labels, weights)
    timings = []
    for _ in range(5):
        t = time.monotonic(); objective(zero, data, labels, weights); timings.append(time.monotonic()-t)
    t = time.monotonic(); theta, fitted = fit(data, labels, weights)
    fit_seconds = time.monotonic()-t
    require(fitted['converged'] and fitted['gradient_max_abs'] < 1e-6, 'synthetic convex fit did not converge')
    require(fitted['objective'] < objective(zero, data, labels, weights)[0], 'synthetic fit did not improve its objective')
    output.mkdir(parents=True)
    result = {'schema': 'recipe-transition-synthetic-audit-v1', 'state': 'complete', 'passed': True,
        'source_report_path': str(report_path), 'source_report_sha256': sha,
        'source_receipts_sha256': raw['files']['receipts.json'], 'own_static_sha256': STATIC_SHA,
        'verified_native_candidate_receipts': len(arrays['receipts']), 'verified_native_fixtures': 6,
        'verified_item_stat_records': len(payload['native_items']), 'native_world': '16.18',
        'steelcaps_from_boots_and_cloth_descriptor_example': example,
        'synthetic_solver': {'rows': n, 'parameters': len(zero), 'objective_gradient_seconds': timings,
            'fit_seconds': fit_seconds, 'fit': fitted, 'input_array_bytes': sum(a.nbytes for a in (prior, h, f))},
        'code_sha256': {str(p): digest(p) for p in (Path(__file__), ROOT/'winprob/recipe_transition_v1.py',
                                                  ROOT/'winprob/ipo_residual.py')},
        'game_payload_reads': False, 'real_purchase_fitting': False, 'source_runtime_imported': False,
        'elapsed_s': time.monotonic()-start,
        'limits': ['Synthetic numerical and operational validation only; no measured M2 purchase gain.',
                   'Descriptors exclude passives and do not equal effective champion combat statistics.',
                   'Receipt conservation is verified; native engine legality is inherited from source execution.']}
    path = output/'report.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    (output/'report.sha256').write_text(digest(path)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'verified_native_candidate_receipts', 'synthetic_solver', 'elapsed_s')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
