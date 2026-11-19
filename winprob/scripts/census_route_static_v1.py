"""Public-only recipe-frontier geometry and synthetic marginal-update timing."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.full_action_catalogue import load_catalogue
from winprob.route_marginal_v1 import RecipeFrontiers, bridge, joint_bridge, kl_divergence, require

CATALOGUE = ROOT.parent/'build_tracker/data/exports/winprob-action-catalogue/full-action-catalogue-20260930-v1/catalogue.json'
STATIC = ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'
DEFINITION = ROOT/'artifacts/goal-definition/20260930-v2/definition.json'
PROTOCOL = ROOT/'docs/route_static_census_protocol_v1.md'


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n', encoding='ascii')


def run(output):
    started = time.monotonic(); output = output.resolve(); require(not output.exists(), 'new immutable static census required')
    bindings = {}
    expected = {CATALOGUE: '14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8',
        STATIC: '5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0',
        DEFINITION: '4457a015b1d08a0893acfc19b8016b7a7104d98f220f2c893949d055144a9072'}
    values = {}
    for path, want in expected.items():
        value, sha = bound_json(path, bindings); require(sha == want, 'public artifact changed'); values[path] = value
    catalogue, _ = load_catalogue(CATALOGUE); definition = values[DEFINITION]
    require(definition['future_action_payloads_read'] is False and definition['training_performed'] is False, 'definition was not public-only')
    baskets = catalogue['group_main']; forests = [[]]+[r['goal_counts'] for r in definition['label_vocabulary'][2:]]
    require(len(baskets) == 9333 and len(forests) == 258 and forests[0] == [], 'fixed public ontology changed')
    items = values[STATIC]['patches']['16.18']['items']
    roots = {i for b in baskets+forests for i, _ in b}; recipes = {}; pending = list(roots)
    while pending:
        node = pending.pop()
        if node in recipes:
            continue
        require(str(node) in items, 'native public item missing')
        raw = items[str(node)].get('from', [])
        require(isinstance(raw, list) and all(isinstance(v, str) and v.isdecimal() for v in raw), 'invalid listed recipe')
        children = tuple(int(v) for v in raw); recipes[node] = children; pending.extend(children)
    matcher = RecipeFrontiers(recipes); output.mkdir(parents=True)
    code = {str(p.resolve()): file_hash(p) for p in (Path(__file__), PROTOCOL, ROOT/'docs/route_marginal_design_v1.md',
            ROOT/'winprob/route_marginal_v1.py', ROOT/'winprob/full_action_catalogue.py', ROOT/'winprob/iql_dataset_contract.py')}
    def status(stage, **extra):
        require(time.monotonic()-started < 1200, 'fixed20minute static worker bound reached')
        value = {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra}
        (output/'status.json').write_text(json.dumps(value)+'\n', encoding='utf-8'); print(json.dumps(value), flush=True)
    matrix = np.lib.format.open_memmap(output/'fraction_units.npy', mode='w+', dtype=np.uint8, shape=(9333, 258))
    timings = []
    for i, basket in enumerate(baskets):
        t = time.monotonic(); match = matcher.match_many(basket, forests); size = sum(c for _, c in basket)
        matrix[i] = match*(60//size) if size else np.zeros(258, dtype=np.uint8)
        require(not size or np.max(match) <= size, 'frontier double-counted the basket')
        if i < 128:
            timings.append(time.monotonic()-t)
        if i == 127:
            remaining = 1200-(time.monotonic()-started); projected = 1.5*float(np.mean(timings))*9333+120
            preflight = {'passed': projected <= .8*remaining, 'projected_seconds': projected, 'remaining_seconds': remaining,
                         'timed_public_baskets': 128, 'mean_seconds': float(np.mean(timings))}
            write(output/'preflight.json', preflight); require(preflight['passed'], 'full public census exceeds resource bound')
        if i % 500 == 0:
            status('compiling_public_geometry', baskets=i+1)
    matrix.flush()
    classes = defaultdict(list)
    for i, row in enumerate(matrix):
        classes[row.tobytes()].append(i)
    examples = []
    for group in sorted(classes.values(), key=lambda r: (-len(r), r[0]))[:8]:
        examples.append({'equivalent_baskets': len(group), 'examples': [{'main_group': i, 'items': baskets[i]} for i in group[:3]]})
    # A synthetic prior/route belief only; not any match or model inference.
    score = np.asarray(matrix, dtype=np.float64)/60
    r = np.exp(score-score.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    p = np.full((1, 9333), 1/9333, dtype=np.float64); prior_goal = p @ r
    neutral, _ = bridge(p, r, prior_goal); require(np.array_equal(neutral, p), 'neutral whole-catalogue identity failed')
    rng = np.random.default_rng(20261003); alpha = rng.dirichlet(np.ones(258))[None, :]
    measurements = []
    for _ in range(8):
        t = time.monotonic(); updated, _ = bridge(p, r, alpha); measurements.append(time.monotonic()-t)
    before, after, q, m = joint_bridge(p[0], r, alpha[0])
    goal_kl = kl_divergence(alpha[0], m); joint_kl = kl_divergence(after, before); action_kl = kl_divergence(q, p[0])
    require(abs(goal_kl-joint_kl) <= 1e-11 and action_kl <= goal_kl+1e-11, 'synthetic KL identity/contraction failed')
    maxima = np.asarray(matrix).max(1)
    for path, want in {**bindings, **code}.items():
        require(file_hash(path) == want, 'public inputs/code changed')
    report = {'schema': 'route-static-census-v1', 'state': 'complete', 'baskets': 9333, 'route_forests': 258,
        'nonempty_major_item_roots': 106, 'reachable_recipe_nodes': len(recipes), 'matrix_bytes': matrix.nbytes,
        'baskets_with_any_match': int((maxima > 0).sum()), 'nonempty_baskets_fully_matched_by_some_goal': int((maxima == 60).sum()),
        'baskets_with_no_match': int((maxima == 0).sum()), 'exact_compatibility_equivalence_classes': len(classes),
        'largest_equivalence_class': max(map(len, classes.values())), 'public_equivalence_examples': examples,
        'synthetic_bridge': {'batch': 1, 'full_catalogue_actions': 9333, 'timings_seconds': measurements,
            'neutral_exact': True, 'goal_KL': goal_kl, 'joint_KL': joint_kl, 'action_KL': action_kl},
        'bindings_sha256': bindings, 'code_sha256': code, 'files': {'fraction_units.npy': file_hash(output/'fraction_units.npy')},
        'game_payload_reads': False, 'purchase_or_future_labels_read': False, 'model_loaded_or_trained': False,
        'source_runtime_imported': False, 'test_or_confirmation_reads': False, 'accuracy_computed': False,
        'elapsed_s': time.monotonic()-started,
        'limits': ['Public ontology geometry and synthetic timing only, not action frequencies or attainable gains.',
                   'Old major-item route ontology excludes boots and is not a complete intent ontology.',
                   'No real routing table, route predictor or policy experiment is selected here.']}
    write(output/'report.json', report); status('complete', equivalence_classes=len(classes), unmatched_baskets=report['baskets_with_no_match'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
