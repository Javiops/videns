"""Independent public antichain spot audit and exact synthetic capacity."""
from __future__ import annotations
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import file_hash
from winprob.route_capacity_v1 import conditional_experts
from winprob.route_marginal_v1 import require, kl_divergence

CENSUS = ROOT/'data/runs/route-static-census-20261003-v1/census/report.json'
PROTOCOL = ROOT/'docs/route_capacity_protocol_v1.md'


def antichain_score(basket, forest, items):
    # No RecipeFrontiers import. Expand individual occurrences and their
    # ancestor/descendant conflicts, then search assignments of <=6 copies.
    nodes = []; ancestors = []; ends = []
    def expand(item, parent_mask):
        idx = len(nodes); nodes.append(item); ancestors.append(parent_mask)
        ends.append(-1)
        for child in items[str(item)].get('from', []):
            expand(int(child), parent_mask | (1 << idx))
        ends[idx] = len(nodes)
    for item, count in forest:
        for _ in range(count):
            expand(item, 0)
    conflicts = [ancestors[i] | (((1 << ends[i])-1) ^ ((1 << i)-1))
                 for i in range(len(nodes))]
    options = []
    for item, count in basket:
        choices = tuple(i for i, node in enumerate(nodes) if node == item)
        options.extend([choices]*count)
    options.sort(key=len)
    @lru_cache(None)
    def search(pos, forbidden):
        if pos == len(options):
            return 0
        best = search(pos+1, forbidden)
        for node in options[pos]:
            if not ((forbidden >> node) & 1):
                best = max(best, 1+search(pos+1, forbidden | conflicts[node]))
        return best
    return search(0, 0)


def run(output):
    start = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'new immutable route audit required')
    require(file_hash(CENSUS) == '08b426d97260521b52c9271c235f705a60c3b899ac9d449fddf67a1a8c951def', 'census changed')
    report = json.loads(CENSUS.read_text(encoding='utf-8'))
    require(report['state'] == 'complete' and not report['game_payload_reads'], 'wrong census stage')
    bindings = {**report['bindings_sha256'], **report['code_sha256'], str(CENSUS): file_hash(CENSUS)}
    code = {str(p.resolve()): file_hash(p) for p in (Path(__file__), PROTOCOL,
            ROOT/'winprob/route_capacity_v1.py', ROOT/'winprob/route_marginal_v1.py',
            ROOT/'winprob/iql_dataset_contract.py')}
    for path, sha in {**bindings, **code}.items():
        require(file_hash(path) == sha, 'bound public input/code changed')
    def public(name):
        paths = [p for p in report['bindings_sha256'] if Path(p).name == name]
        require(len(paths) == 1, 'ambiguous public artifact')
        return json.loads(Path(paths[0]).read_text(encoding='utf-8'))
    catalogue = public('catalogue.json'); baskets = catalogue['group_main']
    definition = public('definition.json'); forests = [[]]+[r['goal_counts'] for r in definition['label_vocabulary'][2:]]
    items = public('boots_static.json')['patches']['16.18']['items']
    matrix_path = CENSUS.parent/'fraction_units.npy'
    require(file_hash(matrix_path) == report['files']['fraction_units.npy'], 'matrix changed')
    matrix = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
    require(matrix.dtype == np.uint8 and matrix.shape == (9333, 258), 'invalid matrix')
    maxima = matrix.max(1); _, counts = np.unique(matrix, axis=0, return_counts=True)
    observed = {'baskets_with_any_match': int((maxima > 0).sum()),
        'nonempty_baskets_fully_matched_by_some_goal': int((maxima == 60).sum()),
        'baskets_with_no_match': int((maxima == 0).sum()),
        'exact_compatibility_equivalence_classes': len(counts),
        'largest_equivalence_class': int(counts.max())}
    for key, value in observed.items():
        require(value == report[key], 'census count/equivalence mismatch: '+key)
    rng = np.random.default_rng(20261003)
    pairs = [(int(i), int(j)) for i, j in zip(rng.integers(9333, size=512), rng.integers(258, size=512))]
    pairs.extend((i, int(matrix[i].argmax())) for i in range(128))
    # Hash-seeded random zeros, rather than letting one background column
    # dominate the additional negative fixtures.
    zero_pairs = np.argwhere(matrix == 0)
    pairs.extend(tuple(map(int, z)) for z in zero_pairs[rng.choice(len(zero_pairs), size=128, replace=False)])
    output.mkdir(parents=True)
    for index, (i, z) in enumerate(pairs):
        require(time.monotonic()-start < 600, 'fixed10minute audit deadline')
        score = antichain_score(baskets[i], forests[z], items)
        size = sum(count for _, count in baskets[i])
        want = score*(60//size) if size else 0
        require(int(matrix[i, z]) == want, 'independent antichain mismatch')
        if index % 128 == 0:
            print(json.dumps({'stage': 'antichain_audit', 'pairs': index+1, 'elapsed_s': time.monotonic()-start}), flush=True)
    raw = np.asarray(matrix, dtype=np.float64)/60
    routing = np.exp(raw-raw.max(1, keepdims=True)); routing /= routing.sum(1, keepdims=True)
    prior = np.full(9333, 1/9333)
    experts, marginal = conditional_experts(prior, routing)
    np.testing.assert_allclose(experts.sum(0), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(experts @ marginal, prior, atol=1e-15, rtol=0)
    kl = np.array([kl_divergence(e, prior) for e in experts.T])
    tv = .5*np.sum(np.abs(experts-prior[:, None]), axis=0)
    synthetic = {'prior': 'uniform, NOT native M2', 'routing': 'softmax(fraction), feasibility only',
        'maximum_attainable_KL': float(kl.max()), 'maximum_attainable_total_variation': float(tv.max()),
        'maximum_action_probability_amplification': float((experts/prior[:, None]).max()),
        'largest_pairwise_routing_ratio': float((routing.max(0)/routing.min(0)).max()),
        'vertices': len(marginal), 'interpretation': 'Exact convex-family capacity for this synthetic prior/kernel; not accuracy or expected gain.'}
    for path, sha in {**bindings, **code}.items():
        require(file_hash(path) == sha, 'bound public input/code changed during audit')
    require(file_hash(matrix_path) == report['files']['fraction_units.npy'], 'matrix changed during audit')
    result = {'schema': 'route-static-audit-v1', 'state': 'complete', 'passed': True,
        'all_matrix_summary_counts_recomputed': observed,
        'independent_antichain_pairs': len(pairs), 'distinct_antichain_pairs': len(set(pairs)),
        'all_matrix_entries_independently_recomputed': False,
        'synthetic_capacity': synthetic, 'bindings_sha256': bindings, 'code_sha256': code,
        'matrix_sha256': report['files']['fraction_units.npy'], 'elapsed_s': time.monotonic()-start,
        'game_payload_reads': False, 'purchase_or_future_labels_read': False,
        'model_loaded_or_trained': False, 'test_or_confirmation_reads': False,
        'accuracy_computed': False, 'breakthrough_established': False}
    path = output/'report.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(file_hash(path)+' report.json\n', encoding='ascii')
    print(json.dumps({'stage': 'complete', 'elapsed_s': time.monotonic()-start, 'synthetic_capacity': synthetic}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
