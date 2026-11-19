"""Independent scaler/marginal checks, model replay, stationarity and metrics."""
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
from winprob.recipe_transition_data_v1 import require, array_hash, value_hash
from winprob.route_probe_io_v1 import CompletedRoute
from winprob.route_probe_v1 import ARMS, kernels, ContextScaler, objective, predict, normalized_prior
from winprob.route_probe_data_v1 import CENSUS, PROTOCOL_SHA
from winprob.route_probe_evaluation_v1 import evaluate
from winprob.bounded_npy_rows_v1 import BoundedNpyRows


def compare(actual, expected, path='root'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'audit keys differ: '+path)
        for key in expected: compare(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'audit list differs: '+path)
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)): compare(a, e, path+'.'+str(i))
    elif isinstance(expected, float):
        require(np.isclose(actual, expected, atol=1e-12, rtol=1e-12), 'audit numeric value differs: '+path)
    else:
        require(actual == expected, 'audit value differs: '+path)


def run(report_path, output):
    start = time.monotonic(); report_path = report_path.resolve(); output = output.resolve()
    require(not output.exists() and report_path.is_relative_to(ROOT/'data/runs'), 'new own audit output required')
    report, report_sha = bound_json(report_path, {})
    require(report['schema'] == 'route-probe-evaluation-v1' and report['state'] == 'complete'
            and report['practical_breakthrough_established'] is False, 'completed development report required')
    for p, sha in report['code_sha256'].items(): require(file_hash(p) == sha, 'scoring code changed')
    src = report['source_manifests']
    data = CompletedRoute(report['fit_directory'], src['16.18']['manifest_path'], src['16.17']['manifest_path'])
    require(data.fit_sha == report['fit_report_sha256'] and data.dist_sha == report['distribution_manifest_sha256'], 'scored fit/distribution changed')
    for world in src: require(data.source[world]['manifest_sha256'] == src[world]['manifest_sha256'], 'scored source manifest changed')
    labels, full_labels = data.load_targets()
    cache = data.directory/'input_cache'; feature = data.input_lock['route_feature_digests']
    for name, sha in feature['cache_files'].items():
        require((cache/name).resolve().parent == cache and file_hash(cache/name) == sha, 'training cache changed')
        data.bindings[str(cache/name)] = sha
    public_matrix = CENSUS.parent/'fraction_units.npy'
    require(file_hash(public_matrix) == 'd5157ae14c1b7267306686dc3bdcca32fd2f40b1442375cd4736db6d402cf572', 'public matrix changed')
    data.bindings[str(public_matrix)] = file_hash(public_matrix)
    raw = np.load(public_matrix, mmap_mode='r', allow_pickle=False)
    sizes = np.array([sum(c for _, c in b) for b in data.base.groups], dtype=np.int32)
    rebuilt, permutation = kernels(raw, sizes)
    require(np.array_equal(permutation, np.load(cache/'row_permutation.npy', allow_pickle=False)), 'control permutation does not reproduce')
    for arm in ARMS:
        require(np.array_equal(rebuilt[arm], np.load(cache/f'kernel_{arm}.npy', allow_pickle=False))
                and array_hash(rebuilt[arm]) == feature['kernels'][arm], 'public kernel does not reproduce')
    # Full-prior row hashes were verified by the source exporter. Recheck the
    # whole bound array and every derived route marginal, not new M2 forwards.
    prior_path = Path(data.dist['prior_manifest_path']).parent/'full_probability.npy'
    require(file_hash(prior_path) == data.prior['files'][prior_path.name], 'full source prior changed')
    data.bindings[str(prior_path)] = data.prior['files'][prior_path.name]
    prior = BoundedNpyRows(prior_path, expected_shape=(56288, 9333), expected_dtype=np.float32)
    marginals = {a: np.load(cache/f'marginal_{a}.npy', mmap_mode='r', allow_pickle=False) for a in ARMS}
    max_marginal_error = 0.
    def guard(): require(time.monotonic()-start < 3600, 'fixed60minute route audit deadline')
    for lo in range(0, len(prior), 64):
        guard(); p = normalized_prior(prior[lo:min(lo+64, len(prior))])
        for arm in ARMS:
            # Different64-row audit batches versus128-row preparation.
            expected = p @ rebuilt[arm]; got = marginals[arm][lo:lo+64]
            error = float(np.max(np.abs(expected-got))); max_marginal_error = max(max_marginal_error, error)
            require(error < 2e-14, 'full-catalogue route marginal differs')
    independent_scales = {}; weights_by_fold = {}; max_scale_error = 0.; checks = {}
    replayed = 0; direct_formula_error = 0.
    for fold in (0, 1):
        guard(); train = np.flatnonzero(data.base.folds != fold); positions = np.flatnonzero(data.folds == fold)
        require(not set(data.base.game_ids[train]) & set(data.base.game_ids[data.query[positions]]), 'whole-game OOF overlap')
        # Independently construct equal-game weights and moments using sums.
        ids = data.base.game_ids[train]; counts = {g: int(np.sum(ids == g)) for g in set(ids)}
        w = np.array([1/counts[g] for g in ids], dtype=np.float64); w /= w.sum(); weights_by_fold[fold] = w
        x = data.base.inputs.context[train]; mean = np.sum(x*w[:, None], axis=0)
        var = np.sum((x-mean)**2*w[:, None], axis=0); scale = np.sqrt(np.maximum(var, 0))
        scale[var < 1e-12] = 1.; mean[0] = 0.; scale[0] = 1.
        independent_scales[fold] = (mean, scale)
        for arm_index, arm in enumerate(ARMS):
            name = f'route_{arm}_fold{fold}.json'; spec = json.loads((data.directory/name).read_text(encoding='utf-8'))
            receipt = spec['fit']; saved_scale = ContextScaler.from_dict(spec['scaler'])
            require(spec['arm'] == arm and spec['evaluation_fold'] == fold and spec['protocol_sha256'] == PROTOCOL_SHA
                    and spec['prior_manifest_sha256'] == data.prior_sha and spec['train_indices_sha256'] == array_hash(train)
                    and spec['train_game_ids_sha256'] == value_hash(sorted(set(ids)))
                    and spec['train_states'] == len(train) and spec['train_games'] == len(set(ids))
                    and spec['scaler'] == data.scale_lock['scalers'][str(fold)], 'fitted identity/scaler changed')
            for a, b in ((mean, saved_scale.mean), (scale, saved_scale.scale)):
                error = float(np.max(np.abs(a-b))); max_scale_error = max(max_scale_error, error)
                require(error <= 2e-12, 'independent training-only moments differ')
            require(receipt['success'] and receipt['parameters'] == 6708 and receipt['ridge'] == .01/258
                    and receipt['maxiter'] == 500 and receipt['gtol'] == 1e-7 and receipt['ftol'] == 0
                    and receipt['maxls'] == 50 and receipt['initialization'] == 'zero'
                    and receipt['global_optimum_claimed'] is False, 'fixed fitting recipe changed')
            theta = np.asarray(spec['coefficients'], dtype=np.float64); require(theta.shape == (26, 258), 'model dimensions changed')
            transformed = saved_scale.transform(x); m = marginals[arm][train]; y = rebuilt[arm][full_labels['main'][train]]
            value, gradient = objective(theta.ravel(), transformed, m, y, w, ridge=.01/258)
            maximum = float(np.max(np.abs(gradient)))
            require(maximum < 1e-6 and abs(maximum-receipt['gradient_max_abs']) <= 2e-12
                    and abs(value-receipt['relative_regularized_objective']) <= 2e-12, 'stationarity/relative objective receipt differs')
            # Directional finite differences independently check calculus at
            # each real saved solution; they make no global-optimum claim.
            rng = np.random.default_rng(20261003+fold*2+arm_index); direction = rng.normal(size=theta.size)
            direction /= np.linalg.norm(direction); eps = 1e-4
            plus, _ = objective(theta.ravel()+eps*direction, transformed, m, y, w, ridge=.01/258)
            minus, _ = objective(theta.ravel()-eps*direction, transformed, m, y, w, ridge=.01/258)
            derivative_error = abs((plus-minus)/(2*eps)-float(gradient @ direction))
            require(derivative_error < 1e-8, 'real-model directional derivative failed')
            checks[name] = {'relative_regularized_objective': value, 'gradient_max_abs': maximum,
                            'directional_derivative_error': derivative_error, 'global_optimum_claimed': False}
            del m, y, transformed
            for lo in range(0, len(positions), 32):
                guard(); local = positions[lo:lo+32]; source_indices = data.query[local]
                original = np.asarray(prior[source_indices]); context = saved_scale.transform(data.base.inputs.context[source_indices])
                require(np.array_equal(original, data.matrices['m2'][local]), 'original focal prior differs')
                q, neutral = predict(original, rebuilt[arm], context, theta)
                require(np.array_equal(q, data.matrices[arm][local])
                        and np.array_equal(neutral, data.neutral[arm_index, local]), 'exact route prediction/neutrality replay differs')
                # Alternative direct exp/logit form, independent of alpha/bridge.
                p = normalized_prior(original); u = context @ theta; e = np.exp(u-u.max(1, keepdims=True))
                direct = p*(e @ rebuilt[arm].T)/np.sum((p @ rebuilt[arm])*e, axis=1, keepdims=True)
                error = float(np.max(np.abs(direct-q))); direct_formula_error = max(direct_formula_error, error)
                require(error < 2e-14, 'independent direct route formula differs')
                replayed += len(local)
    scores = data.scores(labels)
    metrics = evaluate(labels, scores, data.base.game_ids[data.query].tolist(), data.folds, data.legality)
    for key, expected in metrics.items(): compare(report[key], expected, key)
    # Independent Python counts from native option records, including full/kind.
    independent_counts = {}
    for arm in ('m2', *ARMS):
        first = []; top3 = []; complete = []; typed = []
        for i, row in enumerate(data.rows['16.18']):
            chosen = row['options'][arm]; hit = chosen[0]['group'] == int(labels['main'][i])
            first.append(hit); top3.append(any(o['group'] == int(labels['main'][i]) for o in chosen))
            complete.append(hit and not bool(labels['contains_wards'][i]))
            typed.append(hit and (int(labels['main'][i]) != 0 or bool(labels['is_no_buy'][i])))
        counts = dict(zip(('main_top1', 'main_top3', 'complete_top1', 'typed_main_top1'), map(sum, (first, top3, complete, typed))))
        require(all(counts[k] == report['arms'][arm][k]['correct'] for k in counts), 'independent exactness counts differ')
        independent_counts[arm] = counts
    data.revalidate(); require(file_hash(report_path) == report_sha, 'scored report changed during audit')
    output.mkdir(parents=True)
    result = {'schema': 'route-probe-result-audit-v1', 'state': 'complete', 'passed': True,
        'evaluation_report_sha256': report_sha, 'models_verified': 4, 'correction_queries_replayed_exactly': replayed,
        'baseline_queries_verified': 2900, 'all56288_route_marginals_recomputed': True,
        'maximum_marginal_error': max_marginal_error, 'maximum_independent_scaler_error': max_scale_error,
        'maximum_independent_direct_formula_error': direct_formula_error, 'stationarity_checks': checks,
        'independent_exactness_counts': independent_counts, 'metrics_recomputed': metrics,
        'metric_gate_passed': metrics['metric_gate_passed'], 'test_payload_reads': False,
        'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'elapsed_s': time.monotonic()-start, 'auditor_sha256': file_hash(__file__),
        'limits': ['Replays own route calculations against bound native probabilities and decoder receipts; no fresh source neural forwards or native decoder rerun.',
                   'No global optimum, forward/live/causal gain, or goal-completion claim.']}
    target = output/'report.json'; target.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    target.with_suffix('.sha256').write_text(file_hash(target)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'correction_queries_replayed_exactly', 'metric_gate_passed', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--report', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.report, a.output)
