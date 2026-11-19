"""Replay every correction and independently verify scalers, optima and metrics."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import file_hash
from winprob.recipe_transition_data_v1 import RecipeData, array_hash, value_hash, require
from winprob.recipe_transition_probe_v1 import Standardizer, ARMS, game_weights
from winprob.recipe_transition_v1 import objective
from winprob.recipe_transition_evaluation_v1 import predict_correction, compact_prediction, evaluate


def independent_scaler(data, weights):
    h, f = data.context, data.candidates
    hm = np.sum(h*weights[:, None], axis=0)
    hv = np.sum((h-hm)**2*weights[:, None], axis=0)
    hs = np.sqrt(np.maximum(hv, 0)); hs[hv < 1e-12] = 1.; hm[0] = 0.; hs[0] = 1.
    valid = data.reference[:, :3] > 0
    fw = (weights[:, None]/valid.sum(axis=1, keepdims=True)*valid).reshape(-1)
    flat = f.reshape(-1, f.shape[-1])
    fm = np.sum(flat*fw[:, None], axis=0)
    fv = np.sum((flat-fm)**2*fw[:, None], axis=0)
    fs = np.sqrt(np.maximum(fv, 0)); fs[fv < 1e-12] = 1.
    return Standardizer(hm, hs, fm, fs)


def compare_nested(actual, expected, name='root'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'summary keys differ: '+name)
        for key in expected:
            compare_nested(actual[key], expected[key], name+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'summary length differs: '+name)
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare_nested(a, e, name+'.'+str(i))
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and np.isclose(actual, expected, rtol=1e-12, atol=1e-12), 'numeric summary differs: '+name)
    else:
        require(actual == expected, 'summary differs: '+name)


def run(directory, output):
    started = time.monotonic(); directory = directory.resolve(); output = output.resolve()
    require(directory.is_relative_to(ROOT/'data/runs') and not output.exists(), 'audit path or immutable output violation')
    def guard():
        require(time.monotonic()-started < 1800, 'fixed30minute audit bound reached')
    report_path = directory/'report.json'; report = json.loads(report_path.read_text(encoding='utf-8'))
    rh = file_hash(report_path)
    require((directory/'report.sha256').read_text().split() == [rh, 'report.json']
            and report['schema'] == 'recipe-transition-experiment-v1' and report['state'] == 'complete', 'experiment not complete/bound')
    for flag in ('source_runtime_imported', 'test_payload_reads', 'confirmation_payload_reads', 'outcomes_analyzed', 'promotion_authorized', 'practical_breakthrough_established'):
        require(report[flag] is False, 'experiment boundary changed: '+flag)
    for path, want in report['code_sha256'].items():
        require(file_hash(path) == want, 'consumed experiment code changed')
        snapshot = directory/'code_snapshot'/Path(path).resolve().relative_to(ROOT)
        require(file_hash(snapshot) == want, 'code snapshot differs')
    require(file_hash(directory/'predictions.npz') == report['predictions_sha256']
            and file_hash(directory/'training_lock.json') == report['training_lock_sha256']
            and file_hash(directory/'prediction_lock.json') == report['prediction_lock_sha256'], 'prediction/model locks changed')
    training_lock = json.loads((directory/'training_lock.json').read_text(encoding='utf-8'))
    prediction_lock = json.loads((directory/'prediction_lock.json').read_text(encoding='utf-8'))
    require(training_lock['state'] == 'all_models_frozen' and training_lock['prediction_started'] is False
            and training_lock['models'] == report['models'] and len(report['models']) == 6
            and prediction_lock['state'] == 'predictions_frozen_before_scoring'
            and prediction_lock['predictions_sha256'] == report['predictions_sha256'], 'fit/prediction freeze contract differs')
    data = RecipeData(Path(report['native_manifest_path']).parent); guard()
    require(data.digest == report['native_manifest_sha256'] and data.feature_digests == report['feature_digests'], 'input reproduction differs')
    labels = data.load_targets(directory/'inputs_lock.json')
    with np.load(directory/'predictions.npz', allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    query = data.focal; qfold = data.folds[query]; n = len(query)
    require(n == 2900 and np.array_equal(saved['query_indices'], query)
            and np.array_equal(saved['candidate_groups'], data.group_ids[query]) and np.array_equal(saved['folds'], qfold), 'saved query population changed')
    replay = {arm: {'probabilities': np.zeros((n, 4)), 'log_probabilities': np.full((n, 4), -np.inf),
                    'candidate_order': np.zeros((n, 3), dtype=np.int64)} for arm in ARMS}
    scaler_error = 0.; optimality = {}; models = 0
    for fold in (0, 1):
        train = np.flatnonzero(data.folds != fold); local = np.flatnonzero(qfold == fold); held = query[local]
        require(not set(data.game_ids[train]) & set(data.game_ids[held]), 'fit/evaluation game overlap')
        train_inputs = data.batch(train); weights = game_weights(data.game_ids[train])
        independent = independent_scaler(train_inputs, weights)
        for arm in ARMS:
            guard(); name = f'correction_{arm}_fold{fold}.json'
            require(file_hash(directory/name) == report['models'][name], 'saved coefficient artifact changed')
            spec = json.loads((directory/name).read_text(encoding='utf-8'))
            require(spec['evaluation_fold'] == fold and spec['arm'] == arm and spec['train_indices_sha256'] == array_hash(train)
                    and spec['train_game_ids_sha256'] == value_hash(sorted(set(data.game_ids[train])))
                    and spec['train_game_count'] == len(set(data.game_ids[train])) and spec['train_state_count'] == len(train), 'model trained on wrong fold/population')
            require(spec['fit']['ridge'] == .01 and spec['fit']['converged'] and spec['fit']['parameters'] == 676, 'fixed fit settings changed')
            scaler = Standardizer.from_dict(spec['standardizer'])
            for key in scaler.__dataclass_fields__:
                difference = float(np.max(np.abs(getattr(scaler, key)-getattr(independent, key))))
                scaler_error = max(scaler_error, difference)
                require(difference <= 2e-12, 'independent train-only scaler disagrees')
            theta = np.asarray(spec['coefficients'], dtype=np.float64)
            require(theta.shape == (26, 26) and np.isfinite(theta).all(), 'coefficient shape/finite check failed')
            if arm == 'raw':
                require(np.array_equal(theta[:, 16:], np.zeros((26, 10))), 'raw control uses net channels')
            transformed = scaler.transform(train_inputs, arm, data.permutations[train] if arm == 'shuffled_net' else None)
            loss, gradient = objective(theta.ravel(), transformed, labels['coarsened'][train], weights, ridge=.01)
            stationarity = float(np.max(np.abs(gradient)))
            require(stationarity < 1e-6 and abs(loss-spec['fit']['objective']) < 1e-12
                    and abs(stationarity-spec['fit']['gradient_max_abs']) < 1e-12, 'saved model fails convex optimality receipt')
            optimality[name] = {'objective': loss, 'gradient_max_abs': stationarity}
            result = predict_correction(data.batch(held), data.permutations[held], scaler, theta, arm)
            for key, value in result.items():
                replay[arm][key][local] = value
            models += 1
            del transformed
        del train_inputs
    prior = data.inputs.reference[query]
    log_prior = np.full_like(prior, -np.inf); np.log(prior, out=log_prior, where=prior > 0)
    replay['m2'] = {'probabilities': prior.copy(), 'log_probabilities': log_prior,
                    'candidate_order': np.tile([0, 1, 2], (n, 1))}
    for arm, values in replay.items():
        for key, value in values.items():
            require(np.array_equal(value, saved[f'{arm}__{key}']), 'exact prediction replay differs: '+arm+'/'+key)
    focal_labels = {key: value[query] for key, value in labels.items()}
    compact = {arm: compact_prediction(p, data.group_ids[query], focal_labels['coarsened']) for arm, p in replay.items()}
    metrics = evaluate(focal_labels, compact, data.group_ids[query], data.game_ids[query].tolist(), qfold)
    for key, value in metrics.items():
        compare_nested(report[key], value, key)
    data.revalidate(); guard()
    output.mkdir(parents=True)
    result = {'schema': 'recipe-transition-result-audit-v1', 'state': 'complete', 'passed': True,
        'experiment_report_sha256': rh, 'native_manifest_sha256': data.digest,
        'models_checked': models, 'correction_queries_replayed': 3*n, 'baseline_queries_verified': n,
        'exact_probability_logp_order_replay': True, 'maximum_independent_scaler_error': scaler_error,
        'convex_stationarity': optimality, 'metric_gate_passed': metrics['metric_gate_passed'],
        'metrics_recomputed': metrics, 'source_runtime_imported': False, 'test_payload_reads': False,
        'confirmation_payload_reads': False, 'elapsed_s': time.monotonic()-started,
        'auditor_sha256': file_hash(__file__),
        'limits': ['Replays every own correction, not every source M2 neural forward pass.',
                   'Source native recommendations/receipts are bound artifacts with a200-state mutation check.',
                   'Internal exposed retrospective screen, not a forward M2 deployment result.']}
    out = output/'report.json'
    out.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'report.sha256').write_text(file_hash(out)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'models_checked', 'correction_queries_replayed', 'metric_gate_passed', 'elapsed_s')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--directory', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.directory, a.output)
