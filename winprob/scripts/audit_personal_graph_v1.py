"""Independent RMS/optimality verification and exact four-model prediction replay."""
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
from winprob.recipe_transition_data_v1 import array_hash, value_hash, require
from winprob.recipe_transition_probe_v1 import game_weights
from winprob.recipe_transition_v1 import objective
from winprob.personal_graph_data_v1 import GraphData
from winprob.personal_graph_models_v1 import ARMS, RMSScaler, predict_graph
from winprob.personal_graph_evaluation_v1 import compact_prediction, evaluate


def independent_rms(inputs, weights):
    h = np.sqrt(np.maximum(np.sum(inputs.context**2*weights[:, None], axis=0), 0))
    present = inputs.reference[:, :3] > 0
    fw = (weights[:, None]/present.sum(1, keepdims=True)*present).reshape(-1)
    f = np.sqrt(np.maximum(np.sum(inputs.candidates.reshape(-1, 16)**2*fw[:, None], axis=0), 0))
    h[h < 1e-12] = 1.; f[f < 1e-12] = 1.
    return RMSScaler(h, f)


def compare(actual, expected, path='root'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and actual.keys() == expected.keys(), 'summary keys differ: '+path)
        for key in expected:
            compare(actual[key], expected[key], path+'.'+key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'summary list differs: '+path)
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, path+'.'+str(i))
    elif isinstance(expected, float):
        require(np.isclose(actual, expected, atol=1e-12, rtol=1e-12), 'summary value differs: '+path)
    else:
        require(actual == expected, 'summary differs: '+path)


def run(directory, output):
    started = time.monotonic(); directory = directory.resolve(); output = output.resolve()
    require(directory.is_relative_to(ROOT/'data/runs') and not output.exists(), 'audit path/immutable output violation')
    rp = directory/'report.json'; report = json.loads(rp.read_text(encoding='utf-8')); rh = file_hash(rp)
    require((directory/'report.sha256').read_text().split() == [rh, 'report.json']
            and report['state'] == 'complete' and report['schema'] == 'personal-graph-experiment-v1', 'experiment incomplete or unbound')
    for key in ('source_runtime_imported', 'test_payload_reads', 'confirmation_payload_reads', 'outcomes_analyzed',
                'promotion_authorized', 'practical_breakthrough_established'):
        require(report[key] is False, 'experiment boundary failed: '+key)
    for raw, expected in report['code_sha256'].items():
        require(file_hash(raw) == expected and file_hash(directory/'code_snapshot'/Path(raw).resolve().relative_to(ROOT)) == expected, 'consumed code/snapshot changed')
    for name, field in (('predictions.npz', 'predictions_sha256'), ('models_lock.json', 'models_lock_sha256'),
                        ('predictions_lock.json', 'predictions_lock_sha256'), ('inputs_lock.json', 'inputs_lock_sha256')):
        require(file_hash(directory/name) == report[field], 'lock or prediction file changed')
    model_lock = json.loads((directory/'models_lock.json').read_text(encoding='utf-8'))
    pred_lock = json.loads((directory/'predictions_lock.json').read_text(encoding='utf-8'))
    scale_lock = json.loads((directory/'scales_lock.json').read_text(encoding='utf-8'))
    require(model_lock['state'] == 'all_models_frozen_before_prediction' and model_lock['models'] == report['models']
            and len(report['models']) == 4 and pred_lock['state'] == 'predictions_frozen_before_scoring'
            and pred_lock['sha256'] == report['predictions_sha256'] and scale_lock['targets_opened'] is False, 'freeze ordering contract differs')
    data = GraphData(); require(data.graph_feature_digests == report['graph_feature_digests'], 'new profile tensors do not reproduce')
    labels = data.load_targets(directory/'inputs_lock.json')
    with np.load(directory/'predictions.npz', allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    query = data.focal; qfold = data.folds[query]; n = len(query)
    require(n == 2900 and np.array_equal(saved['query_indices'], query) and np.array_equal(saved['candidate_groups'], data.group_ids[query])
            and np.array_equal(saved['folds'], qfold), 'saved focal population changed')
    predictions = {arm: {'probabilities': np.zeros((n, 4)), 'log_probabilities': np.full((n, 4), -np.inf),
                         'candidate_order': np.zeros((n, 3), dtype=np.int64)} for arm in ARMS}
    zero_profiles = {arm: data.zero_profiles[arm][query] for arm in ARMS}; max_scale_error = 0.; checks = {}
    for fold in (0, 1):
        train = np.flatnonzero(data.folds != fold); local = np.flatnonzero(qfold == fold); held = query[local]
        require(not set(data.game_ids[train]) & set(data.game_ids[held]), 'train/evaluation game overlap')
        weights = game_weights(data.game_ids[train]); common_scale = None
        for arm in ARMS:
            require(time.monotonic()-started < 2700, 'fixed45minute audit limit reached')
            name = f'personal_graph_{arm}_fold{fold}.json'; require(file_hash(directory/name) == report['models'][name], 'model file changed')
            spec = json.loads((directory/name).read_text(encoding='utf-8')); receipt = spec['fit']
            require(spec['arm'] == arm and spec['evaluation_fold'] == fold and spec['train_indices_sha256'] == array_hash(train)
                    and spec['train_game_ids_sha256'] == value_hash(sorted(set(data.game_ids[train])))
                    and spec['train_games'] == len(set(data.game_ids[train])) and spec['train_states'] == len(train), 'wrong fitted fold or identity')
            require(receipt['solver_version'] == 'loss-stop-disabled-v2' and receipt['solver'] == 'L-BFGS-B'
                    and receipt['ridge'] == .01 and receipt['ftol'] == 0. and receipt['gtol'] == 1e-7
                    and receipt['max_iter'] == 300 and receipt['maxls'] == 50 and receipt['parameters'] == 272
                    and receipt['converged'], 'fixed solver settings changed')
            inputs = data.batch(train, arm); independent = independent_rms(inputs, weights); scaler = RMSScaler.from_dict(spec['scaler'])
            require(spec['scaler'] == scale_lock['scales'][f'{fold}:{arm}'], 'scaler changed after target read')
            for key in scaler.__dataclass_fields__:
                error = float(np.max(np.abs(getattr(scaler, key)-getattr(independent, key)))); max_scale_error = max(max_scale_error, error)
                require(error <= 2e-12, 'independent uncentered RMS calculation differs')
            if common_scale is not None:
                require(np.array_equal(common_scale, scaler.candidate_scale), 'candidate scales differ between arms')
            common_scale = scaler.candidate_scale
            theta = np.asarray(spec['coefficients'], dtype=np.float64); require(theta.shape == (17, 16), 'coefficient axes changed')
            loss, grad = objective(theta.ravel(), scaler.transform(inputs), labels['coarsened'][train], weights, ridge=.01)
            maximum = float(np.max(np.abs(grad)))
            require(maximum < 1e-6 and abs(loss-receipt['objective']) < 1e-12
                    and abs(maximum-receipt['gradient_max_abs']) < 1e-12, 'convex stationary-point receipt differs')
            checks[name] = {'objective': loss, 'gradient_max_abs': maximum}
            pred = predict_graph(data.batch(held, arm), scaler, theta)
            for key, value in pred.items():
                predictions[arm][key][local] = value
    prior = data.reference[query]; lp = np.full_like(prior, -np.inf); np.log(prior, out=lp, where=prior > 0)
    predictions['m2'] = {'probabilities': prior.copy(), 'log_probabilities': lp, 'candidate_order': np.tile([0, 1, 2], (n, 1))}
    for arm, values in predictions.items():
        for key, value in values.items():
            require(np.array_equal(value, saved[arm+'__'+key]), 'exact prediction replay differs: '+arm+'/'+key)
        if arm in ARMS:
            zero = zero_profiles[arm]
            require(np.array_equal(zero, saved[arm+'__zero_profile']) and np.array_equal(values['probabilities'][zero], prior[zero])
                    and np.array_equal(values['log_probabilities'][zero], lp[zero])
                    and np.array_equal(values['candidate_order'][zero], predictions['m2']['candidate_order'][zero]), 'zero-profile reference parity failed')
    focal_labels = {key: value[query] for key, value in labels.items()}
    compact = {arm: compact_prediction(p, data.group_ids[query], focal_labels['coarsened']) for arm, p in predictions.items()}
    metrics = evaluate(focal_labels, compact, data.group_ids[query], data.game_ids[query].tolist(), qfold, zero_profiles)
    for key, expected in metrics.items():
        compare(report[key], expected, key)
    data.revalidate(); output.mkdir(parents=True)
    result = {'schema': 'personal-graph-result-audit-v1', 'state': 'complete', 'passed': True,
        'experiment_report_sha256': rh, 'models_verified': len(checks), 'correction_queries_replayed': 2*n,
        'baseline_queries_verified': n, 'exact_probability_logp_order_replay': True,
        'zero_profile_exact_reference_parity': True, 'maximum_independent_RMS_error': max_scale_error,
        'convex_stationarity': checks, 'metric_gate_passed': metrics['metric_gate_passed'], 'metrics_recomputed': metrics,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'source_runtime_imported': False,
        'elapsed_s': time.monotonic()-started, 'auditor_sha256': file_hash(__file__),
        'limits': ['Replays own corrections on exposed internal data; native M2/source graphs are previously bound/audited artifacts.',
                   'No forward, live, causal or goal-completion claim.']}
    target = output/'report.json'; target.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'report.sha256').write_text(file_hash(target)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'models_verified', 'correction_queries_replayed', 'metric_gate_passed', 'elapsed_s')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--directory', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.directory, a.output)
