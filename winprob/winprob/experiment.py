"""Frozen W1 models; no runtime imports from sibling projects."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from winprob.metrics import game_weights, summarize, paired_comparison
from winprob.features import ALL_FEATURES

SEED = 20260929
BASELINE = ('time_min', 'gold_diff', 'kills_diff', 'towers_diff', 'dragons_diff')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def sample_training_indices(game, timestamp_ms):
    """Uniform hash selection among frames per game / five-minute interval."""
    chosen = {}
    for i, (g, t) in enumerate(zip(game, timestamp_ms)):
        group = (int(g), int(t)//300000)
        score = hashlib.sha256(f'{SEED}:{g}:{t}'.encode()).digest()
        if group not in chosen or score < chosen[group][0]:
            chosen[group] = (score, i)
    return np.array(sorted(value[1] for value in chosen.values()), dtype=int)


class CalibratedModel:
    def __init__(self, kind, indices):
        self.kind = kind
        self.indices = list(indices)
        self.scaler = None
        self.calibrator = None

    def fit(self, x, y, games):
        features = x[:, self.indices]
        weight = game_weights(games)
        weight *= len(weight)/weight.sum()
        if self.kind == 'logistic':
            self.scaler = StandardScaler().fit(features, sample_weight=weight)
            features = self.scaler.transform(features)
            self.model = LogisticRegression(C=1, max_iter=500, random_state=SEED)
        else:
            self.model = HistGradientBoostingClassifier(
                max_iter=150, max_leaf_nodes=15, learning_rate=.06,
                l2_regularization=10, min_samples_leaf=100,
                early_stopping=False, random_state=SEED)
        self.model.fit(features, y, sample_weight=weight)
        return self

    def raw(self, x):
        features = x[:, self.indices]
        if self.scaler is not None:
            features = self.scaler.transform(features)
        return self.model.predict_proba(features)[:, 1]

    @staticmethod
    def logit(probability):
        p = np.clip(probability, 1e-7, 1-1e-7)
        return np.log(p/(1-p)).reshape(-1, 1)

    def calibrate(self, x, y, games):
        weight = game_weights(games)
        weight *= len(weight)/weight.sum()
        self.calibrator = LogisticRegression(C=1e6, max_iter=500, random_state=SEED)
        self.calibrator.fit(self.logit(self.raw(x)), y, sample_weight=weight)
        return self

    def predict(self, x):
        return self.calibrator.predict_proba(self.logit(self.raw(x)))[:, 1]


def permute_game_labels(y, games, start_ms, patch, seed):
    """Shuffle whole games within patch/week; never independently shuffle rows."""
    rng = np.random.default_rng(seed)
    group_indices = {}
    ids, first = np.unique(games, return_index=True)
    for gid, index in zip(ids, first):
        key = (str(patch[index]), int(start_ms[index])//(7*86400000))
        group_indices.setdefault(key, []).append(index)
    mapping = {}
    for indices in group_indices.values():
        labels = rng.permutation(y[indices])
        for index, label in zip(indices, labels):
            mapping[int(games[index])] = label
    return np.array([mapping[int(g)] for g in games], dtype=np.int8)


def run_experiment(cache_path, output_dir, resamples=2000, resource_check=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    code_paths = sorted(root.glob('winprob/*.py')) + [root/'docs/protocol_v1.md']
    code_hashes = {str(p.relative_to(root)): sha256(p) for p in code_paths}
    def progress(stage, **extra):
        write_json(output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra})
    progress('loading')
    data = np.load(cache_path, allow_pickle=False)
    x, y, games, roles = (data[key] for key in ('x', 'y', 'game', 'role'))
    timestamps = data['timestamp_ms']
    names = data['feature_names'].tolist()
    if tuple(names) != ALL_FEATURES or x.ndim != 2 or x.shape[1] != len(names):
        raise ValueError('cache feature schema differs from the declared allowlist')
    for key in ('y', 'game', 'role', 'timestamp_ms', 'start_ms', 'patch'):
        if len(data[key]) != len(x):
            raise ValueError(f'cache row length mismatch: {key}')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('cache labels are not binary')
    if not set(np.unique(roles)) <= {'fit', 'calibration', 'development'}:
        raise ValueError('cache contains a forbidden role')
    if not np.isfinite(x).all():
        raise ValueError('nonfinite feature')
    for gid in np.unique(games):
        same = games == gid
        if len(np.unique(roles[same])) != 1 or len(np.unique(y[same])) != 1:
            raise ValueError('game split or label inconsistency')
    masks = {role: roles == role for role in ('fit', 'calibration', 'development')}
    for role, mask in masks.items():
        if len(np.unique(games[mask])) < 100 or len(np.unique(y[mask])) != 2:
            raise ValueError(f'insufficient population: {role}')
    split_time = data['split_time_ms'] if 'split_time_ms' in data else data['start_ms']
    if not (split_time[masks['fit']].max() < split_time[masks['calibration']].min()
            and split_time[masks['calibration']].max() < split_time[masks['development']].min()):
        raise ValueError('temporal partitions overlap or are reversed')
    fit_indices = np.flatnonzero(masks['fit'])
    fit_indices = fit_indices[sample_training_indices(games[fit_indices], timestamps[fit_indices])]
    cal, dev = masks['calibration'], masks['development']
    snapshot = [i for i, name in enumerate(names) if 'slope' not in name and 'lag' not in name]
    groups = {'logistic': [names.index(n) for n in BASELINE],
              'snapshot': snapshot, 'trajectory': list(range(len(names)))}
    report = {'schema': 1, 'stage': 'development_pilot',
              'causal_claim': False, 'sealed_test_read': False,
              'source_cache': str(Path(cache_path).resolve()), 'cache_sha256': sha256(cache_path),
              'code_sha256': code_hashes, 'seed': SEED, 'features': names,
              'population': {r: {'games': int(len(np.unique(games[m]))), 'frames': int(m.sum()),
                                'first_start_ms': int(data['start_ms'][m].min()),
                                'last_start_ms': int(data['start_ms'][m].max())} for r, m in masks.items()},
              'fit_sampled_frames': len(fit_indices), 'models': {}, 'comparisons': {},
              'interval_scope': 'paired game bootstrap, conditional on fitted models; development only'}
    for role, mask in masks.items():
        report['population'][role]['patch_games'] = {
            str(p): len(np.unique(games[mask & (data['patch'] == p)]))
            for p in np.unique(data['patch'][mask])}
    prevalence = float(np.average(y[fit_indices], weights=game_weights(games[fit_indices])))
    predictions = {'constant': np.full(int(dev.sum()), prevalence)}
    report['models']['constant'] = summarize(y[dev], predictions['constant'], games[dev], resamples)
    with threadpool_limits(limits=1):
        for kind, columns in groups.items():
            if resource_check is not None:
                resource_check()
            progress('fitting', model=kind)
            model = CalibratedModel(kind, columns).fit(x[fit_indices], y[fit_indices], games[fit_indices])
            model.calibrate(x[cal], y[cal], games[cal])
            predictions[kind] = model.predict(x[dev])
            report['models'][kind] = summarize(y[dev], predictions[kind], games[dev], resamples)
            report['models'][kind]['uncalibrated'] = summarize(y[dev], model.raw(x[dev]), games[dev], resamples)
            path = output/f'{kind}.joblib'
            joblib.dump(model, path, compress=3)
            report['models'][kind]['artifact_sha256'] = sha256(path)
        progress('negative_control')
        if resource_check is not None:
            resource_check()
        shuffled = y.copy()
        for j, mask in enumerate(masks.values()):
            shuffled[mask] = permute_game_labels(y[mask], games[mask], data['start_ms'][mask], data['patch'][mask], SEED+j)
        control = CalibratedModel('snapshot', snapshot).fit(x[fit_indices], shuffled[fit_indices], games[fit_indices])
        control.calibrate(x[cal], shuffled[cal], games[cal])
        control_p = control.predict(x[dev])
        control_base = np.full(int(dev.sum()), float(np.average(shuffled[fit_indices], weights=game_weights(games[fit_indices]))))
        report['negative_control'] = paired_comparison(shuffled[dev], control_base, control_p, games[dev], resamples)
        report['negative_control']['requires_investigation'] = report['negative_control']['log_loss']['ci95'][0] > 0
    for candidate, reference in (('snapshot', 'logistic'), ('trajectory', 'snapshot')):
        report['comparisons'][f'{candidate}_vs_{reference}'] = paired_comparison(
            y[dev], predictions[reference], predictions[candidate], games[dev], resamples)
    report['time_bands'] = {}
    for lower, upper in ((3, 10), (10, 20), (20, 30), (30, 41)):
        keep = (timestamps[dev] >= lower*60000) & (timestamps[dev] < upper*60000)
        if len(np.unique(games[dev][keep])) < 30:
            report['time_bands'][f'{lower}-{min(upper,40)}min'] = {
                'status': 'not_measured_insufficient_games',
                'measured': False,
                'games': len(np.unique(games[dev][keep])), 'frames': int(keep.sum())}
            continue
        report['time_bands'][f'{lower}-{min(upper,40)}min'] = {
            kind: summarize(y[dev][keep], p[keep], games[dev][keep], resamples)
            for kind, p in predictions.items()}
    trajectory = report['comparisons']['trajectory_vs_snapshot']
    snapshot_delta = report['comparisons']['snapshot_vs_logistic']
    report['gates'] = {
        'trajectory_predictive_signal': (trajectory['relative_log_loss_reduction'] >= .005
                                         and trajectory['log_loss']['ci95'][0] > 0
                                         and report['models']['trajectory']['ece20']
                                         <= report['models']['snapshot']['ece20'] + .005),
        'snapshot_clears_logistic': (snapshot_delta['log_loss']['ci95'][0] > 0
                                     and snapshot_delta['brier']['ci95'][0] > 0),
        'negative_control_clear': not report['negative_control']['requires_investigation'],
        'practical_decision_benefit_established': False,
        'confirmatory_evaluation_done': False,
    }
    # Persist predictions for exactly the evaluation population, never reserved games.
    np.savez_compressed(output/'development_predictions.npz', y=y[dev], game=games[dev],
                        timestamp_ms=timestamps[dev], **predictions)
    if any(sha256(root/rel) != digest for rel, digest in code_hashes.items()):
        raise RuntimeError('code changed during experiment')
    if sha256(cache_path) != report['cache_sha256']:
        raise RuntimeError('input cache changed during experiment')
    report['elapsed_s'] = time.monotonic()-started
    report['limitations'] = ['Retrospective OMNI only; no live input claim.',
                             'Source-training games re-split for a development pilot.',
                             'Prediction gains do not establish better purchase outcomes.',
                             'ECE is descriptive and bin-sensitive; proper scores are primary.']
    write_json(output/'report.json', report)
    progress('completed', report=str((output/'report.json').resolve()))
    return report
