"""Frozen, start-only setback retrieval experiment on a supplied clean W1 cache."""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from winprob import admission
from winprob.features import ALL_FEATURES
from winprob.metrics import game_weights, summarize, paired_comparison

SEED = 20260929
ANCHORS = tuple(range(6, 37, 3))
CHEAP_METHODS = ('low_current_wp', 'worst_gold_slope')
METRIC_NAMES = ('macro_recall', 'precision_all_selected', 'macro_precision',
                'positive_game_detection', 'all_game_detection')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def focal_sign(match_id):
    return 1 if hashlib.sha256(str(match_id).encode('utf-8')).digest()[0] % 2 == 0 else -1


def orient_features(values, names, signs):
    values = np.asarray(values, dtype=float).copy()
    signs = np.asarray(signs)
    if values.shape != (len(signs), len(names)) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('invalid focal-team feature orientation')
    for column, name in enumerate(names):
        if name not in ('time_min', 'gold_diff_slope_available'):
            values[:, column] *= signs
    return np.column_stack((values, (signs == 1).astype(float)))


def wp_rank_scores(snapshot_blue, trajectory_blue, signs):
    """Complement outputs for red; never invert the inputs of a blue WP model."""
    signs = np.asarray(signs)
    snapshot_blue, trajectory_blue = np.asarray(snapshot_blue), np.asarray(trajectory_blue)
    if (snapshot_blue.shape != signs.shape or trajectory_blue.shape != signs.shape
            or not np.isin(signs, [-1, 1]).all()):
        raise ValueError('invalid WP orientation shapes')
    for values in (snapshot_blue, trajectory_blue):
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError('invalid W1 probability')
    snapshot = np.where(signs == 1, snapshot_blue, 1-snapshot_blue)
    trajectory = np.where(signs == 1, trajectory_blue, 1-trajectory_blue)
    return {'wp_disagreement': snapshot-trajectory, 'low_current_wp': 1-snapshot}


def build_windows(x, game, timestamps, roles, names, game_metadata):
    """Labels read endpoints; candidate features retain start-row indices only."""
    names = list(names)
    gold, kills = names.index('gold_diff'), names.index('kills_diff')
    records = []
    coverage = {}
    metadata = {int(row['game_index']): row for row in game_metadata}
    if len(metadata) != len(game_metadata) or set(metadata) != set(map(int, np.unique(game))):
        raise ValueError('source manifest game indices do not match cache')
    for gid in sorted(metadata):
        idx = np.flatnonzero(game == gid)
        record = metadata[gid]
        role = str(record['role'])
        if len(set(roles[idx].tolist())) != 1 or str(roles[idx[0]]) != role:
            raise ValueError('game role mismatch')
        if len(np.unique(timestamps[idx])) != len(idx):
            raise ValueError('duplicate frame timestamp')
        bins = {}
        for i in idx[np.argsort(timestamps[idx], kind='stable')]:
            bins.setdefault(int(timestamps[i]) // 60000, int(i))
        count = coverage.setdefault(role, Counter())
        count['source_games'] += 1
        count['planned_clock_slots'] += len(ANCHORS)
        count['multiple_frames_in_minute_bins'] += len(idx) - len(bins)
        found = 0
        sign = focal_sign(record['match_id'])
        for anchor in ANCHORS:
            if anchor not in bins:
                count['missing_start_clock_bin'] += 1
                continue
            if anchor + 3 not in bins:
                count['missing_endpoint_clock_bin'] += 1
                continue
            start, end = bins[anchor], bins[anchor + 3]
            gold_change = sign * float(x[end, gold] - x[start, gold])
            kill_change = sign * float(x[end, kills] - x[start, kills])
            records.append((gid, start, end, sign, int(gold_change <= -1500 or kill_change <= -3),
                            gold_change, kill_change, role, str(record['patch'])))
            found += 1
        count['complete_windows'] += found
        count['games_with_complete_windows'] += int(found > 0)
        count['games_without_complete_windows'] += int(found == 0)
        count['games_with_fewer_than_two_complete_windows'] += int(found < 2)
    if not records:
        raise ValueError('no complete observed three-minute windows')
    columns = list(zip(*records))
    result = {name: np.asarray(value, dtype=dtype) for name, value, dtype in zip(
        ('game', 'start_index', 'end_index', 'sign', 'label', 'gold_change', 'kill_change', 'role', 'patch'),
        columns, (int, int, int, int, np.int8, float, float, str, str))}
    result['start_ms'] = timestamps[result['start_index']]
    result['end_ms'] = timestamps[result['end_index']]
    result['x'] = orient_features(x[result['start_index']], names, result['sign'])
    for role, count in coverage.items():
        durations = (result['end_ms'] - result['start_ms'])[result['role'] == role] / 1000
        count = dict(count)
        count['exclusion_interpretation'] = 'unavailable clock bins; no inferred terminal reason'
        count['complete_window_fraction'] = count.get('complete_windows', 0) / count['planned_clock_slots']
        count['actual_horizon_seconds'] = ({'min': float(durations.min()), 'median': float(np.median(durations)),
                                           'max': float(durations.max())} if len(durations) else None)
        coverage[role] = count
    return result, coverage


def select_windows(scores, games, starts, budget=2):
    scores = np.asarray(scores, dtype=float)
    games, starts = np.asarray(games), np.asarray(starts)
    if not len(scores) or len(scores) != len(games) or len(starts) != len(games) or not np.isfinite(scores).all():
        raise ValueError('invalid ranking population')
    selected = np.zeros(len(scores), dtype=bool)
    for gid in np.unique(games):
        idx = np.flatnonzero(games == gid)
        order = np.lexsort((starts[idx], -scores[idx]))
        selected[idx[order[:budget]]] = True
    return selected


def game_retrieval(scores, labels, games, starts, ends):
    labels, games, starts, ends = (np.asarray(values) for values in (labels, games, starts, ends))
    selected = select_windows(scores, games, starts)
    ids = np.unique(games)
    # Columns: eligible positives, retrieved windows, retrieved positives, selected minutes.
    rows = np.zeros((len(ids), 4), dtype=float)
    for j, gid in enumerate(ids):
        keep = games == gid
        take = keep & selected
        rows[j] = (labels[keep].sum(), take.sum(), labels[take].sum(),
                   ((ends[take] - starts[take]) / 60000).sum())
    return ids, rows, selected


def retrieval_point(rows):
    positive = rows[:, 0] > 0
    detected = rows[:, 2] > 0
    return np.array([
        np.mean(rows[positive, 2] / rows[positive, 0]) if positive.any() else np.nan,
        rows[:, 2].sum() / rows[:, 1].sum(),
        np.mean(rows[:, 2] / rows[:, 1]),
        np.mean(detected[positive]) if positive.any() else np.nan,
        np.mean(detected),
    ])


def summarize_retrieval(rows, resamples=2000):
    rng = np.random.default_rng(SEED)
    samples = np.asarray([retrieval_point(rows[rng.integers(0, len(rows), len(rows))]) for _ in range(resamples)])
    point = retrieval_point(rows)
    result = {'eligible_games': len(rows), 'positive_games': int((rows[:, 0] > 0).sum()),
              'eligible_positive_windows': int(rows[:, 0].sum()), 'selected_windows': int(rows[:, 1].sum()),
              'retrieved_positive_windows': int(rows[:, 2].sum()), 'resamples': resamples,
              'mean_actual_selected_minutes': float(rows[:, 3].mean())}
    for j, name in enumerate(METRIC_NAMES):
        finite = samples[np.isfinite(samples[:, j]), j]
        result[name] = {'estimate': float(point[j]) if np.isfinite(point[j]) else None,
                        'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
                        'bootstrap_valid_resamples': len(finite)}
    return result


def paired_retrieval(candidate, reference, resamples=2000):
    if candidate.shape != reference.shape or not np.array_equal(candidate[:, :2], reference[:, :2]):
        raise ValueError('retrieval comparison must use identical games, positive counts and budgets')
    rng = np.random.default_rng(SEED)
    samples = []
    for _ in range(resamples):
        idx = rng.integers(0, len(candidate), len(candidate))
        samples.append(retrieval_point(candidate[idx]) - retrieval_point(reference[idx]))
    samples = np.asarray(samples)
    delta = retrieval_point(candidate) - retrieval_point(reference)
    baseline_recall = retrieval_point(reference)[0]
    result = {'positive_games': int((candidate[:, 0] > 0).sum()), 'resamples': resamples}
    for j, name in enumerate(METRIC_NAMES):
        finite = samples[np.isfinite(samples[:, j]), j]
        result[name] = {'estimate': float(delta[j]) if np.isfinite(delta[j]) else None,
                        'ci95': np.quantile(finite, [.025, .975]).tolist() if len(finite) else None}
    relative = float(delta[0] / baseline_recall) if np.isfinite(baseline_recall) and baseline_recall > 0 else None
    result['relative_recall_gain'] = relative
    recall_ci = result['macro_recall']['ci95']
    result['useful_signal_gate'] = bool(result['positive_games'] >= 100 and relative is not None
                                        and relative >= .1 and recall_ci is not None and recall_ci[0] > 0
                                        and delta[1] >= -.02)
    result['population_status'] = 'powered_minimum_met' if result['positive_games'] >= 100 else 'underpowered'
    return result


def choose_baseline(rows):
    def key(name):
        point = retrieval_point(rows[name])
        recall = float(point[0]) if np.isfinite(point[0]) else -1
        return (-recall, -float(point[1]), name)
    return min(CHEAP_METHODS, key=key)


def risk_fit(x, labels, games, calibration_x, calibration_y, calibration_games, columns, logistic=False):
    weights = game_weights(games)
    weights *= len(weights) / weights.sum()
    xx = x[:, columns]
    scaler = StandardScaler().fit(xx, sample_weight=weights) if logistic else None
    if scaler is not None:
        xx = scaler.transform(xx)
    model = (LogisticRegression(C=1, max_iter=500, random_state=SEED) if logistic else
             HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=.06,
                 l2_regularization=10, min_samples_leaf=100, early_stopping=False, random_state=SEED))
    model.fit(xx, labels, sample_weight=weights)
    cc = calibration_x[:, columns]
    if scaler is not None:
        cc = scaler.transform(cc)
    raw = np.clip(model.predict_proba(cc)[:, 1], 1e-7, 1-1e-7)
    weights = game_weights(calibration_games)
    weights *= len(weights) / weights.sum()
    calibrator = LogisticRegression(C=1e6, max_iter=500, random_state=SEED)
    calibrator.fit(np.log(raw/(1-raw)).reshape(-1, 1), calibration_y, sample_weight=weights)
    return {'model': model, 'scaler': scaler, 'calibrator': calibrator, 'columns': columns,
            'feature_names': list(ALL_FEATURES) + ['focal_is_blue'], 'schema': 'setback-risk-v1'}


def risk_predict(artifact, x):
    xx = x[:, artifact['columns']]
    if artifact['scaler'] is not None:
        xx = artifact['scaler'].transform(xx)
    p = np.clip(artifact['model'].predict_proba(xx)[:, 1], 1e-7, 1-1e-7)
    return artifact['calibrator'].predict_proba(np.log(p/(1-p)).reshape(-1, 1))[:, 1]


def reject_quarantined(path):
    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    for parent in (directory, *directory.parents):
        if (parent / 'QUARANTINED.md').is_file():
            raise ValueError('quarantined input ancestor: ' + str(parent))


def verify_sidecar(path):
    path = Path(path)
    value = digest(path)
    tokens = path.with_suffix('.sha256').read_text(encoding='ascii').split()
    if len(tokens) != 2 or tokens[0] != value or tokens[1] != path.name:
        raise ValueError('SHA256 sidecar mismatch: ' + str(path))
    return value


def verify_admitted_sources(source, metadata):
    """Bind original roles and current reservations; never read raw payloads."""
    manifest_path = Path(source['admission_manifest']).resolve(strict=True)
    if manifest_path != Path(metadata['admission_manifest_path']).resolve(strict=True):
        raise ValueError('cache and source name different admission manifests')
    manifest_hash = verify_sidecar(manifest_path)
    if (manifest_hash != source['admission_manifest_sha256']
            or manifest_hash != metadata['admission_manifest_sha256']):
        raise ValueError('admission manifest binding mismatch')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('state') != 'temporal_manifest_ready':
        raise ValueError('admission has no frozen temporal roles')
    admission.verify_manifest_sources(manifest)
    allowed = {entry['match_id']: entry for entry in manifest['games']}
    if len(allowed) != len(manifest['games']):
        raise ValueError('duplicate admitted match identity')
    # Recompute forbidden historical IDs from verified metadata, independently
    # of a cache's booleans or a manifest entry's historical-validation flag.
    forbidden = set()
    source_root = Path(manifest['source_data_root']).resolve(strict=True)
    for binding in manifest['split_metadata_sources']:
        assignment, observed_digest = admission.read_assignment(source_root / binding['path'])
        if observed_digest != binding['sha256']:
            raise ValueError('split reservation changed during admission check')
        forbidden.update(mid for mid, role in assignment.items() if role in ('test', 'val'))
    role_map = {'train': 'fit', 'calibration': 'calibration', 'development': 'development'}
    seen = set()
    for row in source['games']:
        mid = row['match_id']
        expected = allowed.get(mid)
        if mid in seen or mid in forbidden or expected is None:
            raise ValueError('cache source ID absent, duplicated, or reserved')
        seen.add(mid)
        if (expected.get('historical_validation') is not False
                or expected.get('role') not in role_map
                or row.get('source_role') != expected['role']
                or row.get('role') != role_map[expected['role']]):
            raise ValueError('cache source role is forbidden or differs from admission')
        for key in ('patch', 'game_creation_ms'):
            if row.get(key) != expected.get(key):
                raise ValueError('cache source metadata differs from admission: ' + key)
    return manifest_path, manifest_hash


def verify_inputs(cache, source_manifest, w1_dir):
    for path in (cache, source_manifest, w1_dir):
        reject_quarantined(path)
    meta_path = cache.parent / 'cache_metadata.json'
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    verify_sidecar(meta_path)
    cache_hash, source_hash = digest(cache), digest(source_manifest)
    if metadata.get('state') != 'complete' or metadata['npz_sha256'] != cache_hash or metadata['source_manifest_sha256'] != source_hash:
        raise ValueError('cache/source provenance mismatch')
    source = json.loads(source_manifest.read_text(encoding='utf-8'))
    for key in ('test_payload_reads', 'confirmatory_payload_reads', 'historical_validation_payload_reads'):
        if source.get(key) is not False:
            raise ValueError('forbidden or unverified payload population: ' + key)
    role_map = {'train': 'fit', 'calibration': 'calibration', 'development': 'development'}
    if any(role_map.get(row.get('source_role')) != row.get('role') for row in source['games']):
        raise ValueError('source manifest contains a forbidden source role')
    if len(set(row['match_id'] for row in source['games'])) != len(source['games']):
        raise ValueError('duplicate source match identity')
    manifest_path, manifest_hash = verify_admitted_sources(source, metadata)
    report_path = w1_dir / 'report.json'
    w1_report = json.loads(report_path.read_text(encoding='utf-8'))
    if w1_report.get('quarantined') or w1_report.get('status') == 'quarantined':
        raise ValueError('quarantined W1 input')
    if w1_report.get('cache_sha256') != cache_hash or w1_report.get('sealed_test_read') is not False:
        raise ValueError('W1 report does not establish this clean cache')
    for relative, expected in w1_report['code_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError('W1 implementation changed since fitting: ' + relative)
    files = {str(cache): cache_hash, str(source_manifest): source_hash,
             str(meta_path): digest(meta_path), str(report_path): digest(report_path),
             str(manifest_path): manifest_hash,
             str(manifest_path.with_suffix('.sha256')): digest(manifest_path.with_suffix('.sha256'))}
    for name in ('snapshot', 'trajectory'):
        path = w1_dir / (name + '.joblib')
        artifact_hash = digest(path)
        if artifact_hash != w1_report['models'][name]['artifact_sha256']:
            raise ValueError('W1 model digest mismatch')
        files[str(path)] = artifact_hash
    return source, files


def check_resources(started, deadline):
    if time.monotonic() - started > deadline:
        raise RuntimeError('bounded experiment deadline exceeded')
    if os.name == 'nt':
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in ('total', 'available', 'page_total', 'page_available',
                                                      'virtual_total', 'virtual_available', 'extended')]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('cannot establish available memory')
        if status.available < 2.5 * 1024**3:
            raise RuntimeError('available memory below the 2.5 GiB gate')


def set_below_normal_priority():
    if os.name == 'nt':
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.argtypes = []
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.SetPriorityClass.restype = wintypes.BOOL
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000):
            raise ctypes.WinError(ctypes.get_last_error())


def run(args):
    started = time.monotonic()
    set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen_files = {str(Path(__file__).resolve()): digest(__file__),
                    str(ROOT/'docs/setback_protocol_v1.md'): digest(ROOT/'docs/setback_protocol_v1.md')}
    def progress(stage, **extra):
        write_json(args.output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra})
    def check():
        check_resources(started, args.deadline_seconds)
    try:
        progress('verifying_clean_inputs')
        check()
        source, input_files = verify_inputs(args.cache, args.source_manifest, args.w1_artifacts)
        frozen_files.update(input_files)
        data = np.load(args.cache, allow_pickle=False)
        x, game, roles, timestamps = (data[key] for key in ('x', 'game', 'role', 'timestamp_ms'))
        names = data['feature_names'].tolist()
        if tuple(names) != ALL_FEATURES or x.shape != (len(game), len(names)) or not np.isfinite(x).all():
            raise ValueError('unexpected or invalid feature cache')
        if len(np.unique(game)) > 6000 or not set(np.unique(roles)) <= {'fit', 'calibration', 'development'}:
            raise ValueError('cache exceeds population bounds')
        split_time = data['split_time_ms']
        if not (split_time[roles == 'fit'].max() < split_time[roles == 'calibration'].min()
                and split_time[roles == 'calibration'].max() < split_time[roles == 'development'].min()):
            raise ValueError('temporal roles overlap')
        windows, coverage = build_windows(x, game, timestamps, roles, names, source['games'])
        fit, cal, dev = (windows['role'] == role for role in ('fit', 'calibration', 'development'))
        for role, keep in (('fit', fit), ('calibration', cal)):
            if len(np.unique(windows['game'][keep])) < 100 or len(np.unique(windows['label'][keep])) != 2:
                raise ValueError('insufficient eligible games/classes for ' + role)
        if len(np.unique(windows['game'][dev])) < 30:
            raise ValueError('fewer than 30 eligible development games')
        progress('scoring_frozen_wp', complete_windows=len(windows['game']))
        check()
        models = {name: joblib.load(args.w1_artifacts/(name+'.joblib')) for name in ('snapshot', 'trajectory')}
        blue_wp = {name: model.predict(x[windows['start_index']]) for name, model in models.items()}
        scores = wp_rank_scores(blue_wp['snapshot'], blue_wp['trajectory'], windows['sign'])
        scores['worst_gold_slope'] = -windows['x'][:, names.index('gold_diff_slope_per_min')]
        def retrieve(method, keep):
            return game_retrieval(scores[method][keep], windows['label'][keep], windows['game'][keep],
                                  windows['start_ms'][keep], windows['end_ms'][keep])
        calibration_rows = {name: retrieve(name, cal)[1] for name in CHEAP_METHODS}
        baseline = choose_baseline(calibration_rows)
        write_json(args.output/'selection_lock.json', {'schema': 'setback-selection-v1',
                   'selected_cheap_baseline': baseline, 'selection_partition': 'calibration',
                   'calibration_point_metrics': {name: dict(zip(METRIC_NAMES, retrieval_point(rows).tolist()))
                                                 for name, rows in calibration_rows.items()},
                   'selection_frozen_before_development_evaluation': True,
                   'input_and_protocol_sha256': frozen_files})
        full_names = names + ['focal_is_blue']
        snapshot_columns = [i for i, name in enumerate(full_names) if 'slope' not in name and 'lag' not in name]
        risk_artifacts = {}
        with threadpool_limits(limits=1):
            for kind, columns, logistic in (('risk_logistic', snapshot_columns, True),
                                           ('risk_snapshot', snapshot_columns, False),
                                           ('risk_history', list(range(len(full_names))), False)):
                progress('fitting_direct_risk', model=kind)
                check()
                artifact = risk_fit(windows['x'][fit], windows['label'][fit], windows['game'][fit],
                                    windows['x'][cal], windows['label'][cal], windows['game'][cal],
                                    columns, logistic)
                scores[kind] = risk_predict(artifact, windows['x'])
                path = args.output/(kind+'.joblib')
                joblib.dump(artifact, path, compress=3)
                risk_artifacts[kind] = {'path': str(path.resolve()), 'sha256': digest(path)}
        check()
        progress('evaluating_development')
        report = {'schema': 'setback-development-v1', 'stage': 'exploratory_development',
                  'report_path': str((args.output/'report.json').resolve()), 'causal_claim': False,
                  'coaching_benefit_established': False, 'confirmatory_evaluation_done': False,
                  'selected_cheap_baseline': baseline, 'source_coverage': coverage,
                  'source_cache_exclusions': dict(Counter(row['reason'] for row in source.get('exclusions', []))),
                  'source_cache_selected_games': source.get('selected_games'),
                  'population': {}, 'methods': {}, 'comparisons': {}, 'risk_probability_diagnostics': {},
                  'artifacts': risk_artifacts, 'input_and_protocol_sha256': frozen_files,
                  'interval_scope': 'paired whole-game bootstrap; fitted models fixed; exploratory comparisons',
                  'limitations': ['Retrospective OMNI complete-window population only.',
                                  'Missing endpoints are unavailable, never inferred terminal.',
                                  'Retrieval endpoints do not establish coachability or causal value.',
                                  'Two prespecified hypotheses; no development winner called confirmation.']}
        for role in ('fit', 'calibration', 'development'):
            keep = windows['role'] == role
            report['population'][role] = {'games': len(np.unique(windows['game'][keep])),
                'windows': int(keep.sum()), 'positive_windows': int(windows['label'][keep].sum()),
                'patch_games': {str(p): len(np.unique(windows['game'][keep & (windows['patch'] == p)]))
                                for p in np.unique(windows['patch'][keep])}}
        method_rows = {}
        save_arrays = {key: windows[key][dev] for key in ('game', 'start_ms', 'end_ms', 'label', 'sign',
                                                         'gold_change', 'kill_change', 'patch')}
        for method in scores:
            ids, rows, selected = retrieve(method, dev)
            method_rows[method] = rows
            report['methods'][method] = summarize_retrieval(rows, args.resamples)
            save_arrays[method+'_score'] = scores[method][dev]
            save_arrays[method+'_selected'] = selected
        fixed_pairs = [('wp_disagreement', baseline), ('risk_history', baseline),
                       ('risk_snapshot', baseline), ('risk_logistic', baseline),
                       ('risk_history', 'risk_snapshot'), ('risk_snapshot', 'risk_logistic'),
                       ('risk_history', 'risk_logistic')]
        for candidate, reference in fixed_pairs:
            report['comparisons'][candidate+'_vs_'+reference] = paired_retrieval(
                method_rows[candidate], method_rows[reference], args.resamples)
        for method in risk_artifacts:
            report['risk_probability_diagnostics'][method] = summarize(
                windows['label'][dev], scores[method][dev], windows['game'][dev], args.resamples)
        report['risk_probability_comparisons'] = {'history_vs_snapshot': paired_comparison(
            windows['label'][dev], scores['risk_snapshot'][dev], scores['risk_history'][dev],
            windows['game'][dev], args.resamples)}
        report['by_patch'] = {}
        for patch in np.unique(windows['patch'][dev]):
            keep = dev & (windows['patch'] == patch)
            n_games = len(np.unique(windows['game'][keep]))
            if n_games < 30:
                report['by_patch'][str(patch)] = {'games': n_games, 'status': 'not_reported_below_30_games'}
                continue
            patch_rows = {name: retrieve(name, keep)[1] for name in scores}
            report['by_patch'][str(patch)] = {'games': n_games,
                'methods': {name: summarize_retrieval(rows, args.resamples) for name, rows in patch_rows.items()},
                'comparisons': {candidate+'_vs_'+reference: paired_retrieval(patch_rows[candidate], patch_rows[reference], args.resamples)
                                for candidate, reference in fixed_pairs}}
        np.savez_compressed(args.output/'development_windows.npz', **save_arrays)
        report['development_windows_sha256'] = digest(args.output/'development_windows.npz')
        report['gates'] = {
            'h1_disagreement_retrieval': report['comparisons']['wp_disagreement_vs_'+baseline]['useful_signal_gate'],
            'h2_history_retrieval_vs_cheap': report['comparisons']['risk_history_vs_'+baseline]['useful_signal_gate'],
            'h2_history_adds_to_snapshot': report['comparisons']['risk_history_vs_risk_snapshot']['useful_signal_gate'],
            'practical_coaching_benefit_established': False,
        }
        check()
        if any(digest(path) != expected for path, expected in frozen_files.items()):
            raise RuntimeError('an input or protocol changed during the experiment')
        admission.verify_manifest_sources(json.loads(Path(source['admission_manifest']).read_text(encoding='utf-8')))
        for path in (args.cache, args.source_manifest, args.w1_artifacts):
            reject_quarantined(path)
        report['elapsed_s'] = time.monotonic() - started
        write_json(args.output/'report.json', report)
        progress('completed', report=str((args.output/'report.json').resolve()))
        return report
    except Exception as exc:
        progress('failed', error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--w1-artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--deadline-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.resamples < 100 or args.deadline_seconds <= 0:
        parser.error('at least 100 resamples and a positive deadline are required')
    for attr in ('cache', 'source_manifest', 'w1_artifacts', 'output'):
        setattr(args, attr, getattr(args, attr).resolve())
    report = run(args)
    print(json.dumps({'report': report['report_path'], 'gates': report['gates'],
                      'elapsed_s': report['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
