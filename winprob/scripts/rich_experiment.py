"""Fixed matched-cohort rich-state WP and three-minute setback experiment."""
from __future__ import annotations

import argparse
import gzip
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
sys.path.insert(0, str(ROOT / 'scripts'))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import setback_experiment as setback
from winprob import admission
from winprob.experiment import CalibratedModel, sample_training_indices
from winprob.features import ALL_FEATURES
from winprob.metrics import paired_comparison, summarize

EXPECTED_GAMES = 6000
SEED = 20260929
SCHEMA = 'rich-state-development-v1'


def read_frozen_payload(binding, endpoint, match_id, source_root):
    """Check compressed bytes before decompression; no alternate archive lookup."""
    path = Path(binding['path']).resolve(strict=True)
    expected = (source_root / 'raw_match_v5' / endpoint / (match_id + '.json.gz')).resolve(strict=True)
    if path != expected:
        raise ValueError('original source path differs from expected admitted archive path')
    raw = path.read_bytes()
    if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('original compressed source digest changed: ' + match_id + '/' + endpoint)
    envelope = json.loads(gzip.decompress(raw).decode('utf-8'))
    if (not isinstance(envelope, dict) or type(envelope.get('schema')) is not int
            or envelope['schema'] != 1 or envelope.get('endpoint') != endpoint
            or envelope.get('match_id') != match_id or not isinstance(envelope.get('payload'), dict)):
        raise ValueError('invalid frozen source envelope: ' + match_id)
    payload = envelope['payload']
    if payload.get('metadata', {}).get('matchId') != match_id:
        raise ValueError('frozen payload identity mismatch: ' + match_id)
    return payload


def orient_rich_features(x, names, signs, sign_features):
    names = tuple(names)
    signs = np.asarray(signs)
    x = np.asarray(x, dtype=float).copy()
    if x.shape != (len(signs), len(names)) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('invalid rich focal orientation')
    if not set(sign_features).issubset(names):
        raise ValueError('unknown signed rich feature')
    for name in sign_features:
        x[:, names.index(name)] *= signs
    return np.column_stack((x, (signs == 1).astype(float)))


def ece_bootstrap(y, predictions, games, resamples):
    """Equal-game ECE20 and paired ECE reductions with whole-game resampling."""
    y, games = np.asarray(y), np.asarray(games)
    ids, inverse, counts = np.unique(games, return_inverse=True, return_counts=True)
    matrices = {}
    for name, probability in predictions.items():
        p = np.clip(np.asarray(probability, dtype=float), 0, 1)
        bins = np.minimum((p * 20).astype(int), 19)
        flat = inverse * 20 + bins
        matrices[name] = np.bincount(flat, weights=(p-y)/counts[inverse],
                                     minlength=len(ids)*20).reshape(len(ids), 20)
    points = {name: float(np.abs(matrix.sum(axis=0)).sum()/len(ids)) for name, matrix in matrices.items()}
    samples = {name: np.empty(resamples) for name in matrices}
    rng = np.random.default_rng(SEED)
    for index in range(resamples):
        draws = rng.integers(0, len(ids), len(ids))
        for name, matrix in matrices.items():
            samples[name][index] = np.abs(matrix[draws].sum(axis=0)).sum()/len(ids)
    summaries = {name: {'estimate': points[name], 'ci95': np.quantile(samples[name], [.025, .975]).tolist(),
                        'games': len(ids), 'resamples': resamples} for name in matrices}
    paired = {}
    for name in matrices:
        if name == 'original_trajectory':
            continue
        delta = samples['original_trajectory'] - samples[name]
        paired[name + '_vs_original_trajectory'] = {
            'estimate': points['original_trajectory']-points[name],
            'ci95': np.quantile(delta, [.025, .975]).tolist(), 'games': len(ids), 'resamples': resamples}
    return summaries, paired


def verify_setback_controls(directory, source_files, cache_hash, windows, dev):
    setback.reject_quarantined(directory)
    report_path = directory / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('stage') != 'exploratory_development' or report.get('confirmatory_evaluation_done') is not False:
        raise ValueError('setback control is not a completed development-only experiment')
    bindings = report['input_and_protocol_sha256']
    for path, expected in source_files.items():
        if bindings.get(path) != expected:
            raise ValueError('setback controls were not fitted on the same frozen W1 inputs: ' + path)
    for path, expected in bindings.items():
        if setback.digest(path) != expected:
            raise ValueError('setback control input/code changed: ' + path)
    saved_path = directory / 'development_windows.npz'
    if setback.digest(saved_path) != report['development_windows_sha256']:
        raise ValueError('persisted setback prediction digest mismatch')
    scores = {}
    with np.load(saved_path, allow_pickle=False) as saved:
        for key in ('game', 'start_ms', 'end_ms', 'label', 'sign', 'gold_change', 'kill_change', 'patch'):
            if not np.array_equal(saved[key], windows[key][dev]):
                raise ValueError('persisted setback control windows are not matched: ' + key)
        for name in ('risk_history', 'risk_logistic'):
            probability = saved[name + '_score'].copy()
            if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
                raise ValueError('invalid persisted setback probability')
            expected_selected = setback.select_windows(probability, windows['game'][dev], windows['start_ms'][dev])
            if not np.array_equal(expected_selected, saved[name + '_selected']):
                raise ValueError('persisted retrieval selection does not reproduce')
            scores[name] = probability
    files = {str(report_path.resolve()): setback.digest(report_path),
             str(saved_path.resolve()): setback.digest(saved_path), **bindings}
    for name in ('risk_history', 'risk_logistic'):
        artifact = report['artifacts'][name]
        if setback.digest(artifact['path']) != artifact['sha256']:
            raise ValueError('setback model artifact changed: ' + name)
        files[str(Path(artifact['path']).resolve())] = artifact['sha256']
    return scores, files


def build_rich_matrix(data, source, source_root, rich, check, progress):
    """Preserve source game order and exact original frame row positions."""
    names = tuple(rich.ALL_RICH_FEATURES)
    if len(names) != len(set(names)) or not set(ALL_FEATURES).issubset(names):
        raise ValueError('rich schema must contain every distinct base feature')
    base_indices = [names.index(name) for name in ALL_FEATURES]
    games = source['games']
    if len(games) != EXPECTED_GAMES or len(np.unique(data['game'])) != EXPECTED_GAMES:
        raise ValueError('rich protocol requires the original matched 6000 games')
    bindings = {row['match_id']: row for row in source['compressed_sources']}
    if len(bindings) != len(source['compressed_sources']):
        raise ValueError('duplicate original compressed source identity')
    if [row['game_index'] for row in games] != list(range(EXPECTED_GAMES)):
        raise ValueError('source game mapping/order changed')
    x = np.empty((len(data['x']), len(names)), dtype=np.float32)
    covered = np.zeros(len(x), dtype=bool)
    for number, game in enumerate(games, 1):
        mid = game['match_id']
        binding = bindings[mid]
        match = read_frozen_payload(binding['match'], 'match', mid, source_root)
        info = match.get('info', {})
        if info.get('queueId') != 420:
            raise ValueError('wrong queue in frozen source')
        if 'gameCreation' in info and info['gameCreation'] != game['game_creation_ms']:
            raise ValueError('frozen source creation timestamp differs')
        timeline = read_frozen_payload(binding['timeline'], 'timeline', mid, source_root)
        rows = rich.extract_rich_game_states(match, timeline)
        del match, timeline
        row_map = {row['timestamp_ms']: row for row in rows}
        if len(row_map) != len(rows):
            raise ValueError('duplicate rich feature timestamp')
        original = np.flatnonzero(data['game'] == game['game_index'])
        if len(original) != game['rows'] or set(row_map) != set(map(int, data['timestamp_ms'][original])):
            raise ValueError('rich features changed the original eligible frame population: ' + mid)
        ordered = [row_map[int(timestamp)] for timestamp in data['timestamp_ms'][original]]
        values = np.asarray([[row['x'][name] for name in names] for row in ordered], dtype=np.float32)
        if not np.array_equal(values[:, base_indices], data['x'][original]):
            raise ValueError('rich extractor changed a base19 float32 value: ' + mid)
        if not np.array_equal(np.asarray([row['y'] for row in ordered], dtype=np.int8), data['y'][original]):
            raise ValueError('rich extractor changed a label: ' + mid)
        for row, index in zip(ordered, original):
            if (row['game_id'] != mid or row['patch'] != str(data['patch'][index])
                    or row['game_start_ms'] != int(data['start_ms'][index])
                    or game['role'] != str(data['role'][index])):
                raise ValueError('rich row identity/metadata changed: ' + mid)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite enriched features: ' + mid)
        x[original] = values
        covered[original] = True
        if number % 50 == 0 or number == len(games):
            progress('enriching_frozen_sources', processed_games=number, total_games=len(games))
            check()
    if not covered.all():
        raise ValueError('not every original row was enriched')
    return x


def run(args):
    from winprob import features_rich as rich
    started = time.monotonic()
    setback.set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = {str(path.resolve()): setback.digest(path) for path in (
        Path(__file__), ROOT/'docs/rich_protocol_v1.md', Path(rich.__file__), ROOT/'scripts/setback_experiment.py')}
    def progress(stage, **extra):
        setback.write_json(args.output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra})
    def check():
        setback.check_resources(started, args.deadline_seconds)
    try:
        check()
        progress('verifying_matched_inputs')
        source, original_files = setback.verify_inputs(args.cache, args.source_manifest, args.w1_artifacts)
        frozen.update(original_files)
        manifest = json.loads(Path(source['admission_manifest']).read_text(encoding='utf-8'))
        source_root = Path(manifest['source_data_root']).resolve(strict=True)
        with np.load(args.cache, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        if tuple(data['feature_names']) != ALL_FEATURES or data['x'].dtype != np.float32:
            raise ValueError('unexpected original feature schema/dtype')
        x, y, game, roles, timestamps = (data[key] for key in ('x', 'y', 'game', 'role', 'timestamp_ms'))
        if len(x) != len(y) or not set(np.unique(roles)) <= {'fit', 'calibration', 'development'}:
            raise ValueError('invalid original population')
        masks = {role: roles == role for role in ('fit', 'calibration', 'development')}
        cal, dev = masks['calibration'], masks['development']
        for role, mask in masks.items():
            if len(np.unique(game[mask])) < 100 or len(np.unique(y[mask])) != 2:
                raise ValueError('insufficient matched WP population: ' + role)
        split_time = data['split_time_ms']
        if not (split_time[masks['fit']].max() < split_time[cal].min()
                and split_time[cal].max() < split_time[dev].min()):
            raise ValueError('matched temporal roles overlap')
        windows, coverage = setback.build_windows(x, game, timestamps, roles, ALL_FEATURES, source['games'])
        risk_fit_mask, risk_cal, risk_dev = (windows['role'] == role for role in ('fit', 'calibration', 'development'))
        for role, mask in (('fit', risk_fit_mask), ('calibration', risk_cal)):
            if len(np.unique(windows['game'][mask])) < 100 or len(np.unique(windows['label'][mask])) != 2:
                raise ValueError('insufficient matched setback population: ' + role)
        control_scores, control_files = verify_setback_controls(args.setback_artifacts, original_files,
                                                               setback.digest(args.cache), windows, risk_dev)
        frozen.update(control_files)
        progress('enriching_frozen_sources', processed_games=0, total_games=EXPECTED_GAMES)
        rich_x = build_rich_matrix(data, source, source_root, rich, check, progress)
        rich_names = tuple(rich.ALL_RICH_FEATURES)
        snapshot_names = tuple(rich.STATE_RICH_FEATURES)
        sign_names = tuple(rich.SIGNED_RICH_FEATURES)
        invariant_names = tuple(rich.INVARIANT_RICH_FEATURES)
        if not set(snapshot_names).issubset(rich_names) or len(set(snapshot_names)) != len(snapshot_names):
            raise ValueError('invalid rich snapshot schema')
        if (set(sign_names) & set(invariant_names) or set(sign_names) | set(invariant_names) != set(rich_names)
                or set(ALL_FEATURES) & set(sign_names) != set(ALL_FEATURES) - {'time_min', 'gold_diff_slope_available'}):
            raise ValueError('rich focal parity mapping is incomplete or changes base orientation')
        snapshot_columns = [rich_names.index(name) for name in snapshot_names]
        cache_path = args.output/'rich_states.npz'
        np.savez_compressed(cache_path, x=rich_x, y=y, game=game, role=roles, timestamp_ms=timestamps,
                            start_ms=data['start_ms'], split_time_ms=split_time, patch=data['patch'],
                            feature_names=np.asarray(rich_names))
        cache_hash = setback.digest(cache_path)
        fit_indices = np.flatnonzero(masks['fit'])
        fit_indices = fit_indices[sample_training_indices(game[fit_indices], timestamps[fit_indices])]
        lock = {'schema': SCHEMA, 'stage': 'locked_before_new_development_predictions',
                'input_and_protocol_sha256': frozen, 'rich_cache_sha256': cache_hash,
                'original_population_unchanged': True, 'games': EXPECTED_GAMES, 'frames': len(x),
                'wp_fit_sampled_frames': len(fit_indices), 'rich_feature_names': list(rich_names),
                'rich_snapshot_feature_names': list(snapshot_names), 'focal_sign_feature_names': list(sign_names),
                'wp_models': ['rich_snapshot', 'rich_trajectory'], 'risk_models': ['rich_risk_history'],
                'risk_controls': ['risk_history', 'risk_logistic', 'latest_two'], 'hyperparameter_search': False,
                'latest_two_status': 'diagnostic chosen after the previous experiment; locked before this rich run'}
        setback.write_json(args.output/'experiment_lock.json', lock)
        probabilities = {}
        artifacts = {}
        with threadpool_limits(limits=1):
            original_model = joblib.load(args.w1_artifacts/'trajectory.joblib')
            probabilities['original_trajectory'] = original_model.predict(x[dev])
            for name, kind, columns in (('rich_snapshot', 'snapshot', snapshot_columns),
                                        ('rich_trajectory', 'trajectory', list(range(len(rich_names))))):
                check()
                progress('fitting_rich_wp', model=name)
                model = CalibratedModel(kind, columns).fit(rich_x[fit_indices], y[fit_indices], game[fit_indices])
                model.calibrate(rich_x[cal], y[cal], game[cal])
                probabilities[name] = model.predict(rich_x[dev])
                path = args.output/(name+'.joblib')
                joblib.dump(model, path, compress=3)
                artifacts[name] = {'path': str(path.resolve()), 'sha256': setback.digest(path),
                                   'feature_names': list(rich_names), 'selected_columns': columns}
            check()
            progress('fitting_rich_setback')
            oriented = orient_rich_features(rich_x[windows['start_index']], rich_names, windows['sign'], sign_names)
            risk = setback.risk_fit(oriented[risk_fit_mask], windows['label'][risk_fit_mask], windows['game'][risk_fit_mask],
                                     oriented[risk_cal], windows['label'][risk_cal], windows['game'][risk_cal],
                                     list(range(oriented.shape[1])))
            risk['feature_names'] = list(rich_names) + ['focal_is_blue']
            risk['schema'] = 'rich-setback-risk-v1'
            risk['focal_sign_features'] = list(sign_names)
            risk_score = setback.risk_predict(risk, oriented[risk_dev])
            path = args.output/'rich_risk_history.joblib'
            joblib.dump(risk, path, compress=3)
            artifacts['rich_risk_history'] = {'path': str(path.resolve()), 'sha256': setback.digest(path)}
        check()
        progress('evaluating_matched_development')
        wp_summaries = {name: summarize(y[dev], p, game[dev], args.resamples) for name, p in probabilities.items()}
        wp_comparisons = {name+'_vs_original_trajectory': paired_comparison(
            y[dev], probabilities['original_trajectory'], probabilities[name], game[dev], args.resamples)
            for name in ('rich_snapshot', 'rich_trajectory')}
        wp_comparisons['rich_trajectory_vs_rich_snapshot'] = paired_comparison(
            y[dev], probabilities['rich_snapshot'], probabilities['rich_trajectory'], game[dev], args.resamples)
        ece_summaries, ece_deltas = ece_bootstrap(y[dev], probabilities, game[dev], args.resamples)
        risk_scores = {**control_scores, 'latest_two': windows['start_ms'][risk_dev].astype(float),
                       'rich_risk_history': risk_score}
        retrieval_rows, risk_predictions = {}, {}
        for name, score in risk_scores.items():
            ids, rows, selected = setback.game_retrieval(score, windows['label'][risk_dev], windows['game'][risk_dev],
                                                        windows['start_ms'][risk_dev], windows['end_ms'][risk_dev])
            retrieval_rows[name] = rows
            risk_predictions[name+'_score'] = score
            risk_predictions[name+'_selected'] = selected
        risk_summaries = {name: setback.summarize_retrieval(rows, args.resamples) for name, rows in retrieval_rows.items()}
        risk_comparisons = {'rich_risk_history_vs_'+name: setback.paired_retrieval(
            retrieval_rows['rich_risk_history'], retrieval_rows[name], args.resamples)
            for name in ('risk_history', 'risk_logistic', 'latest_two')}
        trajectory = wp_comparisons['rich_trajectory_vs_original_trajectory']
        wp_gate = bool(trajectory['relative_log_loss_reduction'] >= .005
                       and trajectory['log_loss']['ci95'][0] > 0 and trajectory['brier']['ci95'][0] > 0)
        risk_gate = all(comparison['useful_signal_gate'] for comparison in risk_comparisons.values())
        report = {'schema': SCHEMA, 'stage': 'exploratory_development',
                  'report_path': str((args.output/'report.json').resolve()),
                  'causal_claim': False, 'confirmatory_evaluation_done': False,
                  'coaching_benefit_established': False, 'sealed_test_read': False,
                  'input_and_protocol_sha256': frozen, 'rich_cache_sha256': cache_hash,
                  'original_cache_sha256': setback.digest(args.cache), 'artifacts': artifacts,
                  'feature_names': list(rich_names), 'snapshot_features': list(snapshot_names),
                  'focal_sign_features': list(sign_names), 'wp_fit_sampled_frames': len(fit_indices),
                  'population': {role: {'games': len(np.unique(game[mask])), 'frames': int(mask.sum())}
                                 for role, mask in masks.items()},
                  'windows_population': {role: {'games': len(np.unique(windows['game'][windows['role']==role])),
                                                'windows': int((windows['role']==role).sum())}
                                         for role in masks},
                  'source_coverage': coverage, 'wp_methods': wp_summaries, 'wp_comparisons': wp_comparisons,
                  'wp_ece20_bootstrap': ece_summaries, 'wp_paired_ece20_reduction': ece_deltas,
                  'risk_methods': risk_summaries, 'risk_comparisons': risk_comparisons,
                  'risk_probability_diagnostics': {name: summarize(windows['label'][risk_dev], score,
                        windows['game'][risk_dev], args.resamples) for name, score in risk_scores.items() if name != 'latest_two'},
                  'risk_probability_comparisons': {'rich_risk_history_vs_'+name: paired_comparison(
                        windows['label'][risk_dev], risk_scores[name], risk_score, windows['game'][risk_dev], args.resamples)
                        for name in ('risk_history', 'risk_logistic')},
                  'gates': {'rich_wp_improves_original_trajectory': wp_gate,
                            'rich_retrieval_improves_both_strong_controls_and_latest': risk_gate,
                            'practical_coaching_benefit_established': False},
                  'interval_scope': 'paired whole-game bootstrap conditional on fitted models; repeated exploratory development',
                  'limitations': ['Same development cohort already used for earlier hypotheses; no untouched confirmation claim.',
                                  'Additional retrospective spectator state is not a live-POV feature set.',
                                  'Complete-window endpoint retrieval does not establish coachability or improved outcomes.',
                                  'Latest-two comparator was added after the earlier risk results, before this enriched run.',
                                  'All retrieval uses retrospective complete replay pools; no early-alert benefit is established.',
                                  'Two fixed models plus one direct risk model; no post-result search or policy promotion.']}
        report['by_patch'] = {}
        for patch in np.unique(data['patch'][dev]):
            wp_keep = data['patch'][dev] == patch
            risk_keep = windows['patch'][risk_dev] == patch
            patch_games = len(np.unique(game[dev][wp_keep]))
            if patch_games < 30:
                report['by_patch'][str(patch)] = {'games': patch_games, 'status': 'not_reported_below_30_games'}
                continue
            patch_report = {'games': patch_games, 'wp_comparisons': {name+'_vs_original_trajectory': paired_comparison(
                y[dev][wp_keep], probabilities['original_trajectory'][wp_keep], probabilities[name][wp_keep],
                game[dev][wp_keep], args.resamples) for name in ('rich_snapshot', 'rich_trajectory')}}
            if len(np.unique(windows['game'][risk_dev][risk_keep])) >= 30:
                rows_by_name = {name: setback.game_retrieval(score[risk_keep], windows['label'][risk_dev][risk_keep],
                    windows['game'][risk_dev][risk_keep], windows['start_ms'][risk_dev][risk_keep],
                    windows['end_ms'][risk_dev][risk_keep])[1] for name, score in risk_scores.items()}
                patch_report['risk_comparisons'] = {'rich_risk_history_vs_'+name: setback.paired_retrieval(
                    rows_by_name['rich_risk_history'], rows_by_name[name], args.resamples) for name in ('risk_history', 'risk_logistic', 'latest_two')}
            report['by_patch'][str(patch)] = patch_report
        report['fixed_clock_bands'] = {}
        for lower, upper in ((3, 10), (10, 20), (20, 30), (30, 41)):
            wp_keep = (timestamps[dev] >= lower*60000) & (timestamps[dev] < upper*60000)
            risk_keep = (windows['start_ms'][risk_dev] >= lower*60000) & (windows['start_ms'][risk_dev] < upper*60000)
            band = {'wp_games': len(np.unique(game[dev][wp_keep])),
                    'risk_games': len(np.unique(windows['game'][risk_dev][risk_keep])),
                    'risk_interpretation': 'same fixed start-clock range and complete windows; no future-pool comparison across ranges'}
            if band['wp_games'] >= 30:
                band['wp_comparisons'] = {name+'_vs_original_trajectory': paired_comparison(
                    y[dev][wp_keep], probabilities['original_trajectory'][wp_keep], probabilities[name][wp_keep],
                    game[dev][wp_keep], args.resamples) for name in ('rich_snapshot', 'rich_trajectory')}
            if band['risk_games'] >= 30:
                band['risk_probability_diagnostics'] = {name: summarize(windows['label'][risk_dev][risk_keep],
                    score[risk_keep], windows['game'][risk_dev][risk_keep], args.resamples)
                    for name, score in risk_scores.items() if name != 'latest_two'}
                band['risk_probability_comparisons'] = {'rich_risk_history_vs_'+name: paired_comparison(
                    windows['label'][risk_dev][risk_keep], risk_scores[name][risk_keep], risk_score[risk_keep],
                    windows['game'][risk_dev][risk_keep], args.resamples) for name in ('risk_history', 'risk_logistic')}
            report['fixed_clock_bands'][f'{lower}-{min(upper,40)}min'] = band
        np.savez_compressed(args.output/'development_wp_predictions.npz', y=y[dev], game=game[dev],
                            timestamp_ms=timestamps[dev], **probabilities)
        np.savez_compressed(args.output/'development_windows.npz',
            **{key: windows[key][risk_dev] for key in ('game','start_ms','end_ms','label','sign','gold_change','kill_change','patch')},
            **risk_predictions)
        report['development_wp_predictions_sha256'] = setback.digest(args.output/'development_wp_predictions.npz')
        report['development_windows_sha256'] = setback.digest(args.output/'development_windows.npz')
        check()
        if any(setback.digest(path) != expected for path, expected in frozen.items()):
            raise RuntimeError('an input, implementation, or protocol changed during rich experiment')
        admission.verify_manifest_sources(manifest)
        for path in (args.cache, args.source_manifest, args.w1_artifacts, args.setback_artifacts):
            setback.reject_quarantined(path)
        if setback.digest(cache_path) != cache_hash:
            raise RuntimeError('rich cache changed during fitting')
        report['elapsed_s'] = time.monotonic()-started
        setback.write_json(args.output/'report.json', report)
        (args.output/'report.sha256').write_text(setback.digest(args.output/'report.json')+'  report.json\n', encoding='ascii')
        progress('completed', report=report['report_path'], gates=report['gates'])
        return report
    except Exception as error:
        progress('failed', error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'source-manifest', 'w1-artifacts', 'setback-artifacts', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--deadline-seconds', type=int, default=2400)
    args = parser.parse_args()
    if args.resamples != 2000 or args.deadline_seconds <= 0:
        parser.error('the fixed protocol requires 2000 bootstrap resamples and a positive deadline')
    for name in ('cache', 'source_manifest', 'w1_artifacts', 'setback_artifacts', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    print(json.dumps({'report': report['report_path'], 'gates': report['gates'], 'elapsed_s': report['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
