"""Fit small residual arms only after exact source-M2 and population gates."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from winprob import admission
from winprob.ipo_residual import (ACTION_NAMES, CONTEXT_NAMES, ContextStandardizer, action_features,
    canonical_reference, context_features, fit_ipo_ridge, fit_winner_only, pair_design, residual_policy, residual_scores)
from winprob.preference_reference import load_reference
from winprob.preference_evaluation import evaluate_preferences
from boots_support import load_static
from setback_experiment import digest, write_json, reject_quarantined, set_below_normal_priority, check_resources


def qualify_training_inputs(rows, probabilities, usable, audit):
    if audit.get('minimum_population_gate') is not True:
        raise ValueError('insufficient postmapping/context-qualified fit or unseen challenge population; no training')
    usable = np.asarray(usable, dtype=bool)
    if usable.shape != (len(rows),) or probabilities.shape[0] != len(rows):
        raise ValueError('reference rows/mask/tensor mismatch')
    selected = [row for row, keep in zip(rows, usable) if keep]
    if any(row['role'] not in ('fit', 'calibration', 'development') for row in selected):
        raise ValueError('forbidden preference partition')
    if any(row['patch'] != '16.19' for row in selected if row['role'] == 'development'):
        raise ValueError('development reference must be M2-untrained patch16.19')
    for row in selected:
        if row.get('action_a') == row.get('action_b') or row.get('group_a') == row.get('group_b'):
            raise ValueError('same-action control cannot enter directional training')
    return selected, probabilities[usable]


def zero_residual_gate(context, actions, probabilities, group_a, group_b):
    theta = np.zeros((context.shape[1], actions.shape[1]))
    actual = residual_policy(context, actions, theta, probabilities)
    expected = np.asarray(probabilities)
    if (actual.dtype != expected.dtype or not np.array_equal(actual, expected)
            or actual.tobytes() != expected.tobytes()):
        raise ValueError('actual adapter zero residual changes native reference bytes')
    # Full ranking must also match the unmodified exported tensor. Stable tie
    # handling here tests ordering preservation, not a replacement source decoder.
    order_mismatches = 0
    for old, new in zip(probabilities, actual):
        order_mismatches += not np.array_equal(np.argsort(old, kind='stable'), np.argsort(new, kind='stable'))
    raw_odds = np.log(probabilities[np.arange(len(context)), group_b].astype(float))-np.log(probabilities[np.arange(len(context)), group_a].astype(float))
    new_odds = np.log(actual[np.arange(len(context)), group_b].astype(float))-np.log(actual[np.arange(len(context)), group_a].astype(float))
    odds_error = float(np.max(np.abs(raw_odds-new_odds)))
    if order_mismatches or odds_error > 1e-12:
        raise ValueError('zero residual changes native reference order or pair odds')
    return {'probability_identity_bytes_exact': True, 'probability_dtype_preserved': str(actual.dtype),
            'full_rank_order_mismatches': int(order_mismatches), 'maximum_pair_log_odds_error': odds_error,
            'maximum_reference_renormalization_applied_at_zero': 0.,
            'native_options_identity': 'required separately on every row by the source artifact consumer'}


def fit_retained_frequency(rows):
    if not rows or any(row.get('role') != 'fit' for row in rows):
        raise ValueError('frequency baseline accepts fit rows only')
    global_counts, cells = Counter(), defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        champion = str(int(row['source_projection']['champion_id']))
        item = str(int(row['action_b']))
        patch = row['patch']
        tuple(map(int, patch.split('.')))  # fail rather than invent patch order
        global_counts[item] += 1
        cells[champion][patch][item] += 1
    return {'schema': 'retained-revision-frequency-v1', 'smoothing': .5,
            'population': 'unweighted retained B counts from the same permitted fit revision pairs; not ordinary-shop frequencies',
            'global_counts': dict(global_counts),
            'by_champion_patch': {champ: {patch: dict(counts) for patch, counts in patches.items()}
                                  for champ, patches in cells.items()}}


def frequency_pair_odds(model, rows):
    result, modes = [], Counter()
    for row in rows:
        champion = str(int(row['source_projection']['champion_id']))
        cells = model['by_champion_patch'].get(champion, {})
        if row['patch'] in cells:
            counts, mode = cells[row['patch']], 'same_patch_champion'
        elif cells:
            latest = max(cells, key=lambda value: tuple(map(int, value.split('.'))))
            counts, mode = cells[latest], 'latest_fit_patch_champion'
        else:
            counts, mode = model['global_counts'], 'global_fit'
        a, b = str(int(row['action_a'])), str(int(row['action_b']))
        result.append(np.log(counts.get(b, 0)+model['smoothing'])-np.log(counts.get(a, 0)+model['smoothing']))
        modes[mode] += 1
    return np.asarray(result), dict(modes)


def policy_drift(context, actions, theta, probabilities, weights, group_a, group_b):
    weights = np.asarray(weights, dtype=float); weights = weights/weights.sum()
    kl, changes, maximum = [], [], []
    for lo in range(0, len(context), 64):
        hi = min(lo+64, len(context))
        base = canonical_reference(probabilities[lo:hi])
        current = residual_policy(context[lo:hi], actions, theta, probabilities[lo:hi]).astype(float)
        current /= current.sum(axis=1, keepdims=True)
        positive = current > 0
        terms = np.zeros_like(current)
        terms[positive] = current[positive]*(np.log(current[positive])-np.log(base[positive]))
        kl.extend(terms.sum(axis=1))
        changes.extend(np.argmax(current, axis=1) != np.argmax(base, axis=1))
        maximum.extend(np.max(np.abs(residual_scores(context[lo:hi], actions, theta)), axis=1))
    correction = pair_design(context, actions, group_b, group_a)@theta.ravel()
    kl = np.asarray(kl); changes = np.asarray(changes); maximum = np.asarray(maximum)
    return {'weighted_mean_kl_candidate_to_m2': float(weights@kl), 'maximum_kl': float(kl.max()),
            'weighted_canonical_group_top1_change_fraction': float(weights@changes),
            'weighted_mean_absolute_pair_log_odds_correction': float(weights@np.abs(correction)),
            'maximum_absolute_pair_log_odds_correction': float(np.abs(correction).max()),
            'maximum_absolute_full_catalogue_residual': float(maximum.max()),
            'scope': 'full canonical probabilities; not a displayed-policy legality or win-effect claim'}


def run(args):
    started = time.monotonic()
    set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(stage, **extra):
        write_json(args.output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra})
    try:
        status('validating_exact_reference_and_populations')
        check_resources(started, args.deadline_seconds)
        for path in (args.reference, args.admission, args.static_export):
            reject_quarantined(path)
        rows, probabilities, groups, usable, consumer_audit = load_reference(args.reference, args.admission)
        write_json(args.output/'reference_consumer_audit.json', consumer_audit)
        rows, probabilities = qualify_training_inputs(rows, probabilities, usable, consumer_audit)
        static, bindings = load_static(args.static_export)
        if static.get('16.18', {}).get('version') != '16.18.1':
            raise ValueError('action features require M2-pinned static world16.18.1')
        reference_manifest = json.loads((args.reference/'manifest.json').read_text(encoding='utf-8'))
        for name, expected in reference_manifest['files'].items():
            bindings[str((args.reference/name).resolve())] = expected
        for path in (args.reference/'manifest.json', args.reference/'manifest.sha256', args.admission,
                     args.admission.with_suffix('.sha256'), Path(__file__).resolve(),
                     ROOT/'winprob/ipo_residual.py', ROOT/'winprob/preference_reference.py',
                     ROOT/'winprob/preference_evaluation.py', ROOT/'winprob/admission.py',
                     ROOT/'scripts/boots_support.py', ROOT/'scripts/setback_experiment.py',
                     ROOT/'docs/ipo_experiment_design_v1.md', ROOT/'docs/ipo_feature_map_v1.md'):
            bindings[str(path.resolve())] = digest(path)
        roles = np.asarray([row['role'] for row in rows])
        fit, dev = roles == 'fit', roles == 'development'
        weights = np.asarray([row['random_focal_pair_weight'] for row in rows], dtype=float)
        group_a = np.asarray([row['group_a'] for row in rows], dtype=int)
        group_b = np.asarray([row['group_b'] for row in rows], dtype=int)
        raw_context = context_features(rows, {patch: data['champion_tags'] for patch, data in static.items()})
        actions = action_features(groups, static['16.18']['items'])
        scaler = ContextStandardizer.fit(raw_context[fit], weights[fit])
        context = scaler.transform(raw_context)
        identity = zero_residual_gate(context, actions, probabilities, group_a, group_b)
        write_json(args.output/'pre_fit_gate.json', {'reference_consumer': consumer_audit,
            'adapter_zero_residual': identity, 'bindings_sha256': bindings, 'feature_schema': 'ipo-pov-static-v1',
            'fit_context_standardization_only': True, 'real_training_started': False})
        check_resources(started, args.deadline_seconds)
        with threadpool_limits(limits=1):
            status('fitting_ipo_weighted_ridge')
            ipo, ipo_fit = fit_ipo_ridge(context[fit], actions, group_b[fit], group_a[fit], weights[fit], tau=.5, ridge=.01)
            write_json(args.output/'fit_diagnostics.json', {'ipo': ipo_fit, 'winner_only': None, 'development_evaluated': False})
            if ipo_fit['gradient_max_abs'] > 1e-7:
                raise ValueError('IPO linear system failed its stationarity check')
            check_resources(started, args.deadline_seconds)
            status('fitting_true_winner_only_full_catalogue')
            winner, winner_fit = fit_winner_only(context[fit], actions, probabilities[fit], group_b[fit], weights[fit], ridge=.01, max_iter=250)
            write_json(args.output/'fit_diagnostics.json', {'ipo': ipo_fit, 'winner_only': winner_fit, 'development_evaluated': False})
            if not winner_fit['converged']:
                raise ValueError('winner-only full-catalogue optimizer did not converge; no comparison: '
                                 f"{winner_fit['message']}; gradient_max_abs={winner_fit['gradient_max_abs']}")
        check_resources(started, args.deadline_seconds)
        model_path = args.output/'residual_models.npz'
        np.savez_compressed(model_path, theta_ipo=ipo, theta_winner_only=winner,
                            context_mean=scaler.mean, context_scale=scaler.scale, action_features=actions)
        model_hash = digest(model_path)
        frequency_model = fit_retained_frequency([row for row, keep in zip(rows, fit) if keep])
        write_json(args.output/'retained_frequency_model.json', frequency_model)
        report = {'schema': 'ipo-residual-experiment-v1', 'stage': 'training_only',
                  'report_path': str((args.output/'report.json').resolve()), 'feature_schema': 'ipo-pov-static-v1',
                  'context_names': list(CONTEXT_NAMES), 'action_names': list(ACTION_NAMES),
                  'reference_consumer_audit': consumer_audit, 'adapter_zero_residual': identity,
                  'bindings_sha256': bindings, 'model_sha256': model_hash,
                  'context_features': context.shape[1], 'action_features': actions.shape[1],
                  'parameters_per_arm': int(ipo.size), 'catalogue_groups': len(groups),
                  'action_feature_rank': int(np.linalg.matrix_rank(actions)),
                  'fit_pairs': int(fit.sum()), 'fit_games': len({row['match_id'] for row, keep in zip(rows, fit) if keep}),
                  'fit_diagnostics': {'ipo': ipo_fit, 'winner_only': winner_fit},
                  'training_performed': True, 'development_evaluated': False,
                  'causal_claim': False, 'win_benefit_established': False, 'preference_benefit_established': False,
                  'feature_collision_pairs': int(np.sum(np.all(actions[group_a] == actions[group_b], axis=1)))}
        # Checkpoint is frozen before any development prediction metric.
        write_json(args.output/'training_lock.json', {'model_sha256': model_hash,
            'training_complete': True, 'development_metric_inspected': False, 'bindings_sha256': bindings})
        if args.evaluate_development:
            status('evaluating_frozen_development_preferences')
            selected = [row for row, keep in zip(rows, dev) if keep]
            reference_odds = np.log(probabilities[dev, :][np.arange(dev.sum()), group_b[dev]].astype(float))-np.log(probabilities[dev, :][np.arange(dev.sum()), group_a[dev]].astype(float))
            design = pair_design(context[dev], actions, group_b[dev], group_a[dev])
            odds = {'m2': reference_odds, 'ipo': reference_odds+design@ipo.ravel(),
                    'winner_only': reference_odds+design@winner.ravel()}
            price_delta = actions[group_b[dev], 0]-actions[group_a[dev], 0]
            odds['price_cheaper'], odds['price_dearer'] = -price_delta, price_delta
            odds['retained_item_frequency'], frequency_fallbacks = frequency_pair_odds(frequency_model, selected)
            evaluator_args = ([row['match_id'] for row in selected], weights[dev],
                              [row['action_a'] for row in selected], [row['action_b'] for row in selected], odds)
            report['development'] = evaluate_preferences(*evaluator_args)
            report['cheap_baseline_check'] = evaluate_preferences(*evaluator_args,
                baselines=('price_cheaper', 'price_dearer', 'retained_item_frequency'), minimum_gain=0.)
            report['frequency_baseline'] = {'model_sha256': digest(args.output/'retained_frequency_model.json'),
                'fit_population': frequency_model['population'], 'development_fallback_counts': frequency_fallbacks}
            report['price_direction_slices'] = {}
            for name, keep in (('B_cheaper', price_delta < 0), ('B_dearer', price_delta > 0), ('equal_price', price_delta == 0)):
                game_ids = [row['match_id'] for row, take in zip(selected, keep) if take]
                counts = {'pairs': int(keep.sum()), 'games': len(set(game_ids)),
                          'price_world': 'pinned M2 16.18.1 item totals', 'scope': 'exploratory; not used for selection'}
                if counts['games'] < 30:
                    report['price_direction_slices'][name] = {**counts, 'status': 'not_measured_below_30_games'}
                else:
                    report['price_direction_slices'][name] = {**counts, 'status': 'measured',
                        'metrics': evaluate_preferences(game_ids, weights[dev][keep],
                            np.asarray([row['action_a'] for row in selected])[keep],
                            np.asarray([row['action_b'] for row in selected])[keep],
                            {arm: value[keep] for arm, value in odds.items()})}
            report['development_policy_drift'] = {
                name: policy_drift(context[dev], actions, theta, probabilities[dev], weights[dev], group_a[dev], group_b[dev])
                for name, theta in (('ipo', ipo), ('winner_only', winner))}
            report['stage'] = 'exploratory_preference_development'
            report['development_evaluated'] = True
            report['preference_benefit_established'] = bool(report['development']['metric_gate_passed']
                and report['cheap_baseline_check']['metric_gate_passed'])
            report['benefit_scope'] = 'audited revision challenge only; not causal value, broad coaching or confirmation'
            np.savez_compressed(args.output/'development_pair_predictions.npz',
                pair_id=np.asarray([row['pair_id'] for row in selected]), game=np.asarray([row['match_id'] for row in selected]),
                action_a=np.asarray([row['action_a'] for row in selected]), action_b=np.asarray([row['action_b'] for row in selected]),
                weights=weights[dev], **{name+'_total_log_odds': value for name, value in odds.items()})
            report['development_predictions_sha256'] = digest(args.output/'development_pair_predictions.npz')
        if any(digest(path) != expected for path, expected in bindings.items()) or digest(model_path) != model_hash:
            raise RuntimeError('bound input/code/model changed during experiment')
        admission.verify_manifest_sources(json.loads(args.admission.read_text(encoding='utf-8')))
        report['elapsed_s'] = time.monotonic()-started
        write_json(args.output/'report.json', report)
        status('completed', report=report['report_path'])
        return report
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Complete source-owned full M2 probability artifact')
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--static-export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evaluate-development', action='store_true', help='One frozen comparison after both fitted artifacts are locked')
    parser.add_argument('--deadline-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.deadline_seconds <= 0:
        parser.error('deadline must be positive')
    for name in ('reference', 'admission', 'static_export', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    print(json.dumps({'report': report['report_path'], 'training_performed': report['training_performed'],
                      'development_evaluated': report['development_evaluated'],
                      'preference_benefit_established': report['preference_benefit_established']}, indent=2))


if __name__ == '__main__':
    main()
