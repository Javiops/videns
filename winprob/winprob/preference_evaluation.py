"""Paired game-cluster evaluation of audited revealed preferences.

Inputs must already pass admission, context and reference-support checks.
Log odds are TOTAL policy log p(B)-log p(A), never residual margins alone.
This module cannot certify those upstream checks or a win-rate improvement.
"""
from __future__ import annotations

import numpy as np


def pair_credit(total_log_odds):
    values = np.asarray(total_log_odds, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('finite one-dimensional total log odds required')
    return (values > 0).astype(np.float64) + .5 * (values == 0)


def evaluate_preferences(game_ids, weights, action_a, action_b, total_log_odds_by_arm,
                         *, bootstrap_samples=2000, seed=20260929,
                         candidate='ipo', baselines=('m2', 'winner_only'),
                         minimum_gain=.05):
    """Estimate weighted pair accuracy and paired uncertainty by whole game.

    A resampled game's complete weighted numerator/denominator is repeated.
    Consequently long games do not acquire more weight simply by contributing
    more pairs if the upstream 1/(10*k) player-game weights are respected.
    The estimand is conditional on an eligible sampled player having a pair;
    different numbers of eligible players can give games different total mass.
    """
    ids = np.asarray(game_ids, dtype=str)
    w = np.asarray(weights, dtype=np.float64)
    a, b = np.asarray(action_a), np.asarray(action_b)
    if (ids.ndim != 1 or w.shape != ids.shape or a.shape != ids.shape or b.shape != ids.shape
            or not len(ids) or any(not item for item in ids)
            or not np.isfinite(w).all() or np.any(w <= 0)):
        raise ValueError('nonempty aligned identities/actions and positive finite weights required')
    if np.any(a == b):
        raise ValueError('same-item roundtrips are controls, not directional preferences')
    if (type(bootstrap_samples) is not int or bootstrap_samples < 1
            or not np.isfinite(minimum_gain) or minimum_gain < 0):
        raise ValueError('invalid evaluation configuration')
    names = list(total_log_odds_by_arm)
    if candidate not in names or any(name not in names for name in baselines):
        raise ValueError('candidate and every declared baseline must be present')
    if not baselines or candidate in baselines or len(set(baselines)) != len(baselines):
        raise ValueError('distinct baseline arms required')
    credits = []
    for name in names:
        credit = pair_credit(total_log_odds_by_arm[name])
        if credit.shape != ids.shape:
            raise ValueError('predictions not aligned to the exact same pairs')
        credits.append(credit)
    credits = np.column_stack(credits)
    games, group = np.unique(ids, return_inverse=True)
    if len(games) < 2:
        raise ValueError('whole-game uncertainty requires at least two games')
    # Scaling all weights is immaterial; normalization avoids numeric overflow.
    w = w / w.sum()
    mass = np.bincount(group, weights=w, minlength=len(games))
    sums = np.column_stack([np.bincount(group, weights=w*credits[:, i], minlength=len(games))
                            for i in range(len(names))])
    point = sums.sum(axis=0) / mass.sum()
    counts = np.random.default_rng(seed).multinomial(
        len(games), np.full(len(games), 1/len(games)), size=bootstrap_samples)
    draws = (counts @ sums) / (counts @ mass)[:, None]
    arms = {name: {'weighted_pair_accuracy': float(point[i]),
                   'interval95': np.quantile(draws[:, i], [.025, .975]).tolist(),
                   'unweighted_ties': int(np.count_nonzero(credits[:, i] == .5))}
            for i, name in enumerate(names)}
    contrasts = {}
    ci = names.index(candidate)
    for baseline in baselines:
        bi = names.index(baseline)
        difference = float(point[ci]-point[bi])
        low, high = np.quantile(draws[:, ci]-draws[:, bi], [.025, .975])
        contrasts[baseline] = {
            'accuracy_gain': difference, 'interval95': [float(low), float(high)],
            'practical_threshold': float(minimum_gain),
            'passes_metric_gate': bool(difference >= minimum_gain and low > 0),
        }
    return {
        'pairs': len(ids), 'games': len(games), 'arms': arms,
        'candidate': candidate, 'paired_contrasts': contrasts,
        'metric_gate_passed': all(row['passes_metric_gate'] for row in contrasts.values()),
        'bootstrap': {'unit': 'whole game', 'samples': bootstrap_samples, 'seed': seed,
                      'interval': 'paired percentile, fitted policies held fixed'},
        'estimand': 'weighted accuracy predicting retained B over rejected A among eligible revisions',
        'limits': ['Admission, support, feature and minimum-population gates are checked upstream.',
                   'Development preference prediction is not causal action value or win benefit.'],
    }
