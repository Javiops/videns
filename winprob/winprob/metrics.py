"""Game-weighted prediction metrics and paired cluster uncertainty."""
from __future__ import annotations

import numpy as np


def game_weights(games):
    """Total mass one per game; the caller normalizes across games."""
    _, inverse, counts = np.unique(games, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def per_game_mean(values, games):
    ids, inverse, counts = np.unique(games, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=float))
    return ids, sums / counts


def losses(y, probability):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError('invalid prediction population')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('binary labels required')
    return {
        'log_loss': -(y * np.log(p) + (1-y) * np.log1p(-p)),
        'brier': (y-p)**2,
        'accuracy': ((p >= .5) == y).astype(float),
    }


def bootstrap_mean(values, resamples=2000, seed=20260929):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError('at least two independent games required')
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples)
    for i in range(resamples):
        estimates[i] = values[rng.integers(0, len(values), len(values))].mean()
    return {
        'estimate': float(values.mean()),
        'ci95': [float(v) for v in np.quantile(estimates, [.025, .975])],
        'games': len(values), 'resamples': resamples,
    }


def reliability(y, probability, games, bins=20):
    p = np.clip(np.asarray(probability), 0, 1)
    y = np.asarray(y)
    weights = game_weights(games)
    assignment = np.minimum((p*bins).astype(int), bins-1)
    rows = []
    error = 0.0
    for b in range(bins):
        keep = assignment == b
        if not keep.any():
            continue
        weight = weights[keep].sum()
        predicted = np.average(p[keep], weights=weights[keep])
        observed = np.average(y[keep], weights=weights[keep])
        error += weight*abs(predicted-observed)
        rows.append({'bin_lower': b/bins, 'bin_upper': (b+1)/bins,
                     'mean_probability': float(predicted), 'win_fraction': float(observed),
                     'frames': int(keep.sum()), 'games': len(np.unique(games[keep])),
                     'game_weight': float(weight)})
    return {'ece20': float(error/weights.sum()), 'bins': rows}


def summarize(y, probability, games, resamples=2000):
    result = {'games': len(np.unique(games)), 'frames': len(y),
              'weighting': 'equal games, equal qualifying frames within game'}
    for metric, values in losses(y, probability).items():
        _, grouped = per_game_mean(values, games)
        result[metric] = bootstrap_mean(grouped, resamples)
    result.update(reliability(y, probability, games))
    return result


def paired_comparison(y, reference, candidate, games, resamples=2000):
    """Positive always means candidate improves on reference."""
    ref = losses(y, reference)
    new = losses(y, candidate)
    result = {}
    for name in ref:
        delta = new[name]-ref[name] if name == 'accuracy' else ref[name]-new[name]
        _, grouped = per_game_mean(delta, games)
        result[name] = bootstrap_mean(grouped, resamples)
    _, baseline_loss = per_game_mean(ref['log_loss'], games)
    result['relative_log_loss_reduction'] = result['log_loss']['estimate']/float(baseline_loss.mean())
    return result
