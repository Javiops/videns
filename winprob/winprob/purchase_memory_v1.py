"""Target-independent retrieval and fixed semi-parametric purchase prediction."""
from __future__ import annotations

from collections import defaultdict
import numpy as np

K = 32
MIX = .5
SEED = 20261001


def normalized(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('finite nonempty embedding matrix required')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError('zero embedding has no cosine direction')
    return x / norm


def neighbors(query, memory, *, memory_rows, memory_games, query_game, k=K):
    """No label parameter. Caller supplies only SAME-champion training memory."""
    x = np.asarray(memory, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    rows, games = np.asarray(memory_rows), np.asarray(memory_games)
    if (x.ndim != 2 or q.shape != (x.shape[1],) or rows.shape != (len(x),)
            or games.shape != rows.shape or len(np.unique(rows)) != len(rows)
            or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0)
            or not np.isfinite(x).all() or not np.isfinite(q).all()
            or type(k) is not int or k <= 0):
        raise ValueError('invalid retrieval identity/embedding input')
    if np.any(games == query_game):
        raise ValueError('query game entered retrieval memory')
    if not len(x):
        return np.empty(0, dtype=np.int64), np.empty(0), np.empty(0)
    # Direct squared differences avoid tiny negative distances/cancellation.
    distance = np.sum((x-q)**2, axis=1)
    order = np.lexsort((rows, distance))
    chosen, used = [], set()
    for position in order:
        game = int(games[position])
        if game not in used:
            chosen.append(int(position)); used.add(game)
            if len(chosen) == k:
                break
    chosen = np.asarray(chosen, dtype=np.int64)
    distance = distance[chosen]
    temperature = max(float(distance[-1])/4., 1e-6)
    weight = np.exp(-(distance-distance[0])/temperature)
    weight /= weight.sum()
    return rows[chosen].astype(np.int64), distance, weight


def distribution(labels, weights, class_count):
    y, w = np.asarray(labels), np.asarray(weights, dtype=np.float64)
    if (y.ndim != 1 or w.shape != y.shape or not np.issubdtype(y.dtype, np.integer)
            or np.any(y < 0) or np.any(y >= class_count) or not np.isfinite(w).all()
            or np.any(w < 0) or (len(w) and not np.isclose(w.sum(), 1., atol=1e-12))):
        raise ValueError('invalid neighbor purchase distribution')
    return np.bincount(y, weights=w, minlength=class_count)


def mix(base, labels, weights):
    p = np.asarray(base, dtype=np.float64)
    if (p.ndim != 1 or len(p) < 3 or not np.isfinite(p).all()
            or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-12)):
        raise ValueError('normalized base probabilities required')
    r = distribution(labels, weights, len(p))
    return (1.-MIX)*p + MIX*r if len(labels) else p.copy()


def shuffled_labels(labels, rows, champions, patches, timestamps, fold):
    """Permute only declared training rows, within fixed public context bins."""
    y, rows = np.asarray(labels), np.asarray(rows)
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or len(np.unique(rows)) != len(rows) or np.any(rows < 0)
            or np.any(rows >= len(y)) or fold not in (0, 1)):
        raise ValueError('unique training state IDs required')
    buckets = defaultdict(list)
    for row in rows:
        buckets[(int(champions[row]), int(patches[row]), int(timestamps[row]//180000))].append(int(row))
    result = np.full(len(y), -1, dtype=y.dtype)
    rng = np.random.default_rng(SEED+fold)
    singleton = 0
    for key in sorted(buckets):
        members = np.asarray(sorted(buckets[key]), dtype=np.int64)
        result[members] = y[rng.permutation(members)]
        singleton += int(len(members) == 1)
    return result, {'bins': len(buckets), 'singleton_bins': singleton,
                    'training_rows': len(rows), 'unchanged_labels': int(np.sum(result[rows] == y[rows]))}


def compact(probabilities, labels):
    p, y = np.asarray(probabilities, dtype=np.float64), np.asarray(labels)
    if (p.ndim != 2 or y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer)
            or np.any(y < 0) or np.any(y >= p.shape[1]) or not np.isfinite(p).all()
            or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1., atol=1e-12)):
        raise ValueError('aligned normalized prediction probabilities required')
    truep = p[np.arange(len(p)), y]
    if np.any(truep <= 0):
        raise ValueError('finite target NLL requires positive target mass')
    top3 = np.argsort(-p, axis=1, kind='stable')[:, :3]
    return {'top1': top3[:, 0], 'top3': top3, 'target_log_probability': np.log(truep)}


def evaluate(labels, scores, seed=SEED, draws=2000):
    """One query per game is enforced upstream; bootstrap identical game rows."""
    y = np.asarray(labels)
    if set(scores) != {'state', 'memory', 'shuffled'} or not len(y):
        raise ValueError('all fixed arms required')
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(y), size=(draws, len(y)))
    metrics = {}; values = {}
    for arm, score in scores.items():
        if (np.shape(score['top1']) != y.shape or np.shape(score['top3']) != (len(y), 3)
                or np.shape(score['target_log_probability']) != y.shape):
            raise ValueError('unaligned compact predictions')
        v = {'top1': (score['top1'] == y).astype(float),
             'top3': (score['top3'] == y[:, None]).any(axis=1).astype(float),
             'nll': -np.asarray(score['target_log_probability'])}
        values[arm] = v
        metrics[arm] = {k: {'value': float(a.mean()),
                            'interval95': np.quantile(a[indexes].mean(axis=1), [.025, .975]).tolist()}
                        for k, a in v.items()}
    contrasts = {}
    for control in ('state', 'shuffled'):
        contrasts[control] = {}
        for key in ('top1', 'top3', 'nll'):
            d = values['memory'][key]-values[control][key]
            contrasts[control][key] = {'difference': float(d.mean()),
                'interval95': np.quantile(d[indexes].mean(axis=1), [.025, .975]).tolist()}
        d = values['memory']['top1']-values[control]['top1']
        contrasts[control]['corrected'] = int(np.sum(d > 0))
        contrasts[control]['broken'] = int(np.sum(d < 0))
    gate = (contrasts['state']['top1']['difference'] >= .03
            and contrasts['state']['top1']['interval95'][0] > 0
            and contrasts['shuffled']['top1']['interval95'][0] > 0
            and contrasts['state']['nll']['difference'] <= 0)
    return {'games': len(y), 'arms': metrics, 'memory_minus_control': contrasts,
            'metric_gate_passed': bool(gate), 'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'whole game'}}
