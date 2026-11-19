"""Log-space Bayesian evidence and target-free probability-ratio transfer."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp


def require(ok, message):
    if not ok: raise ValueError(message)


def simplex(values, dimension):
    p = np.asarray(values, dtype=np.float64)
    require(p.ndim == dimension and all(p.shape) and np.isfinite(p).all() and np.all(p >= 0)
            and np.all(np.abs(p.sum(-1)-1) <= 1e-12), 'normalized finite probabilities required')
    return p


def posterior(log_likelihoods, prior):
    """Likelihoods of already-closed past observations, ordered [visit, expert]."""
    p = simplex(prior, 1); values = np.asarray(log_likelihoods, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == len(p)
            and not np.isnan(values).any() and np.all(values <= 1e-12), 'past log-likelihood matrix required')
    if len(values) == 0: return p.copy()
    logs = np.full_like(p, -np.inf); np.log(p, out=logs, where=p > 0)
    for row in values:
        logs = logs+row; normalizer = logsumexp(logs)
        require(np.isfinite(normalizer), 'past observation impossible under the proxy mixture')
        logs -= normalizer
    result = np.exp(logs); result /= result.sum()
    if np.all(values == values[:, :1]): return p.copy()
    return result


def anchor(reference, proxy_forecasts, prior, updated_weights):
    """No current target parameter. Returns q and the proxy density ratio."""
    p = simplex(reference, 1); experts = simplex(proxy_forecasts, 2)
    old = simplex(prior, 1); new = simplex(updated_weights, 1)
    require(experts.shape == (len(old), len(p)) and old.shape == new.shape
            and np.all(old > 0), 'aligned experts and positive prior required')
    before = old @ experts; after = new @ experts
    require(np.all(before[p > 0] > 0), 'proxy prior misses native action support')
    ratio = np.ones_like(p); np.divide(after, before, out=ratio, where=before > 0)
    if np.array_equal(old, new) or np.all(experts == experts[:1]):
        return p.copy(), np.ones_like(p)
    q = p*ratio; require(q.sum() > 0, 'posterior has no overlap with native support')
    q /= q.sum()
    require(np.isfinite(q).all() and not np.any((p == 0) & (q != 0)), 'native support conservation failed')
    return q, ratio


def closed_real_prefix(events, cutoff):
    """Metadata helper. SAVE removal precedes closure tests and order/counting.

    Each record carries a stable physical id, start/end, and an already bound
    real-purchase marker. No purchase contents, forecaster output or label used.
    """
    require(type(cutoff) is int, 'integer decision cutoff required')
    real = [e for e in events if e['real_purchase']]
    require(all(type(e['start']) is int and type(e['end']) is int and e['end'] >= e['start'] for e in real), 'invalid real visit chronology')
    selected = [e for e in real if e['end'] < cutoff]
    require(len({e['id'] for e in selected}) == len(selected), 'duplicate physical past visit')
    return tuple(e['id'] for e in sorted(selected, key=lambda e: (e['start'], e['end'], e['id'])))
