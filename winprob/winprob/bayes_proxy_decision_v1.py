"""Fixed controls and target-free anchored posterior decisions."""
from __future__ import annotations
import hashlib
import numpy as np
from winprob.bayes_proxy_ratio_v1 import posterior, anchor, require

ARMS = ('history', 'scrambled', 'static_memory')


def swap_evidence(match_id, participant_id, start_ms):
    key = f'bayes-proxy-v1/sign/{match_id}|{int(participant_id)}|{int(start_ms)}'
    return bool(hashlib.sha256(key.encode('utf-8')).digest()[0] & 1)


def decisions(native, state_proxy, memory_proxy, past_likelihoods, swaps):
    raw = np.asarray(native); p = np.asarray(raw, dtype=np.float64)
    require(p.ndim == 1 and np.isfinite(p).all() and np.all(p >= 0) and abs(p.sum()-1) < 2e-6, 'native prior required')
    p = p/p.sum(); evidence = np.asarray(past_likelihoods, dtype=np.float64); signs = np.asarray(swaps)
    require(evidence.ndim == 2 and evidence.shape[1] == 2 and signs.shape == (len(evidence),)
            and signs.dtype == bool, 'aligned closed evidence/control identities required')
    altered = evidence.copy(); altered[signs] = altered[signs, ::-1]
    prior = np.array([.5, .5]); weights = {'history': posterior(evidence, prior), 'scrambled': posterior(altered, prior),
        'static_memory': np.array([0., 1.]) if len(evidence) else prior.copy()}
    forecasts = np.stack([state_proxy, memory_proxy]); result = {}
    for arm in ARMS:
        q, _ = anchor(p, forecasts, prior, weights[arm])
        result[arm] = {'probabilities': q, 'weights': weights[arm], 'neutral': bool(np.array_equal(q, p))}
    if not len(evidence): require(all(r['neutral'] for r in result.values()), 'empty history changed M2')
    return result
