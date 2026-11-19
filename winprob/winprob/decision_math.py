"""Mathematical kernels for proposed decision learning; not a trained policy.

The reference is a policy regularizer, not an assumed historical propensity.
Neither reward shaping nor these losses establish causal action quality.
"""
from __future__ import annotations

import math
import numpy as np


def _finite(values):
    result = np.asarray(values, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError('finite values required')
    return result


def tilted_policy(reference, score, temperature=1.0, legal=None):
    """Argmax_p E_p[score] - temperature * KL(p || conditioned reference).

    A zero score returns the reference conditional on the same legal set.
    Reference zeros remain zero. Legal membership is supplied, never inferred.
    This is an exact finite-action identity, not empirical overlap evidence.
    """
    reference, score = _finite(reference), _finite(score)
    if reference.ndim != 1 or reference.shape != score.shape or not len(reference):
        raise ValueError('matching nonempty vectors required')
    if np.any(reference < 0) or not np.isclose(reference.sum(), 1.0):
        raise ValueError('reference must be a normalized probability vector')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive temperature required')
    legal = np.ones(len(reference), dtype=bool) if legal is None else np.asarray(legal, dtype=bool)
    if legal.shape != reference.shape:
        raise ValueError('legal set shape mismatch')
    support = legal & (reference > 0)
    if not support.any():
        raise ValueError('no reference-supported legal action')
    result = np.zeros_like(reference)
    logits = np.log(reference[support]) + score[support] / temperature
    if not np.isfinite(logits).all():
        raise ValueError('score/temperature overflow')
    mass = np.exp(logits - logits.max())
    result[support] = mass / mass.sum()
    return result


def episodic_wp_rewards(before_values, outcome):
    """Gamma=1 redistribution, paying terminal outcome exactly once.

    before_values[t] is the frozen potential before transition t. The last
    transition gets outcome - before_values[-1]. Sum = outcome - initial WP.
    Equivalent to r + Phi(next) - Phi(current) with original terminal r=Y
    and Phi(terminal)=0; do not add another terminal Y to these rewards.
    """
    values = _finite(before_values)
    if values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1)):
        raise ValueError('nonempty probability sequence required')
    if outcome not in (0, 1):
        raise ValueError('terminal outcome must be binary')
    return np.diff(np.concatenate((values, [float(outcome)])))


def expectile_loss(residual, tau=.7):
    """IQL asymmetric squared loss; an expectile is not a quantile."""
    residual = _finite(residual)
    if not 0 < tau < 1:
        raise ValueError('expectile level must lie strictly between zero and one')
    return np.abs(tau - (residual < 0).astype(float)) * residual**2


def advantage_weights(advantage, temperature=1.0, cap=100.0):
    """Clipped advantage-weighted cloning weights, not action probabilities."""
    advantage = _finite(advantage)
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(cap) or cap <= 0:
        raise ValueError('positive finite temperature/cap required')
    return np.exp(np.minimum(advantage / temperature, math.log(cap)))


def preference_margin(log_chosen, log_rejected, log_reference_chosen, log_reference_rejected):
    return (_finite(log_chosen) - _finite(log_reference_chosen)
            - _finite(log_rejected) + _finite(log_reference_rejected))


def ipo_loss(relative_margin, tau=1.0):
    """Identity Preference Optimization's finite relative-log-odds target."""
    margin = _finite(relative_margin)
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('positive finite regularization required')
    return tau * (margin - 1/(2*tau))**2


def dpo_loss(relative_margin, beta=1.0):
    margin = _finite(relative_margin)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('positive finite scale required')
    return np.logaddexp(0.0, -beta * margin)
