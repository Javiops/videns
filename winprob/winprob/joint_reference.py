"""Lift a grouped policy with a declared full-action completion distribution.

The result is a new full-action reference M2 x nu, whose group marginal is M2.
It is not a known human logging propensity. Full-action argmax followed by
projection need not equal group-marginal argmax or the native displayed policy.
"""
from __future__ import annotations

import numpy as np


def group_marginal(full_mass, full_to_group, group_count):
    mass = np.asarray(full_mass, dtype=np.float64)
    mapping = np.asarray(full_to_group)
    if (mass.ndim != 2 or mapping.shape != (mass.shape[1],)
            or not np.issubdtype(mapping.dtype, np.integer)
            or type(group_count) is not int or group_count < 1
            or np.any(mapping < 0) or np.any(mapping >= group_count)
            or not np.isfinite(mass).all() or np.any(mass < 0)):
        raise ValueError('invalid full-action mass or group mapping')
    result = np.zeros((mass.shape[0], group_count), dtype=np.float64)
    np.add.at(result, (np.arange(len(mass))[:, None], mapping[None, :]), mass)
    return result


def lift_group_policy(group_probability, raw_full_logits, full_to_group):
    """Return P(g)*softmax(raw logits restricted to variants of g).

    Groupwise max subtraction avoids losing a rare group's conditional
    distribution to global softmax underflow. No smoothing invents support.
    """
    base = np.asarray(group_probability, dtype=np.float64)
    logits = np.asarray(raw_full_logits, dtype=np.float64)
    mapping = np.asarray(full_to_group)
    if (base.ndim != 2 or not all(base.shape) or not np.isfinite(base).all()
            or np.any(base < 0) or not np.allclose(base.sum(axis=1), 1., atol=2e-6, rtol=0)
            or logits.ndim != 2 or logits.shape[0] != len(base)
            or np.isnan(logits).any() or np.isposinf(logits).any()
            or mapping.shape != (logits.shape[1],) or not np.issubdtype(mapping.dtype, np.integer)
            or set(mapping.tolist()) != set(range(base.shape[1]))):
        raise ValueError('complete group reference, raw logits and covering map required')
    maximum = np.full(base.shape, -np.inf)
    np.maximum.at(maximum, (np.arange(len(base))[:, None], mapping[None, :]), logits)
    finite = np.isfinite(logits)
    centered = np.full_like(logits, -np.inf)
    np.subtract(logits, maximum[:, mapping], out=centered, where=finite)
    unnormalized = np.exp(centered)
    totals = group_marginal(unnormalized, mapping, base.shape[1])
    if np.any((base > 0) & (totals == 0)):
        raise ValueError('positive reference group has no supported full variant')
    conditional = np.divide(unnormalized, totals[:, mapping], out=np.zeros_like(logits),
                            where=totals[:, mapping] > 0)
    joint = base[:, mapping] * conditional
    error = np.max(np.abs(group_marginal(joint, mapping, base.shape[1])-base))
    if error > 1e-12:
        raise ValueError('lift failed group-marginal conservation')
    return joint, conditional
