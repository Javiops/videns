"""Capacity of a route bridge, and a purchase-supervised residual gate.

Pure probability calculations. No data, model, future-goal or filesystem reads.
"""
from __future__ import annotations
import numpy as np
from winprob.route_marginal_v1 import require, simplex


def conditional_experts(prior, routing):
    """Columns are p0(action | route); unsupported columns remain zero."""
    p = simplex(prior, 'prior', 1)
    r = simplex(routing, 'routing', 2)
    require(len(p) == len(r), 'action axes differ')
    marginal = p @ r
    experts = np.zeros_like(r)
    np.divide(p[:, None]*r, marginal[None, :], out=experts,
              where=marginal[None, :] > 0)
    return experts, marginal


def pairwise_margin_range(prior, routing, first, second):
    """Exact attainable min/max of q[first]-q[second] over all route beliefs.

    Beating each competitor separately is NOT simultaneous top1 feasibility.
    """
    experts, marginal = conditional_experts(prior, routing)
    require(type(first) is int and type(second) is int
            and 0 <= first < len(experts) and 0 <= second < len(experts),
            'valid integer action indices required')
    delta = experts[first, marginal > 0]-experts[second, marginal > 0]
    return float(delta.min()), float(delta.max())


def residual_belief(prior_marginal, logits):
    """alpha proportional to m0*exp(logits); zero/shared logits are neutral."""
    m = simplex(prior_marginal, 'prior route marginal', 2)
    u = np.asarray(logits, dtype=np.float64)
    require(u.shape == m.shape and np.isfinite(u).all(), 'finite aligned logits required')
    # Exclude unsupported routes even from the numerical shift.
    shift = np.max(np.where(m > 0, u, -np.inf), axis=1, keepdims=True)
    ex = np.exp(np.where(m > 0, u-shift, -np.inf))
    alpha = m*ex; alpha /= alpha.sum(1, keepdims=True)
    neutral = np.ptp(np.where(m > 0, u, shift), axis=1) == 0
    alpha[neutral] = m[neutral]
    return alpha


def purchase_loss_gradient(theta, context, prior_marginal, target_routing,
                           weights, ridge):
    """Relative main-action NLL and gradient for a linear residual gate.

    loss = log(sum_z m0_z exp(x theta)_z)
           - log(sum_z R(target,z) exp(x theta)_z) + ridge*||theta||^2.
    Adding -log(p0(target)) gives the actual NLL. This function omits that
    constant and MUST NOT be reported as actual NLL. Routing is a structural
    feature of the *observed current purchase*, not a future-goal target.
    The difference of two log-sum-exp terms is generally NONCONVEX.
    """
    x = np.asarray(context, dtype=np.float64)
    t = np.asarray(theta, dtype=np.float64)
    m = simplex(prior_marginal, 'prior route marginal', 2)
    y = simplex(target_routing, 'observed purchase routing', 2)
    w = simplex(weights, 'weights', 1)
    require(x.ndim == 2 and t.ndim == 2 and t.shape == (x.shape[1], m.shape[1])
            and len(x) == len(m) == len(w) and m.shape == y.shape
            and np.isfinite(x).all() and np.isfinite(t).all()
            and np.isfinite(ridge) and ridge >= 0, 'invalid gate objective inputs')
    # A target with nonzero p0 cannot attribute mass to an unsupported route.
    require(not np.any((m == 0) & (y > 0)), 'target route outside prior support')
    u = x @ t
    require(np.isfinite(u).all(), 'nonfinite gate logits')
    def log_partition_and_probability(base):
        lb = np.full_like(base, -np.inf)
        np.log(base, out=lb, where=base > 0)
        score = lb+u; shift = score.max(1, keepdims=True)
        exp = np.exp(score-shift); total = exp.sum(1, keepdims=True)
        return (shift+np.log(total))[:, 0], exp/total
    log_m, alpha = log_partition_and_probability(m)
    log_y, posterior = log_partition_and_probability(y)
    loss = np.dot(w, log_m-log_y)+ridge*np.sum(t*t)
    grad = x.T @ (w[:, None]*(alpha-posterior))+2*ridge*t
    require(np.isfinite(loss) and np.isfinite(grad).all(), 'nonfinite gate objective')
    return float(loss), grad
