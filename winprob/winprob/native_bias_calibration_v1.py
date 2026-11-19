"""Strictly convex, identity-anchored class-bias calibration of a fixed predictor.

The caller admits games and verifies paired target support before aggregating
TRAIN counts. Prediction accepts only a prior and frozen biases. This is the
W=I subfamily of softmax(W log p + b), not an identified label-shift estimator.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def require(ok, message):
    if not ok:
        raise ValueError(message)


def normalized_prior(prior):
    p = np.asarray(prior, dtype=np.float64)
    require(p.ndim == 2 and all(p.shape) and np.isfinite(p).all() and np.all(p >= 0),
            'finite nonnegative probability rows required')
    total = p.sum(1, keepdims=True)
    require(np.all(total > 0) and np.all(np.abs(total-1) <= 2e-6), 'normalized native probabilities required')
    return p/total


def training_counts(prior, targets):
    p = normalized_prior(prior); y = np.asarray(targets)
    require(y.shape == (len(p),) and np.issubdtype(y.dtype, np.integer)
            and np.all((y >= 0) & (y < p.shape[1])), 'aligned integer TRAIN targets required')
    require(np.all(p[np.arange(len(p)), y] > 0), 'actual TRAIN purchase has zero native support')
    return np.bincount(y, minlength=p.shape[1])


def predict(prior, bias):
    """Preserve support; zero bias returns exact float64-normalized native p."""
    p = normalized_prior(prior); b = np.asarray(bias, dtype=np.float64)
    require(b.shape == (p.shape[1],) and np.isfinite(b).all(), 'one finite bias per catalogue group required')
    if np.all(b == 0):
        return p
    with np.errstate(divide='ignore'):
        logits = np.log(p)+b
    normalizer = logsumexp(logits, axis=1, keepdims=True)
    q = np.exp(logits-normalizer)
    require(np.isfinite(q).all() and np.all(q[p == 0] == 0), 'calibration changed zero support or became nonfinite')
    return q


class BiasObjective:
    """Sum log partitions - TRAIN counts dot b + ridge/2 ||b||^2.

The fixed baseline NLL constant is omitted. Paired support must be verified
by training_counts (or the bounded admitted reader) before construction.
The Hessian is ridge*I plus a sum of categorical covariance matrices.
"""
    def __init__(self, prior, counts, ridge=1., batch_size=256):
        p = normalized_prior(prior); counts = np.asarray(counts)
        require(counts.shape == (p.shape[1],) and np.issubdtype(counts.dtype, np.integer)
                and np.all(counts >= 0) and counts.sum() == len(p), 'TRAIN class counts must sum to training rows')
        require(np.all((p > 0).any(0) | (counts == 0)), 'positive count for unsupported class')
        require(np.isfinite(ridge) and ridge > 0 and type(batch_size) is int and batch_size > 0,
                'positive fixed ridge and bounded batch required')
        with np.errstate(divide='ignore'):
            self.logp = np.log(p)
        self.counts = counts.astype(np.float64); self.ridge = float(ridge); self.batch_size = batch_size

    def __call__(self, bias):
        b = np.asarray(bias, dtype=np.float64)
        require(b.shape == self.counts.shape and np.isfinite(b).all(), 'invalid bias vector')
        value = .5*self.ridge*float(b@b)-float(self.counts@b)
        gradient = self.ridge*b-self.counts
        for lo in range(0, len(self.logp), self.batch_size):
            logits = self.logp[lo:lo+self.batch_size]+b
            z = logsumexp(logits, axis=1)
            value += float(z.sum())
            gradient += np.exp(logits-z[:, None]).sum(0)
        require(np.isfinite(value) and np.isfinite(gradient).all(), 'nonfinite calibrated objective')
        return value, gradient

    def hessian_vector(self, bias, vector):
        b = np.asarray(bias, dtype=np.float64); v = np.asarray(vector, dtype=np.float64)
        require(b.shape == v.shape == self.counts.shape and np.isfinite(b).all() and np.isfinite(v).all(),
                'finite Hessian-vector arguments required')
        product = self.ridge*v.copy()
        for lo in range(0, len(self.logp), self.batch_size):
            logits = self.logp[lo:lo+self.batch_size]+b
            q = np.exp(logits-logsumexp(logits, axis=1, keepdims=True))
            product += (q*(v-(q@v)[:, None])).sum(0)
        return product


def fit(objective, guard=lambda: None):
    """Fixed optimizer; no retry, sweep, candidate selection or target prediction."""
    calls = 0
    def checked(b):
        nonlocal calls
        guard()
        require(calls < 2000, 'fixed 2000 optimizer-evaluation cap reached')
        calls += 1
        return objective(b)
    initial = np.zeros_like(objective.counts)
    result = minimize(checked, initial, method='L-BFGS-B', jac=True,
        options={'maxiter': 500, 'maxfun': 2000, 'maxls': 20, 'maxcor': 20, 'gtol': 1e-6, 'ftol': 0.})
    guard(); value, gradient = objective(result.x)
    residual = float(np.max(np.abs(gradient)))
    receipt = {'success': bool(result.success), 'qualified': bool(result.success and residual < 1e-5),
        'message': str(result.message), 'iterations': int(result.nit), 'optimizer_evaluations': calls,
        'gradient_rechecks': 1, 'evaluations': calls+1,
        'gradient_max_abs': residual, 'relative_penalized_nll_sum': value, 'ridge': objective.ridge,
        'numerical_strong_convexity_gap_bound': float(gradient@gradient/(2*objective.ridge)),
        'numerical_distance_to_unique_minimum_bound': float(np.linalg.norm(gradient)/objective.ridge),
        'bias_sum': float(result.x.sum()), 'parameters': len(result.x),
        'bound_scope': 'Analytic strong-convexity bounds evaluated with float64 gradients, not interval arithmetic.',
        'accuracy_improvement_claimed': False}
    return result.x, receipt


def control_permutation(basket_sizes, seed=20261004):
    sizes = np.asarray(basket_sizes)
    require(sizes.ndim == 1 and np.issubdtype(sizes.dtype, np.integer)
            and np.all((sizes >= 0) & (sizes <= 6)), 'public counted main-basket sizes required')
    rng = np.random.default_rng(seed); permutation = np.arange(len(sizes))
    for size in range(7):
        rows = np.flatnonzero(sizes == size); permutation[rows] = rng.permutation(rows)
    require(np.array_equal(np.sort(permutation), np.arange(len(sizes)))
            and np.array_equal(sizes[permutation], sizes), 'control changed basket-size strata')
    return permutation
