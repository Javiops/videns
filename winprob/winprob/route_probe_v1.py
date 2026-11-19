"""Fixed real/permuted full-catalogue route experiment calculations."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from winprob.route_marginal_v1 import require, simplex, bridge
from winprob.route_capacity_v1 import residual_belief, purchase_loss_gradient
from winprob.recipe_transition_probe_v1 import game_weights

ARMS = ('recipe', 'permuted')
SEED = 20261003


def kernels(fraction_units, basket_sizes):
    score = np.asarray(fraction_units)
    sizes = np.asarray(basket_sizes)
    require(score.ndim == 2 and score.dtype == np.uint8 and np.all(score <= 60)
            and sizes.shape == (len(score),) and np.issubdtype(sizes.dtype, np.integer)
            and np.all((sizes >= 0) & (sizes <= 6)), 'invalid public compatibility geometry')
    raw = score.astype(np.float64)/60
    r = np.exp(raw-raw.max(1, keepdims=True)); r /= r.sum(1, keepdims=True)
    permutation = np.arange(len(r)); rng = np.random.default_rng(SEED)
    for size in range(7):
        rows = np.flatnonzero(sizes == size)
        permutation[rows] = rng.permutation(rows)
    require(np.array_equal(np.sort(permutation), np.arange(len(r)))
            and np.array_equal(sizes[permutation], sizes), 'control changed count strata')
    return {'recipe': r, 'permuted': r[permutation]}, permutation


def normalized_prior(values):
    p = np.asarray(values, dtype=np.float64)
    require(p.ndim == 2 and all(p.shape) and np.isfinite(p).all() and np.all(p >= 0)
            and np.all(np.abs(p.sum(1)-1) <= 2e-6), 'normalized native catalogue prior required')
    return p/p.sum(1, keepdims=True)


@dataclass(frozen=True)
class ContextScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values, game_ids):
        x = np.asarray(values, dtype=np.float64)
        require(x.ndim == 2 and all(x.shape) and np.isfinite(x).all()
                and np.array_equal(x[:, 0], np.ones(len(x))) and len(game_ids) == len(x),
                'finite training-only contexts with intercept required')
        weight = game_weights(game_ids); mean = weight @ x
        var = weight @ ((x-mean)**2); scale = np.sqrt(np.maximum(var, 0))
        scale[var < 1e-12] = 1.; mean[0] = 0.; scale[0] = 1.
        return cls(mean, scale)

    def transform(self, values):
        x = np.asarray(values, dtype=np.float64)
        require(self.mean.ndim == 1 and self.scale.shape == self.mean.shape
                and np.isfinite(self.mean).all() and np.isfinite(self.scale).all()
                and np.all(self.scale > 0) and self.mean[0] == 0 and self.scale[0] == 1
                and x.ndim == 2 and x.shape[1] == len(self.mean) and np.isfinite(x).all()
                and np.array_equal(x[:, 0], np.ones(len(x))), 'invalid context scaler or intercept')
        return (x-self.mean)/self.scale

    def as_dict(self):
        return {'mean': self.mean.tolist(), 'scale': self.scale.tolist()}

    @classmethod
    def from_dict(cls, spec):
        require(set(spec) == {'mean', 'scale'}, 'complete scaler specification required')
        return cls(np.asarray(spec['mean'], dtype=np.float64), np.asarray(spec['scale'], dtype=np.float64))


def objective(flat, context, marginal, target_routing, weights, *, ridge, batch_size=512):
    x = np.asarray(context, dtype=np.float64); m = np.asarray(marginal, dtype=np.float64)
    y = np.asarray(target_routing, dtype=np.float64); w = simplex(weights, 'game weights', 1)
    require(x.ndim == 2 and m.ndim == 2 and y.shape == m.shape and len(x) == len(m) == len(w)
            and np.size(flat) == x.shape[1]*m.shape[1] and type(batch_size) is int and batch_size > 0
            and np.isfinite(ridge) and ridge >= 0, 'aligned bounded objective required')
    theta = np.asarray(flat, dtype=np.float64).reshape(x.shape[1], m.shape[1])
    value = ridge*np.sum(theta*theta); gradient = 2*ridge*theta
    for lo in range(0, len(x), batch_size):
        sl = slice(lo, lo+batch_size); mass = w[sl].sum()
        if mass == 0:
            continue
        v, g = purchase_loss_gradient(theta, x[sl], m[sl], y[sl], w[sl]/mass, 0.)
        value += mass*v; gradient += mass*g
    return float(value), gradient.ravel()


def fit(context, marginal, target_routing, weights, *, max_iter=500, guard=None):
    x = np.asarray(context, dtype=np.float64); m = np.asarray(marginal, dtype=np.float64)
    require(type(max_iter) is int and 1 <= max_iter <= 500, 'fixed bounded optimizer required')
    ridge = .01/m.shape[1]; shape = (x.shape[1], m.shape[1])
    def checked_objective(t):
        if guard is not None:
            guard()
        return objective(t, x, m, target_routing, weights, ridge=ridge)
    result = minimize(checked_objective,
        np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 0., 'maxls': 50})
    receipt = {'success': bool(result.success), 'iterations': int(result.nit),
        'evaluations': int(result.nfev), 'message': str(result.message),
        'relative_regularized_objective': float(result.fun),
        'gradient_max_abs': float(np.max(np.abs(result.jac))), 'parameters': int(result.x.size),
        'ridge': ridge, 'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 0., 'maxls': 50,
        'global_optimum_claimed': False, 'initialization': 'zero', 'actual_NLL_reported': False}
    return result.x.reshape(shape), receipt


def predict(prior, routing, context, theta):
    """No target argument. All catalogue actions remain in the distribution."""
    p = normalized_prior(prior); r = simplex(routing, 'routing', 2)
    x = np.asarray(context, dtype=np.float64); t = np.asarray(theta, dtype=np.float64)
    require(x.ndim == 2 and t.shape == (x.shape[1], r.shape[1]) and len(x) == len(p)
            and p.shape[1] == r.shape[0] and np.isfinite(x).all() and np.isfinite(t).all(),
            'invalid target-free predictor inputs')
    u = x @ t; m = p @ r
    neutral = np.ptp(u, axis=1) == 0
    alpha = residual_belief(m, u); q, _ = bridge(p, r, alpha)
    require(np.array_equal(q[neutral], p[neutral]), 'neutral full-catalogue prior parity failed')
    return q, neutral
