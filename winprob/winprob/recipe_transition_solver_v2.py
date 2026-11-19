"""Same convex objective, stronger function-change stopping precision."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from winprob.recipe_transition_v1 import _supervision, _objective


def fit(inputs, target, weights, ridge=.01, max_iter=300):
    y, w = _supervision(inputs, target, weights, ridge)
    if type(max_iter) is not int or not 1 <= max_iter <= 300:
        raise ValueError('fixed bounded optimizer iteration count required')
    result = minimize(_objective, np.zeros(np.prod(inputs.parameter_shape)),
        args=(inputs, y, w, ridge, 512), jac=True, method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 0., 'maxls': 50})
    return result.x.reshape(inputs.parameter_shape), {
        'solver': 'L-BFGS-B', 'solver_version': 'loss-stop-disabled-v2',
        'ftol': 0., 'gtol': 1e-7, 'max_iter': max_iter, 'maxls': 50,
        'converged': bool(result.success), 'iterations': int(result.nit),
        'function_evaluations': int(result.nfev), 'gradient_evaluations': int(result.njev),
        'objective': float(result.fun), 'gradient_max_abs': float(np.max(np.abs(result.jac))),
        'message': str(result.message), 'parameters': len(result.x), 'ridge': ridge,
        'rows': len(y), 'outside_shortlist_rows': int(np.sum(y == 3)),
        'likelihood': 'exact_coarsened_candidates_plus_OTHER', 'source_runtime_imported': False}
