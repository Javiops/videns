"""Same route likelihood/analytic gradient with fixed bases cached in CUDA FP64."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
from scipy.optimize import minimize
from winprob.route_marginal_v1 import require, simplex


def configure():
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'idle CUDA required; no implicit CPU fallback')
    return {'backend': 'cuda-fp64-analytic-v1', 'torch_version': torch.__version__,
            'device': torch.cuda.get_device_name(0), 'deterministic': True, 'tf32': False}


class Objective:
    def __init__(self, context, marginal, target_routing, weights, ridge):
        import torch
        self.torch = torch
        x = np.asarray(context, dtype=np.float64)
        m = simplex(marginal, 'prior route marginal', 2)
        y = simplex(target_routing, 'target route attribution', 2)
        w = simplex(weights, 'game weights', 1)
        require(x.ndim == 2 and np.isfinite(x).all() and m.shape == y.shape
                and len(x) == len(m) == len(w) and np.all(m > 0) and np.all(y > 0)
                and np.isfinite(ridge) and ridge >= 0, 'fixed positive route-kernel inputs required')
        require(torch.cuda.is_available() and torch.are_deterministic_algorithms_enabled(), 'deterministic CUDA must be configured first')
        def tensor(a): return torch.from_numpy(np.array(a, dtype=np.float64, order='C', copy=True)).to('cuda')
        self.x = tensor(x); self.log_m = tensor(m).log(); self.log_y = tensor(y).log(); self.w = tensor(w)
        self.shape = (x.shape[1], m.shape[1]); self.ridge = float(ridge)
        torch.cuda.synchronize()

    def __call__(self, flat):
        torch = self.torch; coefficients = np.asarray(flat, dtype=np.float64)
        require(coefficients.size == np.prod(self.shape) and np.isfinite(coefficients).all(), 'finite aligned route coefficients required')
        with torch.inference_mode():
            theta = torch.from_numpy(np.array(coefficients.reshape(self.shape), copy=True)).to('cuda')
            u = self.x @ theta; sm = self.log_m+u; sy = self.log_y+u
            loss = torch.sum(self.w*(torch.logsumexp(sm, 1)-torch.logsumexp(sy, 1)))+self.ridge*torch.sum(theta*theta)
            alpha = torch.softmax(sm, 1); posterior = torch.softmax(sy, 1)
            gradient = self.x.T @ (self.w[:, None]*(alpha-posterior))+2*self.ridge*theta
            value = float(loss.item()); grad = gradient.cpu().numpy().ravel().copy()
        require(np.isfinite(value) and np.isfinite(grad).all(), 'nonfinite CUDA route objective')
        return value, grad

    def close(self):
        for key in ('x', 'log_m', 'log_y', 'w'):
            if hasattr(self, key): delattr(self, key)
        self.torch.cuda.empty_cache()


def fit(context, marginal, target_routing, weights, *, max_iter=500, guard=None):
    require(type(max_iter) is int and 1 <= max_iter <= 500, 'fixed bounded optimizer required')
    ridge = .01/np.shape(marginal)[1]
    calculation = Objective(context, marginal, target_routing, weights, ridge)
    shape = calculation.shape
    def checked(t):
        if guard is not None: guard()
        return calculation(t)
    try:
        result = minimize(checked, np.zeros(np.prod(shape)), jac=True, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 0., 'maxls': 50})
    finally:
        calculation.close()
    return result.x.reshape(shape), {'success': bool(result.success), 'iterations': int(result.nit),
        'evaluations': int(result.nfev), 'message': str(result.message),
        'relative_regularized_objective': float(result.fun), 'gradient_max_abs': float(np.max(np.abs(result.jac))),
        'parameters': int(result.x.size), 'ridge': ridge, 'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 0., 'maxls': 50,
        'global_optimum_claimed': False, 'initialization': 'zero', 'actual_NLL_reported': False,
        'objective_backend': 'cuda-fp64-analytic-v1'}
