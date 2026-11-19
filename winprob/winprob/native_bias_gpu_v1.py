"""Bounded CUDA FP64 evaluation of the fixed class-bias objective and gradient."""
from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
from winprob.native_bias_calibration_v1 import normalized_prior, require


def configure():
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.is_available(), 'idle CUDA required; no implicit CPU fallback')
    return {'backend': 'native-bias-cuda-fp64-v1', 'torch_version': torch.__version__,
            'device': torch.cuda.get_device_name(0), 'deterministic': True, 'tf32': False}


class CudaBiasObjective:
    def __init__(self, reader, rows, counts, *, guard=lambda: None, status=lambda **kw: None):
        import torch
        require(torch.cuda.is_available() and torch.are_deterministic_algorithms_enabled(), 'configure deterministic CUDA first')
        self.torch = torch; self.rows = np.asarray(rows, dtype=np.int64).copy(); self.ridge = 1.; self.batch_size = 256
        require(self.rows.ndim == 1 and len(self.rows) > 0 and np.all(np.diff(self.rows) > 0)
                and np.all((self.rows >= 0) & (self.rows < reader.shape[0])), 'ascending unique bounded TRAIN rows required')
        self.classes = reader.shape[1]; self.guard = guard
        self.set_counts(counts)
        self.logp = torch.empty((len(self.rows), self.classes), dtype=torch.float64, device='cuda')
        for lo in range(0, len(self.rows), self.batch_size):
            guard(); hi = min(lo+self.batch_size, len(self.rows))
            p = normalized_prior(reader[self.rows[lo:hi]])
            with np.errstate(divide='ignore'):
                logs = np.log(p)
            self.logp[lo:hi].copy_(torch.from_numpy(logs).to('cuda'))
            if hi % 8192 == 0 or hi == len(self.rows): status(completed=hi, rows=len(self.rows))
        torch.cuda.synchronize()

    def set_counts(self, counts):
        c = np.asarray(counts)
        require(c.shape == (self.classes,) and np.issubdtype(c.dtype, np.integer)
                and np.all(c >= 0) and c.sum() == len(self.rows), 'TRAIN class counts must match cached rows')
        self.counts = c.astype(np.float64)

    def __call__(self, bias):
        self.guard(); torch = self.torch; b = np.asarray(bias, dtype=np.float64)
        require(b.shape == (self.classes,) and np.isfinite(b).all(), 'finite aligned bias required')
        with torch.inference_mode():
            bt = torch.from_numpy(b.copy()).to('cuda')
            value = torch.tensor(.5*float(b@b)-float(self.counts@b), dtype=torch.float64, device='cuda')
            gradient = torch.from_numpy(b-self.counts).to('cuda')
            for lo in range(0, len(self.rows), self.batch_size):
                self.guard(); logits = self.logp[lo:lo+self.batch_size]+bt
                normalizer = torch.logsumexp(logits, dim=1)
                value += normalizer.sum()
                gradient += torch.exp(logits-normalizer[:, None]).sum(dim=0)
            f = float(value.item()); g = gradient.cpu().numpy().copy()
        require(np.isfinite(f) and np.isfinite(g).all(), 'nonfinite GPU calibration objective')
        return f, g

    def close(self):
        if hasattr(self, 'logp'): del self.logp
        self.torch.cuda.empty_cache()
