"""Exact bounded leading-row reads for numeric C-order NPY tensors."""
from pathlib import Path
import numpy as np


class BoundedNpyArray:
    def __init__(self, path, expected_shape, expected_dtype, max_rows=256):
        self.path = Path(path).resolve(strict=True); self.max_rows = max_rows
        if type(max_rows) is not int or not 1 <= max_rows <= 8192: raise ValueError('bounded row limit required')
        with self.path.open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            if version not in ((1, 0), (2, 0)): raise ValueError('fixed NPY v1/v2 header required')
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream); self.offset = stream.tell()
        self.shape = tuple(shape); self.dtype = np.dtype(dtype); self.ndim = len(self.shape)
        if (self.shape != tuple(expected_shape) or not self.shape or min(self.shape) <= 0
                or fortran or self.dtype != np.dtype(expected_dtype) or self.dtype.kind not in 'fiu'):
            raise ValueError('fixed numeric C-order tensor schema required')
        self.row_bytes = int(np.prod(self.shape[1:], dtype=np.int64))*self.dtype.itemsize
        if self.path.stat().st_size != self.offset+self.row_bytes*self.shape[0]: raise ValueError('NPY file length changed')

    def __len__(self): return self.shape[0]

    def __getitem__(self, index):
        scalar = isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))
        if scalar: rows = np.array([index], dtype=np.int64)
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start; stop = len(self) if index.stop is None else index.stop
            if index.step not in (None, 1) or type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= len(self):
                raise ValueError('explicit bounded contiguous slice required')
            if stop-start > self.max_rows: raise ValueError('too many requested rows')
            rows = np.arange(start, stop, dtype=np.int64)
        else: rows = np.asarray(index)
        if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(rows) > self.max_rows
                or np.any(rows < 0) or np.any(rows >= len(self))): raise ValueError('integer indices outside row bounds')
        out = np.empty((len(rows), *self.shape[1:]), dtype=self.dtype)
        if len(rows):
            boundaries = np.r_[0, np.flatnonzero(np.diff(rows) != 1)+1, len(rows)]
            with self.path.open('rb') as stream:
                for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
                    count = int(hi-lo); stream.seek(self.offset+int(rows[lo])*self.row_bytes)
                    raw = stream.read(count*self.row_bytes)
                    if len(raw) != count*self.row_bytes: raise ValueError('short tensor read')
                    out[lo:hi] = np.frombuffer(raw, dtype=self.dtype).reshape((count, *self.shape[1:]))
        return out[0] if scalar else out
