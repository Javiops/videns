"""Bounded exact row writes to a new C-order NPY file, without a live mmap."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from winprob.bounded_npy_rows_v1 import BoundedNpyRows


class NpyRowWriter:
    def __init__(self, path, shape, dtype):
        self.path = Path(path); self.shape = tuple(shape); self.dtype = np.dtype(dtype)
        if self.path.exists() or len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError('new positive two-dimensional output required')
        mapping = np.lib.format.open_memmap(self.path, mode='w+', shape=self.shape, dtype=self.dtype)
        mapping.flush(); mapping._mmap.close(); del mapping
        info = BoundedNpyRows(self.path, expected_shape=self.shape, expected_dtype=self.dtype)
        self.offset = info.offset; self.row_bytes = info.row_bytes
        self.seen = np.zeros(self.shape[0], dtype=bool); self.stream = self.path.open('r+b')

    def write(self, indices, values):
        rows = np.asarray(indices); a = np.asarray(values)
        if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(rows) > 512
                or len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= self.shape[0])
                or self.seen[rows].any() or a.shape != (len(rows), self.shape[1]) or a.dtype != self.dtype):
            raise ValueError('bounded unique unwritten row batch with exact dtype required')
        if not len(rows): return
        if len(rows) == 1 or np.all(np.diff(rows) == 1):
            self.stream.seek(self.offset+int(rows[0])*self.row_bytes)
            self.stream.write(np.ascontiguousarray(a).tobytes())
        else:
            for index, row in zip(rows, a, strict=True):
                self.stream.seek(self.offset+int(index)*self.row_bytes); self.stream.write(np.ascontiguousarray(row).tobytes())
        self.seen[rows] = True

    def close(self, require_complete=True):
        self.stream.flush(); self.stream.close()
        if require_complete and not self.seen.all(): raise ValueError('incomplete NPY output')
