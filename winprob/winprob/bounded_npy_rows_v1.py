"""Read small exact row batches from a C-order NPY file without whole-file mmap."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def require(ok, message):
    if not ok: raise ValueError(message)


class BoundedNpyRows:
    def __init__(self, path, *, expected_shape, expected_dtype, max_rows=512):
        self.path = Path(path).resolve(strict=True)
        require(type(max_rows) is int and 1 <= max_rows <= 512, 'bounded maximum row count required')
        self.max_rows = max_rows
        with self.path.open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            require(version in ((1, 0), (2, 0)), 'only fixed numeric NPY v1/v2 headers accepted')
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = reader(stream)
            self.offset = stream.tell()
        self.shape = tuple(shape); self.dtype = np.dtype(dtype)
        require(self.shape == tuple(expected_shape) and len(self.shape) == 2 and all(v > 0 for v in self.shape)
                and self.dtype == np.dtype(expected_dtype) and self.dtype.kind in 'fiu'
                and not self.dtype.hasobject and not fortran, 'fixed numeric C-order NPY shape/dtype required')
        self.row_bytes = self.shape[1]*self.dtype.itemsize
        require(self.path.stat().st_size == self.offset+self.shape[0]*self.row_bytes, 'NPY byte length disagrees with header')

    def __len__(self): return self.shape[0]

    def _rows(self, indices):
        count = len(indices)
        require(count <= self.max_rows and np.all((indices >= 0) & (indices < len(self))), 'requested rows exceed bounds')
        result = np.empty((count, self.shape[1]), dtype=self.dtype)
        if not count: return result
        with self.path.open('rb') as stream:
            if count == 1 or np.all(np.diff(indices) == 1):
                stream.seek(self.offset+int(indices[0])*self.row_bytes)
                raw = stream.read(count*self.row_bytes)
                require(len(raw) == count*self.row_bytes, 'short NPY row read')
                result[:] = np.frombuffer(raw, dtype=self.dtype).reshape(result.shape)
            else:
                for i, row in enumerate(indices):
                    stream.seek(self.offset+int(row)*self.row_bytes); raw = stream.read(self.row_bytes)
                    require(len(raw) == self.row_bytes, 'short NPY row read')
                    result[i] = np.frombuffer(raw, dtype=self.dtype)
        return result

    def __getitem__(self, index):
        if isinstance(index, tuple):
            require(len(index) == 2, 'paired row/column indexing only')
            rows, columns = index
            values = self[rows]; cols = np.asarray(columns)
            require(values.ndim == 2 and cols.shape == (len(values),) and np.issubdtype(cols.dtype, np.integer)
                    and np.all((cols >= 0) & (cols < self.shape[1])), 'aligned paired column indices required')
            return values[np.arange(len(values)), cols]
        scalar = isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))
        if scalar:
            indices = np.array([index], dtype=np.int64)
        elif isinstance(index, slice):
            require(index.step in (None, 1), 'only contiguous slices accepted')
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            require(type(start) is int and type(stop) is int and 0 <= start <= stop <= len(self)
                    and stop-start <= self.max_rows, 'slice exceeds explicit row bounds')
            indices = np.arange(start, stop, dtype=np.int64)
        else:
            indices = np.asarray(index)
            require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer), 'one-dimensional integer row indices required')
        values = self._rows(indices)
        return values[0] if scalar else values
