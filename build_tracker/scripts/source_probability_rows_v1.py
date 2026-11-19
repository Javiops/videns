"""Source-owned exact small probability-row reads; no consumer imports."""
from pathlib import Path
import numpy as np


class ProbabilityRows:
    def __init__(self, path, shape, dtype):
        self.path = Path(path).resolve(strict=True)
        with self.path.open('rb') as stream:
            version = np.lib.format.read_magic(stream)
            if version not in ((1, 0), (2, 0)): raise ValueError('fixed NPY numeric header required')
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            actual, fortran, actual_dtype = reader(stream); self.offset = stream.tell()
        self.shape = tuple(actual); self.dtype = np.dtype(actual_dtype)
        if self.shape != tuple(shape) or len(self.shape) != 2 or fortran or self.dtype != np.dtype(dtype) or self.dtype.kind != 'f':
            raise ValueError('native probability shape/precision/layout changed')
        self.row_bytes = self.shape[1]*self.dtype.itemsize
        if self.path.stat().st_size != self.offset+self.shape[0]*self.row_bytes: raise ValueError('probability file length changed')

    def __getitem__(self, index):
        scalar = isinstance(index, (int, np.integer)) and not isinstance(index, (bool, np.bool_))
        if scalar: start, stop = int(index), int(index)+1
        elif isinstance(index, slice) and index.step in (None, 1): start, stop = index.start, index.stop
        else: raise ValueError('bounded scalar/contiguous probability rows required')
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.shape[0] or stop-start > 16:
            raise ValueError('probability batch outside bounds')
        with self.path.open('rb') as stream:
            stream.seek(self.offset+start*self.row_bytes); raw = stream.read((stop-start)*self.row_bytes)
        if len(raw) != (stop-start)*self.row_bytes: raise ValueError('short probability read')
        result = np.frombuffer(raw, dtype=self.dtype).reshape(stop-start, self.shape[1]).copy()
        return result[0] if scalar else result
