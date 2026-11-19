from pathlib import Path
import tempfile
import unittest
import numpy as np
from winprob.bounded_npy_array_v1 import BoundedNpyArray


class TensorRowsTests(unittest.TestCase):
    def test_tensor_shapes_exact_bits_and_irregular_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            for shape, dtype in (((17, 10, 22), np.float32), ((17, 37), np.float32), ((17, 10), np.int32), ((17,), np.int64)):
                values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
                path = Path(directory)/'input.npy'; np.save(path, values)
                data = BoundedNpyArray(path, shape, dtype, max_rows=8)
                for index in (0, 16, slice(2, 9), np.array([4, 5, 6, 0, 0, 16]), np.array([], dtype=np.int64)):
                    expected = values[index]; actual = data[index]
                    self.assertEqual(actual.shape, expected.shape); self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_bounds_layout_dtype_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.npy'; a = np.zeros((17, 10, 22), dtype=np.float32); np.save(path, a)
            data = BoundedNpyArray(path, a.shape, a.dtype, max_rows=4)
            for index in (slice(None), np.arange(5), -1, 17, np.array([.5])):
                with self.assertRaises(ValueError): data[index]
            with self.assertRaises(ValueError): BoundedNpyArray(path, a.shape, np.float64)
            np.save(path, np.asfortranarray(a))
            with self.assertRaises(ValueError): BoundedNpyArray(path, a.shape, a.dtype)
            np.save(path, a)
            with path.open('r+b') as stream: stream.truncate(path.stat().st_size-1)
            with self.assertRaises(ValueError): BoundedNpyArray(path, a.shape, a.dtype)


if __name__ == '__main__': unittest.main()
