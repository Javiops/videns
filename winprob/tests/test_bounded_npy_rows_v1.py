from pathlib import Path
import tempfile
import unittest
import numpy as np
from winprob.bounded_npy_rows_v1 import BoundedNpyRows


class BoundedRowsTests(unittest.TestCase):
    def test_exact_all_supported_index_forms_and_order(self):
        values = np.random.default_rng(20261003).normal(size=(31, 13)).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.npy'; np.save(path, values, allow_pickle=False)
            disk = BoundedNpyRows(path, expected_shape=values.shape, expected_dtype=np.float32, max_rows=8)
            for index in (0, 30, slice(4, 12), slice(31, 31), np.array([30, 0, 5, 5, 2]), np.arange(9, 15)):
                np.testing.assert_array_equal(disk[index], values[index])
            rows = np.array([30, 0, 5, 5]); columns = np.array([2, 1, 9, 7])
            np.testing.assert_array_equal(disk[rows, columns], values[rows, columns])
            self.assertEqual(len(disk), len(values)); self.assertEqual(disk.dtype, np.dtype(np.float32))

    def test_whole_file_negative_and_oversized_requests_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.npy'; np.save(path, np.zeros((20, 3), dtype=np.float32))
            disk = BoundedNpyRows(path, expected_shape=(20, 3), expected_dtype=np.float32, max_rows=4)
            for index in (slice(None), slice(0, 5), -1, 20, np.arange(5), np.array([1, -1]), np.array([.5]), slice(0, 4, 2)):
                with self.subTest(index=index), self.assertRaises(ValueError): disk[index]

    def test_dtype_shape_layout_and_truncation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'rows.npy'; values = np.zeros((5, 3), dtype=np.float32); np.save(path, values)
            with self.assertRaises(ValueError): BoundedNpyRows(path, expected_shape=(4, 3), expected_dtype=np.float32)
            with self.assertRaises(ValueError): BoundedNpyRows(path, expected_shape=(5, 3), expected_dtype=np.float64)
            np.save(path, np.asfortranarray(values))
            with self.assertRaises(ValueError): BoundedNpyRows(path, expected_shape=(5, 3), expected_dtype=np.float32)
            np.save(path, values)
            with path.open('r+b') as stream: stream.truncate(path.stat().st_size-1)
            with self.assertRaises(ValueError): BoundedNpyRows(path, expected_shape=(5, 3), expected_dtype=np.float32)


if __name__ == '__main__': unittest.main()
