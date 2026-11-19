from pathlib import Path
import tempfile
import unittest
import numpy as np
from source_probability_rows_v1 import ProbabilityRows


class ProbabilityReadTests(unittest.TestCase):
    def test_native_and_correction_precisions_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            for dtype in (np.float32, np.float64):
                values = np.random.default_rng(6).dirichlet(np.ones(13), size=25).astype(dtype)
                path = Path(directory)/'p.npy'; np.save(path, values)
                rows = ProbabilityRows(path, values.shape, dtype)
                self.assertEqual(rows[4].tobytes(), values[4].tobytes())
                self.assertEqual(rows[6:12].tobytes(), values[6:12].tobytes())
                with self.assertRaises(ValueError): rows[0:25]
                with self.assertRaises(ValueError): rows[-1]
                with self.assertRaises(ValueError): ProbabilityRows(path, (24, 13), dtype)


if __name__ == '__main__': unittest.main()
