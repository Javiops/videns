import tempfile
from pathlib import Path
import unittest
import numpy as np
from export_route_prior_v1 import verify_probability_row, reference, coarsen


class FullPriorReplayTests(unittest.TestCase):
    def test_undisplayed_mass_change_is_detected_despite_equal_coarsening(self):
        p = np.zeros(9333, dtype=np.float32); p[:5] = [.5, .25, .125, .0625, .0625]
        original = reference.array_hash(p)
        verify_probability_row(p, original)
        changed = p.copy(); changed[3] = .03125; changed[4] = .09375
        self.assertEqual(coarsen(p, [0, 1, 2]), coarsen(changed, [0, 1, 2]))
        with self.assertRaises(ValueError):
            verify_probability_row(changed, original)

    def test_ordinal_disk_retention_is_bit_exact_and_swaps_are_detected(self):
        rng = np.random.default_rng(104)
        values = rng.dirichlet(np.ones(9333), size=3).astype(np.float32)
        digests = [reference.array_hash(p) for p in values]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'prior.npy'
            mapped = np.lib.format.open_memmap(path, mode='w+', shape=values.shape, dtype=np.float32)
            for ordinal in (2, 0, 1): mapped[ordinal] = values[ordinal]
            mapped.flush(); del mapped
            saved = np.load(path, mmap_mode='r', allow_pickle=False)
            for ordinal, sha in enumerate(digests): verify_probability_row(saved[ordinal], sha)
            with self.assertRaises(ValueError): verify_probability_row(saved[1], digests[0])
            del saved

    def test_dtype_and_action_dimension_are_bound(self):
        p = np.full(9333, 1/9333, dtype=np.float32)
        with self.assertRaises(ValueError): verify_probability_row(p.astype(np.float64), reference.array_hash(p))
        with self.assertRaises(ValueError): verify_probability_row(p[:3], reference.array_hash(p[:3]))


if __name__ == '__main__':
    unittest.main()
