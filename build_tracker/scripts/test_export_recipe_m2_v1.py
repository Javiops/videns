import unittest
import numpy as np
from export_recipe_m2_v1 import coarsen


class CoarseningTests(unittest.TestCase):
    def test_tail_includes_all_undisplayed_classes(self):
        p = np.zeros(9333, dtype=np.float32); p[:5] = [.3, .2, .1, .25, .15]
        got = coarsen(p, [3, 0, 4]); want = p.astype(float)/p.sum(dtype=float)
        np.testing.assert_allclose(got, [want[3], want[0], want[4], want[1]+want[2]], rtol=0, atol=1e-15)

    def test_padding_has_zero_mass_and_true_small_tail_is_retained(self):
        p = np.zeros(9333); p[0] = 1.-1e-14; p[6] = 1e-14
        got = coarsen(p, [0])
        self.assertEqual(got[1:3], [0., 0.])
        self.assertAlmostEqual(got[3]/1e-14, 1.)

    def test_rejects_missing_support_or_duplicated_candidates(self):
        p = np.zeros(9333); p[0] = 1.
        for groups in ([0, 0], [], [1], [9333]):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                coarsen(p, groups)


if __name__ == '__main__':
    unittest.main()
