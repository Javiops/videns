import unittest
import numpy as np
from decode_route_probe_v1 import supplied_probability


class NeutralNativeTests(unittest.TestCase):
    def test_neutral_returns_original_precision_and_ties(self):
        p = np.zeros((1, 9333), dtype=np.float32); p[0, :3] = [.4, .4, .2]
        q = p.astype(np.float64); q /= q.sum(1, keepdims=True)
        actual = supplied_probability(p, q, True)
        self.assertIs(actual, p)
        self.assertEqual(actual.dtype, np.float32)

    def test_false_neutrality_and_precision_changes_fail(self):
        p = np.full((1, 9333), 1/9333, dtype=np.float32)
        q = p.astype(np.float64); q /= q.sum(1, keepdims=True)
        changed = q.copy(); changed[0, 0] += 1e-7; changed[0, 1] -= 1e-7
        with self.assertRaises(ValueError): supplied_probability(p, changed, True)
        self.assertIs(supplied_probability(p, changed, False), changed)
        with self.assertRaises(ValueError): supplied_probability(p.astype(np.float64), q, True)


if __name__ == '__main__':
    unittest.main()
