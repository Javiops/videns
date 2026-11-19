import unittest
import numpy as np
from winprob.purchase_memory_v1 import normalized, neighbors, distribution, mix, shuffled_labels, compact, evaluate


class PurchaseMemoryTests(unittest.TestCase):
    def test_game_exclusion_is_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'query game'):
            neighbors([1., 0.], [[1., 0.]], memory_rows=[4], memory_games=[7], query_game=7)

    def test_one_vote_per_game_and_stable_distance_ties(self):
        rows, distances, weights = neighbors([0., 0.], [[0., 0.], [0., 0.], [0., 0.], [1., 0.]],
            memory_rows=np.array([3, 2, 1, 4]), memory_games=[10, 10, 11, 12], query_game=99, k=3)
        np.testing.assert_array_equal(rows, [1, 2, 4])
        np.testing.assert_array_equal(distances, [0., 0., 1.])
        self.assertAlmostEqual(weights.sum(), 1.)
        self.assertAlmostEqual(weights[0], weights[1])

    def test_probability_mass_and_duplicate_labels(self):
        r = distribution(np.array([2, 2, 1]), [.3, .4, .3], 4)
        np.testing.assert_allclose(r, [0., .3, .7, 0.])
        p = mix([.4, .3, .2, .1], np.array([2, 2, 1]), [.3, .4, .3])
        np.testing.assert_allclose(p, [.2, .3, .45, .05])

    def test_empty_memory_exact_fallback(self):
        rows, d, w = neighbors([0., 0.], np.empty((0, 2)), memory_rows=np.array([], dtype=int),
                               memory_games=np.array([], dtype=int), query_game=1)
        np.testing.assert_array_equal(mix([.2, .3, .5], rows, w), [.2, .3, .5])
        self.assertEqual(len(d), 0)

    def test_shuffle_cannot_read_query_labels(self):
        labels = np.arange(6); train = np.array([0, 1, 2])
        args = (train, np.ones(6), np.zeros(6), np.zeros(6), 0)
        a, _ = shuffled_labels(labels, *args)
        labels[3:] = 1000
        b, _ = shuffled_labels(labels, *args)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.sort(a[train]), [0, 1, 2])
        np.testing.assert_array_equal(a[3:], [-1, -1, -1])

    def test_shuffle_preserves_context_bins(self):
        y = np.arange(8); rows = np.arange(6)
        result, facts = shuffled_labels(y, rows, np.array([1, 1, 2, 2, 3, 3, 1, 1]),
                                        np.zeros(8), np.zeros(8), 1)
        for ids in ([0, 1], [2, 3], [4, 5]):
            np.testing.assert_array_equal(np.sort(result[ids]), y[ids])
        self.assertEqual(facts['bins'], 3)

    def test_zero_embedding_and_invalid_mass_fail(self):
        with self.assertRaises(ValueError): normalized([[0., 0.]])
        with self.assertRaises(ValueError): mix([.5, .5, .1], np.array([0]), [1.])
        with self.assertRaises(ValueError): distribution(np.array([3]), [1.], 3)

    def test_identical_predictions_cannot_pass_gate(self):
        y = np.array([0, 1, 2, 0])
        p = np.array([[.6, .3, .1], [.3, .6, .1], [.1, .3, .6], [.6, .1, .3]])
        c = compact(p, y)
        result = evaluate(y, dict.fromkeys(('state', 'memory', 'shuffled'), c), draws=25)
        self.assertFalse(result['metric_gate_passed'])
        self.assertEqual(result['memory_minus_control']['state']['top1']['difference'], 0)


if __name__ == '__main__': unittest.main()
