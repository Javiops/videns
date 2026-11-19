import unittest
import numpy as np
from winprob.metrics import game_weights, per_game_mean, paired_comparison, summarize


class MetricsTest(unittest.TestCase):
    def test_long_game_does_not_dominate(self):
        games = np.array([0, 0, 0, 1])
        self.assertAlmostEqual(game_weights(games)[:3].sum(), 1)
        _, means = per_game_mean(np.array([0., 0., 0., 1.]), games)
        self.assertAlmostEqual(means.mean(), .5)

    def test_identical_predictions_exact_zero(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([.2, .3, .8, .7])
        result = paired_comparison(y, p, p, np.array([0, 0, 1, 1]), 50)
        for name in ('log_loss', 'brier', 'accuracy'):
            self.assertEqual(result[name]['estimate'], 0)
            self.assertEqual(result[name]['ci95'], [0, 0])

    def test_duplicate_within_game_keeps_estimand(self):
        y = np.array([0, 1, 1])
        p = np.array([.2, .6, .7])
        games = np.array([0, 0, 1])
        a = summarize(y, p, games, 40)
        b = summarize(np.repeat(y, [2, 2, 1]), np.repeat(p, [2, 2, 1]),
                      np.repeat(games, [2, 2, 1]), 40)
        for metric in ('log_loss', 'brier', 'accuracy'):
            self.assertAlmostEqual(a[metric]['estimate'], b[metric]['estimate'])
            self.assertEqual(a[metric]['ci95'], b[metric]['ci95'])

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            summarize(np.array([0, 1]), np.array([np.nan, .5]), np.array([0, 1]))


if __name__ == '__main__':
    unittest.main()
