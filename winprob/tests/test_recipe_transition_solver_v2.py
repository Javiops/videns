import unittest
import numpy as np
from winprob.recipe_transition_v1 import ChoiceInputs, predict, objective
from winprob.recipe_transition_solver_v2 import fit


class SolverPrecisionTests(unittest.TestCase):
    def test_known_convex_solution_recovery_and_original_gradient(self):
        data = ChoiceInputs(np.full((4, 4), .25), np.ones((4, 1)),
                            np.tile(np.array([[[-1.], [0.], [1.]]]), (4, 1, 1)))
        weights = predict(data, np.array([[.8]]))[0][0]
        theta, receipt = fit(data, np.arange(4), weights, ridge=1e-8)
        loss, gradient = objective(theta.ravel(), data, np.arange(4), weights, ridge=1e-8)
        self.assertTrue(receipt['converged'])
        self.assertLess(np.max(np.abs(gradient)), 1e-7)
        self.assertAlmostEqual(theta[0, 0], .8, places=5)
        self.assertEqual(loss, receipt['objective'])
        self.assertEqual(receipt['outside_shortlist_rows'], 1)
        self.assertLessEqual(receipt['iterations'], 300)

    def test_zero_feature_directions_remain_exactly_zero(self):
        rng = np.random.default_rng(5); n = 80
        h = rng.normal(size=(n, 3)); f = rng.normal(size=(n, 3, 4)); f[:, :, -1] = 0
        prior = np.full((n, 4), .25)
        data = ChoiceInputs(prior, h, f)
        theta, receipt = fit(data, rng.integers(0, 4, n), np.ones(n))
        self.assertTrue(receipt['converged'])
        self.assertLess(receipt['gradient_max_abs'], 1e-6)
        np.testing.assert_array_equal(theta[:, -1], 0.)


if __name__ == '__main__':
    unittest.main()
