import unittest
import numpy as np
from winprob.route_probe_v1 import objective, predict
from winprob.route_gpu_objective_v1 import Objective, configure, fit


class GpuRouteObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = configure()

    def test_cpu_parity_and_finite_differences(self):
        rng = np.random.default_rng(211)
        x = np.column_stack([np.ones(25), rng.normal(size=(25, 3))])
        r = rng.dirichlet(np.ones(7), size=12); p = rng.dirichlet(np.ones(12), size=25)
        m = p @ r; y = r[rng.integers(12, size=25)]; w = np.arange(1, 26, dtype=float); w /= w.sum()
        calc = Objective(x, m, y, w, .01/7)
        try:
            for theta in (np.zeros((4, 7)), rng.normal(size=(4, 7)), np.full((4, 7), 1000.)):
                a, g = calc(theta.ravel()); expected, gradient = objective(theta.ravel(), x, m, y, w, ridge=.01/7)
                self.assertAlmostEqual(a, expected, delta=2e-10 if np.max(theta) > 100 else 2e-12)
                np.testing.assert_allclose(g, gradient, atol=2e-12, rtol=0)
            theta = rng.normal(size=28); direction = rng.normal(size=28); direction /= np.linalg.norm(direction)
            _, gradient = calc(theta); plus, _ = calc(theta+1e-4*direction); minus, _ = calc(theta-1e-4*direction)
            self.assertAlmostEqual((plus-minus)/2e-4, gradient @ direction, delta=1e-8)
        finally: calc.close()

    def test_saved_fit_has_cpu_stationarity_and_recovers_synthetic_signal(self):
        r = np.array([[.9, .1], [.1, .9]])
        x = np.column_stack([np.ones(40), np.tile([-1., 1.], 20)])
        p = np.full((40, 2), .5); labels = np.tile([0, 1], 20); w = np.full(40, 1/40)
        theta, receipt = fit(x, p @ r, r[labels], w)
        self.assertTrue(receipt['success']); value, gradient = objective(theta.ravel(), x, p @ r, r[labels], w, ridge=.01/2)
        self.assertAlmostEqual(value, receipt['relative_regularized_objective'], delta=2e-12)
        self.assertLess(np.max(np.abs(gradient)), 1e-6)
        np.testing.assert_array_equal(predict(p, r, x, theta)[0].argmax(1), labels)


if __name__ == '__main__': unittest.main()
