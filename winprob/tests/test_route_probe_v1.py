import unittest
import numpy as np
from winprob.route_capacity_v1 import purchase_loss_gradient
from winprob.route_probe_v1 import (kernels, ContextScaler, objective, fit, predict, normalized_prior)


class RouteProbeTests(unittest.TestCase):
    def test_control_preserves_count_bins_but_is_not_a_column_rename(self):
        raw = np.arange(120, dtype=np.uint8).reshape(30, 4) % 61
        sizes = np.repeat(np.arange(6), 5)
        ks, permutation = kernels(raw, sizes)
        np.testing.assert_array_equal(sizes[permutation], sizes)
        np.testing.assert_array_equal(ks['permuted'], ks['recipe'][permutation])
        self.assertFalse(np.array_equal(permutation, np.arange(30)))
        np.testing.assert_array_equal(kernels(raw, sizes)[1], permutation)
        np.testing.assert_allclose(ks['recipe'].sum(1), 1.)

    def test_training_only_scaling_weights_each_game_equally(self):
        x = np.array([[1., 1.], [1., 1.], [1., 5.]])
        scaler = ContextScaler.fit(x, ['a', 'a', 'b'])
        np.testing.assert_allclose(scaler.mean, [0., 3.])
        np.testing.assert_allclose(scaler.scale, [1., 2.])
        self.assertEqual(scaler.transform(np.array([[1., 101.]]))[0, 1], 49.)
        np.testing.assert_array_equal(ContextScaler.from_dict(scaler.as_dict()).transform(x), scaler.transform(x))

    def fixture(self):
        rng = np.random.default_rng(5)
        x = np.column_stack([np.ones(7), rng.normal(size=7)])
        p = rng.dirichlet(np.ones(5), size=7); r = rng.dirichlet(np.ones(3), size=5)
        w = np.arange(1, 8, dtype=float); w /= w.sum()
        y = r[[0, 3, 1, 4, 2, 1, 0]]
        theta = rng.normal(size=(2, 3))
        return x, p, r, w, y, theta

    def test_chunking_preserves_full_objective_and_gradient(self):
        x, p, r, w, y, theta = self.fixture(); m = p @ r
        v, g = purchase_loss_gradient(theta, x, m, y, w, .007)
        for batch in (1, 3, 7, 20):
            got, grad = objective(theta.ravel(), x, m, y, w, ridge=.007, batch_size=batch)
            self.assertAlmostEqual(got, v, places=14)
            np.testing.assert_allclose(grad, g.ravel(), atol=2e-15)

    def test_prediction_neutrality_support_and_no_label_parameter(self):
        x, p, r, w, y, theta = self.fixture()
        p[:, 4] = 0.; p /= p.sum(1, keepdims=True)
        original = p.astype(np.float32)
        neutral, flags = predict(original, r, x, np.zeros((2, 3)))
        np.testing.assert_array_equal(neutral, normalized_prior(original))
        self.assertTrue(flags.all())
        changed, flags = predict(original, r, x, theta)
        np.testing.assert_array_equal(changed[:, 4], 0.)
        np.testing.assert_allclose(changed.sum(1), 1.)
        self.assertFalse(flags.any())

    def test_synthetic_purchase_signal_can_be_fitted(self):
        r = np.array([[.9, .1], [.1, .9]])
        x = np.column_stack([np.ones(40), np.tile([-1., 1.], 20)])
        p = np.full((40, 2), .5); y = np.tile([0, 1], 20)
        w = np.full(40, 1/40)
        theta, receipt = fit(x, p @ r, r[y], w)
        self.assertTrue(receipt['success']); self.assertLess(receipt['gradient_max_abs'], 1e-6)
        q, _ = predict(p, r, x, theta)
        np.testing.assert_array_equal(q.argmax(1), y)
        self.assertLess(-np.log(q[np.arange(40), y]).mean(), np.log(2))

    def test_mixture_loss_is_not_assumed_convex(self):
        # Hessian of log Z_m-log Z_y is negative in this direction at zero.
        x = np.ones((1, 1)); m = np.array([[.6, .4]]); y = np.array([[.5, .5]])
        direction = np.array([[1., -1.]]); eps = 1e-4
        zero, _ = purchase_loss_gradient(np.zeros((1, 2)), x, m, y, np.ones(1), 0.)
        plus, _ = purchase_loss_gradient(eps*direction, x, m, y, np.ones(1), 0.)
        minus, _ = purchase_loss_gradient(-eps*direction, x, m, y, np.ones(1), 0.)
        self.assertLess((plus+minus-2*zero)/eps**2, -.039)


if __name__ == '__main__':
    unittest.main()
