import unittest
import numpy as np
from winprob.native_bias_calibration_v1 import BiasObjective, fit, predict, training_counts, control_permutation


class NativeBiasMathTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(72)
        p = rng.dirichlet(np.ones(6), size=17)
        y = rng.integers(0, 6, 17); b = rng.normal(0, .2, 6)
        return p, y, b

    def test_identity_support_and_extreme_logits(self):
        p = np.array([[.2, .8, 0.], [0., .4, .6]])
        np.testing.assert_array_equal(predict(p, np.zeros(3)), p/p.sum(1, keepdims=True))
        q = predict(p, np.array([1000., -1000., 0.]))
        np.testing.assert_allclose(q.sum(1), 1, atol=1e-14)
        self.assertTrue(np.isfinite(q).all()); self.assertTrue(np.all(q[p == 0] == 0))
        with self.assertRaisesRegex(ValueError, 'zero native support'):
            training_counts(p, np.array([2, 0]))

    def test_direct_likelihood_gradient_and_strong_convexity(self):
        p, y, b = self.fixture(); counts = training_counts(p, y); ridge = 1.
        objective = BiasObjective(p, counts, ridge, batch_size=4)
        value, gradient = objective(b)
        direct = -np.log(predict(p, b)[np.arange(len(y)), y]).sum()+np.log(p[np.arange(len(y)), y]).sum()+.5*(b@b)
        self.assertAlmostEqual(value, direct, places=12)
        eps = 1e-5; numeric = np.zeros_like(b)
        for i in range(len(b)):
            direction = np.eye(len(b))[i]
            numeric[i] = (objective(b+eps*direction)[0]-objective(b-eps*direction)[0])/(2*eps)
        np.testing.assert_allclose(gradient, numeric, rtol=0, atol=2e-9)
        v = np.arange(1., 7.); hv = objective.hessian_vector(b, v)
        numeric_hv = (objective(b+eps*v)[1]-objective(b-eps*v)[1])/(2*eps)
        np.testing.assert_allclose(hv, numeric_hv, rtol=0, atol=2e-8)
        self.assertGreaterEqual(float(v@hv), ridge*float(v@v)-1e-12)

    def test_permuting_training_labels_is_not_a_negative_control(self):
        p, y, b = self.fixture(); permuted_y = y[::-1]
        for labels in (y, permuted_y):
            delta = -np.log(predict(p,b)[np.arange(len(y)),labels]).sum()+np.log(p[np.arange(len(y)),labels]).sum()
            self.assertAlmostEqual(delta, BiasObjective(p, training_counts(p,y))(b)[0]-.5*(b@b), places=12)

    def test_lower_nll_can_reduce_purchase_accuracy(self):
        p = np.array([[.51,.49]]*3+[[.99,.01]]*2); y = np.array([0,0,0,1,1])
        b, receipt = fit(BiasObjective(p, training_counts(p,y)))
        self.assertTrue(receipt['qualified']); self.assertLess(abs(b.sum()), 1e-6)
        q = predict(p,b)
        self.assertLess(-np.log(q[np.arange(5),y]).sum(), -np.log(p[np.arange(5),y]).sum())
        self.assertEqual(int((p.argmax(1)==y).sum()), 3)
        self.assertEqual(int((q.argmax(1)==y).sum()), 0)

    def test_class_control_preserves_public_count_strata(self):
        sizes = np.array([0]+[1]*8+[2]*8+[3]*8)
        before = np.random.get_state(); permutation = control_permutation(sizes); after = np.random.get_state()
        np.testing.assert_array_equal(sizes[permutation], sizes)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(len(sizes)))
        np.testing.assert_array_equal(before[1], after[1]); self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(permutation, control_permutation(sizes))
        self.assertFalse(np.array_equal(permutation, np.arange(len(sizes))))


if __name__ == '__main__': unittest.main()
