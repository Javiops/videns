import unittest
import numpy as np
from winprob.route_marginal_v1 import bridge, kl_divergence
from winprob.route_capacity_v1 import (conditional_experts, pairwise_margin_range,
    residual_belief, purchase_loss_gradient)


class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([.2, .3, .5])
        self.r = np.array([[.8, .2], [.4, .6], [.1, .9]])

    def test_convex_hull_exactly_equals_bridge(self):
        experts, marginal = conditional_experts(self.p, self.r)
        np.testing.assert_allclose(experts.sum(0), 1, atol=1e-15)
        for t in np.linspace(0, 1, 31):
            alpha = np.array([t, 1-t])
            q, _ = bridge(self.p[None], self.r, alpha[None])
            np.testing.assert_allclose(q[0], experts @ alpha, atol=1e-15)
            self.assertLessEqual(kl_divergence(q[0], self.p),
                                max(kl_divergence(e, self.p) for e in experts.T)+1e-14)
        np.testing.assert_allclose(experts @ marginal, self.p, atol=1e-15)

    def test_pairwise_endpoints_are_attainable(self):
        experts, _ = conditional_experts(self.p, self.r)
        lo, hi = pairwise_margin_range(self.p, self.r, 0, 2)
        self.assertAlmostEqual(lo, min(experts[0]-experts[2]))
        self.assertAlmostEqual(hi, max(experts[0]-experts[2]))
        self.assertGreater(hi, 0)
        identical = np.array([[.8, .2], [.8, .2], [.1, .9]])
        self.assertLess(pairwise_margin_range(self.p, identical, 0, 1)[1], 0)

    def test_unsupported_route_cannot_supply_a_false_capacity(self):
        p = np.array([.4, .6, 0.])
        r = np.array([[1., 0., 0.], [.5, .5, 0.], [0., 0., 1.]])
        e, m = conditional_experts(p, r)
        np.testing.assert_array_equal(e[:, 2], 0)
        self.assertEqual(pairwise_margin_range(p, r, 2, 1)[1], -(.3/.7))
        alpha = residual_belief(m[None], np.array([[0., 0., 1e6]]))
        np.testing.assert_array_equal(alpha[0], m)

    def test_pairwise_wins_are_not_a_joint_top1_certificate(self):
        # Target 0 beats each rival somewhere, but they sum to .8 while target
        # stays .2: at least one rival is >=.4 in every convex combination.
        experts = np.array([[.2, .2], [.8, 0], [0, .8]])
        p = experts.mean(1); r = experts*.5/p[:, None]
        self.assertGreater(pairwise_margin_range(p, r, 0, 1)[1], 0)
        self.assertGreater(pairwise_margin_range(p, r, 0, 2)[1], 0)
        for a in np.linspace(0, 1, 101):
            q = experts @ [a, 1-a]
            self.assertLess(q[0], max(q[1:]))


class ResidualGateTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(73)
        x = rng.normal(size=(5, 3))
        r = rng.dirichlet(np.ones(4), size=7)
        p = rng.dirichlet(np.ones(7), size=5)
        labels = np.array([1, 3, 4, 0, 6]); w = np.array([.1, .2, .3, .25, .15])
        return x, r, p, labels, w

    def test_zero_gate_is_bit_exact_and_loss_is_zero(self):
        x, r, p, labels, w = self.fixture(); m = p @ r
        alpha = residual_belief(m, np.zeros_like(m))
        np.testing.assert_array_equal(alpha, m)
        q, _ = bridge(p, r, alpha)
        np.testing.assert_array_equal(q, p)
        value, _ = purchase_loss_gradient(np.zeros((3, 4)), x, m, r[labels], w, .01)
        self.assertAlmostEqual(value, 0., places=14)

    def test_objective_matches_actual_action_NLL_difference(self):
        x, r, p, labels, w = self.fixture(); m = p @ r
        t = np.arange(12).reshape(3, 4)/13-.4
        alpha = residual_belief(m, x @ t); q, _ = bridge(p, r, alpha)
        expected = np.dot(w, np.log(p[np.arange(5), labels])-np.log(q[np.arange(5), labels]))
        value, _ = purchase_loss_gradient(t, x, m, r[labels], w, .02)
        self.assertAlmostEqual(value, expected+.02*np.sum(t*t), places=13)

    def test_gradient_finite_differences_and_common_shift_invariance(self):
        x, r, p, labels, w = self.fixture(); m = p @ r
        t = np.arange(12).reshape(3, 4)/13-.4
        value, grad = purchase_loss_gradient(t, x, m, r[labels], w, .01)
        for i in range(t.size):
            step = np.zeros_like(t); step.flat[i] = 1e-6
            plus, _ = purchase_loss_gradient(t+step, x, m, r[labels], w, .01)
            minus, _ = purchase_loss_gradient(t-step, x, m, r[labels], w, .01)
            self.assertAlmostEqual(grad.flat[i], (plus-minus)/2e-6, places=8)
        a = residual_belief(m, x @ t)
        b = residual_belief(m, x @ t+np.arange(5)[:, None]*100)
        np.testing.assert_allclose(a, b, atol=2e-14)

    def test_large_logits_stay_finite_and_zero_support_rejected(self):
        x = np.ones((1, 1)); m = np.array([[.2, .8, 0.]])
        y = np.array([[.4, .6, 0.]])
        value, grad = purchase_loss_gradient(np.array([[1000., -1000., 1e6]]), x, m, y, np.ones(1), 0)
        self.assertTrue(np.isfinite(value)); self.assertTrue(np.isfinite(grad).all())
        with self.assertRaises(ValueError):
            purchase_loss_gradient(np.zeros((1, 3)), x, m, np.array([[.2, .7, .1]]), np.ones(1), 0)


if __name__ == '__main__':
    unittest.main()
