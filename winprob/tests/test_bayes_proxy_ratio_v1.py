import unittest
import numpy as np
from winprob.bayes_proxy_ratio_v1 import posterior, anchor, closed_real_prefix


class ProxyRatioTests(unittest.TestCase):
    def test_bayes_equals_normalized_likelihood_product(self):
        p = np.array([.4, .6]); observations = np.array([[.2, .7], [.6, .1], [.8, .3]])
        expected = p*np.prod(observations, axis=0); expected /= expected.sum()
        np.testing.assert_allclose(posterior(np.log(observations), p), expected, atol=1e-15)
        np.testing.assert_array_equal(posterior(np.empty((0, 2)), p), p)

    def test_long_prefix_and_impossible_observation(self):
        result = posterior(np.tile([-1000., -1001.], (200, 1)), [.5, .5])
        self.assertTrue(np.isfinite(result).all()); self.assertGreater(result[0], 1-1e-12)
        with self.assertRaises(ValueError): posterior(np.array([[-np.inf, -np.inf]]), [.5, .5])

    def test_exact_neutrality_and_identical_proxies(self):
        p = np.array([.3, .1, .6]); experts = np.array([[.7, .2, .1], [.2, .1, .7]])
        np.testing.assert_array_equal(anchor(p, experts, [.5, .5], [.5, .5])[0], p)
        np.testing.assert_array_equal(anchor(p, np.tile(experts[0], (2, 1)), [.5, .5], [.9, .1])[0], p)
        np.testing.assert_array_equal(posterior(np.array([[-2., -2.], [-3., -3.]]), [.4, .6]), [.4, .6])

    def test_ratio_bounds_normalization_and_support(self):
        p = np.array([.4, .6, 0.]); forecasts = np.array([[.6, .3, .1], [.1, .2, .7]])
        old = np.array([.5, .5]); new = np.array([.8, .2])
        q, ratio = anchor(p, forecasts, old, new)
        self.assertAlmostEqual(q.sum(), 1.); self.assertEqual(q[2], 0.)
        self.assertGreaterEqual(ratio.min(), (new/old).min()-1e-15)
        self.assertLessEqual(ratio.max(), (new/old).max()+1e-15)
        with self.assertRaises(ValueError): anchor(p, np.array([[0., 1., 0.], [0., 1., 0.]]), old, new)

    def test_proxy_cumulative_regret_bound(self):
        rng = np.random.default_rng(51); experts = rng.dirichlet(np.ones(4), size=(60, 3))
        labels = rng.integers(4, size=60); prior = np.array([.5, .25, .25]); weight = prior.copy()
        losses = np.zeros(3); mixture_loss = 0.
        for forecast, label in zip(experts, labels, strict=True):
            mixture_loss -= np.log(weight @ forecast[:, label]); losses -= np.log(forecast[:, label])
            weight = posterior(np.log(forecast[:, label])[None], weight)
            self.assertTrue(np.all(mixture_loss <= losses-np.log(prior)+1e-12))

    def test_proxy_bound_does_not_transfer_to_native_reference(self):
        # Past observations can select a proxy that is wrong at this query;
        # native M2 can already be right. Ratio transfer has no native ln2 bound.
        forecasts = np.array([[1e-8, 1-1e-8], [1-1e-8, 1e-8]])
        weight = posterior(np.log(np.tile(forecasts[:, 1], (4, 1))), [.5, .5])
        native = np.array([.9, .1]); q, _ = anchor(native, forecasts, [.5, .5], weight)
        self.assertGreater(-np.log(q[0])+np.log(native[0]), np.log(2))

    def test_strict_closure_and_synthetic_save_invariance(self):
        events = [{'id': 2, 'start': 20, 'end': 29, 'real_purchase': True},
                  {'id': 1, 'start': 10, 'end': 20, 'real_purchase': True},
                  {'id': 3, 'start': 30, 'end': 40, 'real_purchase': True}]
        self.assertEqual(closed_real_prefix(events, 29), (1,))
        self.assertEqual(closed_real_prefix(events, 30), (1, 2))
        inserted = events+[{'id': 88, 'start': 15, 'end': 9999, 'real_purchase': False}]
        self.assertEqual(closed_real_prefix(inserted, 30), (1, 2))
        self.assertEqual(closed_real_prefix(list(reversed(events)), 30), (1, 2))


if __name__ == '__main__': unittest.main()
