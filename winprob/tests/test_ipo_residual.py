"""Synthetic reference only; no M2 masses or real preference rows are fabricated."""
import unittest
import numpy as np

from winprob.ipo_residual import (ACTION_NAMES, CONTEXT_NAMES, ContextStandardizer,
    action_features, canonical_reference, context_features, fit_ipo_ridge, fit_winner_only,
    pair_design, residual_policy, residual_scores, winner_objective)


class ResidualIPOTests(unittest.TestCase):
    def fixture(self):
        context = np.array([[1., -1.], [1., 0.], [1., 1.], [1., 2.]])
        actions = np.array([[1., 0.], [0., 1.], [1., 1.]])
        base = np.array([[.5, .3, .2], [.2, .6, .2], [.6, .2, .2], [.2, .3, .5]])
        return context, actions, base, np.array([1, 1, 2, 2]), np.array([0, 0, 0, 1]), np.array([.1, .05, .1, .2])

    def test_exact_zero_residual_identity_and_reference_zeros(self):
        c, a, p, _, _, _ = self.fixture(); p[0] = [.7, .3, 0]
        zero = np.zeros((2, 2))
        np.testing.assert_array_equal(residual_policy(c, a, zero, p), p)
        shifted = residual_policy(c, a, np.ones((2, 2)), p)
        self.assertEqual(shifted[0, 2], 0)
        np.testing.assert_allclose(shifted.sum(axis=1), 1)

    def test_nonexact_float32_row_sum_has_byte_identical_zero_policy(self):
        p = np.array([[.1, .2, .7000015]], dtype=np.float32)
        error = abs(float(p.sum(dtype=np.float64))-1)
        self.assertGreater(error, 1e-6)
        self.assertLess(error, 2e-6)
        result = residual_policy(np.ones((1, 1)), np.ones((3, 1)), np.zeros((1, 1)), p)
        self.assertEqual(result.dtype, p.dtype)
        self.assertEqual(result.tobytes(), p.tobytes())
        np.testing.assert_array_equal(result, p)

    def test_pair_design_is_antisymmetric_and_same_action_is_zero(self):
        c, a, _, plus, minus, _ = self.fixture()
        np.testing.assert_array_equal(pair_design(c, a, plus, minus), -pair_design(c, a, minus, plus))
        np.testing.assert_array_equal(pair_design(c, a, plus, plus), np.zeros((4, 4)))

    def test_swapping_pair_orientation_and_target_sign_preserves_coefficients(self):
        c, a, _, plus, minus, weights = self.fixture()
        theta, diag = fit_ipo_ridge(c, a, plus, minus, weights)
        swapped, _ = fit_ipo_ridge(c, a, minus, plus, weights, target_sign=-1)
        np.testing.assert_array_equal(theta, swapped)
        self.assertLess(diag['gradient_max_abs'], 1e-12)
        self.assertFalse(diag['pair_difference_centered'])

    def test_pair_margin_equals_separate_candidate_score_difference(self):
        c, a, _, plus, minus, weights = self.fixture()
        theta, _ = fit_ipo_ridge(c, a, plus, minus, weights)
        scores = residual_scores(c, a, theta)
        np.testing.assert_allclose(pair_design(c, a, plus, minus)@theta.ravel(), scores[np.arange(4), plus]-scores[np.arange(4), minus])

    def test_reference_relative_normalizer_cancels_but_total_odds_include_prior(self):
        c, a, p, plus, minus, weights = self.fixture()
        theta, _ = fit_ipo_ridge(c, a, plus, minus, weights)
        policy = residual_policy(c, a, theta, p)
        relative = np.log(policy[np.arange(4), plus]/policy[np.arange(4), minus])-np.log(p[np.arange(4), plus]/p[np.arange(4), minus])
        np.testing.assert_allclose(relative, pair_design(c, a, plus, minus)@theta.ravel(), atol=1e-14)

    def test_context_standardizer_preserves_intercept_and_does_not_refit(self):
        c, _, _, _, _, weights = self.fixture()
        scaler = ContextStandardizer.fit(c, weights)
        transformed = scaler.transform(c)
        np.testing.assert_array_equal(transformed[:, 0], 1)
        before = scaler.mean.copy()
        scaler.transform(np.array([[1., 999.]]))
        np.testing.assert_array_equal(before, scaler.mean)

    def test_ipo_target_is_finite_and_duplication_does_not_change_fit(self):
        c = np.ones((1, 1)); a = np.array([[0.], [1.]])
        theta, _ = fit_ipo_ridge(c, a, [1], [0], [1.], tau=.5, ridge=.01)
        self.assertAlmostEqual(theta[0, 0], .5/.51)
        doubled, _ = fit_ipo_ridge(np.repeat(c, 2, axis=0), a, [1, 1], [0, 0], [.5, .5])
        np.testing.assert_allclose(theta, doubled)

    def test_winner_only_gradient_matches_finite_difference_full_catalogue(self):
        c, a, p, plus, _, weights = self.fixture()
        p = p.astype(np.float32); p[0, 2] += np.float32(1.5e-6)
        theta = np.array([.2, -.1, .3, .05])
        loss, gradient = winner_objective(theta, c, a, p, plus, weights)
        numerical = []
        for j in range(4):
            step = np.zeros(4); step[j] = 1e-6
            high = winner_objective(theta+step, c, a, p, plus, weights)[0]
            low = winner_objective(theta-step, c, a, p, plus, weights)[0]
            numerical.append((high-low)/(2e-6))
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)
        # A third alternative changes the real normalizer; pair-only CE would omit it.
        changed = p.copy(); changed[:, 2] += .1; changed /= changed.sum(axis=1, keepdims=True)
        self.assertNotEqual(loss, winner_objective(theta, c, a, changed, plus, weights)[0])

    def test_winner_only_solver_converges_without_any_rejected_argument(self):
        c, a, p, plus, _, weights = self.fixture()
        before = winner_objective(np.zeros(4), c, a, p, plus, weights)[0]
        theta, diag = fit_winner_only(c, a, p, plus, weights)
        self.assertTrue(diag['converged'])
        self.assertFalse(diag['rejected_labels_consumed'])
        self.assertEqual(diag['normalizer_actions'], 3)
        self.assertLess(diag['objective'], before)

    def test_context_allowlist_ignores_privileged_fields_and_label_metadata(self):
        others = [{'champion_id': i, 'team_id': 100 if i <= 5 else 200} for i in range(2, 11)]
        row = {'patch': '16.19', 'context_timestamp_ms': 600000,
               'source_projection': {'champion_id': 1, 'team_id': 100, 'level': 8, 'others': others,
                                     'total_gold': 999999, 'positions': 'FORBIDDEN'}, 'action_b': 123}
        tags = {'16.19': {str(i): ['Fighter'] if i < 6 else ['Mage'] for i in range(1, 11)}}
        first = context_features([row], tags)
        row['source_projection']['total_gold'] = -999; row['action_b'] = 555
        np.testing.assert_array_equal(first, context_features([row], tags))
        self.assertEqual(first.shape, (1, 22))
        self.assertEqual(first[0, CONTEXT_NAMES.index('allies_Fighter')], 4)
        self.assertEqual(first[0, CONTEXT_NAMES.index('enemies_Mage')], 5)

    def test_action_scales_are_fixed_and_multisets_preserve_counts(self):
        items = {'1': {'gold': {'total': 1500}, 'stats': {'FlatArmorMod': 50}, 'tags': ['Armor']}}
        vectors = action_features([[], [[1, 1]], [[1, 2]]], items)
        self.assertEqual(vectors.shape, (3, 23))
        self.assertEqual(vectors[0, -1], 1)
        self.assertEqual(vectors[1, 0], .5)
        self.assertEqual(vectors[1, ACTION_NAMES.index('FlatArmorMod')], .5)
        np.testing.assert_allclose(vectors[2, :-1], 2*vectors[1, :-1])
        with self.assertRaises(ValueError):
            action_features([[[999, 1]]], items)


if __name__ == '__main__':
    unittest.main()
