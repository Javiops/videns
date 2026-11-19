"""Synthetic mathematical checks; no actual games, packages or source imports."""
from copy import deepcopy
import inspect
import unittest

import numpy as np
from scipy.special import logsumexp

from winprob.recipe_transition_v1 import (
    ACTION_STATS, CANDIDATE_NAMES, CONTEXT_NAMES, ChoiceInputs, coarsen_reference,
    context_features, fit, hessian_vector, objective, predict, transition_features)


ITEMS = {
    '1': {'stats': {'FlatArmorMod': 10}},
    '2': {'stats': {'FlatArmorMod': 25, 'FlatMovementSpeedMod': 45}},
    '3': {'stats': {'FlatHPPoolMod': 100}},
}


def synthetic_inputs(n=29, h=3, f=4):
    rng = np.random.default_rng(617)
    p = rng.uniform(.1, 1., size=(n, 4)); p /= p.sum(1, keepdims=True)
    context = rng.normal(size=(n, h)); context[:, 0] = 1
    features = rng.normal(size=(n, 3, f))
    return ChoiceInputs(p, context, features).validate()


class ReceiptTests(unittest.TestCase):
    def test_component_subtraction_and_duplicate_counts(self):
        # Two owned armor pieces: one is consumed; the other remains owned.
        value = transition_features([1, 1, 3], [2], 1000., [
            {'item_id': 2, 'cost': 400., 'inventory_after': [1, 2, 3]}], ITEMS)
        armor = [key for key, _ in ACTION_STATS].index('FlatArmorMod')
        self.assertAlmostEqual(value[6 + armor], .25)
        self.assertAlmostEqual(value[16 + armor], .15)
        self.assertEqual(value[4], 1/6)
        self.assertEqual(value[5], 0.)
        np.testing.assert_array_equal(value, transition_features([3, 1, 1], [2], 1000., [
            {'item_id': 2, 'cost': 400., 'inventory_after': [3, 2, 1]}], ITEMS))
        repeated = transition_features([], [1, 1], 500., [
            {'item_id': 1, 'cost': 100., 'inventory_after': [1]},
            {'item_id': 1, 'cost': 100., 'inventory_after': [1, 1]}], ITEMS)
        self.assertAlmostEqual(repeated[6 + armor], .2)
        self.assertAlmostEqual(repeated[16 + armor], .2)
        self.assertEqual(repeated[3], 2/6)

    def test_empty_main_has_no_change(self):
        value = transition_features([1, 3], [], 123., [], ITEMS)
        self.assertEqual(value[0], 1)
        self.assertAlmostEqual(value[2], 123/3000)
        np.testing.assert_array_equal(value[3:], np.zeros(23))

    def test_rejects_forged_or_incomplete_receipts(self):
        invalid = [
            [{'item_id': 2, 'cost': 400, 'inventory_after': [2, 3]}],
            [{'item_id': 2, 'cost': 1001, 'inventory_after': [2]}],
            [{'item_id': 2, 'cost': -1, 'inventory_after': [2]}],
            [{'item_id': 2, 'cost': 400, 'inventory_after': []}],
            [{'item_id': 3, 'cost': 400, 'inventory_after': [3]}],
            [],
        ]
        for steps in invalid:
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                transition_features([1], [2], 1000., steps, ITEMS)
        with self.assertRaises(ValueError):
            transition_features([1], [1, 2], 1000., [
                {'item_id': 1, 'cost': 100., 'inventory_after': [1, 1]},
                {'item_id': 2, 'cost': 400., 'inventory_after': [2]}], ITEMS)

    def test_predecision_allowlist_ignores_labels_and_future(self):
        tags = {'1': ['Tank'], '2': ['Mage']}
        state = {'inventory': [1, 1], 'team_id': 100, 'champion_id': 1,
                 'others': [{'champion_id': 2, 'team_id': 100 if i < 4 else 200} for i in range(9)],
                 'level': 10, 'ts': 600000, 'gold_est': 800.}
        before = context_features(state, ITEMS, tags)
        changed = deepcopy(state)
        changed.update(label_counts={'2': 99}, win=True, role='poison',
                       future_inventory=[2]*9, total_gold=10**12)
        for p in changed['others']:
            p.update(role='poison', gold=10**12, label_counts={'2': 99})
        np.testing.assert_array_equal(before, context_features(changed, ITEMS, tags))
        self.assertEqual(len(before), len(CONTEXT_NAMES))
        self.assertEqual(len(CANDIDATE_NAMES), 26)
        self.assertEqual(len(CONTEXT_NAMES)*len(CANDIDATE_NAMES), 676)


class LikelihoodTests(unittest.TestCase):
    def test_coarsened_distribution_matches_full_catalogue_tilt(self):
        full = np.array([[.2, .12, .3, .08, .18, .12], [.05, .1, .2, .3, .15, .2]])
        ids = np.array([[2, 0, 4], [3, 2, -1]])
        prior = coarsen_reference(full, ids)
        h = np.ones((2, 1)); features = np.array([[[.4], [-.3], [.9]], [[.2], [.6], [0.]]])
        data = ChoiceInputs(prior, h, features)
        got, order = predict(data, np.array([[.7]]))
        logits = np.log(full)
        for i, row in enumerate(ids):
            for j, label in enumerate(row):
                if label >= 0:
                    logits[i, label] += .7 * features[i, j, 0]
        expanded = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        np.testing.assert_allclose(got, coarsen_reference(expanded, ids), atol=1e-15, rtol=0)
        self.assertEqual(order[1, -1], 2)
        np.testing.assert_allclose(got.sum(1), 1., atol=1e-15)

    def test_zero_exact_prior_and_native_order(self):
        data = synthetic_inputs()
        # Native display order is authoritative even for finite-precision ties.
        p, order = predict(data, np.zeros(data.parameter_shape))
        np.testing.assert_array_equal(p, data.reference)
        np.testing.assert_array_equal(order, np.tile([0, 1, 2], (len(p), 1)))
        self.assertEqual(list(inspect.signature(predict).parameters), ['inputs', 'theta'])

    def test_invalid_candidate_references_are_rejected(self):
        p = np.array([[.4, .3, .2, .1]])
        for ids in ([[0, 0, -1]], [[0, -1, 2]], [[-1, -1, -1]], [[0, 1, 9]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                coarsen_reference(p, np.asarray(ids))
        with self.assertRaises(ValueError):
            coarsen_reference(p * .8, np.array([[0, 1, 2]]))

    def test_gradient_and_curvature_against_finite_differences(self):
        data = synthetic_inputs(); rng = np.random.default_rng(82)
        theta = rng.normal(scale=.2, size=np.prod(data.parameter_shape))
        y = rng.integers(0, 4, size=len(data.reference)); weights = rng.uniform(.1, 2., len(y))
        _, gradient = objective(theta, data, y, weights)
        eps = 1e-5; numerical = []
        for j in range(len(theta)):
            step = np.zeros_like(theta); step[j] = eps
            numerical.append((objective(theta+step, data, y, weights)[0] -
                              objective(theta-step, data, y, weights)[0]) / (2*eps))
        np.testing.assert_allclose(gradient, numerical, rtol=2e-7, atol=1e-9)
        direction = rng.normal(size=len(theta))
        curvature = hessian_vector(theta, direction, data, y, weights)
        expected = (objective(theta+eps*direction, data, y, weights)[1] -
                    objective(theta-eps*direction, data, y, weights)[1]) / (2*eps)
        np.testing.assert_allclose(curvature, expected, rtol=2e-7, atol=1e-9)
        self.assertGreaterEqual(direction @ curvature, .02*(direction @ direction)-1e-12)
        # Streaming microbatches preserve the same exact mathematical objective.
        small = objective(theta, data, y, weights, batch_size=4)
        np.testing.assert_allclose(small[1], gradient, rtol=0, atol=1e-14)

    def test_OTHER_rows_contribute_and_are_not_filtered(self):
        p = np.array([[.2, .2, .2, .4]])
        data = ChoiceInputs(p, np.ones((1, 1)), np.ones((1, 3, 1)))
        _, gradient = objective([0.], data, np.array([3]), [1.])
        self.assertAlmostEqual(gradient[0], .6)
        theta, receipt = fit(data, np.array([3]), [1.])
        self.assertTrue(receipt['converged'])
        self.assertEqual(receipt['outside_shortlist_rows'], 1)
        self.assertLess(theta[0, 0], 0)
        self.assertGreater(predict(data, theta)[0][0, 3], .4)

    def test_synthetic_identifiable_correction_is_recovered(self):
        # Deterministic fractional observations exactly generated by theta*=.8.
        base = np.tile(np.array([[.25]*4]), (4, 1))
        h = np.ones((4, 1)); f = np.tile(np.array([[[-1.], [0.], [1.]]]), (4, 1, 1))
        data = ChoiceInputs(base, h, f)
        truth = np.array([[.8]])
        probabilities = predict(data, truth)[0][0]
        fitted, receipt = fit(data, np.arange(4), probabilities, ridge=1e-8)
        self.assertTrue(receipt['converged'])
        self.assertLess(receipt['gradient_max_abs'], 1e-6)
        self.assertAlmostEqual(fitted[0, 0], .8, places=5)

    def test_permutation_equivariance_and_padding(self):
        data = synthetic_inputs(n=8); theta = np.ones(data.parameter_shape)*.2
        p, order = predict(data, theta)
        perm = np.array([2, 0, 1]); inverse = np.argsort(perm)
        changed = ChoiceInputs(data.reference[:, [2, 0, 1, 3]], data.context,
                               data.candidates[:, perm])
        cp, co = predict(changed, theta)
        np.testing.assert_allclose(cp[:, [*inverse, 3]], p, atol=1e-15)
        np.testing.assert_array_equal(perm[co], order)
        ids = np.array([[1, -1, -1]])
        single = ChoiceInputs(coarsen_reference(np.array([[.4, .6]]), ids),
                              np.ones((1, 1)), np.array([[[3.], [0.], [0.]]]))
        actual, ranking = predict(single, np.array([[.7]]))
        self.assertEqual(actual[0, 1], 0)
        self.assertEqual(actual[0, 2], 0)
        np.testing.assert_array_equal(ranking, [[0, 1, 2]])


if __name__ == '__main__':
    unittest.main()
