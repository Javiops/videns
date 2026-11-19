import itertools
import unittest
import numpy as np
from winprob.route_marginal_v1 import bridge, joint_bridge, kl_divergence, RecipeFrontiers


class BridgeTests(unittest.TestCase):
    def test_normalization_joint_marginal_and_neutral_identity(self):
        p = np.array([[.2, .3, .5], [.1, .6, .3]])
        r = np.array([[.8, .2], [.4, .6], [.1, .9]])
        m = p @ r; same, got = bridge(p, r, m)
        np.testing.assert_array_equal(same, p); np.testing.assert_array_equal(got, m)
        original, joint, q, _ = joint_bridge(p[0], r, np.array([.7, .3]))
        np.testing.assert_allclose(joint.sum(0), [.7, .3], atol=1e-15)
        np.testing.assert_allclose(joint.sum(1), q, atol=1e-15)
        self.assertAlmostEqual(q.sum(), 1.)
        self.assertGreater(q[0], p[0, 0])

    def test_zero_support_is_preserved_and_impossible_route_rejected(self):
        p = np.array([[.4, .6, 0.]])
        r = np.array([[1., 0., 0.], [.5, .5, 0.], [0., 0., 1.]])
        q, _ = bridge(p, r, np.array([[.2, .8, 0.]]))
        self.assertEqual(q[0, 2], 0.)
        with self.assertRaises(ValueError):
            bridge(p, r, np.array([[.1, .8, .1]]))

    def test_KL_decomposition_minimum_and_contraction(self):
        p = np.array([.2, .3, .5]); r = np.array([[.8, .2], [.4, .6], [.1, .9]]); alpha = np.array([.7, .3])
        prior, best, q, m = joint_bridge(p, r, alpha)
        self.assertAlmostEqual(kl_divergence(best, prior), kl_divergence(alpha, m), places=13)
        self.assertLessEqual(kl_divergence(q, p), kl_divergence(alpha, m)+1e-14)
        conditional = np.array([[.2, .4], [.3, .5], [.5, .1]])
        alternative = conditional*alpha[None, :]
        penalty = sum(alpha[z]*kl_divergence(conditional[:, z], prior[:, z]/m[z]) for z in range(2))
        self.assertAlmostEqual(kl_divergence(alternative, prior), kl_divergence(alpha, m)+penalty, places=13)
        self.assertGreaterEqual(kl_divergence(alternative, prior), kl_divergence(best, prior)-1e-14)

    def test_identical_routing_rows_keep_action_odds(self):
        p = np.array([[.1, .3, .6]]); r = np.array([[.7, .3], [.7, .3], [.2, .8]])
        q, _ = bridge(p, r, np.array([[.9, .1]]))
        self.assertAlmostEqual(q[0, 0]/q[0, 1], p[0, 0]/p[0, 1], places=14)


class RecipeTests(unittest.TestCase):
    def fixture(self):
        return RecipeFrontiers({1: [], 2: [], 3: [1, 1], 4: [3, 2], 5: [1, 2]})

    def test_duplicate_components_are_real_capacity(self):
        recipe = self.fixture()
        self.assertEqual(recipe.maximum_matched_copies(((1, 2),), ((3, 1),)), 2)
        self.assertEqual(recipe.maximum_matched_copies(((1, 3),), ((3, 1),)), 2)
        self.assertEqual(recipe.maximum_matched_copies(((1, 4),), ((3, 2),)), 4)
        goals = [(), ((3, 1),), ((3, 2),), ((4, 1),)]
        np.testing.assert_array_equal(recipe.match_many(((1, 4),), goals), [0, 2, 4, 2])

    def test_parent_and_children_cannot_double_count_same_branch(self):
        recipe = self.fixture()
        self.assertEqual(recipe.maximum_matched_copies(((1, 2), (3, 1)), ((3, 1),)), 2)
        self.assertEqual(recipe.maximum_matched_copies(((1, 1), (2, 1), (3, 1)), ((4, 1),)), 2)
        self.assertEqual(recipe.maximum_matched_copies(((2, 1), (4, 1)), ((4, 1),)), 1)

    def test_matches_exhaustive_full_frontiers_on_small_forest(self):
        recipe = self.fixture()
        def frontiers(node):
            answer = [{node: 1}]
            if recipe.recipes[node]:
                for parts in itertools.product(*(frontiers(c) for c in recipe.recipes[node])):
                    merged = {}
                    for part in parts:
                        for item, count in part.items(): merged[item] = merged.get(item, 0)+count
                    answer.append(merged)
            return answer
        for counts in itertools.product(range(3), repeat=3):
            bag = tuple((item, count) for item, count in zip((1, 2, 3), counts) if count)
            expected = max(sum(min(count, f.get(item, 0)) for item, count in bag) for f in frontiers(4))
            self.assertEqual(recipe.maximum_matched_copies(bag, ((4, 1),)), expected)

    def test_missing_nodes_cycles_and_noncanonical_counts_rejected(self):
        with self.assertRaises(ValueError): RecipeFrontiers({1: [2]})
        with self.assertRaises(ValueError): RecipeFrontiers({1: [2], 2: [1]})
        with self.assertRaises(ValueError): self.fixture().maximum_matched_copies(((1, 1), (1, 1)), ((3, 1),))


if __name__ == '__main__':
    unittest.main()
