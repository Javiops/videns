from copy import deepcopy
import inspect
import unittest
import numpy as np
from winprob.recipe_transition_v1 import ChoiceInputs
from winprob.personal_graph_models_v1 import RMSScaler, public_profiles, predict_graph


class GraphModelTests(unittest.TestCase):
    def test_public_profile_join_counts_items_and_ignores_forbidden_fields(self):
        teams = np.array([100]*5+[200]*5); champions = np.tile(np.arange(1, 6), 2)
        players = [{'team_id': int(t), 'champion_id': int(c), 'items': [1, 1], 'level': 9} for t, c in zip(teams, champions)]
        row = {'team_id': 100, 'champion_id': 1, 'inventory': [1, 1], 'level': 9, 'others': players[1:]}
        items = {'1': {'stats': {'FlatArmorMod': 10}}}; tags = {str(i): ['Mage'] for i in range(1, 6)}
        original = public_profiles(row, teams, champions, 1, items, tags)
        changed = deepcopy(row); changed.update(role='poison', gold=1e9, outcome=True, target={'label_counts': {'2': 9}})
        for p in changed['others']:
            p.update(role='poison', gold=1e9, future_items=[2])
        np.testing.assert_array_equal(original, public_profiles(changed, teams, champions, 1, items, tags))
        self.assertAlmostEqual(original[0, 10], .2)
        self.assertEqual(original[0, -1], .5)

    def test_noncentering_RMS_keeps_zero_information_zero(self):
        rng = np.random.default_rng(9); n = 7
        h = rng.normal(size=(n, 17)); h[:2] = 0
        f = rng.normal(size=(n, 3, 16)); p = np.full((n, 4), .25)
        inputs = ChoiceInputs(p, h, f); scaler = RMSScaler.fit(inputs, list('aaabbcd'))
        transformed = scaler.transform(inputs)
        np.testing.assert_array_equal(transformed.context[:2], 0.)
        np.testing.assert_array_equal(inputs.context, h)
        restored = RMSScaler.from_dict(scaler.as_dict())
        np.testing.assert_array_equal(restored.transform(inputs).context, transformed.context)

    def test_zero_profiles_and_zero_coefficients_preserve_native_order_exactly(self):
        rng = np.random.default_rng(5)
        prior = np.array([[.4, .4, .1, .1], [.7, .1, .1, .1], [.5, 0, 0, .5]])
        h = rng.normal(size=(3, 17)); h[0] = 0; h[2] = 0
        f = rng.normal(size=(3, 3, 16)); f[2, 1:] = 0
        inputs = ChoiceInputs(prior, h, f); scaler = RMSScaler.fit(inputs, ['a', 'b', 'c'])
        output = predict_graph(inputs, scaler, rng.normal(size=(17, 16))*10)
        np.testing.assert_array_equal(output['probabilities'][[0, 2]], prior[[0, 2]])
        np.testing.assert_array_equal(output['candidate_order'][[0, 2]], [[0, 1, 2], [0, 1, 2]])
        zero = predict_graph(inputs, scaler, np.zeros((17, 16)))
        np.testing.assert_array_equal(zero['probabilities'], prior)
        np.testing.assert_array_equal(zero['candidate_order'], np.tile([0, 1, 2], (3, 1)))
        self.assertNotIn('target', inspect.signature(predict_graph).parameters)


if __name__ == '__main__':
    unittest.main()
