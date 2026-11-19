import unittest
import numpy as np
from scripts.audit_recipe_transition_v1 import independent_scaler, compare_nested
from winprob.recipe_transition_v1 import ChoiceInputs
from winprob.recipe_transition_probe_v1 import Standardizer, game_weights


class AuditTests(unittest.TestCase):
    def test_independent_scaler_respects_game_and_candidate_weights(self):
        rng = np.random.default_rng(18)
        h = rng.normal(size=(7, 26)); h[:, 0] = 1; h[:, 3] = 6
        f = rng.normal(size=(7, 3, 26)); f[0, 1:] = 0
        prior = np.tile([.2, .2, .2, .4], (7, 1)); prior[0] = [.6, 0, 0, .4]
        inputs = ChoiceInputs(prior, h, f)
        ids = ['a', 'a', 'a', 'a', 'b', 'c', 'c']
        expected = Standardizer.fit(inputs, ids)
        actual = independent_scaler(inputs, game_weights(ids))
        for name in expected.__dataclass_fields__:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=0, atol=2e-12)
        self.assertEqual(actual.context_scale[3], 1)

    def test_nested_comparison_rejects_changed_decision_or_missing_interval(self):
        expected = {'gate': False, 'paired': {'difference': .01, 'interval': [-.01, .03]}}
        compare_nested(expected, expected)
        with self.assertRaises(ValueError):
            compare_nested({**expected, 'gate': True}, expected)
        with self.assertRaises(ValueError):
            compare_nested({'gate': False, 'paired': {'difference': .01}}, expected)


if __name__ == '__main__':
    unittest.main()
