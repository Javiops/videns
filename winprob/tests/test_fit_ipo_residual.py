"""Trainer front-door and actual adapter identity checks with synthetic arrays."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_ipo_residual import qualify_training_inputs, zero_residual_gate, fit_retained_frequency, frequency_pair_odds, policy_drift


class FitResidualGuardTests(unittest.TestCase):
    def fixture(self):
        rows = [{'role': 'fit', 'patch': '16.18', 'action_a': 1, 'action_b': 2, 'group_a': 0, 'group_b': 1},
                {'role': 'development', 'patch': '16.19', 'action_a': 1, 'action_b': 2, 'group_a': 0, 'group_b': 1}]
        return rows, np.array([[.4, .6], [.7, .3]], dtype=np.float32), np.array([True, True])

    def test_population_failure_stops_before_training(self):
        rows, p, keep = self.fixture()
        with self.assertRaisesRegex(ValueError, 'no training'):
            qualify_training_inputs(rows, p, keep, {'minimum_population_gate': False})

    def test_m2_trained_development_cannot_be_substituted(self):
        rows, p, keep = self.fixture(); rows[1]['patch'] = '16.18'
        with self.assertRaisesRegex(ValueError, 'untrained'):
            qualify_training_inputs(rows, p, keep, {'minimum_population_gate': True})

    def test_roundtrips_cannot_become_training_preferences(self):
        rows, p, keep = self.fixture(); rows[0]['action_b'] = 1
        with self.assertRaisesRegex(ValueError, 'control'):
            qualify_training_inputs(rows, p, keep, {'minimum_population_gate': True})

    def test_support_mask_is_applied_to_identical_rows_and_probabilities(self):
        rows, p, keep = self.fixture(); keep[0] = False
        selected, probability = qualify_training_inputs(rows, p, keep, {'minimum_population_gate': True})
        self.assertEqual(selected, [rows[1]])
        np.testing.assert_array_equal(probability, p[1:])

    def test_actual_zero_adapter_retains_full_order_and_pair_odds(self):
        _, p, _ = self.fixture()
        result = zero_residual_gate(np.array([[1., 2.], [1., 3.]]), np.eye(2), p, np.array([0, 0]), np.array([1, 1]))
        self.assertTrue(result['probability_identity_bytes_exact'])
        self.assertEqual(result['probability_dtype_preserved'], 'float32')
        self.assertEqual(result['full_rank_order_mismatches'], 0)
        self.assertLess(result['maximum_pair_log_odds_error'], 1e-12)

    def test_frequency_uses_fit_last_patch_and_global_without_challenge_labels(self):
        fit = [{'role': 'fit', 'patch': '16.17', 'source_projection': {'champion_id': 1}, 'action_b': 10},
               {'role': 'fit', 'patch': '16.18', 'source_projection': {'champion_id': 1}, 'action_b': 20}]
        model = fit_retained_frequency(fit)
        dev = [{'role': 'development', 'patch': '16.19', 'source_projection': {'champion_id': 1}, 'action_a': 10, 'action_b': 20},
               {'role': 'development', 'patch': '16.19', 'source_projection': {'champion_id': 999}, 'action_a': 10, 'action_b': 20}]
        odds, modes = frequency_pair_odds(model, dev)
        self.assertGreater(odds[0], 0)
        self.assertEqual(odds[1], 0)
        self.assertEqual(modes, {'latest_fit_patch_champion': 1, 'global_fit': 1})
        with self.assertRaisesRegex(ValueError, 'fit rows only'):
            fit_retained_frequency(fit+dev)

    def test_zero_policy_drift_is_zero_on_full_reference(self):
        context, actions = np.array([[1., 2.], [1., 3.]]), np.eye(2)
        probability = np.array([[.4, .6], [.7, .3]], dtype=np.float32)
        result = policy_drift(context, actions, np.zeros((2, 2)), probability, [.1, .2], np.array([0, 0]), np.array([1, 1]))
        self.assertAlmostEqual(result['weighted_mean_kl_candidate_to_m2'], 0)
        self.assertEqual(result['weighted_canonical_group_top1_change_fraction'], 0)


if __name__ == '__main__':
    unittest.main()
