import unittest
import numpy as np

from winprob.preference_evaluation import evaluate_preferences, pair_credit


class PreferenceEvaluationTests(unittest.TestCase):
    def args(self):
        return dict(game_ids=['g1', 'g2', 'g3', 'g4'], weights=[.1]*4,
                    action_a=[100]*4, action_b=[200]*4,
                    total_log_odds_by_arm={'ipo': [1, 1, -1, 1],
                                          'm2': [-1, 1, -1, 1],
                                          'winner_only': [1, -1, -1, 1]},
                    bootstrap_samples=100)

    def test_total_odds_and_ties(self):
        np.testing.assert_array_equal(pair_credit([-2, 0, 2]), [0, .5, 1])
        # A positive correction +1 still loses against reference log odds -2.
        self.assertEqual(pair_credit(np.array([-2.])+1).item(), 0)

    def test_repeated_pairs_do_not_invent_independent_games(self):
        args = self.args()
        original = evaluate_preferences(**args)
        for key in ('game_ids', 'action_a', 'action_b'):
            args[key] = np.repeat(args[key], 10)
        args['weights'] = np.repeat(np.asarray(args['weights'])/10, 10)
        args['total_log_odds_by_arm'] = {k: np.repeat(v, 10)
                                       for k, v in args['total_log_odds_by_arm'].items()}
        repeated = evaluate_preferences(**args)
        self.assertEqual(repeated['games'], 4)
        for name in original['arms']:
            np.testing.assert_allclose(original['arms'][name]['interval95'],
                                       repeated['arms'][name]['interval95'], atol=1e-14)
        self.assertEqual(original['metric_gate_passed'], repeated['metric_gate_passed'])

    def test_same_policy_paired_interval_exact_zero(self):
        args = self.args()
        args['total_log_odds_by_arm']['m2'] = args['total_log_odds_by_arm']['ipo']
        report = evaluate_preferences(**args)
        self.assertEqual(report['paired_contrasts']['m2']['interval95'], [0., 0.])
        self.assertFalse(report['metric_gate_passed'])

    def test_must_beat_both_baselines(self):
        args = self.args()
        args['total_log_odds_by_arm'] = {'ipo': [1]*4, 'm2': [-1]*4, 'winner_only': [1]*4}
        report = evaluate_preferences(**args)
        self.assertTrue(report['paired_contrasts']['m2']['passes_metric_gate'])
        self.assertFalse(report['metric_gate_passed'])

    def test_weight_scale_invariance(self):
        args = self.args()
        before = evaluate_preferences(**args)
        args['weights'] = [10]*4
        after = evaluate_preferences(**args)
        self.assertEqual(before, after)

    def test_controls_and_nonfinite_rejected(self):
        args = self.args()
        args['action_b'][0] = args['action_a'][0]
        with self.assertRaises(ValueError):
            evaluate_preferences(**args)
        with self.assertRaises(ValueError):
            pair_credit([np.inf])

    def test_no_empty_comparator_success(self):
        with self.assertRaisesRegex(ValueError, 'baseline'):
            evaluate_preferences(**self.args(), baselines=())


if __name__ == '__main__':
    unittest.main()
