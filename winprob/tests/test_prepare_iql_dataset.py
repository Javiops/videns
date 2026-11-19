import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_iql_dataset import game_fold, population_gate, encode_episode


class IQLDatasetGateTests(unittest.TestCase):
    def gate_inputs(self):
        games = [{'match_id': 'synthetic'+str(i), 'fold': i % 2} for i in range(1000)]
        counts = np.zeros((2, 13166), dtype=np.int64); counts[:, 1] = 50000
        return (games, [None]*7000, 100000, {'all_source_complete': 10000}, counts,
                np.array([50000, 50000]), np.array([[100, 100], [100, 100]]))

    def test_declared_minima_pass_and_feature_drops_keep_denominator(self):
        args = list(self.gate_inputs())
        self.assertTrue(population_gate(*args)['passed'])
        args[1] = args[1][:-1]
        report = population_gate(*args)
        self.assertFalse(report['passed'])
        self.assertEqual(report['otherwise_source_complete_player_games'], 10000)

    def test_opposite_fold_support_is_not_own_fold_support(self):
        args = list(self.gate_inputs()); args[4][1, 1] = 0; args[4][1, 2] = 50000
        report = population_gate(*args)
        self.assertFalse(report['checks']['opposite_fold_action_coverage'])

    def test_terminal_classes_not_nonterminal_zero_counts(self):
        args = list(self.gate_inputs()); args[-1][0, 0] = 0
        self.assertFalse(population_gate(*args)['checks']['both_terminal_classes_per_fold'])

    def test_fold_is_game_only_and_deterministic(self):
        self.assertEqual([game_fold('match1') for _ in range(10)], [game_fold('match1')]*10)
        self.assertEqual({game_fold('match'+str(i)) for i in range(100)}, {0, 1})

    def test_bad_late_state_rejects_episode_before_any_result(self):
        class Compiler:
            def state(self, state, patch):
                if state['bad']:
                    raise ValueError('late invalid state')
                return {'placeholder': True}
        record = {'patch': '16.17', 'decisions': [{'prestate': {'bad': False}, 'action': {'full_action_id': 1}},
                                               {'prestate': {'bad': True}, 'action': {'full_action_id': 1}}]}
        with self.assertRaisesRegex(ValueError, 'late invalid'):
            encode_episode(record, [{}, {}], Compiler(), {'action_static_valid': np.ones((3, 2), dtype=bool)})


if __name__ == '__main__':
    unittest.main()
