import unittest
import numpy as np
from winprob.bayes_proxy_data_v1 import LabelGate


class LabelGateTests(unittest.TestCase):
    def make(self):
        calls = []; a = {'full_action_id': np.array([1, 2, 3, 4, 1, 2]), 'main_group_id': np.array([1, 2, 0, 3, 1, 2])}
        def loader(name, rows): calls.append((name, rows.tolist())); return a[name][rows]
        gate = LabelGate(np.array([0, 0, 0, 1, 1, 1]), {0: [0], 1: [3]}, np.array([0, 1, 2, 0, 3]), loader)
        return gate, calls, a

    def test_current_future_or_other_past_rejected_before_any_access(self):
        gate, calls, _ = self.make()
        for rows, purpose in (([1], 'past'), ([2], 'past'), ([3], 'past'), ([0], 'training'), ([1], 'training')):
            with self.assertRaises(ValueError): gate.read(np.array(rows), 0, purpose)
        self.assertEqual(calls, [])

    def test_opposite_fold_training_and_only_published_past_allowed(self):
        gate, calls, _ = self.make()
        np.testing.assert_array_equal(gate.read(np.array([3, 4, 5]), 0, 'training'), [3, 1, 2])
        np.testing.assert_array_equal(gate.read(np.array([0]), 0, 'past'), [1])
        np.testing.assert_array_equal(gate.read(np.array([0, 1, 2]), 1, 'training'), [1, 2, 0])

    def test_query_fold_future_mutation_cannot_change_allowed_labels(self):
        gate, _, a = self.make(); before = gate.read(np.array([0]), 0, 'past')
        a['full_action_id'][1:3] = 4; a['main_group_id'][1:3] = 3
        np.testing.assert_array_equal(gate.read(np.array([0]), 0, 'past'), before)

    def test_synthetic_save_never_becomes_past_evidence(self):
        gate, _, a = self.make(); a['full_action_id'][0] = 0; a['main_group_id'][0] = 0
        with self.assertRaises(ValueError): gate.read(np.array([0]), 0, 'past')


if __name__ == '__main__': unittest.main()
