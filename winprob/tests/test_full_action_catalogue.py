import unittest
from unittest.mock import patch
from winprob.full_action_catalogue import counts_key, encode_complete_episode


class FullActionIdentityTests(unittest.TestCase):
    def test_save_one_ward_and_two_wards_are_distinct(self):
        self.assertEqual(len({counts_key({'999999': 1}), counts_key({'2055': 1}),
                              counts_key({'2055': 2})}), 3)

    def test_counts_and_order_preserved(self):
        self.assertEqual(counts_key({'2055': 2, '1036': 1}), counts_key({'1036': 1, '2055': 2}))
        self.assertNotEqual(counts_key({'2055': 2}), counts_key({'2055': 1}))
        for invalid in ({}, {'999999': 1, '2055': 1}, {'2055': True}, {'02055': 1}):
            with self.assertRaises(ValueError):
                counts_key(invalid)

    def test_one_unknown_action_excludes_whole_episode(self):
        transitions = [{'action': {'label_counts': {'1036': 1}}, 'reward': 0},
                       {'action': {'label_counts': {'1036': 2}}, 'reward': 1}]
        with patch('winprob.full_action_catalogue.compile_transitions', return_value=transitions):
            result = encode_complete_episode({}, {counts_key({'1036': 1}): 7})
        self.assertFalse(result['eligible'])
        self.assertNotIn('transitions', result)
        self.assertEqual(result['original_transitions'], 2)


if __name__ == '__main__':
    unittest.main()
