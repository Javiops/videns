from copy import deepcopy
import sys
import unittest
from winprob.recipe_transition_data_v1 import validate_options, labels_for_action
from winprob.full_action_catalogue import counts_key


class ConsumerContractTests(unittest.TestCase):
    def fixture(self):
        groups = [[], [[1036, 2]], [[3047, 1]]]
        row = {'group_ids': [1, 0], 'm2_options': [
            {'group': 1, 'items': {'1036': 2}, 'kind': 'buy', 'legal': True},
            {'group': 0, 'items': {}, 'kind': 'save', 'legal': True}],
            'receipts': [{'before': [], 'wallet': 1000., 'action': [1036, 1036], 'steps': []},
                         {'before': [], 'wallet': 1000., 'action': [], 'steps': []}],
            'residual_public': {'inventory': [], 'gold_est': 1000.}}
        return groups, row

    def test_option_multisets_retain_duplicates(self):
        groups, row = self.fixture()
        self.assertEqual(validate_options(row, groups), [1, 0])
        wrong = deepcopy(row); wrong['receipts'][0]['action'] = [1036]
        with self.assertRaises(ValueError):
            validate_options(wrong, groups)

    def test_receipt_must_use_same_wallet_and_inventory(self):
        groups, row = self.fixture()
        for key, value in (('wallet', 999.), ('before', [1036])):
            changed = deepcopy(row); changed['receipts'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_options(changed, groups)

    def test_outside_purchase_stays_OTHER_not_replaced(self):
        action = {'kind': 'retained_net_purchase', 'encoding': 'item-counts-v1', 'wards_preserved': True,
                  'label_counts': {'3047': 1}, 'full_action_id': 2, 'main_group_id': 2}
        lookup = {counts_key(action['label_counts']): 2}
        got = labels_for_action(action, {'full_to_group': [0, 1, 2]}, lookup, [1, 0, -1])
        self.assertEqual(got['coarsened'], 3)
        self.assertEqual(got['main'], 2)
        self.assertEqual(got['main_size'], 1)

    def test_ward_only_and_observed_no_buy_share_main_not_full_semantics(self):
        common = {'encoding': 'item-counts-v1', 'wards_preserved': True}
        ward = {**common, 'kind': 'retained_net_purchase', 'label_counts': {'2055': 1}, 'full_action_id': 1, 'main_group_id': 0}
        save = {**common, 'kind': 'observed_no_buy_death', 'label_counts': {'999999': 1}, 'full_action_id': 0, 'main_group_id': 0}
        lookup = {counts_key(a['label_counts']): a['full_action_id'] for a in (ward, save)}
        w = labels_for_action(ward, {'full_to_group': [0, 0]}, lookup, [0, 1, -1])
        s = labels_for_action(save, {'full_to_group': [0, 0]}, lookup, [0, 1, -1])
        self.assertEqual(w['coarsened'], s['coarsened'])
        self.assertTrue(w['contains_wards']); self.assertFalse(w['is_no_buy'])
        self.assertFalse(s['contains_wards']); self.assertTrue(s['is_no_buy'])
        self.assertNotEqual(w['full'], s['full'])

    def test_module_does_not_import_native_runtime(self):
        self.assertNotIn('app.mixture_policy', sys.modules)
        self.assertNotIn('train_prefix', sys.modules)


if __name__ == '__main__':
    unittest.main()
