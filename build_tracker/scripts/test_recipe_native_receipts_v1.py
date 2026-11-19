from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from recipe_native_receipts_v1 import execution_arguments, replay_candidate


class ReceiptTests(unittest.TestCase):
    def test_absent_optional_flag_uses_native_default(self):
        self.assertEqual(execution_arguments({'role': 'TOP', 'bot_quest_complete': False, 'unmodeled_regular_slots': 2}),
                         {'role': 'TOP', 'bot_quest_complete': False, 'unmodeled_regular_slots': 2, 'role_slot_inventory_unknown': False})
        self.assertTrue(execution_arguments({'role_slot_inventory_unknown': True})['role_slot_inventory_unknown'])

    def test_receipt_preserves_counts_costs_and_protected_purchases(self):
        calls = []
        def buy(iid, inventory, wallet, dragon, **kwargs):
            calls.append((wallet, kwargs))
            return 100., Counter(inventory)+Counter([iid])
        modules = {'app.conditional_policy': SimpleNamespace(
            footwear_arguments=lambda *a, **k: {'magical_footwear_selected': False, 'footwear_granted': False},
            purchase_sequence=lambda *a: [{'item_id': 1036, 'cost': 100.}, {'item_id': 1036, 'cost': 100.}]),
            'app.shop_econ': SimpleNamespace(try_purchase=buy)}
        row = {'inventory': [], 'gold_est': 300., 'role': 'TOP', 'bot_quest_complete': False}
        with patch.dict('sys.modules', modules):
            result = replay_candidate(row, [1036, 1036], SimpleNamespace(footwear_state_version='fixture'))
        self.assertEqual(result['steps'][-1]['inventory_after'], [1036, 1036])
        self.assertEqual(calls[1][0], 200.)
        self.assertEqual(calls[1][1]['protected'], Counter({1036: 1}))
        self.assertFalse(calls[0][1]['role_slot_inventory_unknown'])
        self.assertEqual(row['inventory'], [])


if __name__ == '__main__':
    unittest.main()
