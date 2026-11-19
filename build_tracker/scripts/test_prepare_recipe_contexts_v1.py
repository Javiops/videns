from copy import deepcopy
import unittest
from prepare_recipe_contexts_v1 import attach_reference_roles, residual_public, target_projection


class ContextTests(unittest.TestCase):
    def fixture(self):
        others = [{'champion_id': i if i <= 5 else i-5, 'team_id': 100 if i <= 5 else 200,
                   'items': [1001], 'level': 6} for i in range(2, 11)]
        state = {'champion_id': 1, 'team_id': 100, 'inventory': [1036, 1036], 'level': 7,
                 'ts': 600000, 'gold_est': 1000., 'others': others}
        roles = {'reference_roles': {'champion_id': 1, 'team_id': 100, 'role': 'TOP',
                 'others': [{'champion_id': p['champion_id'], 'team_id': p['team_id'], 'role': str(i)}
                            for i, p in enumerate(reversed(others))]}}
        return state, roles

    def test_join_preserves_order_inventory_and_mirror_champion(self):
        state, roles = self.fixture(); original = deepcopy(state)
        merged = attach_reference_roles(state, roles)
        self.assertEqual(merged['inventory'], [1036, 1036])
        self.assertEqual([p['champion_id'] for p in merged['others']], [p['champion_id'] for p in state['others']])
        self.assertEqual(merged['others'][0]['role'], '8')
        self.assertEqual(state, original)
        self.assertEqual(merged['role'], 'TOP')

    def test_rejects_missing_or_wrong_roster(self):
        state, roles = self.fixture(); roles['reference_roles']['others'][0]['champion_id'] = 999
        with self.assertRaises(ValueError):
            attach_reference_roles(state, roles)

    def test_public_context_strips_legacy_roles_targets_and_private_fields(self):
        state, roles = self.fixture(); public = residual_public(state)
        changed = attach_reference_roles(state, roles)
        changed.update(win=True, label_counts={'3089': 1}, future_inventory=[3089], total_gold=999999)
        for p in changed['others']:
            p.update(gold=999999, win=True)
        self.assertEqual(public, residual_public(changed))
        self.assertEqual(set(public), {'inventory', 'champion_id', 'team_id', 'level', 'ts', 'gold_est', 'others'})

    def test_target_multiset_and_explicit_save_remain_distinct(self):
        base = {'encoding': 'item-counts-v1', 'wards_preserved': True, 'kind': 'retained_net_purchase',
                'label_counts': {'1036': 2, '2055': 1}, 'full_action_id': 50, 'main_group_id': 4}
        self.assertEqual(target_projection(base)['label_counts'], base['label_counts'])
        ward = {**base, 'label_counts': {'2055': 1}, 'main_group_id': 0}
        self.assertEqual(target_projection(ward)['kind'], 'retained_net_purchase')
        save = {**base, 'kind': 'observed_no_buy_death', 'label_counts': {'999999': 1}, 'full_action_id': 0, 'main_group_id': 0}
        self.assertEqual(target_projection(save)['kind'], 'observed_no_buy_death')
        with self.assertRaises(ValueError):
            target_projection({**save, 'label_counts': {'999999': 1, '2055': 1}})


if __name__ == '__main__':
    unittest.main()
