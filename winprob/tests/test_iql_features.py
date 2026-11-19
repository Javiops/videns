from copy import deepcopy
import unittest
import numpy as np
from winprob.iql_features import PublicFeatureCompiler, GLOBAL_NAMES


class PublicFeaturesTests(unittest.TestCase):
    def setUp(self):
        items = {'1001': {'gold': {'total': 300}, 'stats': {'FlatHPPoolMod': 250}, 'tags': ['Health']},
                 '2055': {'gold': {'total': 75}, 'stats': {}, 'tags': []}}
        static = {patch: {'items': deepcopy(items), 'champion_tags': {str(i): ['Fighter'] for i in range(1, 11)}}
                  for patch in ('16.17', '16.18', '16.19')}
        self.compiler = PublicFeatureCompiler(static)
        self.state = {'champion_id': 1, 'team_id': 100, 'inventory': [1001, 2055, 2055], 'level': 5,
                      'ts': 300000, 'gold_est': 250., 'gold_est_version': 'prequential-v3',
                      'cs': 30, 'kills': 1, 'deaths': 0, 'assists': 2,
                      'ally_obj': {'towers': 0}, 'enemy_obj': {'towers': 1},
                      'others': [{'champion_id': i, 'team_id': 100 if i <= 5 else 200,
                                  'items': [1001], 'level': 5} for i in range(2, 11)]}

    def test_no_total_gold_roles_or_future_targets_consumed(self):
        before = self.compiler.state(self.state, '16.18')
        altered = deepcopy(self.state)
        altered.update(total_gold=1e12, role='changed', win=True, gold_arrival_true=-999,
                       can_complete=True, bot_quest_complete=True, source_target_id=888)
        for player in altered['others']:
            player.update(gold=1e12, role='changed', bot_quest_complete=True)
        after = self.compiler.state(altered, '16.18')
        for key in before:
            if key == 'inventory_bags':
                self.assertEqual(before[key], after[key])
            else:
                np.testing.assert_array_equal(before[key], after[key])

    def test_roster_and_inventory_order_do_not_change_features(self):
        first = self.compiler.state(self.state, '16.18')
        self.state['others'].reverse(); self.state['inventory'].reverse()
        second = self.compiler.state(self.state, '16.18')
        for key in first:
            if key == 'inventory_bags':
                self.assertEqual(first[key], second[key])
            else:
                np.testing.assert_array_equal(first[key], second[key])

    def test_counts_are_not_truncated_to_slot_limit(self):
        self.state['inventory'] = [2055]*21
        result = self.compiler.state(self.state, '16.18')
        self.assertEqual(result['inventory_bags'][0], ((self.compiler.item_index[2055], 21),))
        self.assertEqual(result['inventory_static'][0, -1], 21/6)

    def test_known_zero_and_missing_objective_differ(self):
        result = self.compiler.state(self.state, '16.18')['global_public']
        self.assertEqual(result[GLOBAL_NAMES.index('ally_towers_present')], 1)
        self.assertEqual(result[GLOBAL_NAMES.index('ally_dragons_present')], 0)
        self.assertEqual(len(result), 37)

    def test_unknown_item_and_incomplete_roster_fail(self):
        self.state['inventory'] = [777777]
        with self.assertRaisesRegex(ValueError, 'native public static'):
            self.compiler.state(self.state, '16.18')
        self.state['inventory'] = []; self.state['others'].pop()
        with self.assertRaisesRegex(ValueError, 'ten-player'):
            self.compiler.state(self.state, '16.18')

    def test_save_and_wards_full_action_features_distinct(self):
        bag, save = self.compiler.full_action([], 'observed_no_buy_death')
        ward_bag, wards = self.compiler.full_action([[2055, 2]], 'retained_net_purchase')
        self.assertEqual(bag, ()); self.assertEqual(save[-1], 1)
        self.assertEqual(ward_bag[0][1], 2); self.assertEqual(wards[-1], 0)

    def test_critic_action_uses_declared_native_patch(self):
        self.compiler.static['16.17']['items']['2055']['gold']['total'] = 90
        _, earlier = self.compiler.full_action([[2055, 1]], 'retained_net_purchase', patch='16.17')
        _, reference = self.compiler.full_action([[2055, 1]], 'retained_net_purchase', patch='16.18')
        self.assertAlmostEqual(float(earlier[0]), 90/3000)
        self.assertAlmostEqual(float(reference[0]), 75/3000)


if __name__ == '__main__':
    unittest.main()
