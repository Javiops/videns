"""Synthetic boot-visit selection and conservative budget tests only."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from boots_support import (analyze_game, boot_categories, find_first_choice,
    focal_events, prior_boot_evidence, select_focal, wallet_lower_bound)


class NoOutcomes(dict):
    def __getitem__(self, key):
        if key in ('win', 'winner', 'teams', 'gameDuration', 'kills', 'deaths', 'assists', 'item0'):
            raise AssertionError('outcome/final-field access: '+key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ('win', 'winner', 'teams', 'gameDuration', 'kills', 'deaths', 'assists', 'item0'):
            raise AssertionError('outcome/final-field access: '+key)
        return super().get(key, default)


def catalogue():
    def item(price, recipe=(), into=()):
        return {'gold': {'total': price, 'purchasable': True}, 'maps': {'11': True},
                'from': list(map(str, recipe)), 'into': list(map(str, into))}
    return {'1001': item(300, into=(3047, 3111, 3020)), '1029': item(300),
            '1033': item(400), '3047': item(1200, (1001, 1029), (3174,)),
            '3111': item(1250, (1001, 1033), (3173,)), '3020': item(1100, (1001,)),
            '3174': item(1800, (3047,)), '3173': item(1850, (3111,)), '2003': item(50)}


def event(timestamp, item, kind='ITEM_PURCHASED', **extra):
    return {'timestamp': timestamp, 'itemId': item, 'type': kind, 'participantId': 1, **extra}


class BootsSupportTests(unittest.TestCase):
    def fixture(self, events=None, wallet=1500):
        participants = [NoOutcomes(participantId=i, championId=i, teamId=100 if i <= 5 else 200,
                                  perks={'styles': [{'selections': [{'perk': 8005}]}]}, win=True)
                        for i in range(1, 11)]
        match = {'metadata': {'matchId': 'SYNTHETIC_1'}, 'info': NoOutcomes(participants=participants, teams='FORBIDDEN')}
        tags = {str(i): ['Fighter'] if i == 1 else ['Mage'] for i in range(1, 11)}
        if events is None:
            events = [event(300000, 1001), event(305000, 1029), event(310000, 3047)]
        timeline = {'info': {'frames': [
            {'timestamp': 240000, 'participantFrames': {'1': {'currentGold': wallet, 'totalGold': 5000, 'xp': 3000, 'level': 6}}, 'events': []},
            {'timestamp': 300000, 'participantFrames': {'1': {'currentGold': 0}}, 'events': events},
        ]}}
        return match, timeline, {'items': catalogue(), 'champion_tags': tags}, {240000: np.array([4., 100.])}, ['time_min', 'gold_diff']

    def test_roster_selector_does_not_read_outcomes_or_actual_role(self):
        match, _, static, _, _ = self.fixture()
        focal, roster = select_focal(match, static['champion_tags'])
        self.assertEqual(focal['participant_id'], 1)
        self.assertEqual(len(roster), 10)
        self.assertIs(focal['magical_footwear_selected'], False)

    def test_previsit_cutoff_excludes_all_chosen_recipe_purchases(self):
        row = analyze_game(*self.fixture())
        self.assertEqual(row['cutoff_ms'], 300000)
        self.assertEqual(row['purchase_ms'], 310000)
        self.assertEqual(row['prior_frame_ms'], 240000)
        self.assertEqual(row['wallet_bound']['lower_bound'], 1500)
        self.assertEqual(row['wallet_bound']['known_full_price_spend_upper_bound'], 0)
        self.assertEqual(row['pre_visit_boot_evidence']['ordinary_boots_explicit_count'], 0)
        self.assertTrue(row['affordability']['primary_full_price_support'])
        self.assertEqual(row['affordability']['slot_legality'], 'unknown')

    def test_wallet_subtracts_only_earlier_known_full_prices(self):
        args = self.fixture([event(260000, 2003), event(300000, 1001), event(305000, 1029), event(310000, 3047)])
        row = analyze_game(*args)
        self.assertEqual(row['wallet_bound']['lower_bound'], 1450)
        self.assertEqual(row['wallet_bound']['known_full_price_spend_upper_bound'], 50)

    def test_chosen_purchase_cost_cannot_reconstruct_arrival_money(self):
        events = [{'type': 'ITEM_PURCHASED', 'itemId': 3047, 'timestamp': 300000}]
        low = wallet_lower_bound(100, 240000, 300000, events, catalogue())
        self.assertEqual(low['lower_bound'], 100)
        items = catalogue(); items['3047']['gold']['total'] = 100000
        self.assertEqual(wallet_lower_bound(100, 240000, 300000, events, items)['lower_bound'], 100)

    def test_same_timestamp_recipe_events_do_not_enter_previsit_features(self):
        args = self.fixture([event(300000, 1001), event(300000, 1029), event(300000, 3047)])
        row = analyze_game(*args)
        self.assertEqual(row['cutoff_ms'], 300000)
        self.assertEqual(row['wallet_bound']['lower_bound'], 1500)
        self.assertEqual(row['pre_visit_boot_evidence']['ordinary_boots_explicit_count'], 0)

    def test_undo_in_visit_is_counted_but_not_supported(self):
        args = self.fixture([event(300000, 3047), event(301000, 0, 'ITEM_UNDO', beforeId=3047, afterId=1001)])
        row = analyze_game(*args)
        self.assertTrue(row['label_audit']['undo_in_visit'])
        self.assertFalse(row['affordability']['primary_full_price_support'])

    def test_undo_or_unknown_cost_between_frame_and_visit_invalidates_bound(self):
        for event_value, reason in [(event(260000, 0, 'ITEM_UNDO'), 'undo_in_wallet_interval'),
                                    (event(260000, 999999), 'unknown_intervening_purchase_price')]:
            args = self.fixture([event_value, event(300000, 3047)])
            row = analyze_game(*args)
            self.assertIsNone(row['wallet_bound']['lower_bound'])
            self.assertIn(reason, row['wallet_bound']['unknown_reasons'])

    def test_existing_ordinary_boots_discount_is_separate_from_primary_fullprice(self):
        row = analyze_game(*self.fixture([event(180000, 1001), event(300000, 3047)], wallet=1000))
        self.assertTrue(row['pre_visit_boot_evidence']['ordinary_boots_proven'])
        self.assertFalse(row['affordability']['primary_full_price_support'])
        self.assertTrue(row['affordability']['secondary_ordinary_boots_support'])

    def test_magical_footwear_or_missing_rune_metadata_is_not_invented(self):
        args = self.fixture([event(180000, 1001), event(300000, 3047)])
        args[0]['info']['participants'][0]['perks']['styles'][0]['selections'][0]['perk'] = 8304
        row = analyze_game(*args)
        self.assertFalse(row['pre_visit_boot_evidence']['ordinary_boots_proven'])
        self.assertFalse(row['affordability']['primary_full_price_support'])
        del args[0]['info']['participants'][0]['perks']
        row = analyze_game(*args)
        self.assertIn('magical_footwear_selection_unknown', row['pre_visit_boot_evidence']['ambiguities'])

    def test_first_other_boot_is_not_replaced_by_later_defensive_purchase(self):
        row = analyze_game(*self.fixture([event(300000, 3020), event(600000, 3047)]))
        self.assertEqual(row['action_item_id'], 3020)
        self.assertFalse(row['affordability']['reliable_unmixed_pair'])

    def test_later_upgrade_without_prior_tier2_has_explicit_status(self):
        row = analyze_game(*self.fixture([event(300000, 3174)]))
        self.assertEqual(row['status'], 'later_boot_upgrade_before_tier2')

    def test_mixed_purchase_visit_has_explicit_cotreatment_flag(self):
        row = analyze_game(*self.fixture([event(300000, 2003), event(301000, 3047)]))
        self.assertTrue(row['label_audit']['mixed_basket'])
        self.assertFalse(row['affordability']['primary_full_price_support'])

    def test_missing_strict_prior_cache_frame_is_not_repaired(self):
        args = list(self.fixture()); args[3] = {180000: np.array([3., 0.])}
        row = analyze_game(*args)
        self.assertEqual(row['feature_status'], 'prior_frame_not_in_clean_cache')
        self.assertNotIn('pre_visit_features', row)

    def test_boundary_wallet_does_not_claim_affordability(self):
        bound = wallet_lower_bound(5000, 299500, 300000, [], catalogue())
        self.assertIsNone(bound['lower_bound'])
        self.assertIn('frame_event_boundary_under_one_second', bound['unknown_reasons'])

    def test_static_graph_retains_all_categories_and_separates_later_upgrades(self):
        tier2, later = boot_categories(catalogue())
        self.assertEqual(tier2, {3020, 3047, 3111})
        self.assertEqual(later, {3173, 3174})

    def test_quest_destroy_in_target_visit_is_audit_only_and_excludes_support(self):
        baseline = analyze_game(*self.fixture([event(300000, 3047)]))
        for timestamp in (300000, 301000):
            row = analyze_game(*self.fixture([event(300000, 3047), event(timestamp, 1201, 'ITEM_DESTROYED')]))
            self.assertTrue(row['label_audit']['role_quest_transform_in_visit'])
            self.assertFalse(row['affordability']['primary_full_price_support'])
            self.assertEqual(row['pre_visit_features'], baseline['pre_visit_features'])
            self.assertEqual(row['wallet_bound'], baseline['wallet_bound'])
            self.assertEqual(row['pre_visit_boot_evidence'], baseline['pre_visit_boot_evidence'])

    def test_tier2_destroy_and_later_upgrade_in_visit_are_ambiguous(self):
        row = analyze_game(*self.fixture([event(300000, 3047), event(301000, 3047, 'ITEM_DESTROYED')]))
        self.assertTrue(row['label_audit']['tier2_destroy_in_visit'])
        self.assertFalse(row['affordability']['primary_full_price_support'])
        row = analyze_game(*self.fixture([event(300000, 3047), event(301000, 3174)]))
        self.assertTrue(row['label_audit']['later_boot_upgrade_in_visit'])
        self.assertFalse(row['affordability']['primary_full_price_support'])


if __name__ == '__main__':
    unittest.main()
