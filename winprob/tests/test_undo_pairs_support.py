"""Synthetic revision-pair/roundtrip controls; no real match data or outcomes."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from undo_pairs_support import analyze_game


class ProtectedFields(dict):
    def __getitem__(self, key):
        if key in ('win', 'winner', 'teams', 'gameDuration', 'kills', 'item0'):
            raise AssertionError('forbidden field access: '+key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ('win', 'winner', 'teams', 'gameDuration', 'kills', 'item0'):
            raise AssertionError('forbidden field access: '+key)
        return super().get(key, default)


def e(ts, kind, pid=1, item=0, **extra):
    return {'timestamp': ts, 'type': kind, 'participantId': pid, 'itemId': item, **extra}


def triplet(start=200000, pid=1, a=1036, b=1029):
    return [e(start, 'ITEM_PURCHASED', pid, a),
            e(start+1000, 'ITEM_UNDO', pid, beforeId=a, afterId=0, goldGain=350),
            e(start+2000, 'ITEM_PURCHASED', pid, b)]


def fixture(events=None, source_end=300000):
    def item(price, recipe=(), tags=()):
        return {'gold': {'total': price, 'purchasable': True}, 'maps': {'11': True},
                'from': list(map(str, recipe)), 'tags': list(tags)}
    items = {'1036': item(350), '1029': item(300), '1033': item(400),
             '999': item(900, (1036,)), '2003': item(50, tags=('Consumable',))}
    people = [ProtectedFields(participantId=i, championId=i, teamId=100 if i <= 5 else 200, win=True)
              for i in range(1, 11)]
    match = {'info': ProtectedFields(participants=people, teams='FORBIDDEN')}
    frames = [{'timestamp': ts, 'participantFrames': {str(i): {'currentGold': 1000, 'totalGold': 3000, 'xp': 500, 'level': 3}
                                                    for i in range(1, 11)}, 'events': []}
              for ts in (180000, 240000, source_end)]
    frames[-1]['events'] = triplet() if events is None else events
    cache = {180000: np.array([3., 100.]), 240000: np.array([4., 200.])}
    return match, {'info': {'frames': frames}}, {'items': items}, cache, ['time_min', 'gold_diff']


class UndoPairsSupportTests(unittest.TestCase):
    def test_isolated_full_refund_replacement_is_revision_not_causal_value(self):
        rows, counts = analyze_game(*fixture())
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row['eligible_revision'])
        self.assertFalse(row['eligible_roundtrip_control'])
        self.assertEqual(row['feature_cutoff_ms'], 200000)
        self.assertEqual(row['decision_timestamp_ms'], 202000)
        self.assertEqual(row['feature_frame_ms'], 180000)
        self.assertEqual(row['random_focal_pair_weight'], .1)
        self.assertEqual(counts['players_with_eligible_revision'], 1)

    def test_same_item_rebuy_is_no_preference_control(self):
        rows, _ = analyze_game(*fixture(triplet(b=1036)))
        self.assertTrue(rows[0]['eligible_roundtrip_control'])
        self.assertFalse(rows[0]['eligible_revision'])
        self.assertEqual(rows[0]['random_focal_pair_weight'], 0)

    def test_ordinary_adjacent_purchases_never_form_rejections(self):
        events = [e(200000, 'ITEM_PURCHASED', item=1036), e(201000, 'ITEM_PURCHASED', item=1029)]
        rows, counts = analyze_game(*fixture(events))
        self.assertEqual(rows, [])
        self.assertEqual(counts.get('undo_events', 0), 0)

    def test_undone_sale_or_mismatched_purchase_is_not_certified(self):
        for before, after, refund in ((0, 1036, -245), (1029, 0, 350), (1036, 0, None)):
            events = triplet(); events[1].update(beforeId=before, afterId=after, goldGain=refund)
            rows, _ = analyze_game(*fixture(events))
            self.assertFalse(rows[0]['strict_transaction_pair'])

    def test_partial_refund_or_returned_component_requires_reconstruction(self):
        for after, refund in ((0, 100), (1001, 350)):
            events = triplet(); events[1].update(afterId=after, goldGain=refund)
            rows, _ = analyze_game(*fixture(events))
            self.assertFalse(rows[0]['eligible_revision'])

    def test_missing_or_null_undo_ids_are_not_repaired_as_zero(self):
        for field in ('beforeId', 'afterId'):
            for missing in (True, False):
                events = triplet()
                if missing:
                    del events[1][field]
                else:
                    events[1][field] = None
                rows, _ = analyze_game(*fixture(events))
                self.assertFalse(rows[0]['strict_transaction_pair'])
                self.assertIn('undo_' + ('before' if field == 'beforeId' else 'after')
                              + '_id_not_explicit_integer', rows[0]['rejection_reasons'])

    def test_boolean_or_float_undo_ids_are_not_integer_certificates(self):
        for field in ('beforeId', 'afterId'):
            for value in (False, True, 0.0):
                events = triplet()
                events[1][field] = value
                rows, _ = analyze_game(*fixture(events))
                self.assertFalse(rows[0]['strict_transaction_pair'])
                self.assertIn('undo_' + ('before' if field == 'beforeId' else 'after')
                              + '_id_not_explicit_integer', rows[0]['rejection_reasons'])

    def test_component_destruction_is_not_mistaken_for_simple_inverse(self):
        events = triplet()+[e(200000, 'ITEM_DESTROYED', item=1001)]
        rows, _ = analyze_game(*fixture(events))
        self.assertIn('inventory_destruction_or_transformation_in_window', rows[0]['rejection_reasons'])

    def test_extra_purchase_and_retained_item_undo_are_rejected(self):
        for extra in (e(203000, 'ITEM_PURCHASED', item=1033),
                      e(203000, 'ITEM_UNDO', beforeId=1029, afterId=0, goldGain=300),
                      e(203000, 'ITEM_DESTROYED', item=1029)):
            rows, _ = analyze_game(*fixture(triplet()+[extra]))
            self.assertFalse(rows[0]['eligible_revision'])

    def test_equal_timestamp_and_long_sequence_are_not_repaired(self):
        for times in ((200000, 200000, 202000), (200000, 207000, 214000)):
            events = triplet()
            for ev, ts in zip(events, times): ev['timestamp'] = ts
            rows, _ = analyze_game(*fixture(events))
            self.assertFalse(rows[0]['strict_transaction_pair'])

    def test_material_change_invalidates_primary_but_other_player_shop_is_flagged(self):
        rows, _ = analyze_game(*fixture(triplet()+[e(201500, 'CHAMPION_KILL', pid=2)]))
        self.assertFalse(rows[0]['eligible_revision'])
        rows, _ = analyze_game(*fixture(triplet()+[e(201500, 'ITEM_PURCHASED', pid=2, item=1033)]))
        self.assertTrue(rows[0]['eligible_revision'])
        self.assertEqual(rows[0]['audit']['nonfocal_item_events_during_pair'], 1)

    def test_nested_recipe_choice_is_counted_but_not_primary_preference(self):
        rows, _ = analyze_game(*fixture(triplet(b=999)))
        self.assertTrue(rows[0]['recipe_nested'])
        self.assertFalse(rows[0]['eligible_revision'])

    def test_retention_window_must_be_observed(self):
        args = list(fixture(source_end=205000)); args[1]['info']['frames'] = [args[1]['info']['frames'][0], args[1]['info']['frames'][-1]]
        rows, _ = analyze_game(*args)
        self.assertIn('retention_window_not_observed_to_closure', rows[0]['rejection_reasons'])

    def test_context_cannot_use_a_frame_between_purchase_and_undo(self):
        args = list(fixture())
        args[1]['info']['frames'].insert(1, {'timestamp': 200500, 'participantFrames': {'1': {'currentGold': 0}}, 'events': []})
        args[3][200500] = np.array([3.3417, 99999.])
        rows, _ = analyze_game(*args)
        self.assertEqual(rows[0]['feature_frame_ms'], 180000)
        self.assertEqual(rows[0]['pre_pair_features']['gold_diff'], 100)

    def test_more_expensive_b_is_certified_at_b_not_backdated_to_a(self):
        rows, _ = analyze_game(*fixture(triplet(b=1033)))
        self.assertTrue(rows[0]['certificate']['both_feasible_at_b_conditional'])
        self.assertFalse(rows[0]['certificate']['b_already_affordable_at_a_by_price_dominance'])

    def test_refund_proof_over_budget_grid(self):
        for original_wallet in range(350, 2001, 50):
            for income in (0, 5, 100):
                restored = original_wallet-350+350+income
                for b_price in (300, 400, 900):
                    if restored >= b_price:  # factual B successfully executed
                        self.assertGreaterEqual(restored, 350)
                        self.assertGreaterEqual(restored, b_price)

    def test_all_player_weights_integrate_over_uniform_focal_selection(self):
        events = triplet()+triplet(260000)+triplet(220000, pid=2)
        rows, counts = analyze_game(*fixture(events))
        eligible = [row for row in rows if row['eligible_revision']]
        self.assertEqual(len(eligible), 3)
        self.assertAlmostEqual(sum(row['random_focal_pair_weight'] for row in eligible), .2)
        self.assertEqual(counts['players_with_eligible_revision'], 2)
        self.assertEqual(sorted(row['random_focal_pair_weight'] for row in eligible), [.05, .05, .1])


if __name__ == '__main__':
    unittest.main()
