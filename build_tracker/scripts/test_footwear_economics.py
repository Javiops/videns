"""Bounded, offline tests of real patch-16.17 footwear recipes and raw identity."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from app.ddragon import DataDragon
from app.shop_econ import combine_cost as original_combine_cost
from app.shop_econ import decision_kind as original_decision_kind
from app.shop_econ import inventory_shape
from app.shop_econ import inventory_state as original_inventory_state
from app.shop_econ import try_purchase as original_try_purchase
from footwear_economics import (
    FootwearDataDragon, buy_cost, combine_cost, components_of, context_inventory,
    decision_kind, grant_economics, has_direct_components, inventory_state,
    is_blocked, net_spent, purchase_targets, try_purchase,
)


def pinned_dragon() -> DataDragon:
    # Bypass DataDragon.__init__: tests must not download data or rewrite the
    # active static cache. Prefer the captured primary-source artifact.
    source = ROOT / "data/acceptance/footwear_repair_20260911/sources/item-16.17.1.json"
    if not source.is_file():
        source = ROOT / "data/cache/item-16.17.1.json"
    raw = source.read_bytes()
    data = json.loads(raw)
    assert data["version"] == "16.17.1"
    dragon = DataDragon.__new__(DataDragon)
    dragon.version = "16.17.1"
    dragon._items = data
    dragon._champions = {"version": "16.17.1", "data": {}}
    dragon._champ_by_id = {}
    dragon.signature = hashlib.sha256(raw).hexdigest()
    dragon._combine_recipes = None
    dragon._gold_cache = {}
    dragon._from_cache = {}
    dragon._classify_cache = {}
    return dragon


class FootwearEconomicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = pinned_dragon()
        cls.dragon = FootwearDataDragon(cls.source)

    def test_context_label_and_sale_identity(self):
        raw = self.source.classify(2422).copy()
        self.assertTrue(raw["skip"])
        self.assertFalse(self.dragon.classify(2422)["skip"])
        self.assertTrue(self.dragon.classify(2422)["target_skip"])
        self.assertFalse(self.dragon.gold_block(2422)["purchasable"])
        self.assertEqual(context_inventory([2422, 1036, 1036], self.dragon), [2422, 1036, 1036])
        self.assertEqual(purchase_targets([2422, 1036, 1036], self.dragon), [1036, 1036])
        self.assertEqual(self.dragon.gold_block(2422)["sell"], 90)
        self.assertEqual(self.dragon.gold_block(1001)["sell"], 210)
        self.assertEqual(self.source.classify(2422), raw)
        self.assertNotEqual(self.dragon.signature, self.source.signature)

    def test_sorcerer_discount_and_physical_consumption(self):
        owned = Counter({2422: 1, 1036: 1})
        original = owned.copy()
        self.assertEqual(combine_cost(3020, owned, self.dragon), 800)
        self.assertEqual(owned, original)
        self.assertEqual(combine_cost(3020, owned, self.dragon, consume=True), 800)
        self.assertEqual(+owned, Counter({1036: 1}))
        self.assertNotIn(1001, owned)
        consumed = original - owned
        self.assertEqual(consumed, Counter({2422: 1}))
        owned.update(consumed)  # A ledger undo restores physical 2422, not 1001.
        self.assertEqual(owned, original)

    def test_recursive_and_partial_component_consumption(self):
        owned = Counter({2422: 1, 1042: 1})
        self.assertEqual(combine_cost(3172, owned, self.dragon, consume=True), 550)
        self.assertEqual(+owned, Counter())
        both_daggers = Counter({2422: 1, 1042: 2})
        self.assertEqual(combine_cost(3006, both_daggers, self.dragon, consume=True), 300)
        self.assertEqual(+both_daggers, Counter())
        self.assertIn(2422, components_of(3172, self.dragon))

    def test_sale_and_combine_accounting(self):
        self.assertEqual(net_spent([2422], [], [2422], self.dragon), -90)
        self.assertEqual(net_spent([2422], [3020], [2422], self.dragon), 800)
        self.assertEqual(net_spent([2422], [1037], [2422], self.dragon), 785)
        self.assertEqual(buy_cost(3020, [2422], self.dragon), 800)

    def test_free_grant_is_not_a_purchase_or_gold_credit(self):
        metadata = grant_economics()
        self.assertEqual(metadata["purchase_cost"], 0)
        self.assertEqual(metadata["gold_delta"], 0)
        self.assertFalse(metadata["purchase_label"])
        self.assertIsNone(try_purchase(2422, [], 10000, self.dragon))
        with self.assertRaisesRegex(ValueError, "system grant"):
            net_spent([], [2422], [], self.dragon)

    def test_pregrant_state_and_normal_boots(self):
        for state in (False, None):
            self.assertTrue(is_blocked(1001, [], self.dragon,
                                      magical_footwear_selected=True, footwear_granted=state))
            self.assertTrue(is_blocked(3020, [], self.dragon,
                                      magical_footwear_selected=True, footwear_granted=state))
        self.assertFalse(is_blocked(3020, [2422], self.dragon,
                                   magical_footwear_selected=True, footwear_granted=True))
        self.assertFalse(is_blocked(1036, [], self.dragon,
                                   magical_footwear_selected=True, footwear_granted=False))
        self.assertEqual(try_purchase(1001, [], 300, self.dragon), (300, Counter({1001: 1})))

    def test_boot_line_legality_and_duplicates_are_not_truncated(self):
        self.assertFalse(is_blocked(3020, [2422], self.dragon))
        self.assertTrue(is_blocked(1001, [2422], self.dragon))
        self.assertTrue(is_blocked(3111, [3020], self.dragon))
        inv = [2422, 2422]
        self.assertEqual(context_inventory(inv, self.dragon), inv)
        self.assertEqual(inventory_shape(inv, self.dragon)["boot_copies"], 2)
        self.assertIsNone(try_purchase(3020, inv, 10000, self.dragon))
        self.assertEqual(inv, [2422, 2422])

    def test_slot_capacity_and_protected_components(self):
        full = [2422] + [1036] * 5
        self.assertEqual(inventory_shape(full, self.dragon)["regular_slots"], 6)
        self.assertIsNone(try_purchase(1029, full, 10000, self.dragon))
        self.assertEqual(try_purchase(3020, full, 800, self.dragon), (800, Counter([3020] + [1036] * 5)))
        self.assertIsNone(try_purchase(3020, [2422], 799, self.dragon))
        self.assertIsNone(try_purchase(3020, [2422], 10000, self.dragon, protected=Counter({2422: 1})))
        self.assertEqual(inventory_shape(full, self.dragon, role="BOTTOM", bot_quest_complete=True)["regular_slots"], 5)

    def test_ordinary_and_nonboot_behavior_matches_original(self):
        for iid, inventory, budget in ((3020, [1001], 800), (3006, [1001, 1042], 550),
                                       (3111, [1001, 1033], 550), (3133, [1036, 1036], 100),
                                       (1036, [], 350), (2055, [2055], 75)):
            with self.subTest(item=iid, inventory=inventory):
                old = Counter(inventory)
                new = old.copy()
                self.assertEqual(combine_cost(iid, new, self.dragon, consume=True),
                                 original_combine_cost(iid, old, self.source, consume=True))
                self.assertEqual(new, old)
                self.assertEqual(try_purchase(iid, inventory, budget, self.dragon),
                                 original_try_purchase(iid, inventory, budget, self.source))

    def test_adapter_rejects_unreviewed_patch(self):
        other = pinned_dragon()
        other.version = "16.18.1"
        with self.assertRaisesRegex(ValueError, "16.17.1"):
            FootwearDataDragon(other)

    def test_direct_combine_features_include_footwear(self):
        owned = Counter({2422: 1, 1042: 2})
        self.assertTrue(has_direct_components(3006, owned, self.dragon))
        self.assertEqual(owned, Counter({2422: 1, 1042: 2}))
        self.assertFalse(has_direct_components(3006, Counter({2422: 1, 1042: 1}), self.dragon))
        # A partially owned T2 recipe can discount a recursive T3 purchase but
        # must not be counted as holding the direct T2 component itself.
        self.assertFalse(has_direct_components(3172, owned, self.dragon))
        self.assertTrue(has_direct_components(3020, Counter({2422: 1}), self.dragon))
        state = inventory_state([2422, 1042, 1042], 300, self.dragon)
        # Real pinned alternatives: Greaves combine for 300 and Scout's
        # Slingshot combines the two daggers for 100.
        self.assertEqual(state, {"can_complete": 1, "n_completable": 2,
                                 "cheapest_complete": 100, "gold_after_complete": 200,
                                 "n_inventory": 3})
        self.assertEqual(inventory_state([2422, 1042], 300, self.dragon)["can_complete"], 0)
        self.assertEqual(decision_kind([3006], [2422, 1042, 1042], self.dragon), "complete")
        self.assertEqual(decision_kind([3006], [2422, 1042], self.dragon), "component")
        self.assertEqual(decision_kind([3020], [2422], self.dragon), "complete")
        self.assertEqual(decision_kind([2422, 1036], [], self.dragon), "start")
        self.assertEqual(decision_kind([2422], [], self.dragon), "save")

    def test_feature_parity_for_ordinary_inventory(self):
        for bought, inventory, gold in (([3006], [1001, 1042, 1042], 300),
                                        ([3006], [1001, 1042], 550),
                                        ([3020], [1001], 800),
                                        ([3133], [1036, 1036], 100),
                                        ([1036], [], 350), ([2055], [2055], 75)):
            with self.subTest(bought=bought, inventory=inventory):
                self.assertEqual(decision_kind(bought, inventory, self.dragon),
                                 original_decision_kind(bought, inventory, self.source))
                self.assertEqual(inventory_state(inventory, gold, self.dragon),
                                 original_inventory_state(inventory, gold, self.source))


if __name__ == "__main__":
    unittest.main(verbosity=2)
