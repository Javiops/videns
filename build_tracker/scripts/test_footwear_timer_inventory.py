"""Pinned-data regressions for isolated purchase-only elixir handling."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from app.config import CACHE_DIR
from app.ddragon import DataDragon
from app.inventory import InventoryReplay
from footwear_economics import FootwearDataDragon, combine_cost
from footwear_timer_inventory import FootwearTimerInventoryReplay


def event(kind, item=0, ts=1000000, **extra):
    return {"type": kind, "itemId": item, "timestamp": ts, **extra}


class ElixirInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for name in ("item", "champion"):
            if not (CACHE_DIR / f"{name}-16.17.1.json").is_file():
                raise AssertionError("Pinned local static data required; no test downloads")
        cls.dragon = FootwearDataDragon(DataDragon("16.17.1"))

    def make(self, inventory, role="TOP"):
        self.inventory = {1: list(inventory)}
        self.original = self.inventory[1]
        self.replay = FootwearTimerInventoryReplay(self.inventory, self.dragon, role_for={1: role})
        return self.replay

    def apply(self, events, restores=None):
        return self.replay.apply_batch(1, 1000000, events, restores)

    def test_reviewed_pinned_flags_for_all_three_elixirs(self):
        for item in (2138, 2139, 2140):
            with self.subTest(item=item):
                raw = self.dragon.item(item)
                self.assertIs(raw["consumed"], True)
                self.assertIs(raw["consumeOnFull"], True)
                self.assertTrue(self.dragon.classify(item)["target_skip"])

    def test_six_regular_slots_auto_consume_each_elixir(self):
        for item in (2138, 2139, 2140):
            with self.subTest(item=item):
                self.make([1036] * 6)
                bought = self.apply([event("ITEM_PURCHASED", item)])
                self.assertEqual(self.inventory[1], [1036] * 6)
                self.assertEqual(bought, Counter())
                self.assertIs(self.inventory[1], self.original)
                record = self.replay.auto_consumptions[0]
                self.assertEqual(record["item_id"], item)
                self.assertEqual(record["slot_state_before"]["regular_slots"], 6)
                self.assertEqual(record["slot_state_if_retained"]["regular_slots"], 7)
                self.assertTrue(record["raw_purchase_remains_chargeable"])
                self.assertEqual(record["gold_refund"], 0)
                self.assertFalse(record["purchase_label"])
                self.assertEqual(combine_cost(item, Counter([1036] * 6), self.dragon), 500)

    def test_five_slots_retains_elixir(self):
        self.make([1036] * 5)
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertEqual(self.inventory[1], [1036] * 5 + [2140])
        self.assertFalse(self.replay.auto_consumptions)

    def test_unknown_static_item_fails_closed_without_erasing(self):
        self.make([1036] * 6 + [999998])
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertIn(2140, self.inventory[1])
        self.assertFalse(self.replay.auto_consumptions)

    def test_existing_overflow_is_not_hidden(self):
        self.make([1036] * 7)
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertEqual(len(self.inventory[1]), 8)
        self.assertFalse(self.replay.auto_consumptions)

    def test_missing_role_disables_shortcut(self):
        self.make([1036] * 6, role="")
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertIn(2140, self.inventory[1])
        self.assertFalse(self.replay.auto_consumptions)

    def test_ordinary_consumable_does_not_use_elixir_rule(self):
        self.make([1036] * 6)
        self.apply([event("ITEM_PURCHASED", 2003)])
        self.assertIn(2003, self.inventory[1])
        self.assertFalse(self.replay.auto_consumptions)

    def test_undo_restoration_uses_base_extend_without_consumption(self):
        self.make([1036] * 6)
        undo = event("ITEM_UNDO", beforeId=0, afterId=2140)
        self.apply([undo], {id(undo): [2140]})
        self.assertEqual(self.inventory[1], [1036] * 6 + [2140])
        self.assertFalse(self.replay.auto_consumptions)

    def test_sale_before_purchase_creates_room_in_same_batch(self):
        self.make([1036] * 6)
        self.apply([event("ITEM_SOLD", 1036), event("ITEM_PURCHASED", 2140)])
        self.assertEqual(self.inventory[1], [1036] * 5 + [2140])
        self.assertFalse(self.replay.auto_consumptions)

    def test_purchase_before_sale_consumes_at_its_own_event(self):
        self.make([1036] * 6)
        self.apply([event("ITEM_PURCHASED", 2140), event("ITEM_SOLD", 1036)])
        self.assertEqual(self.inventory[1], [1036] * 5)
        self.assertEqual(len(self.replay.auto_consumptions), 1)

    def test_slot_count_uses_physical_potion_stacks(self):
        self.make([1036] * 4 + [2003] * 5)
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertIn(2140, self.inventory[1])
        self.assertFalse(self.replay.auto_consumptions)
        self.make([1036] * 5 + [2003] * 5)
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertNotIn(2140, self.inventory[1])
        self.assertEqual(len(self.replay.auto_consumptions), 1)

    def test_support_ward_slot_does_not_look_like_full_regular_inventory(self):
        self.make([1036] * 5 + [2055, 2055], role="UTILITY")
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertIn(2140, self.inventory[1])
        self.assertFalse(self.replay.auto_consumptions)

    def test_completed_bottom_boot_slot_does_not_look_like_six_regular(self):
        self.make([1036] * 5 + [2422], role="BOTTOM")
        self.replay.bot_complete.add(1)
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertIn(2140, self.inventory[1])
        self.assertFalse(self.replay.auto_consumptions)

    def test_boot_alias_combine_still_consumes_physical_2422_after_bot_quest(self):
        self.make([2422], role="BOTTOM")
        self.replay.bot_complete.add(1)
        self.assertTrue(self.replay._consumes(3006, 2422))
        bought = self.apply([event("ITEM_DESTROYED", 2422), event("ITEM_PURCHASED", 3006)])
        self.assertEqual(self.inventory[1], [3006])
        self.assertEqual(bought, Counter({3006: 1}))
        self.assertFalse(self.replay.auto_consumptions)

    def test_metadata_records_are_copied(self):
        self.make([1036] * 6)
        self.apply([event("ITEM_PURCHASED", 2140)])
        records = self.replay.auto_consumption_records()
        records[0]["gold_refund"] = 500
        self.assertEqual(self.replay.auto_consumptions[0]["gold_refund"], 0)

    def test_replay_cleans_active_context_between_batches(self):
        self.make([1036] * 6)
        self.apply([event("ITEM_PURCHASED", 2140)])
        self.assertIsNone(self.replay._active_raw_event)
        self.assertIsNone(self.replay._active_owner)
        self.assertIsNone(self.replay._active_ts)
        self.inventory[1].append(2140)
        self.assertEqual(len(self.replay.auto_consumptions), 1)
        self.assertIn(2140, self.inventory[1])

    def test_non_elixir_batches_match_base_inventory_quest_and_undo_behavior(self):
        scenarios = [
            ([3865], [event("ITEM_DESTROYED", 3865)]),
            ([1036], [event("ITEM_DESTROYED", 1036), event("ITEM_PURCHASED", 3133)]),
            ([3006], [event("ITEM_DESTROYED", 1202), event("ITEM_DESTROYED", 3006)]),
            ([3006], [event("ITEM_DESTROYED", 1201)]),
            ([1036], [event("ITEM_UNDO", beforeId=1036, afterId=1055)]),
        ]
        for initial, batch in scenarios:
            with self.subTest(initial=initial, batch=batch):
                base_inventory = {1: deepcopy(initial)}
                base = InventoryReplay(base_inventory, self.dragon)
                expected_bought = base.apply_batch(1, 1000000, batch)
                self.make(initial)
                actual_bought = self.apply(batch)
                self.assertEqual(actual_bought, expected_bought)
                self.assertEqual(self.inventory, base_inventory)
                self.assertEqual(self.replay.bot_complete, base.bot_complete)
                self.assertEqual(self.replay.mid_complete, base.mid_complete)
                self.assertEqual(self.replay.pending_boots, base.pending_boots)


if __name__ == "__main__":
    unittest.main(verbosity=2)
