"""Synthetic causal and physical-capacity checks for the isolated timer model."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from footwear_timer import TimerFootwearScheduler, physical_slot_state


class Dragon:
    def __init__(self):
        self.items = {
            1001: {"tags": ["Boots"]},
            2422: {"tags": ["Boots"]},
            3006: {"tags": ["Boots"], "from": ["1001"]},
            1036: {"tags": ["Damage"]},
            2003: {"tags": ["Consumable"], "stacks": 5},
            2055: {"tags": ["Consumable"], "stacks": 2},
            772043: {"tags": ["Consumable"], "stacks": 2},
            3340: {"tags": ["Trinket"]},
            3865: {"tags": ["GoldPer"]},
            2419: {"tags": []},
        }

    def item(self, item_id):
        return self.items.get(item_id)

    def classify(self, item_id):
        raw = self.item(item_id) or {}
        return {"is_boots": "Boots" in raw.get("tags", []) and bool(raw.get("from")),
                "is_basic_boots": "Boots" in raw.get("tags", []) and not raw.get("from")}


def participant(pid, team, selected=True):
    return {"participantId": pid, "teamId": team,
            "perks": {"styles": [{"selections": [{"perk": 8304 if selected else 8005}]}]}}


def kill(ts, killer=1, victim=6, assists=None):
    return {"type": "CHAMPION_KILL", "timestamp": ts,
            "killerId": killer, "victimId": victim,
            "assistingParticipantIds": assists or []}


class TimerTests(unittest.TestCase):
    def setUp(self):
        self.dragon = Dragon()
        self.participants = [participant(1, 100), participant(2, 100), participant(6, 200, False)]
        self.inventory = {1: [], 2: [], 6: []}
        self.roles = {1: "TOP", 2: "UTILITY", 6: "BOTTOM"}
        self.completed = {1: False, 2: False, 6: False}
        self.callback_records = []
        self.scheduler = self.make_scheduler()

    def make_scheduler(self):
        def grant(pid, ts, metadata):
            self.inventory[pid].append(2422)
            self.callback_records.append((pid, ts, metadata))
        return TimerFootwearScheduler(
            self.participants, self.dragon, self.inventory.__getitem__,
            self.roles.__getitem__, self.completed.__getitem__, grant)

    def batch(self, ts, events=(), apply=None):
        self.scheduler.advance_before(ts)
        if apply:
            apply()
        self.scheduler.observe_batch(ts, events)

    def test_deadline_is_strict_before_raw_timestamp(self):
        self.scheduler.advance_before(720000)
        self.assertFalse(self.scheduler.grants)
        self.scheduler.observe_batch(720000, [])
        self.assertEqual(self.scheduler.state(1)["granted_at"], 720000)
        self.assertEqual(self.scheduler.grants[0]["phase"], "after_raw_batch")

    def test_idle_gap_schedules_original_deadline(self):
        self.batch(700000)
        self.scheduler.advance_before(800000)
        self.assertEqual(self.scheduler.state(1)["granted_at"], 720000)
        self.assertEqual(self.scheduler.grants[0]["phase"], "before_raw_timestamp")
        self.assertEqual(self.scheduler.grants[0]["current_source_timestamp"], 800000)

    def test_early_takedown_moves_future_deadline(self):
        self.batch(100000, [kill(100000)])
        self.scheduler.advance_before(675000)
        self.assertIsNone(self.scheduler.state(1)["granted_at"])
        self.scheduler.advance_before(675001)
        self.assertEqual(self.scheduler.state(1)["granted_at"], 675000)

    def test_late_takedown_never_backdates_grant(self):
        self.batch(700000, [kill(700000)])
        state = self.scheduler.state(1)
        self.assertEqual((state["eligible_at"], state["granted_at"]), (700000, 700000))
        self.assertEqual(state["nominal_deadline"], 675000)

    def test_assist_and_killer_duplicates_count_once(self):
        event = kill(100000, assists=[1, 1, 2, 2])
        self.batch(100000, [event, deepcopy(event)])
        self.batch(100000, [deepcopy(event)])
        self.assertEqual(self.scheduler.state(1)["takedowns"], 1)
        self.assertEqual(self.scheduler.state(2)["takedowns"], 1)

    def test_enemy_only_and_invalid_victim_are_not_takedowns(self):
        self.batch(100000, [kill(100000, victim=2, assists=[2]),
                            kill(100000, victim=0, assists=[2])])
        self.assertEqual(self.scheduler.state(1)["takedowns"], 0)
        self.assertEqual(self.scheduler.state(2)["takedowns"], 0)

    def test_enemy_assist_on_environment_kill_counts(self):
        self.batch(100000, [kill(100000, killer=0, assists=[2])])
        self.assertEqual(self.scheduler.state(2)["takedowns"], 1)

    def test_full_slots_wait_until_after_real_slot_opening(self):
        self.inventory[1] = [1036] * 6
        self.batch(730000)
        self.assertEqual(self.scheduler.state(1)["eligible_at"], 720000)
        self.assertEqual(self.scheduler.state(1)["pending_reason"], "full_slots")
        self.batch(800000, apply=lambda: self.inventory[1].pop())
        grant = next(g for g in self.scheduler.grants if g["participant_id"] == 1)
        self.assertEqual(grant["timestamp"], 800000)
        self.assertEqual(grant["full_slot_wait_ms"], 80000)
        self.assertEqual(grant["phase"], "after_raw_batch")
        self.assertEqual(len(self.inventory[1]), 6)

    def test_takedowns_continue_while_delivery_pending(self):
        self.inventory[1] = [1036] * 6
        self.batch(730000)
        self.batch(750000, [kill(750000)])
        self.batch(790000, [kill(790000, victim=6)])
        self.batch(800000, apply=lambda: self.inventory[1].pop())
        grant = next(g for g in self.scheduler.grants if g["participant_id"] == 1)
        self.assertEqual(grant["eligible_at"], 720000)
        self.assertEqual(grant["nominal_deadline_at_grant"], 630000)
        self.assertEqual(grant["takedowns_before_grant"], 2)

    def test_grant_is_once_even_after_boot_sale(self):
        self.batch(720001)
        self.batch(730000, apply=lambda: self.inventory[1].remove(2422))
        self.batch(740000, [kill(740000)])
        self.assertEqual(len([g for g in self.scheduler.grants if g["participant_id"] == 1]), 1)
        self.assertNotIn(2422, self.inventory[1])

    def test_grant_is_zero_cost_nonpurchase_and_exposes_boundary(self):
        self.batch(720000)
        grant = self.scheduler.grants[0]
        self.assertEqual((grant["gold_delta"], grant["purchase_cost"], grant["purchase_label"]), (0, 0, False))
        self.assertEqual(grant["event_kind"], "system_grant")
        self.assertFalse(grant["delivery_tick_verified"])
        self.assertTrue(any(b["kind"] == "inferred_grant" and b["timestamp"] == 720000
                            for b in self.scheduler.boundaries))

    def test_completed_bottom_role_slot_accepts_seventh_item(self):
        self.roles[1] = "BOTTOM"
        self.completed[1] = True
        self.inventory[1] = [1036] * 6
        self.batch(720001)
        self.assertEqual(self.scheduler.state(1)["granted_at"], 720000)
        self.assertEqual(self.scheduler.grants[0]["slot_state_after"]["regular_slots"], 6)

    def test_unconfirmed_bottom_quest_does_not_add_slot(self):
        self.roles[1] = "BOTTOM"
        self.completed[1] = None
        self.inventory[1] = [1036] * 6
        self.batch(720001)
        self.assertIsNone(self.scheduler.state(1)["granted_at"])

    def test_causal_quest_completion_releases_boot_slot(self):
        self.roles[1] = "BOTTOM"
        self.inventory[1] = [1036] * 6
        self.batch(720001)
        self.batch(750000, apply=lambda: self.completed.__setitem__(1, True))
        self.assertEqual(self.scheduler.state(1)["granted_at"], 750000)

    def test_unknown_item_blocks_grant_and_remains_a_diagnostic(self):
        self.inventory[1] = [9999]
        self.batch(720001)
        self.assertIsNone(self.scheduler.state(1)["granted_at"])
        self.assertTrue(self.scheduler.state(1)["invalid"])
        self.assertEqual(self.scheduler.state(1)["diagnostics"][0]["slot_state"]["unknown_item_ids"], [9999])

    def test_existing_boot_is_not_duplicated(self):
        self.inventory[1] = [3006]
        self.batch(720001)
        self.assertIsNone(self.scheduler.state(1)["granted_at"])
        self.assertEqual(self.scheduler.state(1)["pending_reason"], "boot_owned_before_inferred_grant")

    def test_inventory_overflow_is_not_ordinary_pending(self):
        self.inventory[1] = [1036] * 7
        self.batch(720001)
        self.assertTrue(self.scheduler.state(1)["invalid"])
        self.assertEqual(self.scheduler.state(1)["pending_reason"], "existing_physical_slot_overflow")

    def test_potion_stack_is_one_slot(self):
        self.inventory[1] = [1036] * 4 + [2003] * 5
        self.batch(720001)
        self.assertEqual(self.scheduler.state(1)["granted_at"], 720000)

    def test_unselected_participant_gets_no_grant(self):
        self.batch(900000)
        self.assertEqual(self.scheduler.state(6)["phase"], "not_selected")
        self.assertEqual(self.inventory[6], [])

    def test_missing_fixed_runes_fail_closed(self):
        del self.participants[0]["perks"]
        with self.assertRaisesRegex(ValueError, "Missing fixed rune selections"):
            self.make_scheduler()

    def test_postgame_mutation_does_not_change_grants(self):
        self.batch(700000, [kill(700000)])
        original = deepcopy(self.scheduler.grants)
        self.inventory = {1: [], 2: [], 6: []}
        for player in self.participants:
            player.update(kills=99, deaths=98, assists=97, item0=3006, goldEarned=999999)
            player["perks"]["styles"][0]["selections"][0].update(var1=99999, var2=99999, var3=99999)
        self.scheduler = self.make_scheduler()
        self.batch(700000, [kill(700000)])
        self.assertEqual(self.scheduler.grants, original)

    def test_order_contract_rejects_missing_before_and_backwards_events(self):
        with self.assertRaisesRegex(ValueError, "advance_before"):
            self.scheduler.observe_batch(700000, [])
        self.batch(700000)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.scheduler.advance_before(699999)
        with self.assertRaisesRegex(ValueError, "supplied timestamp"):
            self.scheduler.observe_batch(700000, [kill(699999)])

    def test_callbacks_and_snapshots_cannot_modify_recorded_metadata(self):
        self.batch(720001)
        self.callback_records[0][2]["purchase_label"] = True
        state = self.scheduler.state(1)
        state["eligible_at"] = 0
        self.assertFalse(self.scheduler.grants[0]["purchase_label"])
        self.assertEqual(self.scheduler.state(1)["eligible_at"], 720000)


class PhysicalSlotTests(unittest.TestCase):
    def setUp(self):
        self.dragon = Dragon()

    def test_all_physical_items_count_including_skipped_placeholders(self):
        result = physical_slot_state([1036, 2419, 3865, 3340, 1202, 2003, 2003], self.dragon)
        self.assertEqual(result["regular_slots"], 4)
        self.assertTrue(result["known"])

    def test_support_wards_are_dedicated_but_potions_are_not(self):
        result = physical_slot_state([1036] * 5 + [2055] * 2 + [2003] * 5, self.dragon, role="UTILITY")
        self.assertEqual(result["regular_slots"], 6)
        self.assertEqual(result["role_slot_item_copies"], 2)

    def test_non_support_wards_stack_in_regular_inventory(self):
        result = physical_slot_state([2055, 2055], self.dragon, role="TOP")
        self.assertEqual(result["regular_slots"], 1)

    def test_unknown_static_id_is_explicit_even_if_slot_count_below_six(self):
        result = physical_slot_state([1036, 9999], self.dragon)
        self.assertFalse(result["known"])
        self.assertEqual(result["unknown_item_ids"], [9999])

    def test_bad_stack_size_fails_closed(self):
        for invalid in (-1, 1.5, "2", False):
            with self.subTest(invalid=invalid):
                self.dragon.items[2003]["stacks"] = invalid
                result = physical_slot_state([2003], self.dragon)
                self.assertFalse(result["known"])
                self.assertEqual(result["invalid_stack_item_ids"], [2003])

    def test_zero_stack_size_means_one_slot_per_item(self):
        self.dragon.items[1036]["stacks"] = 0
        result = physical_slot_state([1036, 1036, 1036], self.dragon)
        self.assertTrue(result["known"])
        self.assertEqual(result["regular_slots"], 3)

    def test_pinned_1617_zero_stack_components_are_unstacked(self):
        import json
        from app.config import CACHE_DIR
        from app.ddragon import DataDragon
        path = CACHE_DIR / "item-16.17.1.json"
        self.assertTrue(path.is_file(), "Pinned static item data must already exist locally")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], "16.17.1")
        # Construct only the read-only item accessor context; no network,
        # champion payload or mutable global cache is needed for this check.
        dragon = object.__new__(DataDragon)
        dragon._items = payload
        dragon._classify_cache = {}
        for item_id in (3133, 3134, 3916):
            with self.subTest(item_id=item_id):
                self.assertEqual(dragon.item(item_id)["stacks"], 0)
                result = physical_slot_state([item_id] * 6, dragon)
                self.assertTrue(result["known"])
                self.assertEqual(result["regular_slots"], 6)
                with_boot = physical_slot_state([item_id] * 6 + [2422], dragon)
                self.assertEqual(with_boot["regular_slots"], 7)

    def test_support_ward_stack_overflow_fails_closed(self):
        result = physical_slot_state([2055] * 3, self.dragon, role="UTILITY")
        self.assertFalse(result["known"])

    def test_regular_ward_stack_overflow_fails_closed(self):
        result = physical_slot_state([2055] * 3, self.dragon, role="TOP")
        self.assertFalse(result["known"])

    def test_payload_dictionaries_are_not_silently_miscounted(self):
        with self.assertRaisesRegex(ValueError, "integer item ids"):
            physical_slot_state([{"item_id": 1036}], self.dragon)


if __name__ == "__main__":
    unittest.main(verbosity=2)
