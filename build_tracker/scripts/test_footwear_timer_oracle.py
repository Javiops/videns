"""Mutation and edge-case checks for independent timer evidence QA."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from test_footwear_economics import pinned_dragon
from validate_footwear_timer_results import (
    grant_boundary_reason, grant_rune_qa, source_footwear_qa, source_recipe_price,
    validate_timer_result,
)


def match(deadline=720):
    minutes, seconds = divmod(deadline, 60)
    return {"info": {"participants": [
        {"participantId": 1, "teamId": 100, "perks": {"styles": [{"selections": [
            {"perk": 8304, "var1": minutes, "var2": seconds // 10, "var3": seconds % 10}]}]}},
        {"participantId": 2, "teamId": 200, "perks": {"styles": []}},
        {"participantId": 3, "teamId": 100, "perks": {"styles": []}},
    ]}}


def kill(stamp, killer=1, assisters=None, victim=2):
    return {"type": "CHAMPION_KILL", "timestamp": stamp, "killerId": killer,
            "assistingParticipantIds": assisters or [], "victimId": victim}


def item(stamp, kind, iid=0, **kwargs):
    return {"type": "ITEM_" + kind, "timestamp": stamp, "participantId": 1,
            "itemId": iid, **kwargs}


def timeline(*events):
    return {"info": {"frames": [{"events": list(events)}]}}


def grant(stamp, owner=1):
    return {"timestamp": stamp, "participant_id": owner,
            "gold_delta": 0, "purchase_cost": 0, "purchase_label": False}


class TimerOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dragon = pinned_dragon()

    def test_late_kill_cannot_backdate_grant(self):
        raw = timeline(kill(405906), kill(591730), kill(596039))
        self.assertTrue(grant_rune_qa(match(585), raw, [grant(596039)])["passed"])
        failed = grant_rune_qa(match(585), raw, [grant(585000)])
        self.assertFalse(failed["passed"])
        self.assertIn("grant_precedes_causal_eligibility", {e["code"] for e in failed["issues"]})

    def test_full_slot_wait_counts_later_takedowns(self):
        # Ezreal counterexample: threshold585000, full until630299, two
        # additional takedowns before delivery move the displayed deadline495s.
        raw = timeline(*(kill(ts) for ts in (100000, 200000, 300000, 625656, 627660)))
        checked = grant_rune_qa(match(495), raw, [grant(630299)])
        self.assertTrue(checked["passed"])
        self.assertEqual(checked["wearers"][0]["expected_postgame_deadline_seconds"], 495)
        self.assertFalse(grant_rune_qa(match(495), raw, [grant(585000)])["passed"])

    def test_credit_is_unique_and_enemy_only(self):
        first = kill(100000, assisters=[1, 1])
        raw = timeline(first, deepcopy(first), kill(200000, killer=3, assisters=[1], victim=1))
        result = grant_rune_qa(match(675), raw, [grant(675000)])
        self.assertTrue(result["passed"])
        self.assertEqual(result["wearers"][0]["credited_takedowns_through_grant_ms"], [100000])

    def test_future_takedown_does_not_change_grant_qa(self):
        raw = timeline(kill(100000), kill(900000))
        result = grant_rune_qa(match(675), raw, [grant(675000)])
        self.assertTrue(result["passed"])

    def test_duplicate_or_nonrune_grant_rejected(self):
        self.assertFalse(grant_rune_qa(match(), timeline(), [grant(720000), grant(730000)])["passed"])
        self.assertFalse(grant_rune_qa(match(), timeline(), [grant(720000, 2)])["passed"])

    def test_no_grant_requires_zero_counter(self):
        self.assertTrue(grant_rune_qa(match(0), timeline(), [])["passed"])
        self.assertFalse(grant_rune_qa(match(), timeline(), [])["passed"])

    def test_free_grant_cannot_credit_gold_or_make_label(self):
        for field, value in (("gold_delta", 300), ("purchase_cost", 300), ("purchase_label", True)):
            changed = grant(720000) | {field: value}
            self.assertFalse(grant_rune_qa(match(), timeline(), [changed])["passed"])

    def test_real_recipe_component_identity(self):
        self.assertEqual(source_recipe_price(3020, [2422], self.dragon), (800, [2422]))
        self.assertEqual(source_recipe_price(3006, [2422, 1042, 1042], self.dragon),
                         (300, [1042, 1042, 2422]))
        with self.assertRaisesRegex(ValueError, "unexplained_consumed_components"):
            source_recipe_price(3020, [2422, 1036], self.dragon)

    def test_observed_undo_restores_physical_boots_and_checks_refund(self):
        # Raw documented upgrade/undo/re-upgrade pattern.
        raw = timeline(item(1051933, "DESTROYED", 2422), item(1051933, "PURCHASED", 3047),
                       item(1052802, "UNDO", beforeId=3047, afterId=0, goldGain=900),
                       item(1053302, "DESTROYED", 2422), item(1053302, "PURCHASED", 3111))
        checked = source_footwear_qa(match(), raw, [grant(720000)], self.dragon)
        self.assertTrue(checked["passed"], checked["issues"])
        bad = deepcopy(raw)
        bad["info"]["frames"][0]["events"][2]["goldGain"] = 1200
        result = source_footwear_qa(match(), bad, [grant(720000)], self.dragon)
        self.assertIn("source_boot_undo_refund_disagrees_with_recipe", {e["code"] for e in result["issues"]})
        no_undo = timeline(*[e for e in raw["info"]["frames"][0]["events"] if e["type"] != "ITEM_UNDO"])
        result = source_footwear_qa(match(), no_undo, [grant(720000)], self.dragon)
        self.assertIn("raw_2422_removal_without_owned_component", {e["code"] for e in result["issues"]})

    def test_sale_undo_restores_90_gold_identity(self):
        raw = timeline(item(800000, "SOLD", 2422),
                       item(801000, "UNDO", beforeId=0, afterId=2422, goldGain=-90),
                       item(900000, "SOLD", 2422))
        self.assertTrue(source_footwear_qa(match(), raw, [grant(720000)], self.dragon)["passed"])
        changed = deepcopy(raw)
        changed["info"]["frames"][0]["events"][1]["goldGain"] = -210
        self.assertFalse(source_footwear_qa(match(), changed, [grant(720000)], self.dragon)["passed"])

    def test_source_removal_requires_strict_prior_grant(self):
        raw = timeline(item(800000, "SOLD", 2422))
        for grants in ([], [grant(800000)], [grant(800001)]):
            self.assertFalse(source_footwear_qa(match(), raw, grants, self.dragon)["passed"])

    def test_bot_role_transfer_retains_free_component(self):
        raw = timeline(item(800000, "DESTROYED", 1202), item(800000, "DESTROYED", 2422),
                       item(900000, "DESTROYED", 2422), item(900000, "PURCHASED", 3020))
        self.assertTrue(source_footwear_qa(match(), raw, [grant(720000)], self.dragon)["passed"])

    def test_mundo_boundary_and_other_shopper_and_death_anchor(self):
        inferred = [grant(630000)]
        for ts in (630000, 630058, 630125, 630250):
            event = {"participant_id": 2, "ts": ts, "ts_end": ts}
            self.assertIsNotNone(grant_boundary_reason(event, inferred, guard_ms=250))
        event = {"participant_id": 2, "ts": 645125, "ts_end": 645125, "is_save": True}
        self.assertIsNotNone(grant_boundary_reason(event, inferred, guard_ms=250))
        event = {"participant_id": 2, "ts": 630251, "ts_end": 630251}
        self.assertIsNone(grant_boundary_reason(event, inferred, guard_ms=250))

    def test_owner_visit_crossing_boundary(self):
        event = {"participant_id": 1, "ts": 719000, "ts_end": 722000}
        self.assertEqual(grant_boundary_reason(event, [grant(720000)], guard_ms=0),
                         "own_visit_near_or_spans_inferred_grant")

    def test_source_ledger_ignores_outcome_counters(self):
        raw = timeline(item(800000, "SOLD", 2422))
        before = source_footwear_qa(match(), raw, [grant(720000)], self.dragon)
        after = source_footwear_qa(match(0), raw, [grant(720000)], self.dragon)
        self.assertEqual(before, after)

    def test_candidate_prices_compared_to_independent_source_recipe(self):
        raw = timeline(item(800000, "DESTROYED", 2422), item(800000, "PURCHASED", 3020))
        debug = {"footwear_grants": [grant(720000)],
                 "footwear_batches": [{"participant_id": 1, "ts": 800000,
                     "before": [2422], "batch": raw["info"]["frames"][0]["events"]}],
                 "gold_events": {1: [(800000, 800, "spend")]}}
        self.assertTrue(validate_timer_result(match(), raw, [], debug, self.dragon)["passed"])
        # Correct endpoint and raw rune summary do not excuse a missing300g
        # discount in the reconstructor's actual causal gold ledger.
        changed = deepcopy(debug)
        changed["gold_events"][1] = [(800000, 1100, "spend")]
        result = validate_timer_result(match(), raw, [], changed, self.dragon)
        self.assertIn("candidate_free_component_transaction_cost_disagrees",
                      {e["code"] for e in result["issues"]})

    def test_missing_candidate_before_component_or_trace_rejected(self):
        raw = timeline(item(800000, "SOLD", 2422))
        debug = {"footwear_grants": [grant(720000)], "footwear_batches": [],
                 "gold_events": {1: [(800000, -90, "spend")]}}
        self.assertFalse(validate_timer_result(match(), raw, [], debug, self.dragon)["passed"])
        debug["footwear_batches"] = [{"participant_id": 1, "ts": 800000,
                                     "before": [], "batch": raw["info"]["frames"][0]["events"]}]
        result = validate_timer_result(match(), raw, [], debug, self.dragon)
        self.assertIn("candidate_did_not_hold_physical_2422_at_source_removal",
                      {e["code"] for e in result["issues"]})

    def test_mid_quest_free_upgrade_self_destruction_is_not_a_recipe_component(self):
        batch = [item(830240, "DESTROYED", 2422), item(830240, "DESTROYED", 3020),
                 item(830240, "PURCHASED", 3020)]
        raw = timeline(item(704409, "DESTROYED", 1201), *batch)
        debug = {"footwear_grants": [grant(675000)], "footwear_batches": [
            {"participant_id": 1, "ts": 830240, "before": [2422], "batch": batch}],
            "gold_events": {1: [(830240, 800, "spend")]}}
        # One earlier takedown makes the source rune/grant consistent.
        raw["info"]["frames"][0]["events"].insert(0, kill(100000))
        result = validate_timer_result(match(675), raw, [], debug, self.dragon)
        self.assertTrue(result["passed"], result)
        missing_quest = timeline(kill(100000), *batch)
        self.assertFalse(validate_timer_result(match(675), missing_quest, [], debug, self.dragon)["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
