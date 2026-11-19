"""Synthetic source-prefix checks for the isolated footwear-window prototype."""
from copy import deepcopy
import unittest

from magical_footwear import boots_purchase_blocked, classify_decision_windows


class Dragon:
    def item(self, item):
        return {"into": [3006, 3047]} if item == 2422 else {}

    def from_ids(self, item):
        return {3006: [1001, 1042, 1042], 3047: [1001, 1029],
                3172: [3006], 9998: [1001]}.get(item, [])

    def classify(self, item):
        return {"is_boots": item in (3006, 3047, 3172, 9998),
                "is_basic_boots": item in (1001, 2422)}


DRAGON = Dragon()


def player(pid, selected=False):
    return {"participantId": pid, "teamId": 100 if pid < 6 else 200,
            "perks": {"styles": [{"selections": [{"perk": 8304 if selected else 8300, "var1": 0}]}]}}


def fixture(wearers=(1,)):
    return {"info": {"participants": [player(pid, pid in wearers) for pid in range(1, 11)]}}


def timeline(*events):
    return {"info": {"frames": [{"timestamp": 1_000_000, "events": list(events)}]}}


def item_event(kind, ts, item=2422, pid=1, **other):
    return {"type": kind, "timestamp": ts, "participantId": pid, "itemId": item, **other}


def kill(ts, killer=1, victim=6, assists=()):
    return {"type": "CHAMPION_KILL", "timestamp": ts,
            "killerId": killer, "victimId": victim, "assistingParticipantIds": list(assists)}


def query(match, source, *times, pid=1):
    return classify_decision_windows(match, source,
        [{"decision_id": f"{pid}:{ts}", "participant_id": pid, "anchor_ts": ts} for ts in times], DRAGON)


class FootwearWindows(unittest.TestCase):
    def test_deadline_does_not_invent_delivery(self):
        rows = query(fixture(), timeline(), 719_999, 720_000, 720_001, 900_000)
        self.assertEqual([r["footwear_state"]["phase"] for r in rows],
                         ["pre_grant", "pre_grant", "grant_unobserved", "grant_unobserved"])
        self.assertFalse(rows[-1]["footwear_window_known"])
        self.assertTrue(all(not r["footwear_state"]["grant_time_observed"] for r in rows))

    def test_late_kill_never_backdates(self):
        rows = query(fixture(), timeline(kill(700_000)), 690_000, 700_000, 700_001)
        self.assertEqual(rows[0]["footwear_state"]["first_possible_grant_at"], 720_000)
        self.assertEqual(rows[1]["footwear_state"]["phase"], "pre_grant")
        self.assertEqual(rows[2]["footwear_state"]["first_possible_grant_at"], 700_000)

    def test_only_prior_champion_takedowns_count_once(self):
        rows = query(fixture(), timeline(
            kill(100_000, assists=(1, 1)), kill(200_000, killer=2, assists=(1, 1)),
            kill(300_000, killer=1, victim=2), kill(400_000, killer=3),
            kill(800_000, killer=1)), 629_999, 630_001, 900_000)
        self.assertEqual(rows[0]["footwear_state"]["first_possible_grant_at"], 630_000)
        self.assertEqual(rows[1]["footwear_state"]["phase"], "grant_unobserved")
        self.assertEqual(rows[2]["footwear_state"]["first_possible_grant_at"], 630_000)

    def test_sale_resolves_only_strictly_after_evidence(self):
        rows = query(fixture(), timeline(item_event("ITEM_SOLD", 800_000)), 800_000, 800_001)
        self.assertEqual(rows[0]["footwear_state"]["phase"], "grant_unobserved")
        self.assertEqual(rows[1]["footwear_state"]["phase"], "resolved")
        self.assertTrue(rows[1]["footwear_window_known"])
        self.assertEqual(rows[1]["footwear_state"]["resolved_by"], "observed_2422_sale")

    def test_same_timestamp_kill_and_resolution_are_both_excluded_from_input(self):
        source = timeline(item_event("ITEM_SOLD", 700_000), kill(700_000))
        rows = query(fixture(), source, 700_000, 700_001)
        self.assertEqual(rows[0]["footwear_state"]["phase"], "pre_grant")
        self.assertIsNone(rows[0]["footwear_state"]["resolved_at"])
        self.assertEqual(rows[1]["footwear_state"]["phase"], "resolved")
        self.assertEqual(rows[1]["footwear_state"]["first_possible_grant_at"], 700_000)

    def test_upgrade_group_survives_interleaved_other_players(self):
        source = timeline(item_event("ITEM_DESTROYED", 800_000),
                          item_event("ITEM_PURCHASED", 800_000, 1055, pid=2),
                          item_event("ITEM_PURCHASED", 800_000, 3006))
        row = query(fixture(), source, 800_001)[0]
        self.assertEqual(row["footwear_state"]["resolved_by"], "observed_2422_upgrade")

    def test_mid_upgrade_descendant_is_compatible(self):
        source = timeline(item_event("ITEM_DESTROYED", 800_000),
                          item_event("ITEM_DESTROYED", 800_000, 1201),
                          item_event("ITEM_PURCHASED", 800_000, 3172))
        self.assertEqual(query(fixture(), source, 800_001)[0]["footwear_state"]["phase"], "resolved")

    def test_role_quest_marker_alone_is_not_grant_evidence(self):
        source = timeline(item_event("ITEM_DESTROYED", 800_000, 1202),
                          item_event("ITEM_DESTROYED", 800_000, 1201))
        self.assertEqual(query(fixture(), source, 800_001)[0]["footwear_state"]["phase"], "grant_unobserved")

    def test_explicit_receipt_requires_separate_inventory_integration(self):
        row = query(fixture(), timeline(item_event("ITEM_PURCHASED", 800_000)), 800_001)[0]
        self.assertEqual(row["footwear_state"]["phase"], "grant_unobserved")
        self.assertIn("2422_receipt_requires_inventory_replay", [d["code"] for d in row["footwear_diagnostics"]])

    def test_standalone_or_incompatible_destroy_does_not_resolve(self):
        for source in (timeline(item_event("ITEM_DESTROYED", 800_000)),
                       timeline(item_event("ITEM_DESTROYED", 800_000),
                                item_event("ITEM_PURCHASED", 800_000, 9998)),
                       timeline(item_event("ITEM_DESTROYED", 800_000),
                                item_event("ITEM_PURCHASED", 800_001, 3006)),
                       timeline(item_event("ITEM_PURCHASED", 800_000, 3006))):
            self.assertEqual(query(fixture(), source, 810_000)[0]["footwear_state"]["phase"], "grant_unobserved")

    def test_early_boot_evidence_invalidates_later_windows(self):
        for kind, iid in (("ITEM_SOLD", 2422), ("ITEM_DESTROYED", 2422),
                          ("ITEM_PURCHASED", 3006), ("ITEM_PURCHASED", 1001)):
            rows = query(fixture(), timeline(item_event(kind, 500_000, iid)), 500_000, 500_001, 900_000)
            self.assertEqual(rows[0]["footwear_state"]["phase"], "pre_grant")
            self.assertEqual(rows[1]["footwear_state"]["phase"], "invalid")
            self.assertFalse(rows[2]["footwear_window_known"])

    def test_later_opaque_undo_invalidates_resolved_window(self):
        rows = query(fixture(), timeline(item_event("ITEM_SOLD", 800_000),
                         item_event("ITEM_UNDO", 810_000, 0, beforeId=0, afterId=0)), 810_000, 810_001)
        self.assertEqual(rows[0]["footwear_state"]["phase"], "resolved")
        self.assertEqual(rows[1]["footwear_state"]["phase"], "invalid")

    def test_known_sale_undo_does_not_erase_grant_history(self):
        row = query(fixture(), timeline(item_event("ITEM_SOLD", 800_000),
                        item_event("ITEM_UNDO", 810_000, 0, beforeId=0, afterId=2422, goldGain=-90)), 820_000)[0]
        self.assertEqual(row["footwear_state"]["phase"], "resolved")
        self.assertFalse(row["footwear_state"]["boot_purchases_blocked"])

    def test_every_wearer_controls_other_players_board(self):
        match = fixture((1, 7))
        source = timeline(item_event("ITEM_SOLD", 800_000, pid=1), item_event("ITEM_SOLD", 850_000, pid=7))
        rows = query(match, source, 700_000, 810_000, 850_000, 850_001, pid=3)
        self.assertEqual([r["footwear_blocking_participant_ids"] for r in rows], [[], [7], [7], []])
        self.assertEqual(rows[1]["footwear_state"]["phase"], "not_selected")

    def test_no_footwear_rune_has_no_footwear_restriction(self):
        row = query(fixture(()), timeline(), 900_000)[0]
        self.assertTrue(row["footwear_window_known"])
        self.assertEqual(row["footwear_board_states"], [])
        self.assertFalse(boots_purchase_blocked(row["footwear_state"], 3006, DRAGON))

    def test_boot_guard_blocks_pregrant_and_unknown_only(self):
        rows = query(fixture(), timeline(item_event("ITEM_SOLD", 800_000)), 500_000, 750_000, 800_001)
        self.assertEqual([boots_purchase_blocked(r["footwear_state"], 3006, DRAGON) for r in rows], [True, True, False])
        self.assertTrue(all(not boots_purchase_blocked(r["footwear_state"], 1055, DRAGON) for r in rows))

    def test_final_dto_outcomes_and_future_events_cannot_change_prefix(self):
        match = fixture()
        original = timeline(kill(100_000), item_event("ITEM_SOLD", 800_000))
        changed_match, changed_source = deepcopy(match), deepcopy(original)
        changed_match["info"].update(gameDuration=99999, gameCreation=99)
        for participant in changed_match["info"]["participants"]:
            participant.update(kills=999, assists=999, deaths=999, item0=2422, roleBoundItem=3006)
            participant["perks"]["styles"][0]["selections"][0].update(var1=888, var2=999, var3=777)
        changed_source["info"]["frames"][0]["timestamp"] = 7_777_777
        changed_source["info"]["frames"][0]["events"][-1]["timestamp"] = 999_000
        changed_source["info"]["frames"][0]["events"].append(kill(900_000))
        self.assertEqual(query(match, original, 675_001), query(changed_match, changed_source, 675_001))

    def test_prefix_truncation_and_input_order(self):
        match = fixture()
        source = timeline(kill(100_000), item_event("ITEM_SOLD", 800_000))
        decisions = [{"participant_id": 1, "ts": 800_001}, {"participant_id": 1, "ts": 500_000},
                     {"participant_id": 1, "ts": 800_000}]
        original_match, original_source, original_decisions = deepcopy(match), deepcopy(source), deepcopy(decisions)
        rows = classify_decision_windows(match, source, decisions, DRAGON)
        for row in rows:
            prefix = timeline(*(e for e in source["info"]["frames"][0]["events"] if e["timestamp"] < row["ts"]))
            expected = classify_decision_windows(match, prefix, [{"participant_id": 1, "ts": row["ts"]}], DRAGON)[0]
            self.assertEqual(row, expected)
        self.assertEqual(match, original_match)
        self.assertEqual(source, original_source)
        self.assertEqual(decisions, original_decisions)

    def test_missing_fixed_runes_fail_closed(self):
        match = fixture()
        del match["info"]["participants"][0]["perks"]
        with self.assertRaises(ValueError):
            query(match, timeline(), 900_000)
        match = fixture()
        match["info"]["participants"][0]["perks"]["styles"][0]["selections"] = []
        with self.assertRaises(ValueError):
            query(match, timeline(), 900_000)


if __name__ == "__main__":
    unittest.main()
