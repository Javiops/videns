"""Synthetic rich-feature contracts. Never read archive or split payloads."""

import copy
import math
import unittest

from winprob.features import ALL_FEATURES, extract_game_states
from winprob.features_rich import (ALL_RICH_FEATURES, HISTORY_RICH_FEATURES, INVARIANT_RICH_FEATURES,
                                   MAP_EXTENT, SIGNED_RICH_FEATURES, STATE_RICH_FEATURES,
                                   extract_rich_game_states)


def fixture():
    match = {"metadata": {"matchId": "EUW1_SYNTHETIC_RICH"}, "info": {
        "gameStartTimestamp": 1700000000000, "gameVersion": "16.19.1",
        "participants": [{"participantId": pid, "teamId": 100 if pid <= 5 else 200} for pid in range(1, 11)],
        "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
    }}
    frames = []
    for minute in range(8):
        raw = {}
        for pid in range(1, 11):
            local = (pid - 1) % 5 + 1
            x, y = 1200 + local * 500 + minute * 30, 1500 + local * 200
            if pid > 5:
                x, y = MAP_EXTENT - x, MAP_EXTENT - y
            raw[str(pid)] = {"participantId": pid, "totalGold": 4000 + minute * 100 + local * 200,
                             "xp": minute * 500 + local * 50, "level": minute + 1,
                             "minionsKilled": minute * local, "jungleMinionsKilled": minute,
                             "damageStats": {"totalDamageDoneToChampions": minute * 30,
                                             "totalDamageTaken": minute * 20},
                             "position": {"x": x, "y": y}}
        frames.append({"timestamp": minute * 60000, "participantFrames": raw, "events": []})
    frames[-1]["events"] = [{"type": "GAME_END", "timestamp": 450000}]
    return match, {"metadata": {"matchId": "EUW1_SYNTHETIC_RICH"},
                   "info": {"frameInterval": 60000, "frames": frames}}


class RichFeatureProperties(unittest.TestCase):
    def test_base_rows_unchanged_and_allowlist_finite(self):
        match, timeline = fixture()
        base = extract_game_states(match, timeline)
        rich = extract_rich_game_states(match, timeline)
        self.assertEqual(len(base), len(rich))
        self.assertEqual(set(ALL_RICH_FEATURES), set(STATE_RICH_FEATURES + HISTORY_RICH_FEATURES))
        self.assertEqual(len(ALL_RICH_FEATURES), len(set(ALL_RICH_FEATURES)))
        self.assertTrue(set(ALL_FEATURES) <= set(ALL_RICH_FEATURES))
        for old, new in zip(base, rich):
            self.assertEqual({key: value for key, value in old.items() if key != "x"},
                             {key: value for key, value in new.items() if key != "x"})
            self.assertEqual(old["x"], {key: new["x"][key] for key in ALL_FEATURES})
            self.assertEqual(set(new["x"]), set(ALL_RICH_FEATURES))
            self.assertTrue(all(math.isfinite(value) for value in new["x"].values()))

    def test_future_positions_and_resources_cannot_change_past(self):
        match, timeline = fixture()
        expected = extract_rich_game_states(match, timeline)[0]
        for frame in timeline["info"]["frames"][4:]:
            frame["participantFrames"]["1"]["position"] = {"x": 12345, "y": 12345}
            frame["participantFrames"]["1"]["totalGold"] = 999999
        self.assertEqual(expected, extract_rich_game_states(match, timeline)[0])

    def test_terminal_containing_frame_is_still_excluded(self):
        match, timeline = fixture()
        timeline["info"]["frames"][4]["events"] = [{"type": "GAME_END", "timestamp": 290000}]
        self.assertEqual([row["timestamp_ms"] for row in extract_rich_game_states(match, timeline)], [180000])

    def test_missing_positions_are_flagged_and_cross_geometry_abstains(self):
        match, timeline = fixture()
        raw = timeline["info"]["frames"][3]["participantFrames"]
        invalid = [None, {"x": 0, "y": 2000}, {"x": 1000, "y": 0},
                   {"x": float("nan"), "y": 2000}, {"x": MAP_EXTENT + 1, "y": 2000}]
        for pid, position in zip(range(1, 6), invalid):
            raw[str(pid)]["position"] = position
        x = extract_rich_game_states(match, timeline)[0]["x"]
        self.assertEqual(x["rich_position_count_diff"], -5)
        self.assertEqual(x["rich_position_count_sum"], 5)
        self.assertEqual(x["rich_position_any_diff"], -1)
        self.assertEqual(x["rich_cross_geometry_available"], 0)
        for metric in ("nearest_enemy_distance_mean",
                       "gold_weighted_enemy_distance_mean", "local_gold_balance_min"):
            self.assertEqual(x[f"rich_{metric}_diff"], 0)
            self.assertEqual(x[f"rich_{metric}_sum"], 0)
        # Recover blue's absent coordinate statistic from diff/sum: it is a
        # flagged numeric placeholder, not an invented position at the corner.
        self.assertEqual((x["rich_pos_x_mean_diff"] + x["rich_pos_x_mean_sum"]) / 2, 0)

    def test_role_champion_health_wallet_and_final_metadata_are_unused(self):
        match, timeline = fixture()
        expected = extract_rich_game_states(match, timeline)[0]["x"]
        for participant in match["info"]["participants"]:
            participant.update({"teamPosition": "JUNGLE", "individualPosition": "TOP", "championId": 999,
                                "kills": 1000, "goldEarned": 999999})
        match["info"].update({"gameDuration": 99999, "gameEndTimestamp": 99999})
        for frame in timeline["info"]["frames"]:
            for player in frame["participantFrames"].values():
                player.update({"currentGold": 999999, "championStats": {"health": 0}, "items": [999999]})
        self.assertEqual(expected, extract_rich_game_states(match, timeline)[0]["x"])

    def test_side_swap_plus_map_rotation_has_declared_parity(self):
        match, timeline = fixture()
        # Break fixture symmetry so signed parity is not a vacuous zero test.
        player = timeline["info"]["frames"][3]["participantFrames"]["1"]
        player["position"] = {"x": 6000, "y": 5000}
        player["totalGold"] += 1000
        expected = extract_rich_game_states(match, timeline)[0]["x"]
        self.assertNotEqual(expected["rich_pos_x_mean_diff"], 0)
        self.assertNotEqual(expected["rich_gold_max_share_diff"], 0)
        for participant in match["info"]["participants"]:
            participant["teamId"] = 300 - participant["teamId"]
        for team in match["info"]["teams"]:
            team["win"] = not team["win"]
        for frame in timeline["info"]["frames"]:
            for player in frame["participantFrames"].values():
                player["position"] = {key: MAP_EXTENT - value for key, value in player["position"].items()}
        rotated = extract_rich_game_states(match, timeline)[0]["x"]
        for name in SIGNED_RICH_FEATURES:
            self.assertAlmostEqual(rotated[name], -expected[name], places=9, msg=name)
        for name in INVARIANT_RICH_FEATURES:
            self.assertAlmostEqual(rotated[name], expected[name], places=9, msg=name)

    def test_gold_position_coupling_distinguishes_equal_aggregate_states(self):
        match, timeline = fixture()
        other = copy.deepcopy(timeline)
        left = other["info"]["frames"][3]["participantFrames"]["1"]
        right = other["info"]["frames"][3]["participantFrames"]["5"]
        left["totalGold"], right["totalGold"] = right["totalGold"], left["totalGold"]
        before = extract_rich_game_states(match, timeline)[0]["x"]
        after = extract_rich_game_states(match, other)[0]["x"]
        for name in ALL_FEATURES:
            self.assertEqual(before[name], after[name])
        for name in ("rich_gold_std_diff", "rich_gold_max_share_diff", "rich_gold_top2_share_diff", "rich_pos_x_mean_diff"):
            self.assertEqual(before[name], after[name])
        self.assertNotEqual(before["rich_gold_weighted_enemy_distance_mean_diff"],
                            after["rich_gold_weighted_enemy_distance_mean_diff"])
        self.assertNotEqual(before["rich_gold_weighted_own_base_distance_mean_diff"],
                            after["rich_gold_weighted_own_base_distance_mean_diff"])

    def test_allocation_identifies_concentration_without_changing_team_gold(self):
        match, timeline = fixture()
        other = copy.deepcopy(timeline)
        raw = other["info"]["frames"][3]["participantFrames"]
        raw["1"]["totalGold"] += 2000
        raw["5"]["totalGold"] -= 2000
        before = extract_rich_game_states(match, timeline)[0]["x"]
        after = extract_rich_game_states(match, other)[0]["x"]
        self.assertEqual(before["gold_diff"], after["gold_diff"])
        self.assertNotEqual(before["rich_gold_max_share_diff"], after["rich_gold_max_share_diff"])

    def test_zero_resources_have_explicit_share_availability(self):
        match, timeline = fixture()
        for pid in range(1, 6):
            player = timeline["info"]["frames"][3]["participantFrames"][str(pid)]
            player.update({"totalGold": 0, "xp": 0, "minionsKilled": 0, "jungleMinionsKilled": 0})
        x = extract_rich_game_states(match, timeline)[0]["x"]
        self.assertEqual(x["rich_gold_share_available_diff"], -1)
        self.assertEqual(x["rich_gold_share_available_sum"], 1)
        self.assertTrue(all(math.isfinite(value) for value in x.values()))

    def test_relabeling_within_team_does_not_change_features(self):
        match, timeline = fixture()
        expected = extract_rich_game_states(match, timeline)[0]["x"]
        for frame in timeline["info"]["frames"]:
            players = frame["participantFrames"]
            players["1"], players["5"] = players["5"], players["1"]
            players["1"]["participantId"], players["5"]["participantId"] = 1, 5
        actual = extract_rich_game_states(match, timeline)[0]["x"]
        for name in ALL_RICH_FEATURES:
            self.assertAlmostEqual(expected[name], actual[name], places=9, msg=name)


if __name__ == "__main__":
    unittest.main()
