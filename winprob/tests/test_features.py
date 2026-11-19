"""Synthetic property tests only; no archive, database, or sealed split reads."""

import copy
import unittest

from winprob.features import ALL_FEATURES, extract_game_states


def fixture():
    match = {
        "metadata": {"matchId": "EUW1_SYNTHETIC"},
        "info": {
            "gameStartTimestamp": 1700000000000, "gameVersion": "16.19.1.123",
            "participants": [{"participantId": pid, "teamId": 100 if pid <= 5 else 200,
                              "kills": 999, "teamPosition": "JUNGLE"} for pid in range(1, 11)],
            "teams": [{"teamId": 100, "win": True}, {"teamId": 200, "win": False}],
        },
    }
    frames = []
    for minute in range(7):
        participants = {}
        for pid in range(1, 11):
            participants[str(pid)] = {
                "participantId": pid, "totalGold": 500 + 100 * minute + (minute * 20 if pid <= 5 else 0),
                "xp": minute * 50, "level": 1 + minute, "minionsKilled": minute,
                "jungleMinionsKilled": 2 * minute,
                "damageStats": {"totalDamageDoneToChampions": minute * 20, "totalDamageTaken": minute * 10},
                "currentGold": 999999, "championStats": {"health": 999999},
            }
        frames.append({"timestamp": minute * 60000, "participantFrames": participants, "events": []})
    frames[-1]["events"] = [{"type": "GAME_END", "timestamp": 390000, "winningTeam": 100}]
    timeline = {"metadata": {"matchId": "EUW1_SYNTHETIC"},
                "info": {"frameInterval": 60000, "frames": frames}}
    return match, timeline


class FeatureProperties(unittest.TestCase):
    def test_exact_metadata_and_feature_allowlist(self):
        match, timeline = fixture()
        rows = extract_game_states(match, timeline)
        self.assertEqual([r["timestamp_ms"] for r in rows], [180000, 240000, 300000])
        self.assertEqual(rows[0]["game_id"], "EUW1_SYNTHETIC")
        self.assertEqual(rows[0]["patch"], "16.19")
        self.assertEqual(rows[0]["y"], 1)
        self.assertEqual(set(rows[0]["x"]), set(ALL_FEATURES))
        self.assertEqual(rows[0]["x"]["gold_diff"], 300.0)
        self.assertEqual(rows[0]["x"]["gold_diff_slope_per_min"], 100.0)

    def test_future_mutation_cannot_change_prior_state(self):
        match, timeline = fixture()
        original = extract_game_states(match, timeline)[0]
        changed = copy.deepcopy(timeline)
        for frame in changed["info"]["frames"][4:]:
            frame["participantFrames"]["1"]["totalGold"] = 999999
            frame["events"].append({"type": "CHAMPION_KILL", "timestamp": frame["timestamp"],
                                     "killerId": 1, "victimId": 6, "assistingParticipantIds": [2, 3]})
        # Even an event bundled with the current frame cannot count early.
        changed["info"]["frames"][3]["events"].append({"type": "CHAMPION_KILL", "timestamp": 180001,
                                                        "killerId": 1, "victimId": 6})
        self.assertEqual(original, extract_game_states(match, changed)[0])

    def test_terminal_container_and_later_frames_excluded(self):
        match, timeline = fixture()
        timeline["info"]["frames"][4]["events"] = [{"type": "GAME_END", "timestamp": 299999}]
        rows = extract_game_states(match, timeline)
        self.assertEqual([r["timestamp_ms"] for r in rows], [180000])

    def test_objectives_credit_attacker_and_event_timestamps(self):
        match, timeline = fixture()
        timeline["info"]["frames"][3]["events"] = [
            {"type": "BUILDING_KILL", "timestamp": 180000, "teamId": 200, "buildingType": "TOWER_BUILDING"},
            {"type": "BUILDING_KILL", "timestamp": 179999, "teamId": 100, "buildingType": "INHIBITOR_BUILDING"},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 180000, "killerTeamId": 100, "monsterType": "DRAGON"},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 170000, "killerId": 6, "monsterType": "BARON_NASHOR"},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 170000, "killerTeamId": 100, "monsterType": "HORDE"},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 170000, "killerTeamId": 200, "monsterType": "RIFTHERALD"},
            {"type": "ELITE_MONSTER_KILL", "timestamp": 180001, "killerTeamId": 200, "monsterType": "DRAGON"},
        ]
        x = extract_game_states(match, timeline)[0]["x"]
        for feature, expected in {"towers_diff": 1, "inhibitors_diff": -1, "dragons_diff": 1,
                                  "barons_diff": -1, "grubs_diff": 1, "heralds_diff": -1}.items():
            self.assertEqual(x[feature], expected)

    def test_kda_counts_include_execution_death(self):
        match, timeline = fixture()
        timeline["info"]["frames"][3]["events"] = [
            {"type": "CHAMPION_KILL", "timestamp": 179000, "killerId": 1, "victimId": 6,
             "assistingParticipantIds": [2, 3]},
            {"type": "CHAMPION_KILL", "timestamp": 179100, "killerId": 0, "victimId": 1},
        ]
        x = extract_game_states(match, timeline)[0]["x"]
        self.assertEqual(x["kills_diff"], 1)
        self.assertEqual(x["deaths_diff"], 0)
        self.assertEqual(x["assists_diff"], 2)

    def test_postmatch_labels_and_forbidden_fields_never_enter_x(self):
        match, timeline = fixture()
        expected = extract_game_states(match, timeline)[0]["x"]
        match["info"]["teams"][0]["win"] = False
        match["info"]["teams"][1]["win"] = True
        match["info"]["gameDuration"] = 99999
        match["info"]["gameEndTimestamp"] = 99999
        match["info"]["participants"][0]["kills"] = 1
        for frame in timeline["info"]["frames"]:
            frame["participantFrames"]["1"]["currentGold"] = 0
            frame["participantFrames"]["1"]["championStats"] = {"health": 0}
        row = extract_game_states(match, timeline)[0]
        self.assertEqual(row["x"], expected)
        self.assertEqual(row["y"], 0)

    def test_incomplete_or_missing_measurement_frame_is_skipped(self):
        match, timeline = fixture()
        del timeline["info"]["frames"][3]["participantFrames"]["10"]
        del timeline["info"]["frames"][4]["participantFrames"]["1"]["damageStats"]
        rows = extract_game_states(match, timeline)
        self.assertEqual([r["timestamp_ms"] for r in rows], [300000])
        self.assertEqual(rows[0]["x"]["gold_diff_slope_per_min"], 100.0)

    def test_identity_mismatch_rejected(self):
        match, timeline = fixture()
        timeline["metadata"]["matchId"] = "OTHER_GAME"
        with self.assertRaisesRegex(ValueError, "identity"):
            extract_game_states(match, timeline)

    def test_missing_terminal_marker_rejected(self):
        match, timeline = fixture()
        timeline["info"]["frames"][-1]["events"] = []
        with self.assertRaisesRegex(ValueError, "GAME_END"):
            extract_game_states(match, timeline)

    def test_duplicate_frame_timestamp_rejected(self):
        match, timeline = fixture()
        timeline["info"]["frames"].append(copy.deepcopy(timeline["info"]["frames"][3]))
        with self.assertRaisesRegex(ValueError, "duplicate frame"):
            extract_game_states(match, timeline)

    def test_maximum_time_is_forty_minutes(self):
        match, timeline = fixture()
        terminal = timeline["info"]["frames"].pop()
        for minute in (40, 41):
            frame = copy.deepcopy(timeline["info"]["frames"][-1])
            frame["timestamp"] = minute * 60000
            timeline["info"]["frames"].append(frame)
        terminal["timestamp"] = 42 * 60000
        terminal["events"][0]["timestamp"] = 42 * 60000 + 1
        timeline["info"]["frames"].append(terminal)
        rows = extract_game_states(match, timeline)
        self.assertEqual(rows[-1]["timestamp_ms"], 40 * 60000)

    def test_unavailable_slope_has_an_explicit_indicator(self):
        match, timeline = fixture()
        timeline["info"]["frames"] = timeline["info"]["frames"][3:]
        row = extract_game_states(match, timeline)[0]
        self.assertEqual(row["x"]["gold_diff_slope_per_min"], 0.0)
        self.assertEqual(row["x"]["gold_diff_slope_available"], 0.0)

    def test_nonfinite_measurement_frame_is_skipped(self):
        match, timeline = fixture()
        timeline["info"]["frames"][3]["participantFrames"]["1"]["totalGold"] = float("nan")
        rows = extract_game_states(match, timeline)
        self.assertEqual(rows[0]["timestamp_ms"], 240000)

    def test_side_swap_negates_team_differences(self):
        match, timeline = fixture()
        original = extract_game_states(match, timeline)[0]
        for participant in match["info"]["participants"]:
            participant["teamId"] = 300 - participant["teamId"]
        for team in match["info"]["teams"]:
            team["win"] = not team["win"]
        swapped = extract_game_states(match, timeline)[0]
        self.assertEqual(swapped["y"], 1 - original["y"])
        for key in ALL_FEATURES:
            expected = original["x"][key] if key in ("time_min", "gold_diff_slope_available") else -original["x"][key]
            self.assertEqual(swapped["x"][key], expected)


if __name__ == "__main__":
    unittest.main()
