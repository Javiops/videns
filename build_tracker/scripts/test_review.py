"""Synthetic review checks; no corpus or frozen test-set reads."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import review
from app.profile import ProfileError


class FakeDragon:
    def __init__(self, version):
        self.version = version
        self._items = {"data": {}}

    def champion_name(self, champion_id):
        return f"Champion {champion_id}"

    def champion_id_name(self, champion_id):
        return f"Champion{champion_id}"

    def item_name(self, item_id):
        return f"Item {item_id}"

    def item(self, item_id):
        return {"stacks": 2 if item_id == 2055 else 1, "tags": []}

    def free_boot_upgrades(self):
        return {}

    def classify(self, item_id):
        return {"is_boots": False, "is_basic_boots": item_id == 1001,
                "target_skip": False}

    def from_ids(self, item_id):
        return []

    def gold_block(self, item_id):
        return {"base": 300, "total": 300}


def fixture():
    participants = [
        {"participantId": pid, "puuid": f"p{pid}", "championId": pid,
         "teamId": 100 if pid <= 5 else 200,
         "teamPosition": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][(pid - 1) % 5],
         "riotIdGameName": f"Player{pid}", "riotIdTagline": "TEST", "win": pid <= 5}
        for pid in range(1, 11)
    ]
    frame_state = {
        str(pid): {"position": {"x": pid * 1000, "y": pid * 1000},
                   "championStats": {"health": 0 if pid == 6 else 100, "healthMax": 100},
                   "level": 2, "minionsKilled": pid, "jungleMinionsKilled": 0}
        for pid in range(1, 11)
    }
    match = {"metadata": {"matchId": "EUW1_1234567890"}, "info": {
        "mapId": 11, "participants": participants, "gameVersion": "16.18.999.1",
        "gameDuration": 120, "queueId": 420}}
    timeline = {"metadata": {"matchId": "EUW1_1234567890"}, "info": {"frames": [
        {"timestamp": 0, "participantFrames": frame_state, "events": []},
        {"timestamp": 60000, "participantFrames": frame_state, "events": [
            {"type": "ITEM_PURCHASED", "timestamp": 54000, "participantId": 1, "itemId": 2055},
            {"type": "ITEM_PURCHASED", "timestamp": 55000, "participantId": 1, "itemId": 1001},
            {"type": "WARD_PLACED", "timestamp": 55500, "creatorId": 1, "wardType": "CONTROL_WARD"},
            {"type": "ITEM_DESTROYED", "timestamp": 56000, "participantId": 5, "itemId": 3865},
            {"type": "CHAMPION_KILL", "timestamp": 57000, "killerId": 1, "victimId": 6,
             "assistingParticipantIds": [2], "position": {"x": 7000, "y": 8000}},
            {"type": "BUILDING_KILL", "timestamp": 58000, "buildingType": "TOWER_BUILDING",
             "teamId": 200, "laneType": "MID_LANE", "towerType": "OUTER_TURRET",
             "killerId": 1, "position": {"x": 9000, "y": 9000}},
            {"type": "ITEM_UNDO", "timestamp": 59000, "participantId": 1,
             "beforeId": 1001, "afterId": 0},
        ]},
    ]}}
    return match, timeline


review._version_for = lambda _: "16.18.1"
review.DataDragon = FakeDragon
review._summoner_spells = lambda _: {}
review._runes = lambda _: {}
match, timeline = fixture()
result = review.shape_review(match, timeline, 1)
assert len(result["players"]) == 10
assert len(result["observations"]) == 2
assert result["observations"][0]["players"]["6"]["health"] == 0
assert result["observations"][0]["players"]["1"]["health"] == 100
assert next(shop for shop in result["shops"] if shop["ts"] == 55000)["bought"] == [1001]
assert next(state for state in result["inventory_states"] if state["ts"] == 55000)["players"]["1"] == [2055, 1001]
assert next(state for state in result["inventory_states"] if state["ts"] == 55500)["players"]["1"] == [1001]
assert result["item_meta"]["2055"]["stacks"] == 2
assert any(row["ts"] == 59000 and row["players"]["1"] == []
           for row in result["inventory_states"])
assert result["inventory_uncertain_from_ms"] is None
assert any(row["players"]["5"] == [3866] for row in result["inventory_states"])
assert [event["kind"] for event in result["events"]] == ["kill", "base", "base"]
assert result["events"][1]["destroyed_team_id"] == 200
assert len(result["structures"]) == 30
assert next(s for s in result["structures"] if s["key"] == "200:tower:MIDDLE:OUTER")["destroyed_at"] == 58000
assert next(s for s in result["structures"] if s["key"] == "200:nexus:BASE:")["destroyed_at"] == result["duration_ms"]
assert result["build_comparison"]["status"] == "unavailable"
try:
    review.shape_review(match, timeline, 11)
except ProfileError as exc:
    assert exc.status == 404
else:
    raise AssertionError("Unknown participant was accepted")

print("review synthetic checks passed")
