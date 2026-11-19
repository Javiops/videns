"""Offline checks of opt-in reconstruction and strict experimental bindings."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import app.ddragon as ddragon
import app.reconstruct as production
from test_footwear_economics import pinned_dragon
from footwear_runtime import (
    FOOTWEAR_STATE_VERSION, RECONSTRUCTION_VERSION, activate, declaration,
    validate_generation_runtime, validate_state,
)


def match():
    return {"metadata": {"matchId": "KR_SYNTHETIC_FOOTWEAR"}, "info": {
        "gameVersion": "16.17.1.1", "gameDuration": 1800, "gameCreation": 1,
        "queueId": 420, "participants": [
            {"participantId": 1, "puuid": "wearer", "championId": 103,
             "championName": "Ahri", "teamId": 100, "teamPosition": "MIDDLE", "win": True,
             "perks": {"styles": [{"selections": [{"perk": 8304, "var1": 12, "var2": 0, "var3": 0}]}]}},
            {"participantId": 2, "puuid": "other", "championId": 64,
             "championName": "LeeSin", "teamId": 200, "teamPosition": "JUNGLE", "win": False,
             "perks": {"styles": [{"selections": [{"perk": 8005}]}]}},
        ]}}


def item(ts, owner=1, iid=1036, kind="PURCHASED"):
    return {"type": "ITEM_" + kind, "timestamp": ts, "participantId": owner, "itemId": iid}


def timeline(events):
    frames = []
    for stamp in range(0, 960001, 60000):
        frames.append({"timestamp": stamp,
            "events": [e for e in events if stamp - 60000 < e["timestamp"] <= stamp],
            "participantFrames": {str(pid): {"participantId": pid, "currentGold": 1200,
                "totalGold": 1500 + stamp // 100, "level": 8, "minionsKilled": 60}
                for pid in (1, 2)}})
    return {"info": {"frames": frames}}


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test static reads stay limited to captured item.json; no network or
        # full database, raw-game, exported training or final-test payloads.
        source = pinned_dragon()
        ddragon.DataDragon = lambda version=None: source
        ddragon.dragon_for_patch.cache_clear()
        activate()
        cls.dragon = ddragon.dragon_for_patch("16.17")

    def replay(self, events, supplied_match=None):
        debug = {}
        game, rows = production.reconstruct_game(supplied_match or match(), timeline(events), self.dragon, debug)
        return rows, debug

    def test_explicit_scoping_and_production_isolation(self):
        self.assertEqual(activate(), declaration())
        self.assertEqual(self.dragon.footwear_state_version, FOOTWEAR_STATE_VERSION)
        self.assertIs(ddragon.DataDragon("16.17.1")._source, self.dragon._source)
        for call in (lambda: ddragon.dragon_for_patch("16.18"), lambda: ddragon.DataDragon("16.18.1")):
            with self.assertRaises(ValueError):
                call()
        result = subprocess.run([sys.executable, "-c", "from app.reconstruct import RECONSTRUCTION_VERSION; print(RECONSTRUCTION_VERSION)"],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "decision-start-v3")

    def test_pregrant_postgrant_and_export_board_state(self):
        events, debug = self.replay([item(30000), item(700000), item(750000, owner=2),
                                    item(850000, iid=2422, kind="DESTROYED"), item(850000, iid=3020)])
        self.assertEqual([g["timestamp"] for g in debug["footwear_grants"]], [720000])
        prior = next(e for e in events if e["ts"] == 700000)
        later = next(e for e in events if e["ts"] == 750000)
        self.assertFalse(prior["footwear_granted"])
        self.assertTrue(prior["magical_footwear_selected"])
        board_wearer = next(p for p in later["board"] if p["participant_id"] == 1)
        self.assertTrue(board_wearer["footwear_granted"])
        self.assertIn(2422, [i["item_id"] for i in board_wearer["items"]])
        self.assertFalse(later["magical_footwear_selected"])
        self.assertEqual(prior["reconstruction_version"], RECONSTRUCTION_VERSION)
        self.assertEqual(prior["inventory_version"], declaration()["inventory_version"])
        import baseline
        row = baseline._example(later, self.dragon)
        self.assertEqual(validate_state(row), validate_state(later))
        self.assertEqual(validate_state(row["others"][0]), validate_state(board_wearer))
        upgrade = next(e for e in events if e["ts"] == 850000)
        self.assertIn((850000, 800.0, "spend"), debug["gold_events"][1])
        self.assertNotIn(2422, [i["item_id"] for i in upgrade["label_bought"]])

    def test_save_uses_death_anchor_before_grant(self):
        death = {"type": "CHAMPION_KILL", "timestamp": 710000, "killerId": 2,
                 "victimId": 1, "assistingParticipantIds": []}
        events, _debug = self.replay([item(30000), death, item(750000, owner=2)])
        saved = next(e for e in events if e.get("is_save"))
        self.assertEqual(saved["ts"], 725000)
        self.assertFalse(saved["footwear_granted"])
        self.assertFalse(next(p for p in saved["board"] if p["participant_id"] == 1)["footwear_granted"])

    def test_same_timestamp_is_not_granted_before_decision(self):
        events, _debug = self.replay([item(30000), item(720000, owner=2), item(740000, owner=2)])
        tied = next(e for e in events if e["ts"] == 720000)
        self.assertFalse(next(p for p in tied["board"] if p["participant_id"] == 1)["footwear_granted"])

    def test_postgame_mutation_does_not_change_inputs(self):
        source = match()
        altered = deepcopy(source)
        for p in altered["info"]["participants"]:
            p.update(kills=999, deaths=999, assists=999, item0=3869, roleBoundItem=3175)
            for s in p["perks"]["styles"]:
                for r in s["selections"]:
                    r.update(var1=999, var2=999, var3=999)
        raw = [item(30000), item(700000), item(750000, owner=2)]
        self.assertEqual(self.replay(raw, source)[0], self.replay(raw, altered)[0])

    def test_unreviewed_explicit_grant_and_patch_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Explicit Magical"):
            self.replay([item(720000, iid=2422)])
        changed = match()
        changed["info"]["gameVersion"] = "16.18.1.1"
        with self.assertRaisesRegex(ValueError, "patch 16.17"):
            self.replay([item(30000)], changed)

    def test_missing_state_and_undeclared_generation_fail_closed(self):
        with self.assertRaises(ValueError):
            validate_state({})
        with TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ("generation_manifest.json", "split_manifest.json"):
                (directory / name).write_text(json.dumps({"patch": "16.17", "footwear_runtime": declaration()}))
            self.assertEqual(validate_generation_runtime(directory), declaration())
            (directory / "split_manifest.json").write_text(json.dumps({"patch": "16.17"}))
            with self.assertRaises(ValueError):
                validate_generation_runtime(directory)


if __name__ == "__main__":
    unittest.main()
