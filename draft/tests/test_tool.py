"""Gold antisymmetry (synthetic) and draft engine invariants (real frozen models, skipped if absent)."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from draft.gold import LaneGoldModel, build_gold_tables, gold_pair  # noqa: E402
from draft.tables import build_tables  # noqa: E402
from test_core import synthetic  # noqa: E402

HAVE_MODELS = (ROOT / "data/runs/v1/models.pkl").exists() and (ROOT / "data/runs/gold_v2/model.pkl").exists()


class GoldTests(unittest.TestCase):
    def test_lane_gold_swaps_sign(self):
        G = synthetic()
        T, GT = build_tables(G), build_gold_tables(G)
        x, xs = gold_pair(GT, T, G.champ, 3)
        y = (G.g12[:, 0, 3] - G.g12[:, 1, 3]).astype(float)
        m = LaneGoldModel().fit(x, xs, y)
        self.assertTrue(np.allclose(m.predict(x, xs), -m.predict(xs, x)))


@unittest.skipUnless(HAVE_MODELS, "frozen models not built")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from draft.engine import DraftEngine
        cls.E = DraftEngine(ROOT)
        cls.ids = {v["name"]: k for k, v in cls.E.champs.items()}

    def test_recommend_excludes_taken_and_sorts(self):
        i = self.ids
        blue, red, bans = [0, i["Lee Sin"], 0, 0, 0], [i["Darius"], 0, 0, 0, 0], [i["Garen"], i["Teemo"]]
        r = self.E.recommend(blue, red, bans, side=0, role=0)
        got = [c["id"] for c in r["candidates"]]
        self.assertTrue(r["opponent_known"])
        self.assertFalse({i["Lee Sin"], i["Darius"], i["Garen"], i["Teemo"]} & set(got))
        p = [c["p_expected"] for c in r["candidates"]]
        self.assertEqual(p, sorted(p, reverse=True))

    def test_recommendation_matches_state_once_picked(self):
        i = self.ids
        blue = [i["Garen"], i["Lee Sin"], i["Ahri"], i["Jinx"], i["Thresh"]]
        red = [i["Darius"], i["Vi"], i["Syndra"], i["Caitlyn"], i["Leona"]]
        for side, role in ((0, 2), (1, 3)):
            r = self.E.recommend(blue, red, [], side=side, role=role)
            chosen = (blue if side == 0 else red)[role]
            row = next(c for c in r["candidates"] if c["id"] == chosen)
            st = self.E.state(blue, red)
            ours = st["p_blue"] if side == 0 else 1 - st["p_blue"]
            self.assertAlmostEqual(row["p_expected"], ours, places=12)
            gold = st["lanes"][role]["gold"] * (1 if side == 0 else -1)
            self.assertAlmostEqual(row["gold"], gold, places=3)  # float32 features

    def test_blind_pick_has_worst_case_below_expected(self):
        r = self.E.recommend([0] * 5, [0] * 5, [], side=1, role=2)
        for c in r["candidates"][:10]:
            self.assertLessEqual(c["p_worst"], c["p_expected"] + 1e-12)
            self.assertNotEqual(c["counter"], c["id"])


if __name__ == "__main__":
    unittest.main()
