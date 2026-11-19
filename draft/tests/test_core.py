"""Synthetic invariants: coherence, neutral average champion, sampler exclusions, sealed split."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from draft.data import Games, load_games  # noqa: E402
from draft.features import match_pair, perspective_features  # noqa: E402
from draft.models import Coherent  # noqa: E402
from draft.tables import build_tables, k_lane  # noqa: E402


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    champ = np.zeros((n, 2, 5), dtype=np.int32)
    for g in range(n):
        champ[g] = rng.choice(np.arange(1, 40), size=10, replace=False).reshape(2, 5)
    strength = rng.normal(0, 0.3, 40)
    logit = strength[champ[:, 0]].sum(1) - strength[champ[:, 1]].sum(1)
    win = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    f = lambda: rng.uniform(1000, 20000, (n, 2, 5)).astype(np.float32)
    g12 = rng.normal(4500, 400, (n, 2, 5)).astype(np.float32)
    return Games(ids=np.array([f"X_{i}" for i in range(n)]), role=np.array(["fit"] * n),
                 t=np.arange(n, dtype=np.int64), patch=np.array(["16.18"] * n),
                 dur=rng.uniform(1200, 2400, n).astype(np.float32), blue_win=win, tl=np.ones(n, bool),
                 champ=champ, player=rng.integers(0, 300, (n, 2, 5)).astype(np.int32),
                 asg=np.zeros((n, 2, 5), np.int8), pdmg=f(), mdmg=f(), tdmg=f(), taken=f(), g12=g12,
                 bans=np.zeros((n, 2, 5), np.int32), exp_champ=rng.integers(0, 20, (n, 2, 5)).astype(np.float32),
                 exp_total=rng.integers(0, 90, (n, 2, 5)).astype(np.float32))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.G = synthetic()
        self.T = build_tables(self.G)

    def test_matchup_table_antisymmetric(self):
        c = self.G.champ
        a = self.T.matchup(k_lane(2, c[:, 0, 2], c[:, 1, 2]))
        b = self.T.matchup(k_lane(2, c[:, 1, 2], c[:, 0, 2]))
        self.assertLess(np.abs(a + b).max(), 1e-12)

    def test_average_champion_is_neutral(self):
        self.assertTrue(np.all(self.T.delta[:, 0] == 0))
        self.assertTrue(np.all(self.T.scale[:, 0] == 0) and np.all(self.T.econ[:, 0] == 0))
        self.assertEqual(self.T.matchup(np.array([k_lane(0, 0, 5)]))[0], 0.0)

    def test_coherent_model_swaps_exactly(self):
        z = np.zeros(self.G.champ.shape, np.float32)
        x, xs = match_pair(self.T, self.G.champ, z, z)
        m = Coherent().fit(x, xs, self.G.blue_win.astype(int))
        self.assertTrue(np.allclose(m.predict(x, xs) + m.predict(xs, x), 1.0, atol=1e-12))

    def test_perspective_shape(self):
        z = np.zeros(self.G.champ.shape, np.float32)
        P = perspective_features(self.T, self.G.champ[:10], z[:10], z[:10])
        self.assertEqual(P.shape[:3], (10, 2, 5))
        self.assertFalse(np.isnan(P).any())

    def test_sampler_respects_exclusions(self):
        from blind_v1 import Sampler
        pick = np.ones((5, 50))
        pick[:, 0] = 0
        s = Sampler(pick, seed=1)
        champ = self.G.champ[:50]
        bans = np.tile(np.arange(40, 50), (50, 1)).reshape(50, 2, 5).astype(np.int32)
        hidden = [(1, r) for r in range(5)]
        out = s.draw(champ, bans, hidden, 8)
        for j in range(8):
            for g in range(50):
                drawn = out[j, g, 1]
                self.assertTrue(np.all(drawn > 0))
                self.assertTrue(set(drawn).isdisjoint(set(champ[g, 0])))
                self.assertTrue(set(drawn).isdisjoint(set(range(40, 50))))
                self.assertEqual(len(set(drawn)), 5)

    def test_confirmation_is_sealed(self):
        with self.assertRaises(PermissionError):
            load_games(ROOT / "data/extract/games_v1.jsonl.gz", ROOT / "data/split_v1.json", roles=("confirm",))


if __name__ == "__main__":
    unittest.main()
