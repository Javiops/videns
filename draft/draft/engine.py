"""Draft tool engine: win probability, lane gold at 12:00 and win-maximising recommendations.

Recommendations rank OUR candidates by our probability of winning. Pick rates are used only
to model the enemy's still-unknown reply in the same lane (expected value) and to restrict
candidates to champions with enough games in the role. Uses the frozen v1 coherent model
without experience (champion information only) and the gold v1 lane models.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .features import MATCH_NAMES, match_pair
from .gold import gold_pair

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
MIN_GAMES = 100        # candidates: champions with at least this many fit games in the role
MIN_GAMES_POOL = 30    # champions from the user's pool are allowed down to this
K_REPLIES = 40         # enemy replies considered for an unknown lane opponent
PHASE_MINUTES = (12, 24, 30)  # early / mid / late shown by the tool (36:00 is reached by ~5% of games)
EXPLAIN_TOP = 12
GROUPS = {
    "Champion strength": ["base_a", "base_b"],
    "Lane matchup": [f"lane_{r}" for r in range(5)],
    "Cross-lane counters": ["cross"],
    "Synergies": ["syn_a", "syn_b"],
    "Lane gold @12": [f"gold_{r}" for r in range(5)],
    "Economy": ["eco_a", "even_a", "snow_a", "eco_b", "even_b", "snow_b"],
    "Scaling": ["early_a", "mid_a", "late_a", "early_b", "mid_b", "late_b"],
    "Damage mix": ["ad_a", "ad_b", "dmg_a", "dmg_b", "taken_a", "taken_b"],
}


class DraftEngine:
    def __init__(self, root: Path):
        with open(root / "data/runs/v1/models.pkl", "rb") as f:
            v1 = pickle.load(f)
        with open(root / "data/runs/gold_v2/model.pkl", "rb") as f:
            self.gold = pickle.load(f)          # {minute: {GT, models, dev_rmse, team_rmse, dev_share}}
        self.minutes = [m for m in PHASE_MINUTES if m in self.gold]
        with open(root / "data/runs/profile_v1/profile.pkl", "rb") as f:
            prof = pickle.load(f)              # damage composition + expected CS per minute
        self.damage, self.cs = prof["damage"], prof["cs"]
        with open(root / "data/runs/profile_v2/profile.pkl", "rb") as f:
            prof2 = pickle.load(f)             # kills, kill participation, team dragons
        self.kills = prof2["kills"]                 # whole-game kills
        with open(root / "data/runs/profile_v3/profile.pkl", "rb") as f:
            prof3 = pickle.load(f)             # kill participation and team dragons by minute
        self.kp, self.drag = prof3["kp"], prof3["dragons"]
        self.T, self.C = v1["T"], v1["C_noexp"]
        self.gold_rmse = self.gold[12]["dev_rmse"]
        data = json.loads((root / "data/static/champion-16.19.1.json").read_text(encoding="utf-8"))["data"]
        self.version = "16.19.1"
        self.champs = {int(c["key"]): {"name": c["name"], "img": c["id"]} for c in data.values()}
        pick = self.T.pick
        self.rate = pick / pick.sum(1, keepdims=True)
        self.games = pick
        self.by_role = [[int(c) for c in np.argsort(-pick[r]) if pick[r, c] >= MIN_GAMES and int(c) in self.champs]
                        for r in range(5)]
        self.group_idx = {g: [MATCH_NAMES.index(n) for n in names] for g, names in GROUPS.items()}

    # ---- model calls -------------------------------------------------------------------
    def _features(self, champ):
        z = np.zeros(champ.shape, np.float32)
        return match_pair(self.T, champ, z, z)

    def p_blue(self, champ: np.ndarray) -> np.ndarray:
        x, xs = self._features(champ)
        return self.C.predict(x, xs)

    def lane_gold(self, champ: np.ndarray, minute: int = 12) -> np.ndarray:
        """(n, 5) predicted blue-minus-red lane gold at `minute` (if the game lasts that long)."""
        g = self.gold[minute]
        return np.column_stack([g["models"][L].predict(*gold_pair(g["GT"], self.T, champ, L)) for L in range(5)])

    # ---- API ---------------------------------------------------------------------------
    def meta(self) -> dict:
        champs = [{"id": cid, "name": c["name"], "img": c["img"],
                   "games": [int(self.games[r, cid]) for r in range(5)]}
                  for cid, c in sorted(self.champs.items(), key=lambda kv: kv[1]["name"])]
        return {"version": self.version, "roles": ROLES, "champions": champs, "min_games": MIN_GAMES,
                "minutes": self.minutes,
                "cs": {str(m): [{str(cid): round(float(self.cs[m][r, cid]), 2) for cid in self.champs}
                                for r in range(5)] for m in self.minutes},
                "kills": [{str(cid): round(float(self.kills[r, cid]), 3) for cid in self.champs} for r in range(5)],
                "kp": {str(m): [{str(cid): round(float(self.kp[m][r, cid]), 4) for cid in self.champs}
                                for r in range(5)] for m in self.minutes}}

    @staticmethod
    def _draft(blue, red) -> np.ndarray:
        champ = np.zeros((1, 2, 5), dtype=np.int32)
        champ[0, 0], champ[0, 1] = blue, red
        return champ

    def state(self, blue, red) -> dict:
        champ = self._draft(blue, red)
        both = [bool(blue[L] and red[L]) for L in range(5)]
        phases = []
        for m in self.minutes:
            gold, g = self.lane_gold(champ, m)[0], self.gold[m]
            phases.append({"minute": m, "reach": float(g["dev_share"]), "team_rmse": float(g["team_rmse"]),
                           "dragons": self._dragons(champ, m),
                           "lanes": [{"gold": float(gold[L]), "rmse": float(g["dev_rmse"][L]), "both_known": both[L]}
                                     for L in range(5)]})
        damage = self.damage.team(champ)[0]          # (2, 3): physical / magic / true shares
        return {"p_blue": float(self.p_blue(champ)[0]), "phases": phases,
                "damage": [[float(v) for v in damage[s]] for s in (0, 1)],
                "lanes": phases[self.minutes.index(12)]["lanes"], "complete": all(blue) and all(red)}

    def _dragons(self, champ, minute) -> list:
        """Expected dragons taken by each team by `minute` (games that last that long)."""
        d, ridx = self.drag[minute], np.arange(5)
        return [float(d["mu"] + d["intercept"] + np.dot(d["coef"], [d["own"][ridx, champ[0, s]].sum(),
                d["enemy"][ridx, champ[0, 1 - s]].sum(), 1.0 - 2 * s])) for s in (0, 1)]

    def recommend(self, blue, red, bans, side: int, role: int, pool=None, explain=True, minute: int = 12) -> dict:
        if minute not in self.gold:
            raise ValueError(f"minute must be one of {self.minutes}")
        base = self._draft(blue, red)
        base[0, side, role] = 0
        # the champion currently in this slot stays a candidate, so it can be compared
        taken = (set(base[0].ravel().tolist()) | set(bans)) - {0}
        if pool:
            pool = {int(c) for c in pool}
            cands = [c for c in self.champs if c in pool and self.games[role, c] >= MIN_GAMES_POOL]
        else:
            cands = list(self.by_role[role])
        cands = [c for c in cands if c not in taken]
        if not cands:
            return {"candidates": [], "opponent_known": bool(base[0, 1 - side, role])}
        n = len(cands)
        ours = (lambda p: p) if side == 0 else (lambda p: 1.0 - p)
        sign = 1.0 if side == 0 else -1.0
        grid = np.repeat(base, n + 1, 0)
        grid[:n, side, role] = cands                           # last row: average champion
        p_now = ours(self.p_blue(grid))
        gold = sign * self.lane_gold(grid, minute)[:, role]
        opp_known = bool(base[0, 1 - side, role])
        out = {"opponent_known": opp_known, "reference_avg": float(p_now[-1]), "minute": minute}
        rows = [{"id": c, "p_vs_avg": float(p_now[i]), "gold": float(gold[i]),
                 "games": int(self.games[role, c])} for i, c in enumerate(cands)]
        if not opp_known:
            replies = [d for d in self.by_role[role] if d not in taken][:K_REPLIES]
            w = self.rate[role, replies]
            g2 = np.repeat(base, (n + 1) * len(replies), 0)
            g2[:, side, role] = np.repeat(cands + [0], len(replies))
            g2[:, 1 - side, role] = np.tile(replies, n + 1)
            V = ours(self.p_blue(g2)).reshape(n + 1, len(replies))
            G = (sign * self.lane_gold(g2, minute)[:, role]).reshape(n + 1, len(replies))
            same = np.array(cands + [0])[:, None] == np.array(replies)[None, :]
            W = np.where(same, 0.0, w[None, :])
            exp_p = (V * W).sum(1) / W.sum(1)
            exp_g = (G * W).sum(1) / W.sum(1)
            Vm = np.where(same, np.inf, V)
            worst = Vm.min(1)
            counter = np.array(replies)[Vm.argmin(1)]
            for i, r in enumerate(rows):
                r.update(p_expected=float(exp_p[i]), p_worst=float(worst[i]), counter=int(counter[i]),
                         gold=float(exp_g[i]))
            out["reference_expected"] = float(exp_p[-1])
            out["replies_considered"] = len(replies)
        else:
            for r in rows:
                r.update(p_expected=r["p_vs_avg"], p_worst=None, counter=None)
            out["reference_expected"] = out["reference_avg"]
        for r in rows:
            r["delta_vs_avg_champion"] = r["p_expected"] - out["reference_expected"]
        rows.sort(key=lambda r: -r["p_expected"])
        if explain:
            self._explain(base, side, role, rows[:EXPLAIN_TOP])
        out["candidates"] = rows
        return out

    def _explain(self, base, side, role, rows):
        """Per group: change in our win probability when that group's features are set back to
        the average-champion draft (other slots as given)."""
        ours = (lambda p: p) if side == 0 else (lambda p: 1.0 - p)
        ref = base.copy()
        xr, xsr = self._features(ref)
        grid = np.repeat(base, len(rows), 0)
        grid[:, side, role] = [r["id"] for r in rows]
        x, xs = self._features(grid)
        p_full = ours(self.C.predict(x, xs))
        for g, idx in self.group_idx.items():
            xg, xsg = x.copy(), xs.copy()
            xg[:, idx], xsg[:, idx] = xr[0, idx], xsr[0, idx]
            delta = p_full - ours(self.C.predict(xg, xsg))
            for r, d in zip(rows, delta):
                r.setdefault("why", {})[g] = float(d)
        for r in rows:
            r["why"] = dict(sorted(r["why"].items(), key=lambda kv: -abs(kv[1]))[:4])
