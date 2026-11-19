"""Lane gold difference at 12:00 from the draft (protocol_gold_v1).

G[L, X, c]: shrunk mean of lane L's gold difference seen from champion c's team when c
plays role X. A lane row is antisymmetric: swapping the teams negates every term.
"""
from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

from .data import NC, Games
from .tables import R, Tables, eb, k_lane

GOLD_NAMES = ["side"] + [f"from_{x}" for x in range(R)] + ["matchup"]


def lane_diff(G: Games) -> np.ndarray:
    """(n, 5) blue minus red gold at 12:00; NaN when the frame is missing."""
    return (G.g12[:, 0, :] - G.g12[:, 1, :]).astype(np.float64)


@dataclass
class GoldTables:
    cross: np.ndarray        # (R, R, NC)
    prior_k: dict


def build_gold_tables(G: Games, d: np.ndarray | None = None) -> GoldTables:
    """d: (n, 5) blue-minus-red lane gold at the target minute (default: 12:00 from the extract)."""
    d = lane_diff(G) if d is None else d
    champ = G.champ.astype(np.int64)
    cross = np.zeros((R, R, NC))
    prior = {}
    for L in range(R):
        ok = ~np.isnan(d[:, L])
        noise = float(np.var(d[ok, L]))
        for X in range(R):
            keys = np.concatenate([champ[ok, 0, X], champ[ok, 1, X]])
            vals = np.concatenate([d[ok, L], -d[ok, L]])
            uk, v, _, prior[f"{L}{X}"] = eb(keys, vals, noise)
            cross[L, X, uk] = v
    return GoldTables(cross, prior)


def gold_rows(GT: GoldTables, T: Tables, a: np.ndarray, b: np.ndarray, L: int, side: float) -> np.ndarray:
    """Row for lane L with team a (on `side`) against team b; a, b: (n, 5)."""
    n = len(a)
    cols = [np.full(n, side)]
    for X in range(R):
        cols.append(GT.cross[L, X, a[:, X]] - GT.cross[L, X, b[:, X]])
    cols.append(0.5 * (T.mgold(k_lane(L, a[:, L], b[:, L])) - T.mgold(k_lane(L, b[:, L], a[:, L]))))
    return np.column_stack(cols).astype(np.float32) / np.array([1] + [1000.0] * (R + 1), np.float32)


def gold_pair(GT, T, champ, L, extra=None):
    x = gold_rows(GT, T, champ[:, 0], champ[:, 1], L, 1.0)
    xs = gold_rows(GT, T, champ[:, 1], champ[:, 0], L, -1.0)
    if extra is not None:
        x, xs = np.hstack([x, extra[0]]), np.hstack([xs, extra[1]])
    return x, xs


class LaneGoldModel:
    """Antisymmetric ridge + LightGBM regressor for one lane (gold in units of 1,000)."""

    def __init__(self, seed: int = 7):
        self.ridge = Ridge(alpha=1.0)
        self.gb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                    min_child_samples=200, random_state=seed, n_jobs=1, verbose=-1)

    def fit(self, x, xs, y):
        X, Y = np.vstack([x, xs]), np.concatenate([y, -y]) / 1000.0
        self.ridge.fit(X, Y)
        self.gb.fit(X, Y)
        return self

    def parts(self, x, xs):
        r = 0.5 * (self.ridge.predict(x) - self.ridge.predict(xs))
        g = 0.5 * (self.gb.predict(x) - self.gb.predict(xs))
        return 1000.0 * r, 1000.0 * g

    def predict(self, x, xs):
        r, g = self.parts(x, xs)
        return 0.5 * (r + g)
