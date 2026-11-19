"""Feature builders.

`perspective_features` reproduces the iTero recipe: one row per (team, lane)
with the picker's lane, counters and synergies, a gold@12 prediction and team
context. `match_features` builds the coherent model's blue-vs-red row; it is
written as f(a, b, side) so that swapping teams negates every antisymmetric
feature and exchanges the team features.
"""
from __future__ import annotations

import numpy as np

from .tables import R, Tables, k_lane, k_pair

EXP_CAP = 50.0


def exp_features(exp_champ: np.ndarray, exp_total: np.ndarray, unknown_missing: bool = False):
    ec = np.log1p(np.minimum(exp_champ, EXP_CAP))
    et = np.log1p(np.minimum(exp_total, 5 * EXP_CAP))
    if unknown_missing:  # no earlier game of this player in the archive -> unknown, not zero
        unknown = exp_total == 0
        ec = np.where(unknown, np.nan, ec)
        et = np.where(unknown, np.nan, et)
    return ec, et


def _team_stats(T: Tables, a: np.ndarray):
    """Per-team arrays for champions a (n, 5)."""
    ridx = np.arange(R)
    delta = T.delta[ridx, a]
    econ = T.econ[ridx, a]            # (n, 5, 3)
    scale = T.scale[ridx, a]          # (n, 5, 3)
    phys = T.phys[ridx, a]
    dmg = T.dmg[ridx, a]
    taken = T.taken[ridx, a]
    return delta, econ, scale, phys, dmg, taken


def _syn_sum(T: Tables, a: np.ndarray) -> np.ndarray:
    total = np.zeros(len(a))
    for r1 in range(R):
        for r2 in range(r1 + 1, R):
            total += 0.5 * (T.syn(k_pair(r1, a[:, r1], r2, a[:, r2])) + T.syn(k_pair(r2, a[:, r2], r1, a[:, r1])))
    return total


MATCH_NAMES = (["side", "base_a", "base_b"] + [f"lane_{r}" for r in range(R)] + ["cross"]
               + ["syn_a", "syn_b"] + [f"gold_{r}" for r in range(R)]
               + ["eco_a", "even_a", "snow_a", "eco_b", "even_b", "snow_b"]
               + ["early_a", "mid_a", "late_a", "early_b", "mid_b", "late_b"]
               + ["ad_a", "ad_b", "dmg_a", "dmg_b", "taken_a", "taken_b"]
               + ["expc_a", "expc_b", "expt_a", "expt_b", "expcmin_a", "expcmin_b"])


def match_features(T: Tables, a, b, ec_a, ec_b, et_a, et_b, side: float) -> np.ndarray:
    """Row for team a (on `side`, +1 blue / -1 red) against team b. a, b: (n, 5)."""
    n = len(a)
    da, ea, sa, pa, ma, ta = _team_stats(T, a)
    db, eb_, sb, pb, mb, tb = _team_stats(T, b)
    lanes = np.stack([T.matchup(k_lane(r, a[:, r], b[:, r])) for r in range(R)], 1)
    cross = np.zeros(n)
    for r1 in range(R):
        for r2 in range(R):
            if r1 != r2:
                cross += 0.5 * (T.cross(k_pair(r1, a[:, r1], r2, b[:, r2]))
                                - T.cross(k_pair(r2, b[:, r2], r1, a[:, r1])))
    ridx = np.arange(R)
    gold = np.stack([T.cgold[r, a[:, r]] - T.cgold[r, b[:, r]]
                     + 0.5 * (T.mgold(k_lane(r, a[:, r], b[:, r])) - T.mgold(k_lane(r, b[:, r], a[:, r])))
                     for r in ridx], 1) / 1000.0
    ad = lambda p, m: (p * m).sum(1) / m.sum(1)
    cols = [np.full(n, side), da.sum(1), db.sum(1), lanes, cross[:, None],
            _syn_sum(T, a)[:, None], _syn_sum(T, b)[:, None], gold,
            ea.sum(1), eb_.sum(1), sa.sum(1), sb.sum(1),
            ad(pa, ma)[:, None], ad(pb, mb)[:, None], ma.sum(1)[:, None], mb.sum(1)[:, None],
            ta.sum(1)[:, None], tb.sum(1)[:, None],
            np.nansum(ec_a, 1)[:, None], np.nansum(ec_b, 1)[:, None],
            np.nansum(et_a, 1)[:, None], np.nansum(et_b, 1)[:, None],
            np.nanmin(ec_a, 1)[:, None], np.nanmin(ec_b, 1)[:, None]]
    out = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols]).astype(np.float32)
    assert out.shape[1] == len(MATCH_NAMES)
    return out


def match_pair(T: Tables, champ, ec, et):
    """(x_blue, x_swapped) for the coherent model. champ/ec/et: (n, 2, 5)."""
    x = match_features(T, champ[:, 0], champ[:, 1], ec[:, 0], ec[:, 1], et[:, 0], et[:, 1], 1.0)
    xs = match_features(T, champ[:, 1], champ[:, 0], ec[:, 1], ec[:, 0], et[:, 1], et[:, 0], -1.0)
    return x, xs


PERSP_NAMES = (["side_blue"] + [f"lane_is_{r}" for r in range(R)] + ["my_delta", "team_wo", "enemy"]
               + [f"counter_{r}" for r in range(R)] + ["counter_avg"]
               + [f"syn_{r}" for r in range(R)] + ["syn_avg", "pred_gold"]
               + ["my_eco", "my_even", "my_snow", "opp_eco", "opp_even", "opp_snow"]
               + ["my_early", "my_mid", "my_late", "team_early", "team_mid", "team_late",
                  "wo_early", "wo_mid", "wo_late", "en_early", "en_mid", "en_late"]
               + ["my_phys", "team_ad", "team_ad_wo", "team_ad_var", "en_ad",
                  "team_dmg", "en_dmg", "team_taken", "en_taken"]
               + ["my_expc", "my_expt"])


def perspective_features(T: Tables, champ, ec, et) -> np.ndarray:
    """(n, 2, 5, F): row for the player on side s in lane L (iTero's t1_LANE view)."""
    n = len(champ)
    out = np.zeros((n, 2, R, len(PERSP_NAMES)), dtype=np.float32)
    for s in (0, 1):
        a, b = champ[:, s], champ[:, 1 - s]
        da, ea, sa, pa, ma, ta = _team_stats(T, a)
        db, eb_, sb, pb, mb, tb = _team_stats(T, b)
        for L in range(R):
            me = a[:, L]
            wo = np.arange(R) != L
            counters = np.stack([T.matchup(k_lane(L, me, b[:, L])) if X == L
                                 else T.cross(k_pair(L, me, X, b[:, X])) for X in range(R)], 1)
            syns = np.stack([np.zeros(n) if X == L else T.syn(k_pair(L, me, X, a[:, X])) for X in range(R)], 1)
            pred_gold = (T.cgold[L, me] - T.cgold[L, b[:, L]] + T.mgold(k_lane(L, me, b[:, L]))) / 1000.0
            ad = (pa * ma).sum(1) / ma.sum(1)
            ad_wo = (pa[:, wo] * ma[:, wo]).sum(1) / ma[:, wo].sum(1)
            onehot = np.zeros((n, R))
            onehot[:, L] = 1
            cols = [np.full(n, 1.0 - s), onehot, da[:, L], da[:, wo].sum(1), db.sum(1),
                    counters, counters.mean(1), syns, syns[:, wo].mean(1), pred_gold,
                    ea[:, L], eb_[:, L], sa[:, L], sa.sum(1), sa[:, wo].sum(1), sb.sum(1),
                    pa[:, L], ad, ad_wo, ad - ad_wo, (pb * mb).sum(1) / mb.sum(1),
                    ma.sum(1), mb.sum(1), ta.sum(1), tb.sum(1), ec[:, s, L], et[:, s, L]]
            out[:, s, L] = np.column_stack([c if c.ndim == 2 else c[:, None] for c in cols])
    return out
