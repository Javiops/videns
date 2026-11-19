"""Empirical-Bayes champion, pair, economy, scaling and damage tables.

Everything is estimated from the games passed in (fit games only). Champion id
0 never occurs in real games, so every lookup at 0 returns the neutral value:
that is the "average champion" used for imputation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import NC, Games

R = 5
GOLD_STATE = 500.0          # lane gold difference at 12:00 splitting behind / even / ahead
DUR_EDGES = (1500.0, 1920.0)  # <25 min early, 25-32 mid, >32 late
DAMAGE_PRIOR = 30.0


def logit(p):
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


FIXED_K: dict | None = None  # exploratory override {table name: k}; None -> method of moments


def eb(keys: np.ndarray, resid: np.ndarray, noise: float, name: str = "") -> tuple:
    """Group residuals by key; shrink each mean by k = noise / tau^2 (method of moments)."""
    ok = ~np.isnan(resid)
    uk, inv = np.unique(keys[ok], return_inverse=True)
    n = np.bincount(inv).astype(np.float64)
    r = np.bincount(inv, weights=resid[ok].astype(np.float64))
    if FIXED_K is not None and name in FIXED_K:
        k = FIXED_K[name]
    else:
        tau2 = max((np.sum(r * r / n) - noise * len(n)) / n.sum(), 1e-9)
        k = noise / tau2
    return uk, r / (n + k), n, k


@dataclass
class PairTable:
    keys: np.ndarray
    vals: np.ndarray

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.keys, keys)
        pos = np.minimum(pos, len(self.keys) - 1)
        hit = self.keys[pos] == keys
        return np.where(hit, self.vals[pos], 0.0)


def k_lane(r, c, o):
    return (np.int64(r) * NC + c) * NC + o


def k_pair(r1, c1, r2, c2):
    return ((np.int64(r1) * NC + c1) * R + r2) * NC + c2


def dense(keys, vals, shape) -> np.ndarray:
    out = np.zeros(int(np.prod(shape)))
    out[keys] = vals
    return out.reshape(shape)


@dataclass
class Tables:
    p_side: float
    delta: np.ndarray
    matchup: PairTable
    cross: PairTable
    syn: PairTable
    cgold: np.ndarray
    mgold: PairTable
    econ: np.ndarray        # (R, NC, 3) behind / even / ahead
    scale: np.ndarray       # (R, NC, 3) early / mid / late
    phys: np.ndarray        # (R, NC) physical share of champion damage
    dmg: np.ndarray         # (R, NC) champion damage dealt (thousands)
    taken: np.ndarray       # (R, NC) damage taken (thousands)
    pick: np.ndarray        # (R, NC) pick counts
    prior_k: dict = field(default_factory=dict)

    def team_p0(self, champ: np.ndarray) -> np.ndarray:
        """Base-model P(blue) from champion-role deltas only."""
        d = self.delta[np.arange(R), champ]  # (n, 2, 5)
        return sigmoid(logit(self.p_side) + 4.0 * (d[:, 0].sum(1) - d[:, 1].sum(1)))


def build_tables(G: Games) -> Tables:
    n = len(G)
    champ = G.champ.astype(np.int64)
    yb = G.blue_win.astype(np.float64)
    y = np.stack([yb, 1 - yb], 1)                                    # (n, 2)
    p_side = float(yb.mean())
    roles = np.broadcast_to(np.arange(R), (n, 2, R))
    prior = {}

    # champion-role deltas, side-adjusted
    e_side = np.array([p_side, 1 - p_side])
    res = np.broadcast_to((y - e_side)[:, :, None], (n, 2, R))
    uk, v, _, prior["delta"] = eb((roles * NC + champ).ravel(), res.ravel(), 0.25, "delta")
    delta = dense(uk, v, (R, NC))

    tmp = Tables(p_side, delta, *(None,) * 11)
    p0 = tmp.team_p0(G.champ)
    res2 = y - np.stack([p0, 1 - p0], 1)                              # (n, 2) residual vs base model
    noise = float(np.mean(p0 * (1 - p0)))

    # same-lane matchups (antisymmetric by construction)
    keys, vals = [], []
    for r in range(R):
        for s in (0, 1):
            keys.append(k_lane(r, champ[:, s, r], champ[:, 1 - s, r]))
            vals.append(res2[:, s])
    uk, v, _, prior["matchup"] = eb(np.concatenate(keys), np.concatenate(vals), noise, "matchup")
    matchup = PairTable(uk, v)

    # cross-lane counters (enemy pairs, different roles) and synergies (ally pairs)
    ck, cv, sk, sv = [], [], [], []
    for r1 in range(R):
        for r2 in range(R):
            if r1 == r2:
                continue
            for s in (0, 1):
                ck.append(k_pair(r1, champ[:, s, r1], r2, champ[:, 1 - s, r2]))
                cv.append(res2[:, s])
                sk.append(k_pair(r1, champ[:, s, r1], r2, champ[:, s, r2]))
                sv.append(res2[:, s])
    uk, v, _, prior["cross"] = eb(np.concatenate(ck), np.concatenate(cv), noise, "cross")
    cross = PairTable(uk, v)
    uk, v, _, prior["syn"] = eb(np.concatenate(sk), np.concatenate(sv), noise, "syn")
    syn = PairTable(uk, v)

    # laning: gold difference to the lane opponent at 12:00
    d = G.g12[:, 0, :] - G.g12[:, 1, :]                               # (n, R) blue minus red
    dd = np.stack([d, -d], 1).astype(np.float64)                      # (n, 2, R)
    gnoise = float(np.nanvar(dd))
    uk, v, _, prior["cgold"] = eb((roles * NC + champ).ravel(), dd.ravel(), gnoise, "cgold")
    cgold = dense(uk, v, (R, NC))
    opp = champ[:, ::-1, :]
    expect = cgold[roles, champ] - cgold[roles, opp]
    gres = dd - expect
    uk, v, _, prior["mgold"] = eb(k_lane(roles, champ, opp).ravel(), gres.ravel(), float(np.nanvar(gres)), "mgold")
    mgold = PairTable(uk, v)

    # economy: win-rate residual by lane gold state, relative to all champions in that state
    ys = np.broadcast_to(y[:, :, None], (n, 2, R))
    state = np.where(dd < -GOLD_STATE, 0, np.where(dd > GOLD_STATE, 2, 1))
    valid = ~np.isnan(dd)
    econ = np.zeros((R, NC, 3))
    for st in range(3):
        m = valid & (state == st)
        base = np.zeros(R)
        for r in range(R):
            base[r] = ys[:, :, r][m[:, :, r]].mean()
        resid = np.where(m, ys - base[roles], np.nan)
        uk, v, _, prior[f"econ{st}"] = eb((roles * NC + champ).ravel(), resid.ravel(), 0.25, f"econ{st}")
        econ[:, :, st] = dense(uk, v, (R, NC))

    # scaling: residual vs base model by game length
    bucket = np.digitize(G.dur, DUR_EDGES)
    scale = np.zeros((R, NC, 3))
    res_slot = np.broadcast_to(res2[:, :, None], (n, 2, R))
    for b in range(3):
        m = np.broadcast_to((bucket == b)[:, None, None], (n, 2, R))
        resid = np.where(m, res_slot, np.nan)
        uk, v, _, prior[f"scale{b}"] = eb((roles * NC + champ).ravel(), resid.ravel(), noise, f"scale{b}")
        scale[:, :, b] = dense(uk, v, (R, NC))

    # damage profile, shrunk toward the pick-weighted role mean (also the value at champion 0)
    tot = (G.pdmg + G.mdmg + G.tdmg).astype(np.float64)
    share = np.where(tot > 0, G.pdmg / np.maximum(tot, 1), np.nan)
    key = (roles * NC + champ).ravel()
    pick = np.bincount(key, minlength=R * NC).reshape(R, NC).astype(np.float64)

    def profile(x):
        x = x.ravel()
        ok = ~np.isnan(x)
        s = np.bincount(key[ok], weights=x[ok], minlength=R * NC).reshape(R, NC)
        c = np.bincount(key[ok], minlength=R * NC).reshape(R, NC)
        role_mean = s.sum(1) / c.sum(1)
        return (s + DAMAGE_PRIOR * role_mean[:, None]) / (c + DAMAGE_PRIOR)

    return Tables(p_side, delta, matchup, cross, syn, cgold, mgold, econ, scale,
                  profile(share), profile(tot / 1000.0), profile(G.taken.astype(np.float64) / 1000.0),
                  pick, prior)
