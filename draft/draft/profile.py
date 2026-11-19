"""Champion-role profiles for the tool: expected CS at a minute and damage composition.

Both are shrunk means per (role, champion) toward the pick-weighted role mean (prior weight
K_PRIOR games). Champion id 0 (an empty slot) gets the role mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import NC, Games

R = 5
K_PRIOR = 30.0


def role_means(values: np.ndarray, champ: np.ndarray, k: float = K_PRIOR) -> np.ndarray:
    """values, champ: (n, 2, 5); NaN values are ignored. Returns (R, NC)."""
    out = np.zeros((R, NC))
    for r in range(R):
        v, c = values[:, :, r].ravel(), champ[:, :, r].ravel().astype(np.int64)
        ok = ~np.isnan(v)
        s = np.bincount(c[ok], weights=v[ok], minlength=NC)
        n = np.bincount(c[ok], minlength=NC).astype(float)
        mean = s.sum() / n.sum()
        out[r] = (s + k * mean) / (n + k)
    return out


@dataclass
class DamageProfile:
    phys: np.ndarray   # (R, NC) physical damage to champions per minute
    magic: np.ndarray
    true: np.ndarray

    def team(self, champ: np.ndarray) -> np.ndarray:
        """champ (n, 2, 5) -> (n, 2, 3) shares of physical / magic / true damage per team."""
        ridx = np.arange(R)
        parts = np.stack([self.phys[ridx, champ], self.magic[ridx, champ], self.true[ridx, champ]], -1).sum(2)
        return parts / parts.sum(-1, keepdims=True)


def build_damage_profile(G: Games) -> DamageProfile:
    per_min = (G.dur / 60.0)[:, None, None]
    return DamageProfile(role_means(G.pdmg / per_min, G.champ), role_means(G.mdmg / per_min, G.champ),
                         role_means(G.tdmg / per_min, G.champ))


def actual_team_shares(G: Games) -> np.ndarray:
    parts = np.stack([G.pdmg.sum(2), G.mdmg.sum(2), G.tdmg.sum(2)], -1).astype(float)
    return parts / parts.sum(-1, keepdims=True)


def build_cs_profile(G: Games, cs: np.ndarray) -> np.ndarray:
    """cs (n, 2, 5) creep score at one minute (NaN when the game did not reach it) -> (R, NC)."""
    return role_means(cs, G.champ)
