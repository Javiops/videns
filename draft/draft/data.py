"""Chronological split and array loader for extracted draft records.

The confirmation role is never materialised unless `allow_confirm=True`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ASG = {"PRIMARY": 0, "SECONDARY": 1, "AUTOFILL": 2}
NC = 1000  # champion ids are < 1000; id 0 is the "average champion" sentinel
FRACTIONS = (0.70, 0.85)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(extract_path: Path, out_path: Path) -> dict:
    keys = []
    with gzip.open(extract_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            keys.append((r["t"], r["id"]))
    keys.sort()
    n = len(keys)
    b1, b2 = int(n * FRACTIONS[0]), int(n * FRACTIONS[1])
    while 0 < b1 < n and keys[b1][0] == keys[b1 - 1][0]:
        b1 += 1
    while 0 < b2 < n and keys[b2][0] == keys[b2 - 1][0]:
        b2 += 1
    roles = {mid: ("fit" if i < b1 else "dev" if i < b2 else "confirm") for i, (_, mid) in enumerate(keys)}
    bounds = {"fit": (keys[0][0], keys[b1 - 1][0]), "dev": (keys[b1][0], keys[b2 - 1][0]),
              "confirm": (keys[b2][0], keys[-1][0])}
    split = {"schema": "draft-split-v1", "extract_sha256": sha256_file(extract_path),
             "counts": {"fit": b1, "dev": b2 - b1, "confirm": n - b2},
             "time_bounds_ms": bounds, "roles": roles}
    raw = (json.dumps(split, sort_keys=True) + "\n").encode()
    out_path.write_bytes(raw)
    out_path.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()} {out_path.name}\n",
                                               encoding="ascii")
    return {k: v for k, v in split.items() if k != "roles"}


@dataclass
class Games:
    ids: np.ndarray
    role: np.ndarray        # 'fit' / 'dev'
    t: np.ndarray
    patch: np.ndarray
    dur: np.ndarray
    blue_win: np.ndarray
    tl: np.ndarray
    champ: np.ndarray       # (n, 2, 5) int
    player: np.ndarray      # (n, 2, 5) int
    asg: np.ndarray         # (n, 2, 5) int, 3 = unknown
    pdmg: np.ndarray
    mdmg: np.ndarray
    tdmg: np.ndarray
    taken: np.ndarray
    g12: np.ndarray         # NaN when no timeline frame
    bans: np.ndarray        # (n, 2, 5) int, 0 = none
    exp_champ: np.ndarray   # prior archive games of this player on this champion
    exp_total: np.ndarray   # prior archive games of this player

    def subset(self, idx) -> "Games":
        return Games(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})

    def __len__(self) -> int:
        return len(self.ids)


def load_games(extract_path: Path, split_path: Path, roles=("fit", "dev"), allow_confirm=False) -> Games:
    if "confirm" in roles and not allow_confirm:
        raise PermissionError("confirmation games are sealed in protocol v1")
    tokens = split_path.with_suffix(".sha256").read_text(encoding="ascii").split()
    if tokens[0] != sha256_file(split_path):
        raise ValueError("split sidecar mismatch")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split["extract_sha256"] != sha256_file(extract_path):
        raise ValueError("extract changed since the split was frozen")
    wanted = set(roles)
    recs = []
    with gzip.open(extract_path, "rt", encoding="utf-8") as f:
        for line in f:
            # role is decided from the frozen split before the record is kept
            mid = line[7:line.index('"', 7)]
            if split["roles"][mid] not in wanted:
                continue
            recs.append(json.loads(line))
    recs.sort(key=lambda r: (r["t"], r["id"]))
    n = len(recs)
    players: dict[str, int] = {}
    arr = {k: np.zeros((n, 2, 5), dtype=np.float32) for k in ("pdmg", "mdmg", "tdmg", "taken", "g12")}
    champ = np.zeros((n, 2, 5), dtype=np.int32)
    player = np.zeros((n, 2, 5), dtype=np.int32)
    asg = np.full((n, 2, 5), 3, dtype=np.int8)
    bans = np.zeros((n, 2, 5), dtype=np.int32)
    for g, r in enumerate(recs):
        for j, p in enumerate(r["p"]):
            s, l = p["side"], ROLES.index(p["role"])
            champ[g, s, l] = p["champ"]
            player[g, s, l] = players.setdefault(p["pl"], len(players))
            asg[g, s, l] = ASG.get(p["asg"], 3)
            for k in ("pdmg", "mdmg", "tdmg", "taken"):
                arr[k][g, s, l] = p[k]
            arr["g12"][g, s, l] = np.nan if p["g12"] is None else p["g12"]
        for s in (0, 1):
            b = [c for c in r["bans"][s] if c > 0][:5]
            bans[g, s, :len(b)] = b
    exp_champ = np.zeros((n, 2, 5), dtype=np.float32)
    exp_total = np.zeros((n, 2, 5), dtype=np.float32)
    per_champ: dict[tuple[int, int], int] = {}
    per_player: dict[int, int] = {}
    for g in range(n):  # chronological: counts use strictly earlier games only
        for s in (0, 1):
            for l in range(5):
                key = (int(player[g, s, l]), int(champ[g, s, l]))
                exp_champ[g, s, l] = per_champ.get(key, 0)
                exp_total[g, s, l] = per_player.get(key[0], 0)
        for s in (0, 1):
            for l in range(5):
                key = (int(player[g, s, l]), int(champ[g, s, l]))
                per_champ[key] = per_champ.get(key, 0) + 1
                per_player[key[0]] = per_player.get(key[0], 0) + 1
    return Games(
        ids=np.array([r["id"] for r in recs]), role=np.array([split["roles"][r["id"]] for r in recs]),
        t=np.array([r["t"] for r in recs], dtype=np.int64), patch=np.array([r["patch"] for r in recs]),
        dur=np.array([r["dur"] for r in recs], dtype=np.float32),
        blue_win=np.array([r["blue_win"] for r in recs], dtype=np.int8),
        tl=np.array([r["tl"] for r in recs]), champ=champ, player=player, asg=asg,
        bans=bans, exp_champ=exp_champ, exp_total=exp_total, **arr)
