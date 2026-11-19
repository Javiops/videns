"""Campamentos: serie de presencia del rombo naranja y muertes de campamento
(port fiel de build_camp_dots + la lógica de camp_kills de macro_v5)."""
import glob
import json
import os
from pathlib import Path

from .regions import all_camps, dist


def dot_series(pass_dir: Path, geom_at) -> dict:
    """Cuenta píxeles naranjas alrededor de cada campamento por frame.
    geom_at(t) -> (mx0, my0, msize) del minimapa dentro del PNG, o None si el
    PNG ya es el minimapa recortado (formato v2)."""
    import cv2
    camps = all_camps()
    series = {c: [] for c in camps}
    for fp in sorted(glob.glob(str(pass_dir / "mm_*.png"))):
        t = int(os.path.basename(fp)[3:9]) / 10.0
        img = cv2.imread(fp)
        if img is None:
            continue
        if geom_at is None:
            mx0, my0 = 0, 0
            msize = min(img.shape[0], img.shape[1])
        else:
            mx0, my0, msize = geom_at(t)
        if img.shape[0] < my0 + msize or img.shape[1] < mx0 + msize:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # ventana proporcional al tamaño del minimapa (fase 1: 10px sobre 508)
        r = max(6, int(round(msize * 10 / 508)))
        for c, (nx, ny) in camps.items():
            x, y = int(mx0 + nx * msize), int(my0 + (1 - ny) * msize)
            w = hsv[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            n = ((w[..., 0] >= 8) & (w[..., 0] <= 26)
                 & (w[..., 1] > 140) & (w[..., 2] > 160)).sum()
            series[c].append((t, int(n)))
    return series


def occupied(traj: dict, t: float, cpos: tuple, rad: float = 0.09) -> bool:
    for pts in traj.values():
        for pt in pts:
            if abs(pt[0] - t) <= 3 and dist(pt[1:3], cpos) < rad:
                return True
    return False


def camp_kills(dots: dict, traj: dict) -> list:
    """[(t, camp)] — muerte de campamento por histéresis del rombo, ignorando
    frames con un campeón encima (el icono tapa el rombo)."""
    camps = all_camps()
    kills = []
    for camp, ser in dots.items():
        cpos = camps[camp]
        alive, prev_absent = True, False
        for t, n in ser:
            if t < 105 or occupied(traj, t, cpos):
                prev_absent = False
                continue
            if n >= 5:
                alive, prev_absent = True, False
            elif n < 3:
                if alive and prev_absent:
                    alive = False
                    kills.append((t, camp))
                prev_absent = True
            else:
                prev_absent = False
    kills.sort()
    return kills


def alive_series(dots: dict, traj: dict) -> dict:
    """{camp: [(t, alive)]} — estado suavizado con la misma histéresis, para
    snapshots (alive=None al inicio hasta el primer dato firme)."""
    camps = all_camps()
    out = {}
    for camp, ser in dots.items():
        cpos = camps[camp]
        alive = None
        prev_absent = False
        states = []
        for t, n in ser:
            if t < 105 or occupied(traj, t, cpos):
                prev_absent = False
            elif n >= 5:
                alive, prev_absent = True, False
            elif n < 3:
                if prev_absent:
                    alive = False
                prev_absent = True
            else:
                prev_absent = False
            states.append((t, alive))
        out[camp] = states
    return out


def alive_at(states: dict, t: float) -> dict:
    """{camp: True/False/None} en el instante t (último estado conocido)."""
    import bisect
    out = {}
    for camp, ser in states.items():
        if not ser:
            out[camp] = None
            continue
        ts = [s[0] for s in ser]
        i = bisect.bisect_right(ts, t) - 1
        out[camp] = ser[i][1] if i >= 0 else None
    return out
