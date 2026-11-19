"""Trayectorias: limpieza de detecciones del minimapa + anclas de 60s de
Match-V5 (port fiel de la lógica auditada de macro_v5)."""
import json
import math
from pathlib import Path

from .config import TRAJ_MIN_SCORE
from .regions import norm_pos


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clean_trajectory(pts: list, own_base: tuple) -> list:
    """Descarta clusters incoherentes (falsos positivos aislados). pts:
    [(t, nx, ny)] ordenado."""
    out, i = [], 0
    while i < len(pts):
        j = i
        while j + 1 < len(pts) and dist(pts[j + 1][1:3], pts[j][1:3]) < 0.25:
            j += 1
        cl = pts[i:j + 1]
        prev_ok = not out or dist(out[-1][1:3], cl[0][1:3]) < 0.30
        if len(cl) > 2 or prev_ok or dist(cl[0][1:3], own_base) < 0.25:
            out += cl
        i = j + 1
    return out


def load_detections(traj_file: Path) -> dict:
    """{champ: [(t, nx, ny)]} filtrado por score, sin limpiar."""
    traj: dict[str, list] = {}
    with open(traj_file, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["score"] >= TRAJ_MIN_SCORE:
                traj.setdefault(r["champ"], []).append((r["t"], r["nx"], r["ny"]))
    return traj


def mv5_anchors(timeline: dict, participants: dict) -> dict:
    """{champ: [(t, nx, ny)]} desde participantFrames (posición exacta cada 60s)."""
    anchors: dict[str, list] = {}
    for f in timeline["info"]["frames"]:
        t = f["timestamp"] / 1000.0
        for pidstr, pf in f["participantFrames"].items():
            if "position" not in pf:
                continue
            champ = participants[int(pidstr)]["champ"]
            anchors.setdefault(champ, []).append((t, *norm_pos(pf["position"])))
    return anchors


def fuse(detections: dict, anchors: dict, enemies: set) -> dict:
    """Limpieza + fusión con anclas, por campeón."""
    traj = {}
    champs = set(detections) | set(anchors)
    for champ in champs:
        base = (1.0, 1.0) if champ in enemies else (0.0, 0.0)
        pts = clean_trajectory(sorted(detections.get(champ, [])), base)
        traj[champ] = sorted(pts + anchors.get(champ, []))
    return traj


def interp_at(traj_champ: list, t: float, max_gap: float = 8.0):
    """Posición interpolada linealmente en t, o None.

    Las tres pasadas muestrean por separado (~3s cada una), así que la muestra
    más cercana de la pasada omni puede tener segundos de desfase y un campeón
    en movimiento aparece lejos. Interpolando entre las dos muestras que
    rodean a t el error baja a unos pocos píxeles. Nunca se extrapola ni se
    cruzan huecos mayores que max_gap (muerte o pérdida de detección).
    """
    import bisect
    if not traj_champ:
        return None
    ts = [p[0] for p in traj_champ]
    i = bisect.bisect_left(ts, t)
    if i == 0:
        return (traj_champ[0][1], traj_champ[0][2]) if ts[0] - t <= 2.0 else None
    if i >= len(traj_champ):
        return (traj_champ[-1][1], traj_champ[-1][2]) if t - ts[-1] <= 2.0 else None
    t0, x0, y0 = traj_champ[i - 1]
    t1, x1, y1 = traj_champ[i]
    if t1 - t0 > max_gap:
        return None
    if t1 == t0:
        return (x0, y0)
    w = (t - t0) / (t1 - t0)
    return (x0 + w * (x1 - x0), y0 + w * (y1 - y0))


def pos_at_causal(traj_champ: list, t: float, max_age: float = 6.0):
    """Última posición conocida en t SIN mirar hacia delante.

    `pos_at` coge la muestra más cercana, que puede ser POSTERIOR a t: vale
    para reconstruir la verdad a posteriori, pero mete futuro en cualquier
    feature de inferencia. Todo lo que alimente al modelo usa esta.
    Devuelve (pos, antigüedad_s) o (None, None).
    """
    import bisect
    if not traj_champ:
        return None, None
    ts = [p[0] for p in traj_champ]
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        return None, None
    tp, x, y = traj_champ[i]
    age = t - tp
    if age > max_age:
        return None, None
    return (x, y), age


def pos_at(traj_champ: list, t: float, max_dt: float = 6.0):
    """Posición más cercana en el tiempo dentro de max_dt, o None (los huecos
    no se rellenan: muerta o sin detección, regla de fase 1)."""
    best, best_dt = None, max_dt + 1
    lo, hi = 0, len(traj_champ)
    # búsqueda binaria del vecino
    import bisect
    i = bisect.bisect_left([p[0] for p in traj_champ], t)
    for j in (i - 1, i, i + 1):
        if 0 <= j < len(traj_champ):
            dt = abs(traj_champ[j][0] - t)
            if dt < best_dt:
                best, best_dt = traj_champ[j], dt
    if best is not None and best_dt <= max_dt:
        return (best[1], best[2])
    return None
