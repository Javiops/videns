"""Iconos reales del minimapa, aprendidos de la pasada omni.

Los retratos de Data Dragon no distinguen bien: con ellos el detector
"encuentra" campeones MUERTOS (que no están en el minimapa) en el 80-87% de
las muestras, porque el retrato de uno casa con el icono de otro. Medido en
EUW1_7986655074; ver docs/history.md.

Aquí se recorta el icono tal y como el juego lo dibuja — con su anillo de
equipo, su tamaño y su borde — usando la pasada SIN NIEBLA, donde sabemos que
todos los vivos están dibujados y dónde están (trayectoria + anclas de
Match-V5). La mediana de muchos recortes limpia el terreno de debajo y deja el
icono.

El icono resultante es el patrón para preguntar, en las pasadas CON NIEBLA,
"¿está este campeón dibujado aquí?".
"""
import glob
import os

import cv2
import numpy as np

from .trajectory import interp_at

PATCH = 56                  # lado del recorte, en px de minimapa de 508
MIN_SAMPLES = 8
MAX_SAMPLES = 80
ISOLATION = 0.055           # nadie más a menos de esto (normalizado)


def _patch_size(msize: int) -> int:
    s = int(round(PATCH * msize / 508.0))
    return s + (s % 2)      # par


def isolated(champ: str, t: float, traj: dict, radius: float = ISOLATION) -> bool:
    """True si no hay ningún otro campeón cerca (para no aprender su icono)."""
    me = interp_at(traj.get(champ, []), t)
    if me is None:
        return False
    for other, pts in traj.items():
        if other == champ:
            continue
        p = interp_at(pts, t)
        if p is None:
            continue
        if (p[0] - me[0]) ** 2 + (p[1] - me[1]) ** 2 < radius ** 2:
            return False
    return True


def _collect(pass_dir, traj: dict, champs: list, dead_at, msize: int,
             radius: float) -> dict:
    side = _patch_size(msize)
    half = side // 2
    buckets = {c: [] for c in champs}
    for fp in sorted(glob.glob(str(pass_dir / "mm_*.png"))):
        t = int(os.path.basename(fp)[3:9]) / 10.0
        if all(len(buckets[c]) >= MAX_SAMPLES for c in champs):
            break
        img = None
        for champ in champs:
            if len(buckets[champ]) >= MAX_SAMPLES or dead_at(champ, t):
                continue
            if not isolated(champ, t, traj, radius):
                continue
            p = interp_at(traj.get(champ, []), t)
            if p is None:
                continue
            if img is None:
                img = cv2.imread(fp)
                if img is None:
                    break
                if img.shape[0] != msize:
                    img = img[:msize, :msize]
            cx, cy = int(p[0] * msize), int((1 - p[1]) * msize)
            if cx - half < 0 or cy - half < 0 or cx + half > msize or cy + half > msize:
                continue
            buckets[champ].append(img[cy - half:cy + half, cx - half:cx + half])
    return buckets


def extract(pass_dir, traj: dict, champs: list, dead_at, msize: int) -> tuple:
    """({champ: icono BGR}, {champ: n_muestras}).

    Se relaja el aislamiento por campeón si no reúne muestras suficientes: un
    jungla que acompaña al equipo, o un support pegado a su ADC, casi nunca
    está solo y se quedaba sin icono (le pasó a Shaco en la primera partida).
    """
    icons, counts = {}, {}
    pending = list(champs)
    for radius in (ISOLATION, 0.040, 0.030):
        if not pending:
            break
        buckets = _collect(pass_dir, traj, pending, dead_at, msize, radius)
        still = []
        for champ, patches in buckets.items():
            counts[champ] = max(counts.get(champ, 0), len(patches))
            if len(patches) >= MIN_SAMPLES:
                icons[champ] = np.median(np.stack(patches), axis=0).astype(np.uint8)
            else:
                still.append(champ)
        pending = still
    return icons, counts


# Barrido medido sobre EUW1_7986655074 (1059 muestras por configuración):
# CCOEFF con disco 0.42 y ventana ±26px da la mejor sensibilidad con <=2% de
# falsos. Ni la búsqueda global (más falsos) ni ventanas mayores mejoran.
MASK_FRAC = 0.42
WIN_PX = 26
MATCH_THRESHOLD = 0.50


def disk_mask(side: int, frac: float = MASK_FRAC) -> np.ndarray:
    m = np.zeros((side, side), np.uint8)
    cv2.circle(m, (side // 2, side // 2), int(side * frac), 255, -1)
    return m


def match_at(mm, icon, px: int, py: int, win: int = WIN_PX) -> float:
    """Mejor correlación del icono en una ventana alrededor de (px, py)."""
    side = icon.shape[0]
    half = side // 2
    x0, y0 = max(0, px - half - win), max(0, py - half - win)
    x1 = min(mm.shape[1], px + half + win)
    y1 = min(mm.shape[0], py + half + win)
    patch = mm[y0:y1, x0:x1]
    if patch.shape[0] < side or patch.shape[1] < side:
        return 0.0
    res = cv2.matchTemplate(patch, icon, cv2.TM_CCOEFF_NORMED,
                            mask=disk_mask(side))
    res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
    return float(cv2.minMaxLoc(res)[1])


def detect_rates(pass_dir, traj: dict, champs: list, icons_by_champ: dict,
                 dead_at, msize: int, stride: int = 1) -> tuple:
    """({champ: [t visible]}, {champ: n_muestras}) en una pasada."""
    seen = {c: [] for c in champs}
    n = {c: 0 for c in champs}
    for fp in sorted(glob.glob(str(pass_dir / "mm_*.png")))[::stride]:
        t = int(os.path.basename(fp)[3:9]) / 10.0
        mm = None
        for champ in champs:
            if champ not in icons_by_champ or dead_at(champ, t):
                continue
            p = interp_at(traj.get(champ, []), t)
            if p is None:
                continue
            if mm is None:
                mm = cv2.imread(fp)
                if mm is None:
                    break
            n[champ] += 1
            px, py = int(p[0] * msize), int((1 - p[1]) * msize)
            if match_at(mm, icons_by_champ[champ], px, py) >= MATCH_THRESHOLD:
                seen[champ].append(t)
    return seen, n
