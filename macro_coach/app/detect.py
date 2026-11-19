"""Detector de campeones sobre frames del minimapa (port del v5 de fase 1).

Dos modos:
- detect_frame: detección global (pasada omni) — anillos por color ->
  componentes -> clasificación local contra retratos.
- verify_at: verificación local (pasadas con niebla) — sabemos dónde está
  realmente el campeón (trayectoria omni); solo comprobamos si su icono
  aparece ahí. Umbral más laxo porque el prior es fuerte.

Retratos: data/portraits_hud/<champ>.png > data/portraits/<champ>.png >
Data Dragon (a escala). El umbral 0.87 se auditó con retratos locales; la
puerta de QA de process_game delata si los de ddragon rinden peor.
"""
from pathlib import Path

import cv2
import numpy as np

from .config import (DATA, DET_CLS_MIN, DET_NMS_DIST, DET_MIN_AREA,
                     DET_SIZES, DET_SIZE_FRACS, DET_ZOOM, VIS_VERIFY_SCORE,
                     VIS_VERIFY_WIN_PX)


def sizes_for(minimap_size: int | None) -> tuple:
    """Tamanos de icono a probar. El icono escala con el minimapa, asi que se
    derivan de su lado; sin tamano conocido, los fijos de fase 1."""
    if not minimap_size:
        return DET_SIZES
    out = sorted({max(12, int(round(minimap_size * f))) for f in DET_SIZE_FRACS})
    return tuple(out)


def portrait_path(champ: str, patch: str | None = None) -> Path:
    for cand in (DATA / "portraits_hud" / f"{champ}.png",
                 DATA / "portraits" / f"{champ}.png"):
        if cand.exists():
            return cand
    if patch:
        from . import ddragon
        return ddragon.portrait(ddragon.version_for_patch(patch), champ)
    raise FileNotFoundError(f"sin retrato para {champ} y sin parche para ddragon")


def build_templates(champs: list, patch: str | None = None,
                    minimap_size: int | None = None) -> dict:
    out = {}
    sizes = sizes_for(minimap_size)
    for champ in champs:
        img = cv2.imread(str(portrait_path(champ, patch)))
        if img is None:
            raise FileNotFoundError(f"retrato ilegible: {champ}")
        h, w = img.shape[:2]
        m = int(h * (1 - 1 / DET_ZOOM) / 2)
        img = img[m:h - m, m:w - m]
        per = []
        for s in sizes:
            t = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            mask = np.zeros((s, s), np.uint8)
            cv2.circle(mask, (s // 2, s // 2), int(s * 0.40), 255, -1)
            per.append((s, t, mask))
        out[champ] = per
    return out


def team_mask(hsv, team: str):
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if team == "ORDER":
        m = (hue >= 92) & (hue <= 112) & (sat > 120) & (val > 120)
    else:
        m = ((hue >= 170) | (hue <= 4)) & (sat > 120) & (val > 120)
    return cv2.dilate((m.astype(np.uint8)) * 255, np.ones((3, 3), np.uint8))


def team_pixels(hsv, team: str) -> int:
    return int((team_mask(hsv, team) > 0).sum())


def detect_frame(mm, hsv, team: str, champs: list, templates: dict,
                 msize: int) -> list:
    """Detección global de un equipo en un frame (port fiel)."""
    if not champs:
        return []
    mask = team_mask(hsv, team)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    hits = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < DET_MIN_AREA:
            continue
        pad = 28
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(msize, x + w + pad), min(msize, y + h + pad)
        patch = mm[y0:y1, x0:x1]
        if patch.shape[0] < 40 or patch.shape[1] < 40:
            continue
        for champ in champs:
            best = None
            for s, tpl, tmask in templates[champ]:
                if patch.shape[0] < s or patch.shape[1] < s:
                    continue
                res = cv2.matchTemplate(patch, tpl, cv2.TM_CCORR_NORMED, mask=tmask)
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                _, sc, _, loc = cv2.minMaxLoc(res)
                if best is None or sc > best[0]:
                    best = (float(sc), x0 + loc[0] + s // 2, y0 + loc[1] + s // 2)
            if best and best[0] >= DET_CLS_MIN:
                hits.append((best[0], champ, best[1], best[2]))
    hits.sort(reverse=True)
    dets, used_c, used_p = [], set(), []
    for sc, champ, cx, cy in hits:
        if champ in used_c:
            continue
        if any((cx - ux) ** 2 + (cy - uy) ** 2 < DET_NMS_DIST ** 2 for ux, uy in used_p):
            continue
        used_c.add(champ)
        used_p.append((cx, cy))
        dets.append({"champ": champ, "cx": cx, "cy": cy, "score": round(sc, 3)})
    return dets


def verify_at(mm, champ: str, templates: dict, px: int, py: int,
              msize: int) -> float:
    """Mejor score del campeón en una ventana alrededor de (px, py) en píxeles
    del minimapa. Para pasadas con niebla: >= VIS_VERIFY_SCORE = visible."""
    win = VIS_VERIFY_WIN_PX
    best = 0.0
    for s, tpl, tmask in templates[champ]:
        half = s // 2 + win
        x0, y0 = max(0, px - half), max(0, py - half)
        x1, y1 = min(msize, px + half), min(msize, py + half)
        patch = mm[y0:y1, x0:x1]
        if patch.shape[0] < s or patch.shape[1] < s:
            continue
        res = cv2.matchTemplate(patch, tpl, cv2.TM_CCORR_NORMED, mask=tmask)
        res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
        _, sc, _, _ = cv2.minMaxLoc(res)
        best = max(best, float(sc))
    return best


VERIFY_THRESHOLD = VIS_VERIFY_SCORE
