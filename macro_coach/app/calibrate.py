"""Calibración de la geometría del minimapa (dónde está y qué tamaño tiene).

Fase 1 asumía un recorte fijo (esquina inferior derecha, 508px). En fase 2 el
minimapa de la replay aparece CENTRADO y puede tener otro tamaño, así que la
geometría no se asume nunca: se mide al inicio de cada pasada.

Método: correlación multiescala del fondo del mapa (`data/map_background.png`)
contra la pantalla. Los iconos encima (campeones, torres, niebla) no impiden
que el terreno domine la correlación.

Descartados por no fiables (docs/history.md):
- Contornos/Canny: eligen cuadrados distintos según el contenido del momento
  (418 / 462 / 483 px en la misma partida) — geometría plausible pero falsa.
- Ajuste por mínimos cuadrados contra las posiciones de Match-V5: los frames
  son de 60s y los campeones se mueven entre el frame y la captura; sirve para
  VERIFICAR (los quietos casan a 1-10px) pero no para ajustar.
"""
import cv2
import numpy as np

from .config import DATA

BG_FILE = DATA / "map_background.png"
# El minimapa por defecto de una replay ronda los 270px; el del owner, movido
# y ampliado, 508px. El rango tiene que cubrir ambos: con SIZE_MIN=340 la
# correlacion caia a 0.24-0.27 simplemente porque nunca se probaba el tamano
# correcto, y el mensaje de error culpaba al arte del mapa.
SIZE_MIN, SIZE_MAX = 200, 680
MIN_SCORE = 0.45
VERIFY_TOL_PX = 14.0        # tolerancia para contar un campeón como "casa"
VERIFY_MIN_OK = 3           # campeones quietos que deben casar


def _background():
    bg = cv2.imread(str(BG_FILE))
    if bg is None:
        raise FileNotFoundError(
            f"falta {BG_FILE}: es la plantilla de calibración. Regenérala con "
            "scripts/make_map_background.py sobre una pasada existente.")
    return bg


def find_minimap(screen_bgr, coarse_step: int = 6) -> dict:
    """{'x0','y0','size','score'} por correlación multiescala. Grueso y fino."""
    bg = _background()
    H, W = screen_bgr.shape[:2]

    def best_at(sizes):
        best = None
        for size in sizes:
            if size < 40 or size > min(H, W):
                continue
            tpl = cv2.resize(bg, (size, size), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(screen_bgr, tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (float(mx), int(size), (int(loc[0]), int(loc[1])))
        return best

    coarse = best_at(range(SIZE_MIN, SIZE_MAX + 1, coarse_step))
    if coarse is None:
        raise RuntimeError("calibración: sin candidatos de tamaño")
    fine = best_at(range(max(SIZE_MIN, coarse[1] - coarse_step),
                         min(SIZE_MAX, coarse[1] + coarse_step) + 1))
    score, size, (x0, y0) = fine if fine[0] >= coarse[0] else coarse
    if score < MIN_SCORE:
        raise RuntimeError(
            f"calibración: correlación {score:.2f} < {MIN_SCORE}. ¿El minimapa "
            "está oculto, o cambió el arte del mapa con el parche? "
            "(regenera data/map_background.png)")
    return {"x0": x0, "y0": y0, "size": size, "score": round(score, 3)}


def crop(screen_bgr, geom: dict):
    return screen_bgr[geom["y0"]:geom["y0"] + geom["size"],
                      geom["x0"]:geom["x0"] + geom["size"]]


def still_aligned(mm, min_score: float = 0.35) -> float:
    """Correlación del recorte actual contra el fondo del mapa a su tamaño.

    Comprobación barata (un matchTemplate del mismo tamaño) para detectar que
    el minimapa se ha movido o cambiado de tamaño A MITAD DE PASADA: si eso
    pasa, todos los frames siguientes se recortarían del sitio equivocado sin
    que nada falle. Devuelve la correlación; el llamante decide."""
    bg = _background()
    n = mm.shape[0]
    if n < 40 or mm.shape[1] != n:
        return 0.0
    tpl = cv2.resize(bg, (n, n), interpolation=cv2.INTER_AREA)
    res = cv2.matchTemplate(mm, tpl, cv2.TM_CCOEFF_NORMED)
    return float(np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).max())


def verify_with_truth(screen_bgr, geom: dict, champ_positions: dict,
                      templates: dict, teams: dict) -> dict:
    """Contrasta la geometría con Match-V5: detecta iconos y mide el error de
    los que mejor casan. Los campeones que se movieron desde el frame de 60s
    dan error grande por definición, así que se exige que los MEJORES casen.
    """
    from . import detect
    mm = crop(screen_bgr, geom)
    size = geom["size"]
    hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
    seen = {}
    for team in ("ORDER", "CHAOS"):
        names = [c for c, t in teams.items() if t == team]
        for d in detect.detect_frame(mm, hsv, team, names, templates, size):
            seen[d["champ"]] = (d["cx"] / size, 1 - d["cy"] / size)
    errs = []
    for champ, (nx, ny) in champ_positions.items():
        if champ in seen:
            dx, dy = seen[champ][0] - nx, seen[champ][1] - ny
            errs.append((float(np.hypot(dx, dy)) * size, champ))
    errs.sort()
    ok = sum(1 for e, _ in errs if e <= VERIFY_TOL_PX)
    return {"n_detected": len(seen), "n_match": ok,
            "best_err_px": [round(e, 1) for e, _ in errs[:5]],
            "ok": ok >= VERIFY_MIN_OK}


def champ_positions_at(timeline: dict, participants: dict, t_s: float) -> tuple:
    """(t_frame, {champ: (nx, ny)}) del frame de 60s más cercano a t_s."""
    from .regions import norm_pos
    best, best_dt = None, None
    for f in timeline["info"]["frames"]:
        ft = f["timestamp"] / 1000.0
        dt = abs(ft - t_s)
        if best_dt is None or dt < best_dt:
            best, best_dt = f, dt
    out = {}
    for pidstr, pf in best["participantFrames"].items():
        if "position" not in pf:
            continue
        out[participants[int(pidstr)]["champ"]] = norm_pos(pf["position"])
    return best["timestamp"] / 1000.0, out
