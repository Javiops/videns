"""Máscara de 13 regiones (fase 1) y utilidades de geometría del mapa.

Coordenadas normalizadas nx,ny en [0,1] con origen abajo-izquierda (como
Match-V5 tras dividir por MAP_MAX). La máscara .npy está en filas de imagen
(origen arriba-izquierda): region() hace la conversión.
"""
import math
from functools import lru_cache

import numpy as np

from .config import LBN, REGION_MASK_FILE, CAMPS_BLUE


@lru_cache(maxsize=1)
def mask():
    m = np.load(REGION_MASK_FILE)
    if m.shape[0] != m.shape[1]:
        raise ValueError(f"máscara no cuadrada: {m.shape}")
    return m


def region(nx: float, ny: float) -> str:
    m = mask()
    n = m.shape[0]
    x = min(max(int(nx * n), 0), n - 1)
    y = min(max(int((1 - ny) * n), 0), n - 1)
    return LBN.get(int(m[y, x]), "JGL_PA")


def all_camps() -> dict:
    """12 campamentos: 6 azules + espejo rojo ('<nombre> enemigo')."""
    out = dict(CAMPS_BLUE)
    out.update({k + " enemigo": (1 - x, 1 - y) for k, (x, y) in CAMPS_BLUE.items()})
    return out


def quad_of_camp(camp: str) -> str:
    x, y = all_camps()[camp]
    if x + y < 1:
        return "JGL_PA" if y > x else "JGL_PR"
    return "JGL_EA" if y < x else "JGL_ER"


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm_pos(p: dict) -> tuple:
    from .config import MAP_MAX
    return (min(max(p["x"] / MAP_MAX, 0.0), 1.0),
            min(max(p["y"] / MAP_MAX, 0.0), 1.0))


def own_base(team_id: int) -> tuple:
    """Esquina de la fuente: azul (100) abajo-izquierda, rojo (200) arriba-derecha."""
    return (0.0, 0.0) if team_id == 100 else (1.0, 1.0)


def mirror_region(reg: str) -> str:
    """La región vista desde el equipo rojo (la máscara está en frame azul)."""
    swap = {"BASE": "BASE_ENEMIGA", "BASE_ENEMIGA": "BASE",
            "JGL_PA": "JGL_EA", "JGL_EA": "JGL_PA",
            "JGL_PR": "JGL_ER", "JGL_ER": "JGL_PR"}
    return swap.get(reg, reg)


def region_for_team(nx: float, ny: float, team_id: int) -> str:
    r = region(nx, ny)
    return r if team_id == 100 else mirror_region(r)


def validate_camps() -> list:
    """Los 12 campamentos deben caer en su cuadrante de jungla (chequeo fase 1)."""
    bad = []
    for camp in all_camps():
        cx, cy = all_camps()[camp]
        reg = region(cx, cy)
        if reg != quad_of_camp(camp):
            bad.append((camp, reg, quad_of_camp(camp)))
    return bad
