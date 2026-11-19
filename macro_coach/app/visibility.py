"""Visibilidad por equipo y métrica de información.

La verdad de visibilidad sale EXCLUSIVAMENTE de las pasadas con niebla: para
cada frame de `pass_<team>` se comprueba si el icono aprendido del enemigo está
dibujado donde sabemos que está. Nunca se aproxima desde datos omniscientes
(regla 2 de CLAUDE.md).

## Qué significa `conf` y qué NO significa

`conf` es **la frescura del último avistamiento CONFIRMADO**, no la
probabilidad de que el enemigo esté oculto:

- muerto → 100 (la posición es pública: fuente + respawn);
- avistado hace ≤ `CONF_VISIBLE_GRACE_S` → 100;
- si no, decaimiento lineal hasta 0 en `CONF_HORIZON_S`.

El detector tiene falsos negativos (~20% medido), así que una no-detección
aislada **no** prueba que el enemigo esté oculto. Por eso:

1. la sensibilidad se mide POR CAMPEÓN en la pasada de su propio equipo (donde
   siempre es visible) y, si baja de `VIS_MIN_SENSITIVITY`, su visibilidad se
   marca **desconocida** (`None`) en vez de "no visible";
2. una pasada sospechosa (`suspect_no_fog`, o ambas pasadas con la misma
   niebla) invalida la visibilidad entera de esa partida;
3. la corrección agregada `visible_frac_corrected` es una **frecuencia**: sirve
   para auditar la partida, nunca para inventar visibilidad instante a
   instante. Aquí no se usa para eso.

Todo lo desconocido se representa con `None`, jamás con un número plausible.
"""
import bisect
import json
from pathlib import Path

from .config import (CONF_HORIZON_S, CONF_VISIBLE_GRACE_S,
                     VIS_MIN_SENSITIVITY, VIS_MIN_USABLE_ENEMIES)


def intervals_from_times(times: list, gap: float = CONF_VISIBLE_GRACE_S) -> list:
    """Colapsa instantes de avistamiento en intervalos [(a, b)]."""
    out = []
    for t in sorted(times):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [tuple(iv) for iv in out]


def sight_state(intervals: list, t: float) -> tuple:
    """(visible_ahora, t_del_ultimo_avistamiento <= t)."""
    last = None
    for a, b in intervals:
        if a > t:
            break
        if a <= t <= b + CONF_VISIBLE_GRACE_S:
            return True, min(t, b)
        last = b
    return False, last


def confidence(t: float, intervals: list, dead: bool) -> float:
    """0-100: frescura del último avistamiento confirmado (ver cabecera)."""
    if dead:
        return 100.0
    visible, last = sight_state(intervals, t)
    if visible:
        return 100.0
    if last is None:
        return 0.0
    return max(0.0, 100.0 * (1.0 - (t - last) / CONF_HORIZON_S))


class TeamVisibility:
    """Visibilidad de los 5 enemigos desde un equipo.

    sensitivity: {champ: 0..1} medida en la pasada del propio equipo del
    campeón. usable: False invalida la partida entera (pasada sospechosa).
    """

    def __init__(self, enemy_intervals: dict, deaths: list,
                 sensitivity: dict | None = None, usable: bool = True,
                 spawn_seed: bool = True, reason: str = ""):
        self.iv = {}
        for champ, ivs in enemy_intervals.items():
            ivs = sorted(tuple(x) for x in ivs)
            if spawn_seed:
                ivs = [(0.0, 0.0)] + ivs
            self.iv[champ] = ivs
        self.deaths = deaths
        self._death_ts = [d[0] for d in deaths]
        self.sensitivity = sensitivity or {}
        self.usable = bool(usable)
        self.reason = reason

    # --- calidad ---
    def champ_usable(self, champ: str) -> bool:
        if not self.usable:
            return False
        s = self.sensitivity.get(champ)
        return s is not None and s >= VIS_MIN_SENSITIVITY

    def dead_at(self, champ: str, t: float) -> bool:
        """Última muestra PASADA: nunca mirar hacia delante."""
        if not self.deaths:
            return False
        i = bisect.bisect_right(self._death_ts, t) - 1
        if i < 0:
            return False
        return bool(self.deaths[i][1].get(champ, False))

    def enemy(self, champ: str, t: float) -> dict:
        """Estado de un enemigo. `known=False` -> conf/visible/last_seen_t son
        None: no sabemos si está oculto o si el detector lo perdió."""
        dead = self.dead_at(champ, t)
        sens = self.sensitivity.get(champ)
        if not self.champ_usable(champ) and not dead:
            return {"champ": champ, "known": False, "conf": None,
                    "visible": None, "dead": dead, "last_seen_t": None,
                    "sensitivity": sens}
        ivs = self.iv.get(champ, [(0.0, 0.0)])
        visible, last = sight_state(ivs, t)
        return {"champ": champ, "known": True,
                "conf": round(confidence(t, ivs, dead), 1),
                "visible": bool(visible), "dead": dead,
                "last_seen_t": (round(last, 2) if last is not None else None),
                "sensitivity": sens}

    def info(self, t: float) -> dict:
        """info_pct = media de los enemigos CONOCIDOS. Si hay menos de
        `VIS_MIN_USABLE_ENEMIES`, no se publica número: None."""
        per = [self.enemy(c, t) for c in sorted(self.iv)]
        known = [e for e in per if e["known"]]
        pct = (round(sum(e["conf"] for e in known) / len(known), 1)
               if len(known) >= VIS_MIN_USABLE_ENEMIES else None)
        return {"info_pct": pct, "n_known": len(known), "n_enemies": len(per),
                "usable": self.usable and pct is not None,
                "reason": self.reason, "enemies": per}


def sensitivity_from_meta(meta: dict, mode: str) -> dict:
    """{champ: sensibilidad} desde el meta de visibility.json."""
    per = (meta or {}).get(mode, {}).get("per_champion", {}) or {}
    return {c: v.get("sensibilidad") for c, v in per.items()}


def mode_usable(meta: dict, mode: str) -> tuple:
    """(usable, motivo) de una pasada según su meta de auditoría."""
    m = (meta or {}).get(mode, {}) or {}
    if m.get("suspect_no_fog"):
        return False, "pasada marcada suspect_no_fog (los 5 rivales >90% visibles)"
    if m.get("same_fog_as_other"):
        return False, "las dos pasadas llevan la misma niebla"
    if not m:
        return False, "sin metadatos de visibilidad"
    return True, ""


def load(vis_file: Path) -> dict:
    """{'blue': {champ: [(a,b)]}, 'red': {...}, 'meta': {...}}.

    `meta` se CONSERVA: lleva la sensibilidad por campeón y las marcas de
    sospecha, sin las cuales no se puede decidir si la visibilidad es
    utilizable (antes se descartaba y el snapshot recibía un número siempre).
    """
    raw = json.loads(Path(vis_file).read_text(encoding="utf-8"))
    out = {team: {c: [tuple(iv) for iv in ivs] for c, ivs in d.items()}
           for team, d in raw.items() if team in ("blue", "red")}
    out["meta"] = raw.get("meta", {})
    return out


def team_visibility_for(vis_data: dict, team_id: int, deaths: list):
    """TeamVisibility del equipo observador, ya con sensibilidad y marcas."""
    if not vis_data:
        return None
    mode = "blue" if team_id == 100 else "red"
    if mode not in vis_data:
        return None
    meta = vis_data.get("meta", {})
    usable, reason = mode_usable(meta, mode)
    return TeamVisibility(vis_data[mode], deaths,
                          sensitivity=sensitivity_from_meta(meta, mode),
                          usable=usable, reason=reason)
