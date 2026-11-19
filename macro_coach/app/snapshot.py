"""Snapshot del estado en un punto de decisión (t = t0_segmento - eps).

Todo lo que hay aquí es observable en las fuentes: builds reconstruidas,
frames 60s (oro marcado _prev_frame: legal pero rancio), rombos, eventos de
objetivos, visibilidad de las pasadas con niebla (o null), y referencias al
frame de minimapa más cercano de cada perspectiva.
"""
import bisect
import glob
import json
import os
from pathlib import Path

from . import inventory
from .config import (DRAGON_RESPAWN_S, DRAGON_FIRST_S, SNAPSHOT_EPS_S,
                     MINIMAP_REF_MAX_DT, SNAPSHOT_V)
from .regions import region, region_for_team
from .segments import participants, jcs_at
from .trajectory import pos_at, pos_at_causal
from .names import internal


def load_playerlist(pass_dir: Path) -> list:
    """[(t, {champ: {dead, respawn, level, cs, items}})] de samples.jsonl."""
    out = []
    f = pass_dir / "samples.jsonl"
    if not f.exists():
        return out
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "players" in r:
                out.append((r["t"], {internal(p["champ"]): p
                                     for p in r["players"]}))
    return out


def deaths_series(playerlist: list) -> list:
    return [(t, {c: p["dead"] for c, p in d.items()}) for t, d in playerlist]


def player_state_at(playerlist: list, champ: str, t: float) -> dict | None:
    """Última muestra PASADA de la playerlist (nunca mirar hacia delante: el
    snapshot es el estado con el que se decidió). Antes de la primera muestra
    (~8s) se admite la primera: todos vivos a nivel 1."""
    if not playerlist:
        return None
    ts = [r[0] for r in playerlist]
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        # borde del inicio: la primera muestra vale si está a <12s; si la
        # playerlist empieza tarde (captura parcial), no hay dato
        if ts[0] - t <= 12:
            i = 0
        else:
            return None
    return playerlist[i][1].get(champ)


def frame_before(timeline: dict, t: float) -> dict:
    """participantFrames del último frame 60s <= t (estado legal, rancio)."""
    best = timeline["info"]["frames"][0]
    for f in timeline["info"]["frames"]:
        if f["timestamp"] / 1000.0 > t:
            break
        best = f
    return best


def objectives_state(timeline: dict, t: float) -> dict:
    """Dragones/larvas/heraldo/barón tomados hasta t + ventana de respawn del
    dragón observada; torres perdidas por equipo."""
    dragons = {100: 0, 200: 0}
    grubs = {100: 0, 200: 0}
    herald = {100: 0, 200: 0}
    baron = {100: 0, 200: 0}
    towers_lost = {100: 0, 200: 0}
    last_dragon_t = None
    soul = None
    for f in timeline["info"]["frames"]:
        for e in f["events"]:
            et = e["timestamp"] / 1000.0
            if et > t:
                break
            if e["type"] == "ELITE_MONSTER_KILL":
                team = e.get("killerTeamId")
                mt = e.get("monsterType")
                if team not in (100, 200):
                    continue
                if mt == "DRAGON":
                    dragons[team] += 1
                    last_dragon_t = et
                elif mt == "HORDE":
                    grubs[team] += 1
                elif mt == "RIFTHERALD":
                    herald[team] += 1
                elif mt == "BARON_NASHOR":
                    baron[team] += 1
            elif e["type"] == "BUILDING_KILL":
                if e.get("teamId") in (100, 200):
                    towers_lost[e["teamId"]] += 1
            elif e["type"] == "DRAGON_SOUL_GIVEN":
                soul = e.get("teamId")
    if last_dragon_t is None:
        next_dragon_in = max(0.0, DRAGON_FIRST_S - t)
    else:
        next_dragon_in = max(0.0, last_dragon_t + DRAGON_RESPAWN_S - t)
    return {"dragons": dragons, "grubs": grubs, "herald": herald, "baron": baron,
            "towers_lost": towers_lost, "soul_team": soul,
            "next_dragon_in": round(next_dragon_in, 1)}


class MinimapIndex:
    """Frame de minimapa más cercano por pasada."""

    def __init__(self, game_dir: Path):
        self.refs = {}
        for name in ("pass_omni", "pass_blue", "pass_red"):
            d = game_dir / name
            files = sorted(glob.glob(str(d / "mm_*.png")))
            self.refs[name] = [(int(os.path.basename(f)[3:9]) / 10.0,
                               f"{name}/{os.path.basename(f)}") for f in files]

    def at(self, t: float) -> dict:
        out = {}
        for name, refs in self.refs.items():
            if not refs:
                out[name] = None
                continue
            ts = [r[0] for r in refs]
            i = bisect.bisect_left(ts, t)
            best = None
            for j in (i - 1, i):
                if 0 <= j < len(refs):
                    dt = abs(refs[j][0] - t)
                    if best is None or dt < best[0]:
                        best = (dt, refs[j][1])
            out[name] = {"file": best[1], "dt": round(best[0], 1)} \
                if best and best[0] <= MINIMAP_REF_MAX_DT else None
        return out


def _observed_for(row: dict, my_team: int, enemy_state: dict, traj: dict,
                  t: float) -> dict:
    """Lo que el equipo del POV puede saber de este jugador en t.

    - Propio equipo: se conoce siempre (posición propia y de los aliados).
    - Enemigo visible ahora: se conoce su posición actual.
    - Enemigo oculto pero avistado antes: su ÚLTIMA posición avistada y la
      antigüedad de ese avistamiento — información pasada, no futura.
    - Enemigo muerto: conocido (la fuente y el respawn son públicos).
    - Enemigo nunca avistado, o visibilidad no fiable: DESCONOCIDO.

    Antes se enviaba al modelo la región real de los cinco enemigos aunque
    estuvieran en niebla: 312 de 315 posiciones ocultas conservaban región en
    EUW1_7986655074. Eso es omnisciencia, no estado observable.
    """
    if row["team"] == my_team:
        return {"known": True, "source": "own_team",
                "region_team": row["region_team"], "pos": row["pos"],
                "age_s": row.get("pos_age_s"), "visible": True}
    e = enemy_state.get(row["champ"])
    if e is None or not e.get("known"):
        return {"known": False, "source": "no_fiable", "region_team": None,
                "pos": None, "age_s": None, "visible": None}
    if e.get("dead"):
        return {"known": True, "source": "muerto", "region_team": None,
                "pos": None, "age_s": 0.0, "visible": False, "dead": True}
    if e.get("visible"):
        return {"known": True, "source": "visible",
                "region_team": row["region_team"], "pos": row["pos"],
                "age_s": row.get("pos_age_s"), "visible": True}
    last_t = e.get("last_seen_t")
    if last_t is None:
        return {"known": False, "source": "nunca_avistado", "region_team": None,
                "pos": None, "age_s": None, "visible": False}
    # posición en el instante del último avistamiento (pasado, no futuro)
    pos, _ = pos_at_causal(traj.get(row["champ"], []), last_t, max_age=6.0)
    if pos is None:
        return {"known": False, "source": "avistado_sin_posicion",
                "region_team": None, "pos": None, "age_s": round(t - last_t, 2),
                "visible": False}
    return {"known": True, "source": "ultimo_avistamiento",
            "region_team": region_for_team(*pos, my_team),
            "pos": [round(pos[0], 4), round(pos[1], 4)],
            "age_s": round(t - last_t, 2), "visible": False}


def snapshot_time(t_decision: float) -> float:
    """Instante causal del snapshot. ÚNICA definición: todo lo que entre en el
    snapshot (campamentos incluidos) se evalúa aquí, no en t_decision."""
    return max(0.0, t_decision - SNAPSHOT_EPS_S)


def build_snapshot(match: dict, timeline: dict, traj: dict, camp_states: dict,
                   playerlist: list, minimaps: MinimapIndex, vis, me: int,
                   t_decision: float, prices: dict) -> dict:
    """Estado en t = t_decision - eps.

    Distingue dos capas y NO las mezcla:
      - `truth`: dónde estaba de verdad cada uno (pasada sin niebla). Sirve
        para etiquetar y para la revisión del owner.
      - `observed`: lo que el equipo del POV podía saber al decidir. Para un
        enemigo oculto es su último avistamiento y su antigüedad; si nunca se
        le vio, o la visibilidad no es fiable, es DESCONOCIDO.
    Las features de inferencia sólo pueden leer `observed` (app/dataset.py).
    """
    from .camps import alive_at
    t = snapshot_time(t_decision)
    camps_at = alive_at(camp_states, t)
    P = participants(match)
    my_team = P[me]["team"]
    inv = inventory.inventories_at(timeline, t)
    frame = frame_before(timeline, t)
    frame_t = frame["timestamp"] / 1000.0

    players = []
    for pid in range(1, 11):
        champ = P[pid]["champ"]
        pf = frame["participantFrames"].get(str(pid), {})
        pl = player_state_at(playerlist, champ, t)
        # causal: nunca una muestra posterior a t (pos_at cogía la más
        # cercana, que podía ser del futuro)
        pos, pos_age = pos_at_causal(traj.get(champ, []), t)
        players.append({
            "pid": pid, "champ": champ, "role": P[pid]["role"],
            "team": P[pid]["team"],
            "pos": [round(pos[0], 4), round(pos[1], 4)] if pos else None,
            "pos_age_s": round(pos_age, 2) if pos_age is not None else None,
            "region": region(*pos) if pos else None,
            "region_team": region_for_team(*pos, my_team) if pos else None,
            "level": pl["level"] if pl else pf.get("level"),
            "cs": pl["cs"] if pl else (pf.get("minionsKilled", 0)
                                       + pf.get("jungleMinionsKilled", 0)),
            "dead": pl["dead"] if pl else None,
            "respawn_in": pl.get("respawn") if pl else None,
            "gold_total_prev_frame": pf.get("totalGold"),
            "gold_current_prev_frame": pf.get("currentGold"),
            "xp_prev_frame": pf.get("xp"),
            "items": inventory.items_list(inv[pid]),
            "item_gold": inventory.gold_value(inv[pid], prices),
        })

    me_row = players[me - 1]
    jcs = jcs_at([(f2["timestamp"] / 1000.0,
                   f2["participantFrames"][str(me)].get("jungleMinionsKilled", 0),
                   f2["participantFrames"][str(me)].get("minionsKilled", 0))
                  for f2 in timeline["info"]["frames"]
                  if str(me) in f2["participantFrames"]], t)

    vision = {"available": False, "usable": False, "info_pct": None,
              "enemies": None, "reason": "sin pasadas de niebla"}
    if vis is not None:
        vision = {"available": True, **vis.info(t)}

    # --- capa observable: lo unico que puede ver el modelo de los enemigos ---
    enemy_state = {e["champ"]: e for e in (vision.get("enemies") or [])}
    for row in players:
        row["observed"] = _observed_for(row, my_team, enemy_state, traj, t)

    return {
        "t": round(t, 2),
        "snapshot_v": SNAPSHOT_V,
        "frame_t": frame_t,
        "me": {"pid": me, "champ": P[me]["champ"], "team": my_team,
               "region_team": me_row["region_team"], "jcs": jcs[0], "cs_lane": jcs[1]},
        "players": players,
        "camps": camps_at,
        "objectives": objectives_state(timeline, t),
        "vision": vision,
        "minimap": minimaps.at(t),
    }
