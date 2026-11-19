"""Segmentación de la timeline de un jugador — port fiel del núcleo auditado
de macro_v5 (fase 1), parametrizado por POV y con salida estructurada.

Reglas auditadas que este módulo preserva (ver docs/history.md):
- La posición de un evento manda sobre la detección.
- Gank = carril con rival cerca (<GANK_RADIUS) o kill/muerte anclada; en la
  mitad propia es pelea, no gank (frontera GANK_ENEMY_HALF en frame de equipo).
- Farmear = rombo del cuadrante muere en la ventana o sube el CS de jungla;
  si no, es cruzar (tránsito).
- Los huecos (muerte / sin detección) no se rellenan.
Novedad fase 2 (pendiente de auditar): centinela = río + CS de jungla.
"""
import math

from .config import (GANK_RADIUS, SEG_MIN_S, SEG_JOIN_S, SEG_MERGE_S,
                     FARM_JCS_MIN, GANK_ENEMY_HALF, CAMP_WINDOW)
from .regions import region, mirror_region, quad_of_camp, norm_pos

POS_SHORT = {"TOP": "TOP", "JUNGLE": "JG", "MIDDLE": "MID",
             "BOTTOM": "ADC", "UTILITY": "SUP"}
LANE_OF_ROLE = {"TOP": "TOP", "MID": "MID", "ADC": "BOT", "SUP": "BOT"}


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def participants(match: dict) -> dict:
    return {p["participantId"]: {"champ": p["championName"],
                                 "role": POS_SHORT.get(p["teamPosition"], "?"),
                                 "team": p["teamId"]}
            for p in match["info"]["participants"]}


def junglers(match: dict) -> list:
    return [pid for pid, v in participants(match).items() if v["role"] == "JG"]


def _tf(p, team):
    """Frame de equipo: azul tal cual; rojo espejado."""
    return p if team == 100 else (1 - p[0], 1 - p[1])


def pov_marks(timeline: dict, P: dict, me: int) -> list:
    """Kills/muertes/asistencias del POV: (t, kind, other_pid, pos)."""
    marks = []
    for f in timeline["info"]["frames"]:
        for e in f["events"]:
            if e["type"] != "CHAMPION_KILL":
                continue
            t = e["timestamp"] / 1000.0
            k, v = e.get("killerId"), e.get("victimId")
            ass = e.get("assistingParticipantIds", []) or []
            if me not in [k, v] + ass:
                continue
            p = norm_pos(e["position"])
            if k == me:
                marks.append((t, "kill", v, p))
            elif v == me:
                marks.append((t, "death", k, p))
            else:
                marks.append((t, "assist", v, p))
    marks.sort()
    return marks


def jcs_series(timeline: dict, me: int) -> list:
    out = []
    for f in timeline["info"]["frames"]:
        pf = f["participantFrames"].get(str(me))
        if pf:
            out.append((f["timestamp"] / 1000.0,
                        pf.get("jungleMinionsKilled", 0), pf.get("minionsKilled", 0)))
    return out


def jcs_at(jcs: list, t: float) -> tuple:
    prev = (0, 0)
    for st, j, mi in jcs:
        if st > t:
            break
        prev = (j, mi)
    return prev


def detect_recalls(ev: list, team: int) -> list:
    """[(t_inicio, t_fin)] — patrón auditado: quieto 2+ muestras y salto largo
    hacia base propia desde fuera de ella (en frame de equipo)."""
    recalls = []
    for i in range(2, len(ev) - 1):
        pts = [(_tf(p[1:3], team)) for p in ev[i - 2:i + 2]]
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
        t0, t3 = ev[i - 2][0], ev[i + 1][0]
        if (dist((x0, y0), (x2, y2)) < 0.02 and dist((x1, y1), (x2, y2)) < 0.02
                and dist((x2, y2), (x3, y3)) > 0.20
                and (x3 + y3) < (x2 + y2) - 0.15 and (x2 + y2) > 0.5):
            recalls.append((t0, t3))
    return recalls


def build_segments(match: dict, timeline: dict, traj: dict, camp_kills: list,
                   me: int) -> list:
    """Segmentos estructurados del POV `me` (participantId)."""
    P = participants(match)
    my_team = P[me]["team"]
    my_champ = P[me]["champ"]
    enemies = {P[i]["champ"] for i in P if P[i]["team"] != my_team}
    marks = pov_marks(timeline, P, me)
    jcs = jcs_series(timeline, me)

    ev = sorted(traj.get(my_champ, []) + [(mk[0], *mk[3]) for mk in marks])
    if not ev:
        return []

    def enemies_near(t, p, rad=GANK_RADIUS, dt=6):
        out = set()
        for champ in enemies:
            for pt in traj.get(champ, []):
                if abs(pt[0] - t) <= dt and dist(p, pt[1:3]) < rad:
                    out.add(champ)
                    break
        return out

    recalls = detect_recalls(ev, my_team)

    # --- etiqueta de región por punto (ventana de mayoría, port fiel) ---
    labels = []
    for i, (t, nx, ny) in enumerate(ev):
        raw = region(nx, ny)
        if raw in ("BASE", "BASE_ENEMIGA"):
            labels.append("BASE")
        elif any(a <= t <= b for a, b in recalls):
            labels.append("BACKEA")
        else:
            win = [region(*p[1:3]) for p in ev[max(0, i - 1):i + 2]]
            win = [w for w in win if w not in ("BASE", "BASE_ENEMIGA")] or [raw]
            labels.append(max(set(win), key=win.count))
    for t, kind, other, p in marks:
        r = region(*p)
        for i, (tt, _, _) in enumerate(ev):
            if abs(tt - t) < 0.01 and r not in ("BASE", "BASE_ENEMIGA"):
                labels[i] = r

    # --- agrupar, filtrar, fundir (umbrales auditados) ---
    segs = []
    for (t, nx, ny), lb in zip(ev, labels):
        if segs and segs[-1]["lab"] == lb and t - segs[-1]["t1"] <= SEG_JOIN_S:
            segs[-1]["t1"] = t
            segs[-1]["pts"].append((t, nx, ny))
        else:
            segs.append({"t0": t, "t1": t, "lab": lb, "pts": [(t, nx, ny)]})

    def anchored(s):
        return any(s["t0"] - 5 <= mk[0] <= s["t1"] + 5 for mk in marks)

    segs = [s for s in segs if s["t1"] - s["t0"] >= SEG_MIN_S or anchored(s)
            or s["lab"] in ("BASE", "BACKEA")]
    merged = []
    for s in segs:
        if merged and merged[-1]["lab"] == s["lab"] and s["t0"] - merged[-1]["t1"] <= SEG_MERGE_S:
            merged[-1]["t1"] = s["t1"]
            merged[-1]["pts"] += s["pts"]
        else:
            merged.append(s)

    # --- monstruos épicos cerca de fosos (para desambiguar Barón/Heraldo/Larvas) ---
    epic = []
    for f in timeline["info"]["frames"]:
        for e in f["events"]:
            if e["type"] == "ELITE_MONSTER_KILL":
                epic.append((e["timestamp"] / 1000.0, e.get("monsterType")))

    # --- estructura final ---
    out = []
    mi = 0
    for s in merged:
        lb = s["lab"]
        my = []
        while mi < len(marks) and marks[mi][0] <= s["t1"] + 5:
            if marks[mi][0] >= s["t0"] - 5:
                my.append(marks[mi])
            mi += 1
        seg = {"t0": round(s["t0"], 2), "t1": round(s["t1"], 2),
               "region": lb, "n_pts": len(s["pts"]),
               "marks": [{"t": round(mk[0], 2), "kind": mk[1],
                          "other_pid": mk[2],
                          "other": P.get(mk[2], {}).get("champ")} for mk in my]}
        fight_lanes = []
        for mk in my:
            other = mk[2]
            if other in P and P[other]["team"] != my_team:
                ln = LANE_OF_ROLE.get(P[other]["role"])
                if ln and ln not in fight_lanes:
                    fight_lanes.append(ln)
        seg["fight_lanes"] = fight_lanes

        if lb == "BASE":
            seg["kind"] = "base"
        elif lb == "BACKEA":
            seg["kind"] = "reset"
        elif lb in ("TOP", "MID", "BOT"):
            near = set()
            for p in s["pts"]:
                near.update(enemies_near(p[0], p[1:3]))
            near.update(P[mk[2]]["champ"] for mk in my
                        if mk[2] in P and P[mk[2]]["team"] != my_team)
            seg["lane"] = lb
            seg["rivals"] = sorted(near)
            if near:
                cx = sum(p[1] for p in s["pts"]) / len(s["pts"])
                cy = sum(p[2] for p in s["pts"]) / len(s["pts"])
                tx, ty = _tf((cx, cy), my_team)
                seg["kind"] = "lane_gank" if tx + ty >= GANK_ENEMY_HALF else "lane_fight"
            else:
                seg["kind"] = "lane_cover"
        elif lb in ("JGL_PA", "JGL_PR", "JGL_EA", "JGL_ER"):
            seg["kind"] = "jungle"
            seg["region_team"] = lb if my_team == 100 else mirror_region(lb)
            dot_deaths = [c for t, c in camp_kills
                          if s["t0"] + CAMP_WINDOW[0] <= t <= s["t1"] + CAMP_WINDOW[1]
                          and quad_of_camp(c) == lb]
            j0, _ = jcs_at(jcs, s["t0"] - 2)
            j1, _ = jcs_at(jcs, s["t1"] + 30)
            seg["farming"] = bool(dot_deaths) or (j1 - j0) >= FARM_JCS_MIN
            seg["camps_taken"] = dot_deaths
        elif lb in ("RIO_TOP", "RIO_BOT"):
            seg["kind"] = "river"
            seg["region_team"] = lb
            j0, _ = jcs_at(jcs, s["t0"] - 2)
            j1, _ = jcs_at(jcs, s["t1"] + 30)
            seg["farming"] = (j1 - j0) >= 1        # centinela (regla fase 2)
        elif lb == "PIT_DRAGON":
            seg["kind"] = "pit_dragon"
        elif lb == "PIT_BARON":
            seg["kind"] = "pit_baron"
            near_epic = [m for t, m in epic if abs(t - s["t0"]) < 25 or abs(t - s["t1"]) < 25]
            seg["monster"] = near_epic[0] if near_epic else None
        else:
            seg["kind"] = "jungle"
            seg["region_team"] = lb
            seg["farming"] = False
        out.append(seg)
    return out
