"""Timeline macro v5 — Match-V5 como capa de eventos, minimapa como trayectoria.

Fuentes:
  - Match-V5 timeline: kills/objetivos/edificios CON POSICION EXACTA, y
    participantFrames cada 60s con posicion y CS exactos de los 10.
  - Minimapa (traj_v7): trayectoria densa (~3s) para lo que la API no ve.
  - Rombos del minimapa (camp_dots): campamentos vivos/muertos.

Reglas:
  - La posicion de un evento manda sobre la deteccion: si mata en BOT, el
    segmento es BOT aunque el detector dijera otra cosa.
  - Los frames de 60s de Match-V5 son anclas exactas de posicion.
  - farmea = rombo de ese cuadrante muere en el segmento o sube jungle CS.
  - Objetivos de ambos equipos como filas propias, con su region real.
"""
import json
import math

S = "C:/Users/Javier/AppData/Local/Temp/claude/C--Users-Javier-build-tracker/19666e3e-e394-403c-9692-3d8eafc40471/scratchpad"
PLAYER = "Evelynn"
MAP_MAX = 14870.0

POS_SHORT = {"TOP": "TOP", "JUNGLE": "JG", "MIDDLE": "MID",
             "BOTTOM": "ADC", "UTILITY": "SUP"}
SHOW = {"Chogath": "Cho'Gath"}

import numpy as np
LBN = {1: "BASE", 2: "BASE_ENEMIGA", 3: "TOP", 4: "MID", 5: "BOT",
       6: "RIO_TOP", 7: "RIO_BOT", 8: "JGL_PA", 9: "JGL_PR",
       10: "JGL_EA", 11: "JGL_ER", 12: "PIT_BARON", 13: "PIT_DRAGON"}
MASK = np.load(f"{S}/region_labels.npy")
N = MASK.shape[0]

CB = {"gromp": (0.142, 0.568), "azul": (0.256, 0.531), "lobos": (0.254, 0.433),
      "raptors": (0.467, 0.342), "rojo": (0.525, 0.272), "krugs": (0.563, 0.181)}
CAMPS = dict(CB)
CAMPS.update({k + " enemigo": (1 - x, 1 - y) for k, (x, y) in CB.items()})


def region(nx, ny):
    x = min(max(int(nx * N), 0), N - 1)
    y = min(max(int((1 - ny) * N), 0), N - 1)
    return LBN.get(int(MASK[y, x]), "JGL_PA")


def mmss(t):
    return f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def norm(p):
    return (min(max(p["x"] / MAP_MAX, 0), 1), min(max(p["y"] / MAP_MAX, 0), 1))


def clean_trajectory(pts, own_base):
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


def main():
    tl = json.load(open(f"{S}/mv5_timeline.json"))
    m = json.load(open(f"{S}/mv5_match.json"))
    P = {p["participantId"]: {"champ": p["championName"],
                              "pos": POS_SHORT.get(p["teamPosition"], "?"),
                              "team": p["teamId"]}
         for p in m["info"]["participants"]}
    ME = next(i for i, v in P.items() if v["champ"] == PLAYER)
    MY_TEAM = P[ME]["team"]
    lab = lambda i: f"{SHOW.get(P[i]['champ'], P[i]['champ'])} ({P[i]['pos']})" if i in P else "?"
    ENEMIES = {P[i]["champ"] for i in P if P[i]["team"] != MY_TEAM}

    frames = tl["info"]["frames"]
    events = [e for f in frames for e in f["events"]]

    # --- capa de eventos (autoritativa) ---
    marks, objectives = [], []
    for e in events:
        t = e["timestamp"] / 1000.0
        if e["type"] == "CHAMPION_KILL":
            k, v = e.get("killerId"), e.get("victimId")
            ass = e.get("assistingParticipantIds", []) or []
            if ME in [k, v] + ass:
                p = norm(e["position"])
                if k == ME:
                    marks.append((t, "kill", f"KILL {lab(v)}", p, v))
                elif v == ME:
                    marks.append((t, "death", f"MUERE ante {lab(k)}", p, k))
                else:
                    marks.append((t, "assist", f"asiste kill de {lab(k)} a {lab(v)}", p, v))
        elif e["type"] == "ELITE_MONSTER_KILL":
            sub = e.get("monsterSubType", "")
            tipo = {"DRAGON": {"EARTH_DRAGON": "Dragón de tierra",
                               "FIRE_DRAGON": "Dragón de fuego",
                               "CHEMTECH_DRAGON": "Dragón quimtec",
                               "WATER_DRAGON": "Dragón de agua",
                               "AIR_DRAGON": "Dragón de viento",
                               "HEXTECH_DRAGON": "Dragón hextec",
                               "ELDER_DRAGON": "Dragón anciano"}.get(sub, "Dragón"),
                    "RIFTHERALD": "Heraldo", "BARON_NASHOR": "Barón",
                    "HORDE": "Larva"}.get(e.get("monsterType"), e.get("monsterType"))
            objectives.append({"t": t, "type": tipo, "team": e.get("killerTeamId"),
                               "mine": e.get("killerId") == ME,
                               "p": norm(e["position"]), "kind": "monster"})
        elif e["type"] == "BUILDING_KILL":
            b = "Inhibidor" if e.get("buildingType") == "INHIBITOR_BUILDING" else "Torre"
            lane = {"TOP_LANE": "TOP", "MID_LANE": "MID", "BOT_LANE": "BOT"}.get(e.get("laneType"), "")
            ass = e.get("assistingParticipantIds", []) or []
            objectives.append({"t": t, "type": f"{b} {lane}".strip(),
                               "team": 200 if e.get("teamId") == 100 else 100,
                               "mine": ME in [e.get("killerId")] + ass,
                               "p": norm(e["position"]), "kind": "building"})
    # agrupa larvas consecutivas del mismo equipo
    grouped, larvae = [], {}
    for o in objectives:
        if o["type"] == "Larva":
            key = o["team"]
            g = larvae.get(key)
            if g and o["t"] - g["t"] < 120:
                g["n"] += 1
                g["mine"] = g["mine"] or o["mine"]
                continue
            o = dict(o, n=1)
            larvae[key] = o
        grouped.append(o)
    objectives = grouped
    for o in objectives:
        if o["type"] == "Larva":
            o["type"] = f"Larvas x{o.get('n', 1)}"
    marks.sort()
    objectives.sort(key=lambda o: o["t"])
    first_tower = min((o["t"] for o in objectives if o["kind"] == "building"), default=None)

    # --- anclas exactas: frames de 60s ---
    anchors = {}
    for f in frames:
        t = f["timestamp"] / 1000.0
        for pidstr, pf in f["participantFrames"].items():
            pid = int(pidstr)
            if "position" not in pf:
                continue
            anchors.setdefault(P[pid]["champ"], []).append((t, *norm(pf["position"])))
    # CS de jungla exacto de Evelynn
    jcs = []
    for f in frames:
        t = f["timestamp"] / 1000.0
        pf = f["participantFrames"].get(str(ME))
        if pf:
            jcs.append((t, pf.get("jungleMinionsKilled", 0), pf.get("minionsKilled", 0)))

    def jcs_at(t):
        prev = (0, 0)
        for st, j, mi in jcs:
            if st > t:
                break
            prev = (j, mi)
        return prev

    # --- trayectorias: minimapa + anclas MV5 ---
    traj = {}
    for r in map(json.loads, open(f"{S}/traj_v7.jsonl")):
        if r["score"] >= 0.895:
            traj.setdefault(r["champ"], []).append((r["t"], r["nx"], r["ny"]))
    for champ in traj:
        base = (1.0, 1.0) if champ in ENEMIES else (0.0, 0.0)
        traj[champ] = clean_trajectory(sorted(traj[champ]), base)
    for champ, pts in anchors.items():
        traj.setdefault(champ, [])
        traj[champ] = sorted(traj[champ] + pts)

    ev = sorted(traj[PLAYER] + [(mk[0], *mk[3]) for mk in marks])

    def enemies_near(t, p, rad=0.15, dt=6):
        out = set()
        for champ in ENEMIES:
            for pt in traj.get(champ, []):
                if abs(pt[0] - t) <= dt and dist(p, pt[1:3]) < rad:
                    out.add(champ)
                    break
        return out

    # rombos -> muertes de camp
    dots = json.load(open(f"{S}/camp_dots.json"))

    def occupied(t, cpos, rad=0.09):
        for pts in traj.values():
            for pt in pts:
                if abs(pt[0] - t) <= 3 and dist(pt[1:3], cpos) < rad:
                    return True
        return False

    camp_kills = []
    for camp, ser in dots.items():
        cpos = CAMPS[camp]
        alive, prev_absent = True, False
        for t, n in ser:
            if t < 105 or occupied(t, cpos):
                prev_absent = False
                continue
            if n >= 5:
                alive, prev_absent = True, False
            elif n < 3:
                if alive and prev_absent:
                    alive = False
                    camp_kills.append((t, camp))
                prev_absent = True
            else:
                prev_absent = False

    def quad_of(camp):
        x, y = CAMPS[camp]
        if x + y < 1:
            return "JGL_PA" if y > x else "JGL_PR"
        return "JGL_EA" if y < x else "JGL_ER"

    # recalls
    recalls = []
    for i in range(2, len(ev) - 1):
        (t0, x0, y0), (t1, x1, y1), (t2, x2, y2), (t3, x3, y3) = ev[i - 2:i + 2]
        if (dist((x0, y0), (x2, y2)) < 0.02 and dist((x1, y1), (x2, y2)) < 0.02
                and dist((x2, y2), (x3, y3)) > 0.20
                and (x3 + y3) < (x2 + y2) - 0.15 and (x2 + y2) > 0.5):
            recalls.append((t0, t3))

    # etiquetas
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
    # la region del evento manda sobre la muestra
    for t, kind, txt, p, other in marks:
        r = region(*p)
        for i, (tt, _, _) in enumerate(ev):
            if abs(tt - t) < 0.01:
                labels[i] = r if r not in ("BASE", "BASE_ENEMIGA") else labels[i]

    segs = []
    for (t, nx, ny), lb in zip(ev, labels):
        if segs and segs[-1]["lab"] == lb and t - segs[-1]["t1"] <= 10:
            segs[-1]["t1"] = t
            segs[-1]["pts"].append((t, nx, ny))
        else:
            segs.append({"t0": t, "t1": t, "lab": lb, "pts": [(t, nx, ny)]})

    def anchored(s):
        return any(s["t0"] - 5 <= mk[0] <= s["t1"] + 5 for mk in marks)

    segs = [s for s in segs if s["t1"] - s["t0"] >= 8 or anchored(s)
            or s["lab"] in ("BASE", "BACKEA")]
    merged = []
    for s in segs:
        if merged and merged[-1]["lab"] == s["lab"] and s["t0"] - merged[-1]["t1"] <= 14:
            merged[-1]["t1"] = s["t1"]
            merged[-1]["pts"] += s["pts"]
        else:
            merged.append(s)

    for o in objectives:
        merged.append({"t0": o["t"], "t1": o["t"], "lab": "OBJETIVO", "obj": o, "pts": []})
    merged.sort(key=lambda s: s["t0"])

    JGL = {"JGL_PA": ("jungla propia · cuadrante azul"),
           "JGL_PR": ("jungla propia · cuadrante rojo"),
           "JGL_EA": ("jungla enemiga · cuadrante azul"),
           "JGL_ER": ("jungla enemiga · cuadrante rojo")}
    NAMES = {"BASE": "en base", "BACKEA": "backea",
             "RIO_TOP": "río superior", "RIO_BOT": "río inferior",
             "PIT_BARON": "foso de Barón", "PIT_DRAGON": "foso de Dragón"}

    out, mi = [], 0
    print(f"TIMELINE MACRO — {PLAYER} ({P[ME]['pos']}, equipo azul)")
    print(f"(fase de lineas hasta {mmss(first_tower)} — primera torre)\n")
    for s in merged:
        lb = s["lab"]
        my = []
        while mi < len(marks) and marks[mi][0] <= s["t1"] + 5:
            if marks[mi][0] >= s["t0"] - 5:
                my.append(marks[mi])
            mi += 1
        detail = ""
        if lb == "OBJETIVO":
            o = s["obj"]
            side = "equipo" if o["team"] == MY_TEAM else "RIVAL"
            action = f"OBJETIVO ({side})"
            detail = o["type"] + (" · lo asegura ella" if o["mine"] else "")
            detail += f" — {NAMES.get(region(*o['p']), JGL.get(region(*o['p']), region(*o['p'])))}"
        elif lb in ("TOP", "MID", "BOT", "PIT_BARON", "PIT_DRAGON"):
            near = set()
            for p in s["pts"]:
                near.update(enemies_near(p[0], p[1:3]))
            near.update(P[mk[4]]["champ"] for mk in my
                        if mk[4] in P and P[mk[4]]["team"] != MY_TEAM)
            zonal = {"PIT_BARON": "foso de Barón", "PIT_DRAGON": "foso de Dragón"}
            if near and lb in zonal:
                action = f"pelea en {zonal[lb]}"
            elif near:
                cx = sum(p[1] for p in s["pts"]) / len(s["pts"])
                cy = sum(p[2] for p in s["pts"]) / len(s["pts"])
                action = f"GANK {lb}" if cx + cy >= 0.97 else f"pelea en {lb} (lado propio)"
                detail = "sobre " + ", ".join(
                    sorted(f"{SHOW.get(c, c)} ({next(P[i]['pos'] for i in P if P[i]['champ'] == c)})"
                           for c in near))
            elif lb in zonal:
                action = zonal[lb]
            else:
                action = f"pisa {lb} (sin rival visto)"
        elif lb in JGL:
            zona = JGL[lb]
            dot_deaths = [c for t, c in camp_kills
                          if s["t0"] - 5 <= t <= s["t1"] + 15 and quad_of(c) == lb]
            j0, _ = jcs_at(s["t0"] - 2)
            j1, _ = jcs_at(s["t1"] + 30)
            farming = bool(dot_deaths) or (j1 - j0) >= 3
            if lb.startswith("JGL_E"):
                action = ("INVADE " if farming else "cruza ") + zona
            else:
                action = ("farmea " if farming else "cruza ") + zona
            # accion compuesta: pelea con un laner rival dentro de la jungla
            # -> se nombra el gank por el carril de la victima y se encadena
            LANE_OF = {"TOP": "TOP", "MID": "MID", "ADC": "BOT", "SUP": "BOT"}
            lanes = []
            for mk in my:
                other = mk[4]
                if other in P and P[other]["team"] != MY_TEAM:
                    ln = LANE_OF.get(P[other]["pos"])
                    if ln and ln not in lanes:
                        lanes.append(ln)
            if lanes:
                action = " → ".join(f"GANK {ln}" for ln in lanes) + " → " + action
        else:
            action = NAMES[lb]
        line = f"{mmss(s['t0'])}-{mmss(s['t1'])}  {action}"
        if detail:
            line += f"  [{detail}]"
        if my:
            line += "   {" + "; ".join(f"{mmss(mk[0])} {mk[2]}" for mk in my) + "}"
        print(line)
        out.append({"t0": s["t0"], "t1": s["t1"], "action": action, "detail": detail,
                    "marks": [{"t": mk[0], "kind": mk[1], "text": mk[2]} for mk in my]})

    stats = next(p for p in m["info"]["participants"] if p["championName"] == PLAYER)
    json.dump({"first_tower": first_tower, "segments": out,
               "kda": f"{stats['kills']}/{stats['deaths']}/{stats['assists']}",
               "cs": stats["totalMinionsKilled"] + stats["neutralMinionsKilled"],
               "win": stats["win"],
               "drag_rival": sum(1 for o in objectives
                                 if o["type"].startswith("Dragón") and o["team"] != MY_TEAM),
               "drag_team": sum(1 for o in objectives
                                if o["type"].startswith("Dragón") and o["team"] == MY_TEAM)},
              open(f"{S}/timeline_evelynn2.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
