"""Detector v5 sobre las capturas de capture_pass (minimapa 508px).

Anillos por color -> componentes -> clasificacion local de campeones vivos.
Usa samples.jsonl para saber quien esta muerto en cada instante.

Uso: python analyze_images.py <dir_pass> <out.jsonl> [t_debug ...]
"""
import cv2
import numpy as np
import json
import sys
import glob
import os
import bisect

S = "C:/Users/Javier/AppData/Local/Temp/claude/C--Users-Javier-build-tracker/19666e3e-e394-403c-9692-3d8eafc40471/scratchpad"
import os as _os
MX0 = int(_os.environ.get("MM_X0", 9))
MY0 = int(_os.environ.get("MM_Y0", 5))
MSIZE = int(_os.environ.get("MM_SIZE", 508))
T_MIN = float(_os.environ.get("MM_TMIN", -1))
T_MAX = float(_os.environ.get("MM_TMAX", 1e9))
ZOOM = 1.2
SIZES = tuple(int(x) for x in _os.environ.get("MM_SIZES", "38,44,50").split(","))
CLS_MIN = 0.87
NMS_DIST = 16
MIN_AREA = 90

TEAMS = {"Nasus": "ORDER", "Evelynn": "ORDER", "Zed": "ORDER",
         "Caitlyn": "ORDER", "Soraka": "ORDER",
         "Chogath": "CHAOS", "Vi": "CHAOS", "Lux": "CHAOS",
         "Jinx": "CHAOS", "Blitzcrank": "CHAOS"}
NAME_FIX = {"Cho'Gath": "Chogath"}


def build_templates():
    out = {}
    for champ in TEAMS:
        img = cv2.imread(f"{S}/portraits_hud/{champ}.png")
        if img is None:
            img = cv2.imread(f"{S}/portraits/{champ}.png")
        h, w = img.shape[:2]
        m = int(h * (1 - 1 / ZOOM) / 2)
        img = img[m:h - m, m:w - m]
        per = []
        for s in SIZES:
            t = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            mask = np.zeros((s, s), np.uint8)
            cv2.circle(mask, (s // 2, s // 2), int(s * 0.40), 255, -1)
            per.append((s, t, mask))
        out[champ] = per
    return out


def team_mask(hsv, team):
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if team == "ORDER":
        m = (hue >= 92) & (hue <= 112) & (sat > 120) & (val > 120)
    else:
        m = ((hue >= 170) | (hue <= 4)) & (sat > 120) & (val > 120)
    return cv2.dilate((m.astype(np.uint8)) * 255, np.ones((3, 3), np.uint8))


def detect_team(mm, hsv, team, champs, templates):
    if not champs:
        return []
    mask = team_mask(hsv, team)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    hits = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        pad = 28
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(MSIZE, x + w + pad), min(MSIZE, y + h + pad)
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
            if best and best[0] >= CLS_MIN:
                hits.append((best[0], champ, best[1], best[2]))
    hits.sort(reverse=True)
    dets, used_c, used_p = [], set(), []
    for sc, champ, cx, cy in hits:
        if champ in used_c:
            continue
        if any((cx - ux) ** 2 + (cy - uy) ** 2 < NMS_DIST ** 2 for ux, uy in used_p):
            continue
        used_c.add(champ)
        used_p.append((cx, cy))
        dets.append({"champ": champ, "cx": cx, "cy": cy, "score": round(sc, 3)})
    return dets


def load_deaths(pass_dir):
    """[(t, {champ: muerto})] ordenado por t."""
    out = []
    for line in open(f"{pass_dir}/samples.jsonl"):
        r = json.loads(line)
        if "players" in r:
            out.append((r["t"], {NAME_FIX.get(p["champ"], p["champ"]): p["dead"]
                                 for p in r["players"]}))
    return out


def dead_at(deaths, t):
    if not deaths:
        return {}
    ts = [d[0] for d in deaths]
    i = min(max(bisect.bisect_left(ts, t) - 1, 0), len(deaths) - 1)
    if abs(deaths[i][0] - t) > 12 and i + 1 < len(deaths):
        i += 1
    return deaths[i][1]


def main():
    pass_dir, out_path = sys.argv[1], sys.argv[2]
    debug_ts = {float(a) for a in sys.argv[3:]}
    templates = build_templates()
    deaths = load_deaths(pass_dir)
    files = sorted(glob.glob(f"{pass_dir}/mm_*.png"))
    rows = []
    for fp in files:
        t = int(os.path.basename(fp)[3:9]) / 10.0
        if not (T_MIN <= t <= T_MAX):
            continue
        img = cv2.imread(fp)
        if img is None:
            continue
        mm = img[MY0:MY0 + MSIZE, MX0:MX0 + MSIZE]
        if mm.shape[0] != MSIZE or mm.shape[1] != MSIZE:
            continue
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        dmap = dead_at(deaths, t)
        dets = []
        for team in ("ORDER", "CHAOS"):
            champs = [c for c, tm in TEAMS.items() if tm == team and not dmap.get(c, False)]
            dets += detect_team(mm, hsv, team, champs, templates)
        if debug_ts and any(abs(t - dt) < 2 for dt in debug_ts):
            vis = mm.copy()
            for d in dets:
                cv2.circle(vis, (d["cx"], d["cy"]), 26, (0, 255, 0), 2)
                cv2.putText(vis, f"{d['champ'][:4]} {d['score']:.2f}",
                            (d["cx"] - 30, d["cy"] - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(f"{S}/debug_v5_t{int(t)}.png", vis)
        for d in dets:
            rows.append({"t": t, "champ": d["champ"],
                         "nx": round(d["cx"] / MSIZE, 4),
                         "ny": round(1 - d["cy"] / MSIZE, 4),
                         "score": d["score"]})
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(files)} frames, {len(rows)} det ({len(rows) / max(len(files), 1):.1f}/10)")


if __name__ == "__main__":
    main()
