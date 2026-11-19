"""Procesa una partida del corpus: detecciones -> trayectorias -> campamentos
-> visibilidad (si hay pasadas con niebla) -> segmentos -> decisiones ->
snapshots. Imprime las puertas de QA y actualiza meta.json.

Uso: python scripts/process_game.py <MATCH_ID> [--redetect]
"""
import argparse
import cv2
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import camps as camps_mod
from app import dataset, inventory, lexicon, segments, snapshot, trajectory, visibility
from app.config import (LEXICON_V, MM_GEOM_DEFAULT, game_dir, ensure_dirs,
                        VIS_VERIFY_SCORE, TRAJ_MIN_COVERAGE)


def patch_of(match: dict) -> str:
    v = match["info"]["gameVersion"].split(".")
    return f"{v[0]}.{v[1]}"


def geom_at_fn(pass_dir: Path):
    """geom_at(t) -> (x0, y0, size) dentro del PNG guardado.

    Formato v2 (fase 2): el PNG ya es el minimapa recortado, así que el
    recorte es (0, 0, lado de la imagen) -> se devuelve None y el llamante usa
    la imagen entera. Formato v1 (fase 1): recorte de pantalla con geometría
    fija o con 'geom_switches' (el minimapa cambió de tamaño a mitad de pasada).
    """
    meta = json.loads((pass_dir / "meta.json").read_text(encoding="utf-8")) \
        if (pass_dir / "meta.json").exists() else {"geom": list(MM_GEOM_DEFAULT)}
    if meta.get("format") == "minimap-crop-v2":
        return None
    switches = meta.get("geom_switches")
    if not switches:
        geom = meta.get("geom", MM_GEOM_DEFAULT)
        if isinstance(geom, dict):
            geom = (geom["x0"], geom["y0"], geom["size"])
        geom = tuple(geom)
        return lambda t: geom
    switches = sorted((s[0], tuple(s[1])) for s in switches)

    def geom_at(t):
        cur = switches[0][1]
        for t0, gm in switches:
            if t >= t0:
                cur = gm
        return cur
    return geom_at


def crop_minimap(img, geom_at, t):
    """Recorte del minimapa de un frame, según el formato de la pasada."""
    if geom_at is None:
        n = min(img.shape[0], img.shape[1])
        return img[:n, :n]
    mx0, my0, msize = geom_at(t)
    mm = img[my0:my0 + msize, mx0:mx0 + msize]
    if mm.shape[0] != msize or mm.shape[1] != msize:
        return None
    return mm


def load_prices(match: dict) -> dict:
    from app import ddragon
    try:
        return ddragon.item_prices(ddragon.version_for_patch(patch_of(match)))
    except Exception as e:
        print(f"AVISO: sin precios de ddragon ({e}); item_gold sera 0")
        return {}


def run_detector(g: Path, match: dict, out_file: Path):
    """Detección global sobre pass_omni (solo si no hay traj.jsonl)."""
    import cv2
    import glob
    import os
    from app import detect
    P = segments.participants(match)
    champs = {v["champ"]: ("ORDER" if v["team"] == 100 else "CHAOS") for v in P.values()}
    geom_at = geom_at_fn(g / "pass_omni")
    pl = snapshot.load_playerlist(g / "pass_omni")
    deaths = snapshot.deaths_series(pl)
    rows = []
    files = sorted(glob.glob(str(g / "pass_omni" / "mm_*.png")))
    probe = cv2.imread(files[0]) if files else None
    size_hint = None
    if probe is not None:
        mmp = crop_minimap(probe, geom_at, 0.0)
        size_hint = mmp.shape[0] if mmp is not None else None
    templates = detect.build_templates(list(champs), patch_of(match), size_hint)
    for k, fp in enumerate(files):
        t = int(os.path.basename(fp)[3:9]) / 10.0
        img = cv2.imread(fp)
        if img is None:
            continue
        mm = crop_minimap(img, geom_at, t)
        if mm is None:
            continue
        msize = mm.shape[0]
        hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
        dmap = {}
        for dt, dd in deaths:
            if abs(dt - t) <= 12:
                dmap = dd
                break
        dets = []
        for team in ("ORDER", "CHAOS"):
            alive = [c for c, tm in champs.items() if tm == team and not dmap.get(c, False)]
            dets += detect.detect_frame(mm, hsv, team, alive, templates, msize)
        for d in dets:
            rows.append({"t": t, "champ": d["champ"],
                         "nx": round(d["cx"] / msize, 4),
                         "ny": round(1 - d["cy"] / msize, 4),
                         "score": d["score"]})
        if k % 100 == 0:
            print(f"  detectando {k}/{len(files)}", flush=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  {len(rows)} detecciones ({len(rows)/max(len(files),1):.1f}/frame)")


def P_team(P: dict, champ: str) -> int:
    return next(v["team"] for v in P.values() if v["champ"] == champ)


def build_visibility(g: Path, match: dict, traj: dict, deaths: list) -> dict | None:
    """Visibilidad por equipo desde las pasadas con niebla.

    Se pregunta, en cada frame con niebla, si el icono APRENDIDO del campeón
    (app/icons.py, recortado de la pasada sin niebla) está dibujado donde
    sabemos que el campeón está. Los retratos genéricos de ddragon no valen:
    con ellos el detector "encontraba" campeones muertos en el 80-87% de las
    muestras.

    El método tiene ~79% de sensibilidad y ~2% de falsos (medido). Esa
    sensibilidad NO es uniforme: depende de cuánto se mueve el campeón. Por eso
    se mide POR CAMPEÓN usando la pasada de SU PROPIO equipo, donde por
    definición siempre es visible. La visibilidad corregida es entonces

        visto_por_el_rival / visto_por_los_suyos

    un cociente pareado que cancela el sesgo de movilidad, en vez de comparar
    contra una constante global.
    """
    import glob
    from app import icons

    P = segments.participants(match)
    team_of = {v["champ"]: v["team"] for v in P.values()}
    champs = list(team_of)

    def dead_at(champ, t):
        if not deaths:
            return False
        best = min(deaths, key=lambda d: abs(d[0] - t))
        return bool(best[1].get(champ, False)) if abs(best[0] - t) <= 12 else False

    omni_files = sorted(glob.glob(str(g / "pass_omni" / "mm_*.png")))
    if not omni_files:
        return None
    msize = cv2.imread(omni_files[0]).shape[0]
    icons_by_champ, counts = icons.extract(g / "pass_omni", traj, champs,
                                           dead_at, msize)
    missing = [c for c in champs if c not in icons_by_champ]
    print(f"  iconos aprendidos: {len(icons_by_champ)}/10"
          + (f"; SIN ICONO {missing}" if missing else ""))

    raw, totals = {}, {}
    for mode in ("blue", "red"):
        pdir = g / f"pass_{mode}"
        if not pdir.exists():
            return None
        # Las tres pasadas TIENEN que compartir la geometria del minimapa: el
        # icono se aprende en la omni y se busca en las de niebla. Con tamanos
        # distintos la correlacion se hunde, y lo peor es que NO falla — la
        # visibilidad se INFLA hacia "siempre visible", que es justo la
        # direccion que le enseñaria al modelo que nadie se esconde nunca.
        # Medido en EUW1_7984516268: con las tres a 508px, el azul veia al rojo
        # 0.747/0.137/0.563/0.88/0.585; recapturando SOLO la omni a 255px paso
        # a 0.909/0.327/0.968/1.0/1.0, y la pasada roja se quedo sin
        # sensibilidad. El guardia de suspect_no_fog pillo el rojo; el azul
        # habria pasado con numeros inventados.
        fog_files = sorted(glob.glob(str(pdir / "mm_*.png")))
        if not fog_files:
            return None
        fsize = cv2.imread(fog_files[0]).shape[0]
        # +-8px es ruido de calibracion (el corpus mezcla 508 y 507); un cambio
        # real de escala del HUD son cientos de px. Ver GEOM_SIZE_TOL en capture.
        if abs(fsize - msize) > 8:
            raise RuntimeError(
                f"pasada {mode} con minimapa de {fsize}px y omni de {msize}px: "
                "no se pueden mezclar geometrias. Recaptura LAS TRES pasadas "
                "de esta partida con el mismo HUD.")
        seen, n = icons.detect_rates(pdir, traj, champs, icons_by_champ,
                                     dead_at, msize)
        raw[mode], totals[mode] = seen, n

    out, meta_vis = {}, {}
    for mode, obs_team in (("blue", 100), ("red", 200)):
        own_mode = "blue" if obs_team == 100 else "red"
        enemies = [c for c in champs if team_of[c] != obs_team]
        rates, corrected = {}, {}
        for c in enemies:
            enemy_pass = mode                      # quien observa: obs_team
            own_pass = "red" if team_of[c] == 200 else "blue"
            n_e = max(totals[enemy_pass].get(c, 0), 1)
            n_o = max(totals[own_pass].get(c, 0), 1)
            obs = len(raw[enemy_pass].get(c, [])) / n_e
            sens = len(raw[own_pass].get(c, [])) / n_o
            rates[c] = {"observado": round(obs, 3), "sensibilidad": round(sens, 3)}
            corrected[c] = round(min(obs / sens, 1.0), 3) if sens > 0.25 else None
        out[mode] = {c: [list(iv) for iv in
                         visibility.intervals_from_times(raw[mode].get(c, []))]
                     for c in enemies}
        meta_vis[mode] = {"method": "learned-icon-paired-v3",
                          "per_champion": rates,
                          "visible_frac_corrected": corrected,
                          "icons_missing": missing}
        bad = [c for c, v in rates.items() if v["sensibilidad"] <= 0.25]
        if bad:
            print(f"  AVISO visibilidad {mode}: sensibilidad insuficiente en "
                  f"{bad} — su visibilidad queda sin valor (null), no inventada")
        # Una pasada capturada por error con la niebla en "Todos" pasa la
        # validacion del momento t=150 si alli se ven todos los rivales, y
        # luego se haria pasar por niebla de equipo. Se delata aqui: sin
        # niebla, TODOS los rivales salen visibles casi siempre.
        vals = [v for v in corrected.values() if v is not None]
        if vals and min(vals) > 0.90:
            # BLOQUEA: la visibilidad de esta pasada no se usa para entrenar
            # (antes solo avisaba y el snapshot recibia numeros igualmente)
            meta_vis[mode]["suspect_no_fog"] = True
            meta_vis[mode]["usable"] = False
            print(f"  AVISO visibilidad {mode}: los 5 rivales salen visibles "
                  f">90% ({corrected}). Es lo que se veria si esa pasada se "
                  f"capturo sin niebla de equipo: NO te fies de su info_pct "
                  f"y recaptura la pasada.")
        print(f"  visibilidad {mode}: rivales visibles (corregido) {corrected}")
    out["meta"] = meta_vis
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_id")
    ap.add_argument("--redetect", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    g = game_dir(args.match_id)
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
    P = segments.participants(match)
    enemies_of_blue = {v["champ"] for v in P.values() if v["team"] == 200}
    pl = snapshot.load_playerlist(g / "pass_omni")
    deaths = snapshot.deaths_series(pl)

    # 1. trayectorias
    traj_file = g / "traj.jsonl"
    if not traj_file.exists() or args.redetect:
        print("1. detector sobre pass_omni...")
        run_detector(g, match, traj_file)
    dets = trajectory.load_detections(traj_file)
    anchors = trajectory.mv5_anchors(timeline, P)
    traj = trajectory.fuse(dets, anchors, enemies_of_blue)
    import glob as _glob
    n_frames = max(len(_glob.glob(str(g / "pass_omni" / "mm_*.png"))), 1)
    coverage = {c: round(len(dets.get(c, [])) / n_frames, 2)
                for c in (v["champ"] for v in P.values())}
    flaco = {c: v for c, v in coverage.items() if v < TRAJ_MIN_COVERAGE}
    print(f"1. trayectorias: {len(traj)} campeones; cobertura {coverage}")
    if flaco:
        # un campeon que casi no se detecta desaparece del dataset EN SILENCIO
        # (le paso a Shaco por un umbral de score heredado de fase 1)
        print(f"   AVISO: cobertura baja en {flaco} — revisa el detector antes "
              "de fiarte de sus POV ni de su visibilidad")

    # 2. campamentos
    camps_file = g / "camps.json"
    if not camps_file.exists():
        print("2. rombos de campamentos...")
        dots = camps_mod.dot_series(g / "pass_omni", geom_at_fn(g / "pass_omni"))
        camps_file.write_text(json.dumps(dots), encoding="utf-8")
    dots = json.loads(camps_file.read_text(encoding="utf-8"))
    ckills = camps_mod.camp_kills(dots, traj)
    cstates = camps_mod.alive_series(dots, traj)
    print(f"2. campamentos: {len(ckills)} muertes de campamento")

    # 3. visibilidad
    vis_file = g / "visibility.json"
    vis_data = None
    if vis_file.exists():
        vis_data = visibility.load(vis_file)
    elif (g / "pass_blue").exists() and (g / "pass_red").exists():
        print("3. visibilidad desde pasadas con niebla...")
        vis_data = build_visibility(g, match, traj, deaths)
        if vis_data:
            vis_file.write_text(json.dumps(vis_data), encoding="utf-8")
    print(f"3. visibilidad: {'OK' if vis_data else 'NO DISPONIBLE (sin pasadas con niebla)'}")

    # 4. builds: QA contra el final oficial (sin visión ni botas: ver inventory.py)
    boots_ids = set()
    try:
        from app import ddragon
        v = ddragon.version_for_patch(patch_of(match))
        boots_ids = {int(i) for i, it in ddragon.items(v)["data"].items()
                     if "Boots" in it.get("tags", [])}
    except Exception as e:
        print(f"AVISO: sin ddragon para excluir botas del QA ({e})")
    diff = inventory.final_diff_vs_match(timeline, match, boots_ids=boots_ids)
    bad = [pid for pid, d in diff.items() if not d["ok"]]
    print(f"4. builds: inventario final coincide en {10 - len(bad)}/10 (comparable)"
          + (f" (difieren: { {p: diff[p] for p in bad} })" if bad else ""))

    # 5. segmentos + decisiones + snapshots por jungla
    prices = load_prices(match)
    pl = snapshot.load_playerlist(g / "pass_omni")
    deaths = snapshot.deaths_series(pl)
    minimaps = snapshot.MinimapIndex(g)
    all_segments, all_decisions = [], []
    for me in segments.junglers(match):
        my_team = P[me]["team"]
        segs = segments.build_segments(match, timeline, traj, ckills, me)
        decs = lexicon.decisions_from_segments(segs)
        # La visibilidad llega con su sensibilidad por campeón y sus marcas de
        # sospecha: si la pasada no es fiable, el snapshot dirá DESCONOCIDO en
        # vez de un número.
        vis_team = visibility.team_visibility_for(vis_data, my_team, deaths)
        if vis_team is not None and not vis_team.usable:
            print(f"   AVISO: visibilidad NO utilizable para el POV "
                  f"{P[me]['champ']}: {vis_team.reason}")
        for d in decs:
            # cstates completo: build_snapshot evalúa los campamentos en SU
            # instante causal (t0 - eps), no en t0
            snap = snapshot.build_snapshot(
                match, timeline, traj, cstates,
                pl, minimaps, vis_team, me, d["t0"], prices)
            all_decisions.append({
                "match_id": args.match_id, "pid": me, "champ": P[me]["champ"],
                "team": my_team, "lexicon_v": LEXICON_V,
                "t0": d["t0"], "t1": d["t1"], "token": d["token"],
                "kind": d["kind"], "chain": d["chain"], "snapshot": snap})
        for s in segs:
            all_segments.append({"match_id": args.match_id, "pid": me, **s})
        toks = Counter(d["token"] for d in decs)
        print(f"5. POV {P[me]['champ']} (pid {me}): {len(segs)} segmentos, "
              f"{len(decs)} decisiones — {dict(toks.most_common(5))}")

    with open(g / "segments.jsonl", "w", encoding="utf-8") as fh:
        for s in all_segments:
            fh.write(json.dumps(s) + "\n")
    with open(g / "decisions.jsonl", "w", encoding="utf-8") as fh:
        for d in all_decisions:
            fh.write(json.dumps(d) + "\n")

    # feature smoke: cada decisión debe vectorizar sin excepciones
    for d in all_decisions:
        dataset.feature_vector(d)

    meta_file = g / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    meta.update({"lexicon_v": LEXICON_V, "patch": patch_of(match),
                 "n_decisions": len(all_decisions), "n_segments": len(all_segments),
                 "vision_available": bool(vis_data),
                 "builds_final_ok": 10 - len(bad)})
    meta_file.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"OK: {len(all_decisions)} decisiones en {g / 'decisions.jsonl'}")


if __name__ == "__main__":
    main()
