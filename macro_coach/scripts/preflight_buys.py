"""Prototipo: leer la cartera JUSTO antes y despues de cada compra saltando la
replay a esos instantes (pasada dirigida: salta -> pausa -> foto del marcador).

Por que: el ~22% de las compras quedan ACOTADAS porque dentro de la ventana de
~8 s entre lecturas OCR entra oro no localizable (el CS no es un evento del
timeline: se sabe CUANTO entro, no CUANDO). Una lectura a tp±MARGEN estrecha
la ventana a ~3 s — casi siempre quieta — y ademas permite MEDIR el coste
pagado en casi todas las compras, no solo en las certificadas por muerte.

Que hace (sin tocar el corpus; escribe en data/preflight_buys/<MID>/):
  1. abre la replay y pone el marcador en vista de oro (verificado leyendo);
  2. recorre los instantes de parada SOLO HACIA DELANTE (un seek largo hacia
     atras cerro el juego en su dia): salta, espera a que el renderer asiente
     (la pantalla debe CAMBIAR respecto a la parada anterior y luego repetirse
     — la Replay API responde antes de que el HUD se redibuje), y recorta el
     marcador;
  3. offline: OCR de los recortes, re-resuelve las compras con las lecturas
     nuevas añadidas y compara contra la resolucion de produccion;
  4. valida contra las anclas de Match-V5 una resolucion que NUNCA las vio
     (solo OCR de pasada + lecturas dirigidas), como hace resolver_partida.

El desfase HUD/reloj (<1 s, ocr.LAG_HUD_S) es la razon del margen: leer EN tp
es ambiguo justo cuando la cartera pega el salto del precio; a tp-1.5 s la
lectura es de un instante conocido y de ahi a la compra solo cabe pasivo.

NUNCA lanzar esto como hijo del shell de un agente (se reaper a mitad, ver
CLAUDE.md): Start-Process con log redirigido, y la maquina sin tocar mientras.

Uso: python scripts/preflight_buys.py <MATCH_ID> [--margin 1.5] [--keep]
"""
import argparse
import bisect
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import mss
import numpy as np

from app import buys, lcu, ocr
from app.capture import (END_MARGIN_S, HUD_STATE, api_get, api_post,
                         ensure_gold_view, ensure_replay_open, focus_game,
                         grab, scoreboard_rect)
from app.config import DATA, game_dir
from preflight_gold import comprueba_partida, espera_api

SALIDA = DATA / "preflight_buys"
# Umbral del veredicto sobre las anclas de Match-V5 (mismo espiritu que el
# err_max ~10 oro de la validacion de produccion, con margen).
ANCLA_MAX_OK = 25.0


def paradas(timeline: dict, margin: float, length: float) -> list:
    """Instantes de lectura: antes y despues de cada evento de tienda de los
    diez, deduplicados (dos lecturas a <margin*0.6 aportan lo mismo) y
    recortados al margen final de la replay (al acabar, el juego se cierra)."""
    ts = sorted({e["timestamp"] / 1000.0
                 for f in timeline["info"]["frames"] for e in f["events"]
                 if e["type"] in ("ITEM_PURCHASED", "ITEM_SOLD", "ITEM_UNDO")
                 and 0 < e.get("participantId", 0) <= 10})
    brutos = sorted(x for t in ts for x in (t - margin, t + margin))
    out = []
    for s in brutos:
        s = round(min(max(s, 0.0), length - END_MARGIN_S), 1)
        if not out or s - out[-1] >= margin * 0.6:
            out.append(s)
    return out


def captura_parada(sct, t_obj: float, pantalla_prev, rect) -> tuple:
    """(crop_marcador, pantalla_completa, info) en la parada t_obj.

    El seek ya esta pedido. Tres esperas encadenadas, todas medibles:
      a) el reloj de la Replay API llega a t_obj y esta en pausa;
      b) la pantalla CAMBIA respecto a la parada anterior (la API va por
         delante del renderer: sin esto se guarda el frame de la parada previa
         con nombre nuevo);
      c) dos capturas seguidas identicas (el redibujado asento).
    Si (b) no llega, se empuja: play un instante y pausa de nuevo.
    """
    x0, y0, x1, y1 = rect
    info = {"t": t_obj, "reloj_ok": False, "redibujada": False, "asentada": False,
            "empujones": 0}
    t0 = time.time()
    while time.time() - t0 < 8.0:
        try:
            pb = api_get("/replay/playback")
            if pb.get("paused") and abs(float(pb.get("time", -99)) - t_obj) <= 0.6:
                info["reloj_ok"] = True
                break
        except Exception:
            pass
        time.sleep(0.2)

    pantalla = None
    for intento in range(16):
        pantalla = grab(sct)
        if pantalla_prev is None or not np.array_equal(pantalla, pantalla_prev):
            info["redibujada"] = True
            break
        if intento in (5, 10):
            # el renderer no redibuja en pausa: empujon de ~0.3s y re-pausa
            api_post("/replay/playback", {"speed": 1, "paused": False})
            time.sleep(0.3)
            api_post("/replay/playback", {"time": t_obj, "paused": True})
            info["empujones"] += 1
        time.sleep(0.25)
    for _ in range(8):
        time.sleep(0.2)
        otra = grab(sct)
        if np.array_equal(otra, pantalla):
            info["asentada"] = True
            break
        pantalla = otra
    return pantalla[y0:y1, x0:x1], pantalla, info


def filtra_dirigidas(base: list, extra: list, eventos: dict,
                     tol: float = 6.0) -> tuple:
    """Depura las lecturas dirigidas contra la fisica del marcador.

    Medido en el prototipo (EUW1_7986441889): tras un seek el marcador puede
    renderizarse VIEJO varios segundos aunque la pantalla ya cambio — totales
    que BAJAN respecto a la lectura vecina y carteras pre-compra 1.5 s despues
    de la compra (una metio un error de 1071 oro en el control de coste). Dos
    reglas que NO usan el modelo de precios, solo el total (oro GANADO,
    monotono por construccion):

      R1  el total nunca baja: una dirigida cuyo total contradiga a una vecina
          legible cae (si la vecina tambien es dirigida, caen las dos);
      R2  en un tramo sin eventos de tienda del pid, cartera y total suben
          IGUAL (todo ingreso suma a ambos): |dcur - dtot| > tol descarta a
          las dirigidas del par. Con evento de tienda en medio no se aplica
          (gastar mueve la cartera y no el total; seria ademas circular con
          el modelo de precios que el control valida).

    base = lecturas de la pasada + anclas mv5 (contexto fiable); solo se
    descartan celdas DIRIGIDAS. Devuelve (extra_filtrado, celdas_descartadas).
    """
    ev_ts = {pid: sorted(e["t"] for e in eventos.get(pid, []))
             for pid in range(1, 11)}
    drop = set()                                   # (t, pid) dirigidas anuladas
    dirigidas = {round(t, 3) for t, _ in extra}
    for pid in range(1, 11):
        seq = []
        for t, fila in list(base) + list(extra):
            g = fila[pid - 1]
            if g is not None:
                seq.append((t, g[0], g[1], round(t, 3) in dirigidas))
        seq.sort()
        for _ in range(2):                 # una caida puede destapar otra
            vivos = [r for r in seq if not (r[3] and (r[0], pid) in drop)]
            cambio = False
            for a, b in zip(vivos, vivos[1:]):
                mal = b[2] < a[2] - 0.5            # R1: el total baja
                if not mal:
                    sin_ev = (bisect.bisect_right(ev_ts[pid], a[0])
                              == bisect.bisect_right(ev_ts[pid], b[0]))
                    mal = (sin_ev
                           and abs((b[1] - a[1]) - (b[2] - a[2])) > tol)
                if mal:
                    for r in (a, b):
                        if r[3] and (r[0], pid) not in drop:
                            drop.add((r[0], pid))
                            cambio = True
            if not cambio:
                break
    out = [(t, [None if (t, pid) in drop else fila[pid - 1]
                for pid in range(1, 11)]) for t, fila in extra]
    return out, len(drop)


def resolver_todos(lecturas: list, eventos: dict, cert) -> list:
    filas = []
    for pid in range(1, 11):
        filas += buys.resolver_pid(lecturas, eventos[pid], pid,
                                   solo_pasivo=cert(pid))
    return filas


def resumen_de(filas: list) -> dict:
    EX = ("ventana_quieta", "tramo_ida", "tramo_vuelta")
    anchos = sorted(f["intervalo"][1] - f["intervalo"][0]
                    for f in filas if f["metodo"] == "acotado")
    return {"compras": len(filas),
            "exactas": sum(1 for f in filas if f["metodo"] in EX),
            "por_metodo": {m: sum(1 for f in filas if f["metodo"] == m)
                           for m in EX + ("acotado", "sin_lectura")},
            "intervalo_mediano": anchos[len(anchos) // 2] if anchos else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_id")
    ap.add_argument("--margin", type=float, default=1.5,
                    help="segundos antes/despues de cada evento (> LAG_HUD_S)")
    ap.add_argument("--keep", action="store_true",
                    help="no borrar los recortes al terminar")
    ap.add_argument("--solo-offline", action="store_true",
                    help="reanalizar una captura previa (--keep) sin recapturar")
    args = ap.parse_args()
    mid = args.match_id
    if args.margin <= ocr.LAG_HUD_S:
        raise SystemExit(f"--margin debe superar LAG_HUD_S={ocr.LAG_HUD_S}: "
                         "leer mas cerca del evento es ambiguo (desfase HUD)")

    g = game_dir(mid)
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
    out = SALIDA / mid
    plantillas = ocr.carga()      # falla YA si no hay plantillas de digitos

    if args.solo_offline:
        if not (out / "lecturas.jsonl").exists():
            raise SystemExit(f"no hay captura previa en {out} (usa --keep)")
        infos, dur_captura, stops = [], 0.0, []
    else:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)

        meta = lcu.replay_metadata(mid) or {}
        if meta.get("state") != "watch":
            print("descargando replay...", flush=True)
            lcu.download_replay(mid, match)
        lcu.watch_replay(mid)
        if not espera_api():
            raise SystemExit("la Replay API no levanta: la replay no ha abierto")
        ensure_replay_open(mid)
        comprueba_partida(match)

        length = float(api_get("/replay/playback")["length"])
        stops = paradas(timeline, args.margin, length)
        print(f"{len(stops)} paradas para {mid} (margen {args.margin}s, "
              f"replay de {length:.0f}s)", flush=True)

        api_post("/replay/render", HUD_STATE)
        focus_game()
        # pausar YA y colocarse en la primera parada: la replay abre
        # reproduciendo y cada segundo es un seek hacia atras mas largo luego
        api_post("/replay/playback",
                 {"time": stops[0], "paused": True, "speed": 1})
        time.sleep(5)

        with mss.MSS() as sct:
            gv = ensure_gold_view(sct)
            print(f"  vista de oro: {gv['gold_view']} "
                  f"(por {gv.get('verified_by')}, "
                  f"{gv.get('celdas', gv.get('colorfulness'))})", flush=True)
            if not gv["gold_view"]:
                raise SystemExit("el marcador NO esta en vista de oro (mouse3);"
                                 " sin ella los recortes no llevan cartera")
            rect = scoreboard_rect()
            pantalla_prev = None
            infos = []
            t_ini = time.time()
            for i, s in enumerate(stops):
                api_post("/replay/playback", {"time": s, "paused": True})
                crop, pantalla_prev, info = captura_parada(sct, s,
                                                           pantalla_prev, rect)
                cv2.imwrite(str(out / f"sb_{int(round(s * 10)):07d}.png"), crop)
                infos.append(info)
                if (i + 1) % 25 == 0:
                    por_min = (i + 1) / max(time.time() - t_ini, 1) * 60
                    print(f"  {i + 1}/{len(stops)} paradas "
                          f"({por_min:.0f}/min; reloj_ok "
                          f"{sum(x['reloj_ok'] for x in infos)}, redibujadas "
                          f"{sum(x['redibujada'] for x in infos)}, empujones "
                          f"{sum(x['empujones'] for x in infos)})", flush=True)
        dur_captura = time.time() - t_ini
        api_post("/replay/playback", {"paused": True})
        print(f"captura dirigida: {len(stops)} paradas en "
              f"{dur_captura/60:.1f} min", flush=True)

    # ---------------- offline: OCR + filtro + re-resolucion ----------------
    extra = []
    celdas_ok = celdas_tot = 0
    if args.solo_offline:
        for line in open(out / "lecturas.jsonl", encoding="utf-8"):
            r = json.loads(line)
            extra.append((r["t"], [tuple(x) if x else None for x in r["gold"]]))
            celdas_ok += sum(1 for x in r["gold"] if x)
            celdas_tot += 10
        stops = [t for t, _ in extra]
    else:
        for p in sorted(out.glob("sb_*.png")):
            t = int(p.stem.split("_")[1]) / 10.0
            r = ocr.lee_marcador(cv2.imread(str(p)), plantillas)
            extra.append((t, [tuple(x) if x else None for x in r["gold"]]))
            celdas_ok += r["leidos"]
            celdas_tot += 10
        (out / "lecturas.jsonl").write_text(
            "\n".join(json.dumps({"t": t,
                                  "gold": [list(x) if x else None for x in fila]})
                      for t, fila in extra) + "\n", encoding="utf-8")

    from app import ddragon
    patch = ".".join(match["info"]["gameVersion"].split(".")[:2])
    data = ddragon.items(ddragon.version_for_patch(patch))["data"]
    prices = {int(k): x["gold"]["total"] for k, x in data.items()}
    sells = {int(k): x["gold"]["sell"] for k, x in data.items()}
    eventos = buys.eventos_de_compra(timeline, prices, sells)
    cert = buys._certificador(mid, match, timeline)
    ocr_l = buys.lecturas_ocr(g)
    mv5 = buys.lecturas_mv5(timeline)

    extra, descartadas = filtra_dirigidas(ocr_l + mv5, extra, eventos)

    antes = resumen_de(resolver_todos(sorted(ocr_l + mv5), eventos, cert))
    filas_mejor = resolver_todos(sorted(ocr_l + mv5 + extra), eventos, cert)
    despues = resumen_de(filas_mejor)
    # validacion honesta: resolucion que NUNCA vio las anclas de Match-V5,
    # contrastada contra ellas (mismo esquema que resolver_partida)
    filas_sin_mv5 = resolver_todos(sorted(ocr_l + extra), eventos, cert)
    valida = buys._valida(filas_sin_mv5, sorted(ocr_l + extra), timeline)

    inf = {"match_id": mid, "margin_s": args.margin, "paradas": len(stops),
           "min_captura": round(dur_captura / 60, 1),
           "paradas_reloj_ok": sum(x["reloj_ok"] for x in infos),
           "paradas_redibujadas": sum(x["redibujada"] for x in infos),
           "paradas_asentadas": sum(x["asentada"] for x in infos),
           "empujones": sum(x["empujones"] for x in infos),
           "frac_celdas_legibles": round(celdas_ok / max(celdas_tot, 1), 4),
           "celdas_dirigidas_descartadas": descartadas,
           "antes": antes, "despues": despues, "validacion": valida}
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / f"{mid}_report.json").write_text(json.dumps(inf, indent=1),
                                               encoding="utf-8")

    print(f"\n--- compras con lecturas dirigidas ({mid}) ---")
    print(f"  paradas: {len(stops)} en {inf['min_captura']} min; "
          f"redibujadas {inf['paradas_redibujadas']}, "
          f"asentadas {inf['paradas_asentadas']}, empujones {inf['empujones']}")
    print(f"  legibilidad OCR de las dirigidas: "
          f"{100 * inf['frac_celdas_legibles']:.1f}%")
    print(f"  celdas dirigidas descartadas por el filtro (HUD viejo tras "
          f"seek): {descartadas}")
    print(f"  exactas: {antes['exactas']}/{antes['compras']} -> "
          f"{despues['exactas']}/{despues['compras']}")
    print(f"  por metodo: {antes['por_metodo']} -> {despues['por_metodo']}")
    print(f"  intervalo mediano de las acotadas restantes: "
          f"{antes['intervalo_mediano']} -> {despues['intervalo_mediano']}")
    print(f"  validacion (sin anclas en la entrada): {valida}")

    anclas_ok = (valida.get("anclas", {}).get("err_max", 1e9) <= ANCLA_MAX_OK)
    coste = valida.get("coste", {})
    mejora = despues["exactas"] > antes["exactas"]
    ok = (inf["frac_celdas_legibles"] > 0.95 and mejora and anclas_ok)
    print(f"  control de coste certificado: n={coste.get('n', 0)}, "
          f"err_max={coste.get('err_max')}")
    print("\nVEREDICTO:", "funciona — escalable a la tanda" if ok
          else "NO concluyente, mirar arriba")
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
