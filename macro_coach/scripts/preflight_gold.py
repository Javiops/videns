"""Preflight antes de una tanda: ¿sale el oro bien de la pantalla, a 32x?

Captura UNA pasada TRUTH-SEEKER completa de una partida a un directorio aparte
(no toca el corpus) y contrasta el oro leido contra Match-V5. Responde de una
vez a las tres preguntas que bloquean una tanda desatendida:

  1. ¿aguanta la replay? (si el cliente se cae, se ve aqui y no a las 3 horas)
  2. ¿esta el marcador en vista de oro? — `ensure_gold_view` lo verifica
     LEYENDO las diez celdas, asi que si el remapeo de mouse3 no esta, falla
     aqui mismo en vez de capturar una tanda entera sin oro;
  3. ¿es legible a 32x con la replay EN MARCHA? Todo lo medido hasta ahora era
     con la replay pausada. `frac_legible` lo contesta.

Uso: python scripts/preflight_gold.py <MATCH_ID> [--speed N]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mss

from app import lcu, ocr
from app.capture import (api_get, capture_pass, ensure_replay_open, prepare,
                         replay_api_up)
from app.config import DATA, game_dir

TOLERANCIA = 25
SALIDA = DATA / "preflight"


def espera_api(timeout_s: float = 240.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if replay_api_up():
            time.sleep(8)
            return True
        time.sleep(3)
    return False


def comprueba_partida(match: dict) -> None:
    """La replay abierta tiene que ser LA de este match_id.

    Si quedo otra abierta de antes, todo lo demas funcionaria y el contraste
    contra Match-V5 saldria catastrofico — pareceria un fallo del OCR cuando en
    realidad estariamos leyendo otra partida. La Replay API no da el id, pero
    la `playerlist` da los diez campeones, y eso identifica la partida.
    """
    from app.names import internal
    esperados = {internal(p["championName"]) for p in match["info"]["participants"]}
    vistos = {internal(p["championName"])
              for p in api_get("/liveclientdata/playerlist")}
    if vistos != esperados:
        raise SystemExit(
            "la replay abierta NO es la pedida.\n"
            f"  esperaba: {sorted(esperados)}\n"
            f"  hay:      {sorted(vistos)}\n"
            "Cierra el juego que este abierto y vuelve a lanzar.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_id")
    ap.add_argument("--keep", action="store_true",
                    help="no borrar la pasada de prueba al terminar")
    args = ap.parse_args()
    mid = args.match_id

    g = game_dir(mid)
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
    out = SALIDA / mid
    if out.exists():
        shutil.rmtree(out)

    meta = lcu.replay_metadata(mid) or {}
    if meta.get("state") != "watch":
        print("descargando replay...", flush=True)
        lcu.download_replay(mid, match)
    lcu.watch_replay(mid)
    if not espera_api():
        raise SystemExit("la Replay API no levanta: la replay no ha abierto")

    ensure_replay_open(mid)
    comprueba_partida(match)
    with mss.MSS() as sct:
        ctx = prepare(match, sct)
    gv = ctx["gold_view"]
    print(f"  vista de oro: {gv['gold_view']} (verificado por "
          f"{gv.get('verified_by')}, {gv})", flush=True)
    if not gv["gold_view"]:
        raise SystemExit(
            "el marcador NO esta en vista de oro. Es el toggle que el owner "
            "remapeo a mouse3: sin el, una tanda entera se captura sin oro "
            "(ya paso con las 40 primeras). Revisa el remapeo antes de seguir.")

    print("pasada TRUTH-SEEKER de prueba...", flush=True)
    pmeta = capture_pass(out, "omni", ctx, timeline)
    print(f"  {pmeta['frames']} frames a {pmeta['speed']}x, "
          f"duplicados {pmeta['dup_frac']}, "
          f"escrituras perdidas {pmeta['writer_dropped']}", flush=True)

    plantillas = ocr.carga()
    filas = []
    for p in sorted(out.glob("sb_*.png")):
        t = int(p.stem.split("_")[1]) / 10.0
        r = ocr.lee_marcador(cv2.imread(str(p)), plantillas)
        filas.append({"t": round(t, 2), "gold": r["gold"], "leidos": r["leidos"]})
    inf = ocr.valida_contra_timeline(filas, timeline, TOLERANCIA)
    inf["speed"] = pmeta["speed"]
    inf["dup_frac"] = pmeta["dup_frac"]
    (SALIDA / f"{mid}_report.json").write_text(json.dumps(inf, indent=1),
                                               encoding="utf-8")

    print(f"\n--- OCR del oro contra Match-V5 ({mid}, {pmeta['speed']}x) ---")
    print(f"  recortes            {inf['muestras']}")
    print(f"  celdas legibles     {inf['celdas_leidas']}/{inf['celdas_totales']}"
          f"  ({100*inf['frac_legible']:.1f}%)")
    print(f"  anclas de Match-V5  {inf['anclas_mv5']}"
          f" (ilegibles {inf['anclas_ilegibles']})")
    if "error_mediana" in inf:
        print(f"  |error| mediana     {inf['error_mediana']:.0f} oro")
        print(f"  |error| p90 / max   {inf['error_p90']:.0f} / {inf['error_max']:.0f}")
        print(f"  dentro de {TOLERANCIA} oro     {100*inf['frac_en_tolerancia']:.2f}%"
              "   (informativo: le afecta el desfase temporal)")
    print(f"  bajadas del total   {inf['bajadas_de_total']}  (deben ser 0)")
    print(f"  fuera de horquilla  {inf['horquilla_fuera']}/"
          f"{inf['horquilla_comprobadas']}  (deben ser 0)")

    # El criterio es la HORQUILLA, no el error contra el anclaje mas cercano:
    # ese error incluye el desfase temporal entre la captura y el frame de 60s,
    # que a 32x es hasta ~1s y vale decenas de oro sin que nadie haya leido mal.
    # La horquilla no depende del tiempo y se exige en todas las muestras.
    ok = (inf["frac_legible"] > 0.99 and inf["bajadas_de_total"] == 0
          and inf["horquilla_fuera"] == 0)
    print("\nVEREDICTO:", "listo para la tanda" if ok else "NO listo, mirar arriba")
    if not args.keep:
        shutil.rmtree(out, ignore_errors=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
