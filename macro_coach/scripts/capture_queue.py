"""Orquestador nocturno: recorre la cola y captura las 3 pasadas de cada
partida — abre la replay (LCU), espera la Replay API, captura omni/blue/red,
cierra el proceso del juego y pasa a la siguiente. ~12 min por partida.

Requisitos: cliente de League abierto y logueado, replays descargadas
(scripts/download_queue.py), EnableReplayApi=1, pantalla sin bloquear y
NADIE tocando el equipo. Si la validación de niebla falla en la primera
partida, el run entero se detiene (riesgo abierto #1: mejor parar que quemar
la noche capturando solo omni).

Aguanta caidas del juego: cada pasada se reintenta 3 veces reabriendo la
replay, y al final se da una SEGUNDA VUELTA sobre las partidas que fallaron
(`--vueltas 1` la desactiva). Como la cola es reanudable, reintentar sale
gratis cuando no hay fallos.

Uso: python scripts/capture_queue.py [--limit N] [--start-at MATCH_ID]
                                     [--vueltas N]
Al acabar: scripts/process_game.py <ID> por partida (o un bucle).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mss

from app import lcu
from app.capture import (api_get, capture_pass, ensure_replay_open, prepare,
                         replay_api_up, set_fog)
from app.config import QUEUE_FILE, game_dir

GAME_EXE = "League of Legends.exe"


def replay_api_up() -> bool:
    try:
        api_get("/replay/playback")
        return True
    except Exception:
        return False


def wait_replay_api(timeout_s: float = 240.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if replay_api_up():
            time.sleep(8)          # deja asentar la carga
            return True
        time.sleep(3)
    return False


def kill_game():
    subprocess.run(["taskkill", "/IM", GAME_EXE, "/F"],
                   capture_output=True)
    t0 = time.time()
    while replay_api_up() and time.time() - t0 < 30:
        time.sleep(2)
    time.sleep(4)


def passes_missing(g: Path) -> list:
    return [m for m in ("omni", "blue", "red")
            if not (g / f"pass_{m}" / "samples.jsonl").exists()]


def captura_partida(mid: str, etiqueta: str) -> str:
    """Las pasadas que falten de una partida. Devuelve 'ok', 'fallo' o 'abort'.

    El juego se cae de vez en cuando por su cuenta (`SentryHandleException`,
    visto tambien en la noche de captura buena). Por eso cada pasada se
    reintenta 3 veces reabriendo la replay: una caida cuesta una pasada, no la
    noche. Lo que no se arregla aqui se arregla con la segunda vuelta de main().
    """
    g = game_dir(mid)
    missing = passes_missing(g)
    if not missing:
        print(f"{etiqueta} {mid}: 3 pasadas ya hechas, salto")
        return "ok"
    print(f"{etiqueta} {mid}: faltan {missing}", flush=True)
    try:
        match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
        timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))
        meta = lcu.replay_metadata(mid) or {}
        if meta.get("state") != "watch":
            print("  descargando replay...", flush=True)
            lcu.download_replay(mid, match)
        lcu.watch_replay(mid)
        if not wait_replay_api():
            raise RuntimeError("la Replay API no levanta (carga fallida)")
        ctx = None
        for mode in missing:
            for attempt in (1, 2, 3):
                # entre pasadas el juego puede haberse cerrado (fin de
                # replay) o caerse; se reabre y se reintenta. Se recalibra
                # siempre: el minimapa puede haberse movido de tamano o
                # sitio entre pasadas.
                try:
                    # prepare() DENTRO del try: una calibracion fallida
                    # tiene que reintentar como cualquier otro fallo
                    ensure_replay_open(mid)
                    with mss.MSS() as sct:
                        ctx = prepare(match, sct)
                    print(f"  minimapa {ctx['geom']['x0']},{ctx['geom']['y0']} "
                          f"{ctx['geom']['size']}px (corr {ctx['geom']['score']})",
                          flush=True)
                    print(f"  pasada {mode} (intento {attempt})...", flush=True)
                    capture_pass(g / f"pass_{mode}", mode, ctx, timeline)
                    break
                except SystemExit:
                    raise
                except Exception as e:
                    print(f"  fallo en {mode}: {e}; reintento", flush=True)
                    ctx = None
                    time.sleep(5)
            else:
                raise RuntimeError(f"pasada {mode} fallida tras 3 intentos")
            time.sleep(3)
        return "ok"
    except SystemExit as e:
        # ABORT de validación de niebla o de foco: parar el run entero
        print(f"\nRUN DETENIDO en {mid}: {e}\n"
              "Si es la niebla: valida a mano el selector del HUD y "
              "documenta el modo correcto antes de relanzar.")
        return "abort"
    except Exception as e:
        print(f"  FALLO {mid}: {e} — sigo con la siguiente")
        return "fallo"
    finally:
        kill_game()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start-at", default=None)
    ap.add_argument("--vueltas", type=int, default=2,
                    help="vueltas sobre los fallos (1 = sin segunda vuelta)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(QUEUE_FILE, encoding="utf-8")]
    if args.start_at:
        idx = next(i for i, r in enumerate(rows) if r["match_id"] == args.start_at)
        rows = rows[idx:]
    if args.limit:
        rows = rows[:args.limit]

    pendientes = [r["match_id"] for r in rows]
    done, failed, abortado = [], [], False
    # Segunda vuelta sobre los fallos: una caida del juego es transitoria mas a
    # menudo que no, y la cola es reanudable (las pasadas hechas se saltan), asi
    # que reintentar al final sale gratis cuando no hay fallos y rescata la
    # partida cuando los hay. Sin esto, una partida que agota sus 3 intentos se
    # pierde para siempre aunque el problema durase un minuto.
    for vuelta in range(1, args.vueltas + 1):
        if not pendientes:
            break
        if vuelta > 1:
            print(f"\n--- vuelta {vuelta}: reintento de {len(pendientes)} "
                  f"fallo(s) ---", flush=True)
            time.sleep(30)
        failed = []
        for i, mid in enumerate(pendientes):
            r = captura_partida(mid, f"[v{vuelta} {i+1}/{len(pendientes)}]")
            if r == "ok":
                done.append(mid)
            elif r == "abort":
                failed.append(mid)
                abortado = True
                break
            else:
                failed.append(mid)
        if abortado:
            break
        pendientes = failed

    print(f"\n{len(done)} partidas capturadas, {len(failed)} fallos")
    if failed:
        print("fallos:", failed)
    if done:
        print("Siguiente: procesar —")
        print("  for %i in (" + " ".join(done[:5]) + " ...) do "
              ".venv\\Scripts\\python.exe scripts\\process_game.py %i")


if __name__ == "__main__":
    main()
