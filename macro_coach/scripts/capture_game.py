"""Las tres pasadas de captura de una partida con la replay YA ABIERTA.

Uso: python scripts/capture_game.py <MATCH_ID> [--passes omni,blue,red]

Reglas de captura: no tocar ratón ni teclado durante las pasadas (el script
usa el ratón para el desplegable de niebla entre pasadas); EnableReplayApi=1
en game.cfg antes de abrir la replay. ~3.5 min por pasada a 8x.

La geometría del minimapa se calibra al principio y cada pasada valida su
niebla mirando qué campeones se ven; si la niebla no es la pedida, aborta.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mss

from app.capture import (api_get, capture_pass, ensure_replay_open, prepare,
                         replay_api_up, set_fog)
from app.config import game_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_id")
    ap.add_argument("--passes", default="omni,blue,red")
    args = ap.parse_args()
    g = game_dir(args.match_id)
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    timeline = json.loads((g / "mv5_timeline.json").read_text(encoding="utf-8"))

    # No se exige la replay abierta: ensure_replay_open la abre por LCU si
    # hace falta (el juego se cierra al terminar una replay).
    ctx = None
    for mode in args.passes.split(","):
        mode = mode.strip()
        out = g / f"pass_{mode}"
        if (out / "samples.jsonl").exists():
            print(f"[{mode}] ya capturada, salto")
            continue
        for attempt in (1, 2, 3):
            ensure_replay_open(args.match_id)
            # se recalibra ANTES DE CADA PASADA: mover el minimapa entre
            # pasadas corromperia en silencio las siguientes
            with mss.MSS() as sct:
                ctx = prepare(match, sct)
            print(f"minimapa calibrado: {ctx['geom']['x0']},{ctx['geom']['y0']} "
                  f"{ctx['geom']['size']}px (corr {ctx['geom']['score']})")
            print(f"[{mode}] pasada completa a 8x (intento {attempt})...")
            try:
                capture_pass(out, mode, ctx, timeline)
                break
            except SystemExit:
                raise                      # abortos de validacion: no reintentar
            except Exception as e:
                print(f"[{mode}] fallo ({e}); reintento tras reabrir", flush=True)
                ctx = None
                time.sleep(5)
        else:
            raise SystemExit(f"[{mode}] no se pudo capturar tras 3 intentos")
        time.sleep(3)
    if replay_api_up():
        set_fog("omni")
    print("capturas listas. Siguiente: scripts/process_game.py " + args.match_id)


if __name__ == "__main__":
    main()
