"""Descarga la replay vía LCU y (opcional) la abre para capturar.

Uso: python scripts/fetch_replay.py <MATCH_ID> [--watch]
Requiere el cliente de League abierto y logueado en EUW, en el mismo parche.
También descarga mv5_match.json y mv5_timeline.json al corpus si faltan.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import lcu
from app.config import game_dir, ensure_dirs
from app.riot import Riot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("match_id")
    ap.add_argument("--watch", action="store_true", help="abre la replay al acabar")
    args = ap.parse_args()
    ensure_dirs()
    g = game_dir(args.match_id)
    g.mkdir(parents=True, exist_ok=True)

    riot = Riot()
    for name, fn in (("mv5_match.json", riot.match), ("mv5_timeline.json", riot.timeline)):
        f = g / name
        if not f.exists():
            data = fn(args.match_id)
            if data is None:
                raise SystemExit(f"404 de la API para {args.match_id} ({name})")
            f.write_text(json.dumps(data), encoding="utf-8")
            print(f"  {name} descargado")

    print("descargando replay (LCU)...")
    match = json.loads((g / "mv5_match.json").read_text(encoding="utf-8"))
    meta = lcu.download_replay(args.match_id, match)
    print(f"  replay lista: {meta.get('state')}")
    if args.watch:
        lcu.watch_replay(args.match_id)
        print("  abriendo replay (~1 min). Luego: scripts/capture_game.py "
              + args.match_id)


if __name__ == "__main__":
    main()
