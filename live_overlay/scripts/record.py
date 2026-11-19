"""Graba snapshots del Live Client API mientras juegas, para reproducirlos después.

Escribe data/snapshots/snap_<YYYYmmdd_HHMMSS>.json cada --interval segundos
(2 por defecto) mientras haya partida. Es el corpus propio de este proyecto:
scripts/replay.py lo reproduce por el panel y las mediciones de docs/design.md
se rehacen sobre él. Ctrl-C para parar.

Uso: .venv\\Scripts\\python.exe scripts\\record.py [--interval 2] [--out data\\snapshots]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import SNAPSHOT_DIR
from app.live_client import fetch_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=SNAPSHOT_DIR)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval debe ser > 0")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Grabando en {args.out} cada {args.interval:g} s. Ctrl-C para parar.")
    written = 0
    while True:
        snap = fetch_snapshot()
        # milisegundos en el nombre: con --interval < 1 no se pisan capturas del mismo segundo
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        if snap:
            (args.out / f"snap_{stamp}.json").write_text(json.dumps(snap), encoding="utf-8")
            written += 1
            game_time = (snap.get("gameData") or {}).get("gameTime") or 0
            print(f"{stamp}  t={float(game_time):7.1f}  ({written} snapshots)")
        else:
            print(f"{stamp}  sin partida")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
