"""Reproduce un directorio de snapshots por el panel, sin partida en marcha.

Sirve la misma app (app.main) con `fetch_snapshot` sustituido por un lector que
avanza por los ficheros del directorio a ritmo real: un snapshot cada --step
segundos (2 por defecto, la cadencia de scripts/record.py). Sirve para ver el
overlay con una partida grabada y para depurar sin abrir League.

Uso: .venv\\Scripts\\python.exe scripts\\replay.py --dir data\\snapshots [--from snap_20260929_213000.json] [--step 2] [--port 8021]
Luego abrir http://127.0.0.1:8021/?overlay=1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="directorio con snap_*.json (allgamedata)")
    parser.add_argument("--from", dest="start", default=None, help="nombre del primer fichero a reproducir")
    parser.add_argument("--step", type=float, default=2.0, help="segundos reales entre snapshots")
    parser.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()

    if args.step <= 0:
        parser.error("--step debe ser > 0")
    names = sorted(p.name for p in args.dir.glob("snap_*.json"))
    if not names:
        raise SystemExit(f"No hay snap_*.json en {args.dir}")
    if args.start:
        start_name = Path(args.start).name
        if start_name not in names:
            raise SystemExit(f"{start_name} no está en {args.dir}; primero {names[0]}, último {names[-1]}")
        names = names[names.index(start_name):]
    print(f"{len(names)} snapshots desde {names[0]}; un snapshot cada {args.step:g} s")

    import app.main as main

    t0: float | None = None   # el reloj arranca con el primer poll del panel, no al lanzar el script
    cache: dict[str, dict | None] = {}

    def fetch(timeout: float = 2.0) -> dict | None:
        nonlocal t0
        if t0 is None:
            t0 = time.monotonic()
        index = min(int((time.monotonic() - t0) / args.step), len(names) - 1)
        name = names[index]
        if name not in cache:
            try:
                cache[name] = json.loads((args.dir / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache[name] = None
        return cache[name]

    main.fetch_snapshot = fetch
    import uvicorn

    print(f"Panel: http://127.0.0.1:{args.port}/?overlay=1")
    uvicorn.run(main.app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
