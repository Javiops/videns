"""Descarga (LCU) las replays de toda la cola, sin abrirlas. Idempotente:
las ya descargadas ('watch') se saltan. Requiere el cliente de League abierto
y en el parche de las partidas. Las replays caducan al cambiar el parche:
ejecutar esto pronto tras encolar.

Uso: python scripts/download_queue.py [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import lcu
from app.config import QUEUE_FILE, game_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not QUEUE_FILE.exists():
        raise SystemExit("cola vacia (data/queue.jsonl no existe)")
    rows = [json.loads(l) for l in open(QUEUE_FILE, encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    ok, fail = [], []
    for i, row in enumerate(rows):
        mid = row["match_id"]
        try:
            meta = lcu.replay_metadata(mid) or {}
            if meta.get("state") == "watch":
                print(f"[{i+1}/{len(rows)}] {mid}: ya descargada")
                ok.append(mid)
                continue
            print(f"[{i+1}/{len(rows)}] {mid}: descargando...", flush=True)
            match = json.loads((game_dir(mid) / "mv5_match.json")
                               .read_text(encoding="utf-8"))
            lcu.download_replay(mid, match)
            ok.append(mid)
            print(f"  OK")
        except lcu.LcuError as e:
            print(f"  FALLO {mid}: {e}")
            fail.append(mid)
    print(f"\n{len(ok)} replays listas, {len(fail)} fallos")
    if fail:
        print("fallos:", fail)


if __name__ == "__main__":
    main()
