"""Construye la cola de partidas challenger a capturar.

Semillas: liga challenger EUW (solo queue) -> historial ranked solo reciente de
cada semilla -> filtro por parche y por tener 2 junglas -> dedup contra corpus
y cola. Definición documentada: ambos junglas de una lobby challenger cuentan
como POV válidos (juegan a MMR challenger aunque su liga personal varíe).

Uso: python scripts/collect_matches.py --patch 16.18 [--games 40] [--seeds 25]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CORPUS, QUEUE_FILE, QUEUE_RANKED_SOLO, ensure_dirs
from app.riot import Riot
from app.segments import participants


def queued_ids() -> set:
    ids = set()
    if QUEUE_FILE.exists():
        for line in open(QUEUE_FILE, encoding="utf-8"):
            ids.add(json.loads(line)["match_id"])
    if CORPUS.exists():
        ids |= {d.name for d in CORPUS.iterdir() if d.is_dir()}
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", required=True, help="p. ej. 16.18 (el del cliente)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--per-seed", type=int, default=10)
    args = ap.parse_args()
    ensure_dirs()

    riot = Riot()
    league = riot.challenger_league()
    entries = sorted(league["entries"], key=lambda e: -e["leaguePoints"])[:args.seeds]
    print(f"{len(entries)} semillas challenger (por LP)")

    have = queued_ids()
    added = 0
    with open(QUEUE_FILE, "a", encoding="utf-8") as fh:
        for e in entries:
            if added >= args.games:
                break
            ids = riot.match_ids(e["puuid"], count=args.per_seed,
                                 queue=QUEUE_RANKED_SOLO) or []
            for mid in ids:
                if added >= args.games or mid in have:
                    continue
                m = riot.match(mid)
                if m is None:
                    continue
                ver = m["info"]["gameVersion"]
                if not ver.startswith(args.patch + "."):
                    continue
                P = participants(m)
                jg = [(pid, v["champ"], v["team"]) for pid, v in P.items()
                      if v["role"] == "JG"]
                if len(jg) != 2:
                    continue
                if m["info"]["gameDuration"] < 900:   # remakes / ff15 fuera
                    continue
                row = {"match_id": mid, "patch": args.patch,
                       "duration": m["info"]["gameDuration"],
                       "junglers": [{"pid": p, "champ": c, "team": t}
                                    for p, c, t in jg],
                       "added": time.strftime("%Y-%m-%d %H:%M")}
                fh.write(json.dumps(row) + "\n")
                have.add(mid)
                added += 1
                print(f"  + {mid} ({row['junglers'][0]['champ']} vs "
                      f"{row['junglers'][1]['champ']}, {ver})")
    print(f"OK: {added} partidas nuevas en la cola "
          f"(descargar las replays pronto: caducan con el parche)")


if __name__ == "__main__":
    main()
