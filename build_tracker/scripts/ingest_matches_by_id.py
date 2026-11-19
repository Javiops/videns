"""Ingest explicit match ids with ALL TEN perspectives (macro_coach bridge).

The ladder ingester only reconstructs visits for ladder members, but the
exporter keeps ten-player matches only. The measured-wallet games
(data/imports/gold_measured_v1.json) must therefore be present with the full
lobby AND all ten per-player perspectives. This script tops up whatever is
missing: raw responses come from the gzip archive when already fetched, so
already-archived games cost no API quota.

Riot dev keys die every 24 h: a 401/403 aborts loudly (project rule), it never
continues half-ingested in silence.

Usage: python scripts/ingest_matches_by_id.py [MATCH_ID ...]
       (no args: every match in data/imports/gold_measured_v1.json)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RANKED_SOLO_QUEUE
from app.db import init_db, insert_reconstruction, match_puuids, upsert_player
from app.ddragon import DataDragon
from app.ladder import _player_from_match, routing_for_match_id
from app.reconstruct import reconstruct_game, reconstruct_visits
from app.riot import RiotClient


def main():
    mids = sys.argv[1:]
    if not mids:
        imp = json.loads(Path("data/imports/gold_measured_v1.json")
                         .read_text(encoding="utf-8"))
        mids = imp["matches"]
    init_db()
    client = RiotClient()
    dragon = DataDragon()
    hechas = completas = 0
    try:
        for mid in mids:
            regional, platform = routing_for_match_id(mid)
            match = client.match(regional, mid)
            if not match:
                print(f"{mid}: sin match (404) — fuera")
                continue
            info = match.get("info") or {}
            if info.get("queueId") != RANKED_SOLO_QUEUE:
                print(f"{mid}: queue {info.get('queueId')} != ranked solo — fuera")
                continue
            participants = info.get("participants") or []
            stored = set(match_puuids(mid) or [])
            missing = [p["puuid"] for p in participants
                       if p.get("puuid") and p["puuid"] not in stored]
            if not missing:
                completas += 1
                continue
            timeline = client.timeline(regional, mid)
            if not timeline:
                print(f"{mid}: sin timeline — fuera")
                continue
            for p in participants:
                upsert_player(_player_from_match(p, platform, regional))
            game, events = reconstruct_game(match, timeline, dragon)
            entries = [reconstruct_visits(match, timeline, puuid, dragon)
                       for puuid in missing]
            insert_reconstruction(game, events, entries)
            hechas += 1
            print(f"{mid}: +{len(entries)} perspectivas "
                  f"(tenia {len(stored)}/10)")
    finally:
        client.close()
    print(f"\n{hechas} partidas completadas ahora, {completas} ya completas, "
          f"total pedido {len(mids)}")


if __name__ == "__main__":
    main()
