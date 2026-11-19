"""Kills/deaths/assists and team dragons for the v1 games -> data/extract/match_stats_v1.jsonl.gz (detached)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.frames import extract_match_stats

print(extract_match_stats(ROOT / "data/extract/games_v1.jsonl.gz", ROOT / "data/extract/match_stats_v1.jsonl.gz",
                          log=lambda m: print(m, flush=True)), flush=True)
