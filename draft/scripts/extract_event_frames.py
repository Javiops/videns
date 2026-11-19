"""Cumulative kills/assists/dragons at 6..36 min for the v1 timeline games (detached)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.frames import extract_event_frames

print(extract_event_frames(ROOT / "data/extract/games_v1.jsonl.gz", ROOT / "data/extract/event_frames_v1.jsonl.gz",
                           log=lambda m: print(m, flush=True)), flush=True)
