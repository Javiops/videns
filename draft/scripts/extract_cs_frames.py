"""CS at 6/12/18/24/30/36 minutes for the v1 timeline games -> data/extract/cs_frames_v1.jsonl.gz (detached)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.frames import extract_cs_frames

print(extract_cs_frames(ROOT / "data/extract/games_v1.jsonl.gz", ROOT / "data/extract/cs_frames_v1.jsonl.gz",
                        log=lambda m: print(m, flush=True)), flush=True)
