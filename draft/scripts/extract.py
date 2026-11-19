"""Extract admitted games -> data/extract/games_v1.jsonl.gz (detached job)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from draft.extract import extract

limit = None
def log(msg):
    print(msg, flush=True)
r = extract(ROOT / "data/admission/admission_v1.json", ROOT / "data/extract/games_v1.jsonl.gz",
            ROOT / "data/salt.hex", log=log)
log(r)
