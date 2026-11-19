"""Write data/admission/admission_v1.json (metadata only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from draft.admission import admit

m = admit(Path(__file__).resolve().parents[1] / "data/admission/admission_v1.json")
print({k: v for k, v in m.items() if k not in ("admitted", "sources")})
print("split sources:", len(m["sources"]["build_tracker_splits"]))
