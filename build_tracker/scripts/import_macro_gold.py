"""Import the macro_coach measured-wallet artifact into build_tracker.

Owner decision (2026-09-20): data crosses the two projects ONLY as a versioned
exported artifact — never shared code, never cross-corpus reads. This script
takes a copy of macro_coach's `buys_export_v1.jsonl` (dropped into
data/imports/ by hand or by the owner's copy step) and produces
`data/imports/gold_measured_v1.json`, the file the trainer consumes via
PREFIX_GOLD_MEASURED.

What it contains: for every shop event of every player in the source games,
the wallet JUST BEFORE the event — exact (`gold`, verified against Match-V5
anchors, err_max ~10) or as an interval [lo, hi] (the ~8 s OCR window; never
an invented number). The trainer joins visits by (match_id, ts, champion_id):
a visit's ts IS its first shop event's Match-V5 timestamp, so measured rows
land exactly on the visits they describe.

POV note: this is the buyer's OWN wallet — the one number a player always
knows exactly — so using it as the budget is POV-legal by construction.

Usage: python scripts/import_macro_gold.py data\\imports\\buys_export_v1.jsonl
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VERSION = "gold-measured-v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_path")
    ap.add_argument("--out", default="data/imports/gold_measured_v1.json")
    args = ap.parse_args()
    src = Path(args.export_path)
    raw = src.read_bytes()
    src_sha = hashlib.sha256(raw).hexdigest()

    meta_path = src.with_suffix("").with_suffix("")  # strip .jsonl
    meta_file = src.parent / (src.stem + ".meta.json")
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("sha256") and meta["sha256"] != src_sha:
            raise SystemExit(
                f"{src} does not match its meta sha256: the artifact was "
                "modified after export. Re-copy both files from macro_coach.")
    else:
        meta = {}
        print(f"WARNING: no {meta_file.name} next to the export; "
              "provenance will carry only the computed sha256")

    rows = {}
    n_exact = n_interval = n_skipped = 0
    matches = set()
    seen = set()
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        matches.add(r["match_id"])
        if r.get("champion_id") is None:
            n_skipped += 1
            continue
        key = f"{r['match_id']}|{r['ts_ms']}|{r['champion_id']}"
        # varias compras pueden compartir el ms (combinar de un clic); la
        # cartera de la VISITA es la del PRIMER evento del timestamp — el
        # export viene en orden del timeline, asi que gana la primera fila
        if key in seen:
            continue
        seen.add(key)
        if r.get("verificado") and r.get("oro_antes") is not None:
            v = int(r["oro_antes"])
            rows[key] = {"gold": v, "lo": v, "hi": v, "exact": True}
            n_exact += 1
        elif r.get("intervalo"):
            lo, hi = r["intervalo"]
            rows[key] = {"gold": None, "lo": int(lo), "hi": int(hi),
                         "exact": False}
            n_interval += 1
        else:
            n_skipped += 1     # sin_lectura: nothing measurable, nothing fake

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "version": VERSION,
        "source": str(src),
        "source_sha256": src_sha,
        "source_meta": meta,
        "matches": sorted(matches),
        "n_exact": n_exact, "n_interval": n_interval, "n_skipped": n_skipped,
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"{out}: {n_exact} exact + {n_interval} interval wallets "
          f"({n_skipped} skipped) from {len(matches)} matches")
    print(f"  source sha256 {src_sha}")
    print("  train with: PREFIX_GOLD_MEASURED=" + str(out))


if __name__ == "__main__":
    main()
