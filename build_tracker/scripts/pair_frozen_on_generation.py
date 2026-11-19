"""Pair the frozen references' rows on a foreign generation's cohort (docs/benchmark_1619_v1.md section 4).

Reads rows_R1_<gen>.jsonl and rows_M2_<gen>.jsonl written by score_frozen_on_generation.py (both gated 0/0/0
on the v4 cohort in their own processes), asserts identical row identity, and publishes: aggregates (exact,
top-3), the paired-by-game M2 - R1 contrast with 95 % CIs (400 + 2,000 draws), per-slice contrasts and
movement counts, together with the two scoring reports. The bar for 16.19 is what M2 achieves here.

    footwear_runtime.py --entry scripts/pair_frozen_on_generation.py --review <v4 review> -- --generation <dir> --out <dir>
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    started = time.monotonic()
    out = args.out.resolve(); gen = args.generation.resolve().name
    from basket_newbf_mixture import load_rows
    from stage2_newgen import compare, beats, now_utc
    reports = {}
    arms = {}
    for model in ("R1", "M2"):
        rep_path = out / f"report_{model}_{gen}.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        rows_path = Path(rep["rows_file"])
        if sha256_file(rows_path) != rep["rows_file_sha256"] or not rep["gate"]["passed"]:
            raise SystemExit(f"{model}: rows digest differs from its report or its gate did not pass")
        reports[model] = rep
        arms[model] = load_rows(rows_path, 10**9)
    n = len(arms["R1"])
    if len(arms["M2"]) != n:
        raise SystemExit("row counts differ between the two references")
    report = {"schema": 1, "protocol": "docs/benchmark_1619_v1.md section 4", "created_utc": now_utc(),
              "target_generation": reports["R1"]["target_generation"], "cohort": reports["R1"]["cohort"],
              "scoring_world": reports["R1"]["scoring_world"], "identities": {m: reports[m]["identity"] for m in reports},
              "gates": {m: reports[m]["gate"] for m in reports}, "cohort_games": len({r["match_id"] for r in arms["R1"]})}
    compare(arms, [("M2", "R1")], report, draw_pairs=[("M2", "R1")])
    pv = report["paired"]["M2_vs_R1"]
    report["reading"] = {"M2_beats_R1_on_this_cohort": beats(report, "M2", "R1"),
                         "M2_vs_R1_points": round(pv["diff"] * 100, 3), "ci95_points": [round(x * 100, 3) for x in pv["ci95"]],
                         "bar_for_16.19": {m: round(report["aggregates"][m]["first_action_exact"]["value"], 5) for m in ("R1", "M2")},
                         "note": "descriptive benchmark of frozen references on a new patch; no training decision is taken here"}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    path = out / f"benchmark_{gen}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": n, "games": report["cohort_games"], "aggregates": {m: round(v["first_action_exact"]["value"], 5) for m, v in report["aggregates"].items()},
                      "M2_vs_R1": report["reading"], "report": str(path)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
