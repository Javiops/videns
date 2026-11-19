"""Stage-2 final report: every candidate against R1, per-segment costs, blocked segments.

Reads the segment reports of one stamp plus the orchestrator status and writes
`stage2_report_{stamp}.json`. Pure JSON; no model, no data. Publishes null and
negative results the same way as positive ones.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "experiments" / "16.18-stage2-v4"
STAGE1 = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v4-bf-stage1"


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stamp", default="full")
    ap.add_argument("--status", default=str(OUT_DIR / "run" / "status.json"))
    args = ap.parse_args()
    stamp = args.stamp
    ref = load(STAGE1 / "frozen_reference_v4.json")
    r1 = ref["metrics_cohort"]["first_action_exact_wardfree"] if ref else None
    status = load(args.status) or {}
    reports = {"A_domains": load(OUT_DIR / f"domains_report_{stamp}.json"),
               "A_features": load(OUT_DIR / f"cache_report_{stamp}.json"),
               "C_wardfree": load(OUT_DIR / f"wardfree_report_{stamp}.json"),
               "B_twin_finetune": load(OUT_DIR / f"finetune_report_{stamp}.json"),
               "D_mixture": load(OUT_DIR / f"mixture_report_{stamp}.json")}
    budget = load(ROOT / "data/experiments/16.18-stage2-budget-v1/budget_pilot.json")
    segments = {}
    for name, rep in reports.items():
        if rep is None:
            segments[name] = {"state": "missing (not run, blocked, or failed before writing a report)"}
            continue
        seg = {"state": "stopped_by_bound" if rep.get("aborted") else "completed", "elapsed_seconds": rep.get("elapsed_seconds")}
        if rep.get("aborted"):
            seg["reason"] = rep["aborted"]
        if "pilot" in rep:
            seg["pilot"] = rep["pilot"]
        if "bench" in rep:
            seg["bench"] = rep["bench"]
        for k in ("train_seconds", "eval_seconds", "fit_seconds", "cohort_seconds", "rows_per_second", "integrity_gate_passed"):
            if k in rep:
                seg[k] = rep[k]
        if "checkpoints" in rep:
            seg["train_seconds_per_arm"] = {a: c.get("train_seconds") for a, c in rep["checkpoints"].items()}
        segments[name] = seg
    candidates = {}
    for seg, name in (("C_wardfree", "wardfree_new"), ("B_twin_finetune", "econ_ft_new"),
                      ("B_twin_finetune", "control_ft_new"), ("D_mixture", "M2")):
        rep = reports.get(seg)
        if not rep or rep.get("aborted") or f"{name}_vs_new_bf" not in rep.get("paired", {}):
            continue
        p = rep["paired"][f"{name}_vs_new_bf"]
        p3 = rep["paired"].get(f"{name}_vs_new_bf_any3", {})
        candidates[name] = {"first_exact": rep["aggregates"][name]["first_action_exact"]["value"],
                            "any_of_three": rep["aggregates"][name]["any_of_three_exact"]["value"],
                            "vs_R1_points": round(p["diff"] * 100, 3), "vs_R1_ci95": [round(x * 100, 3) for x in p["ci95"]],
                            "vs_R1_ci95_2000draws": [round(x * 100, 3) for x in p["ci95_2000draws"]] if "ci95_2000draws" in p else None,
                            "vs_R1_top3_points": round(p3.get("diff", 0) * 100, 3) if p3 else None,
                            "net_decisions_vs_R1": rep["movement"].get(f"{name}_vs_new_bf", {}).get("net"),
                            "beats_R1": bool(p["diff"] > 0 and p["ci95"][0] > 0),
                            "slices_vs_R1": {k.split("_vs_new_bf_")[-1]: {"points": round(v["diff"] * 100, 3), "ci95": [round(x * 100, 3) for x in v["ci95"]]}
                                             for k, v in rep.get("slices", {}).items() if k.startswith(f"{name}_vs_new_bf_")}}
    attribution = None
    if reports["B_twin_finetune"] and not reports["B_twin_finetune"].get("aborted"):
        attribution = reports["B_twin_finetune"].get("decision")
    winners = {k: v for k, v in candidates.items() if v["beats_R1"] and k != "control_ft_new"}
    best = max(winners.items(), key=lambda kv: kv[1]["vs_R1_points"])[0] if winners else None
    summary = {"reference_R1_first_exact": r1, "candidates_scored": sorted(candidates),
               "candidates_beating_R1": sorted(winners), "best_candidate": best,
               "best_points_vs_R1": winners[best]["vs_R1_points"] if best else None,
               "best_ci95": winners[best]["vs_R1_ci95"] if best else None,
               "best_net_decisions": winners[best]["net_decisions_vs_R1"] if best else None,
               "practical_progress": bool(best),
               "statement": ("A candidate demonstrably exceeds the frozen reference on the same cohort." if best else
                             "No candidate beats the frozen reference with the CI excluding zero; the new Bf stays the reference (null/negative published).")}
    out = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "mode": stamp,
           "protocol": "docs/experiment_stage2_econ_wardfree_v1.md", "reference": ref and {"checkpoint": ref["checkpoint"], "first_exact": r1},
           "orchestrator_status": status, "budget_pilot": budget and {k: budget[k] for k in ("A_econ_cache", "B_twin_finetune", "C_wardfree") if k in budget},
           "segments": segments, "candidates_vs_R1": candidates, "attribution_econ_vs_control": attribution,
           "product_summary": summary}
    path = OUT_DIR / f"stage2_report_{stamp}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"product_summary": summary, "segments": {k: v.get("state") for k, v in segments.items()}}, indent=1), flush=True)
    print(f"report: {path}", flush=True)


if __name__ == "__main__":
    main()
