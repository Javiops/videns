"""TRAIN-only target membership after legal filtering of a frozen label-blind pool.

Versioned, parametrized adaptation of the frozen diagnostic
data/reports/strategy_analysis_20260920/basket_legality_probe.py (that file
and its JSON stay untouched as evidence). Checks, for each sampled probe
purchase, whether the bought basket was legal at the supplied budget
(app.conditional_policy.purchase_sequence under the generation's activated
runtime) and whether it was already in the fit-side catalogue.

Reads ONLY the frozen sample produced by scripts/basket_vocabulary_probe.py
(or the original probe). No validation/test payloads, no model inference.

Usage:
  python scripts/basket_legality_probe.py --sample-dir <dir with sample+parent>
      [--generation <gen dir>] [--review <static_review.json>] [--out <file>]
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]


def digest(p: Path) -> str:
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation", default=str(
        ROOT / "data/dataset_generations/16.18-reconstructible-development-v3"))
    ap.add_argument("--sample-dir", required=True,
                    help="dir holding basket_probe_legality_sample_train.jsonl"
                         " and basket_vocabulary_probe.json (the parent)")
    ap.add_argument("--review", default=str(
        ROOT / "data/corpora/16.18-reconstructible-v3/static_review/static_review.json"))
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <sample-dir>/basket_legality_probe.json)")
    ap.add_argument("--budget-seconds", type=float, default=300.0)
    args = ap.parse_args()
    started = time.monotonic()

    gen = Path(args.generation)
    sample_dir = Path(args.sample_dir)
    manifest = json.loads((gen / "generation_manifest.json").read_text(encoding="utf-8"))
    assignment = json.loads((gen / "split_assignment.json").read_text(encoding="utf-8"))
    parent = json.loads((sample_dir / "basket_vocabulary_probe.json").read_text(encoding="utf-8"))

    import footwear_runtime as fr
    fr.validate_generation_runtime(gen)
    runtime = fr.activate_for_manifest(manifest, review_path=Path(args.review))
    from app.conditional_policy import purchase_sequence, require_explicit_slot_state
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch(manifest["patch"])

    sample_path = sample_dir / "basket_probe_legality_sample_train.jsonl"
    counts = Counter()
    arms = {str(k): Counter() for k in (16, 32, 64, 128)}
    by_size = {}
    games = set()
    with sample_path.open(encoding="utf-8") as f:
        for line in f:
            if time.monotonic() - started > args.budget_seconds:
                raise RuntimeError("time budget exceeded; no partial report")
            p = json.loads(line)
            row = p["row"]
            assert assignment[row["match_id"]] == "train"
            require_explicit_slot_state(row)
            action = tuple(sorted(i for k, n in row["label_counts"].items()
                                  for i in [int(k)] * int(n)))
            assert action and 999999 not in action
            budget = float(row["gold_est"])
            assert row["gold_est_version"] == "prequential-v3"
            legal = purchase_sequence(action, row, budget, dragon) is not None
            structural = legal or purchase_sequence(
                action, row, 10_000_000.0, dragon) is not None
            counts["purchases"] += 1
            games.add(row["match_id"])
            counts["target_legal_at_gold_est"] += legal
            counts["target_structurally_legal"] += structural
            counts["budget_blocked"] += structural and not legal
            counts["structurally_blocked"] += not structural
            counts["global_seen"] += p["global_seen"]
            counts["global_seen_and_legal"] += p["global_seen"] and legal
            counts["global_seen_and_structurally_legal"] += p["global_seen"] and structural
            q = str(len(action))
            by_size.setdefault(q, Counter())
            by_size[q]["n"] += 1
            by_size[q]["legal"] += legal
            for k, hit in p["shortlist_membership"].items():
                arms[k]["n"] += 1
                arms[k]["raw_membership"] += hit
                arms[k]["legal_membership"] += hit and legal
                arms[k]["structurally_legal_membership"] += hit and structural
                arms[k]["budget_blocked_member"] += hit and structural and not legal
                arms[k]["structurally_blocked_member"] += hit and not structural
                by_size[q]["k" + k + "_raw"] += hit
                by_size[q]["k" + k + "_legal"] += hit and legal
    assert counts["purchases"] == parent["legality_probe_sample"]["n"]

    report = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": "internal chronological probe within TRAIN",
        "population": {"purchases": counts["purchases"], "games": len(games)},
        "sample_rule": parent["legality_probe_sample"]["rule"],
        "export_fingerprint": manifest["export_fingerprint"],
        "parent_report": str(sample_dir / "basket_vocabulary_probe.json"),
        "sample_sha256": digest(sample_path),
        "runtime": runtime,
        "economy_function": ("app.conditional_policy.purchase_sequence through "
                             "footwear_runtime.activate_for_manifest; runtime "
                             "declaration matches generation"),
        "counts": dict(counts),
        "shortlists": {k: {**v,
                           "raw_coverage": v["raw_membership"] / v["n"],
                           "legal_coverage": v["legal_membership"] / v["n"],
                           "structural_coverage": v["structurally_legal_membership"] / v["n"]}
                       for k, v in arms.items()},
        "by_basket_size": {k: dict(v) for k, v in by_size.items()},
        "limits": [
            "Membership feasibility only, not model accuracy.",
            "Only target membership is checked after freezing label-blind pool; "
            "no candidate-list legality cardinality or runtime measurement.",
            "gold_est is reconstructed, not exact wallet.",
            "No incumbent predictions; current incumbent has trained on this TRAIN probe.",
            "No validation/test payload read.",
        ],
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [
            Path(__file__),
            ROOT / "scripts/basket_vocabulary_probe.py",
            ROOT / "app/conditional_policy.py",
            ROOT / "scripts/footwear_runtime.py",
            ROOT / "scripts/benchmark_1618_runtime.py",
            ROOT / "scripts/reconstructible_1618_runtime.py"]},
        "bindings": {
            "train_export_sha256_actual": digest(gen / "visits_train.jsonl"),
            "train_export_sha256_expected": manifest["files"]["visits_train.jsonl"]["sha256"],
            "parent_report_sha256": digest(sample_dir / "basket_vocabulary_probe.json"),
            "generation_manifest_sha256": digest(gen / "generation_manifest.json"),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    assert (report["bindings"]["train_export_sha256_actual"]
            == report["bindings"]["train_export_sha256_expected"])
    out = Path(args.out) if args.out else sample_dir / "basket_legality_probe.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    c = counts
    print(f"purchases={c['purchases']} legal@gold_est={c['target_legal_at_gold_est']} "
          f"budget_blocked={c['budget_blocked']} global_seen_and_legal={c['global_seen_and_legal']}")


if __name__ == "__main__":
    main()
