"""One predeclared additional-copy experiment on the fixed 16.18 candidate.

Launch through footwear_runtime.py after the full conditional comparison. This
entry point has no test, subset, threshold, alternate artifact or sweep mode.
Both variants use Predictor.predict_options and the shared displayed-policy
harness. Outputs are exclusive immutable files in the existing experiment run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import eval_policy as ep
import train_prefix as tp
from app.dataset_artifacts import canonical_sha, sha256
from app.predictor import Predictor, additional_copy_policy_spec
from app.shop_econ import PINK, SAVE_ITEM, components_of

COPY_METRICS = {
    "excess_copies": "all scored validation decisions: sum max(predicted_count-max(1,true_count),0) in first option",
    "missing_copies": "all scored validation decisions: sum max((true_count-1)-max(predicted_count-1,0),0) over true items",
    "excess_copy_error": "all scored validation decisions: first option has at least one excess additional copy",
    "missing_copy_error": "all scored validation decisions: first option misses at least one true additional copy",
}
PRIMARY = ["first_action_exact", "action_family_match"]
POLICIES = {"original": "argmax-v1", "modified": "count-tail-075-v1"}


def predeclared_experiment() -> dict:
    """Embed this exact declaration in run_plan.json before training starts."""
    return {
        "schema": 1,
        "question": "Does requiring count-head tail mass >=0.75 for additional copies reduce excess copies without worsening missing copies or either primary displayed-policy metric?",
        "original_policy": additional_copy_policy_spec(POLICIES["original"]),
        "modified_policy": additional_copy_policy_spec(POLICIES["modified"]),
        "metrics": PRIMARY + list(COPY_METRICS),
        "split": "val", "games": 0, "bootstrap": 400, "seed": 16,
        "no_retraining": True, "first_copy_ranking_unchanged": True,
        "copy_metric_populations": COPY_METRICS,
        "stopping_rule": "exactly one frozen comparison; no threshold tuning after scoring; retain negative and inconclusive results",
        "supported_win_rule": "both primary modified-minus-original CI95 lower bounds >0, excess and missing mean copies do not increase, and zero illegal or unaffordable displayed options",
    }


def copy_errors(actual: Counter, diagnostics: dict) -> dict:
    """Duplicate-copy errors, separate from first-copy identity mistakes.

    A missing item with truth count three contributes two missing additional
    copies. A wrong singleton contributes zero duplicate-copy errors; primary
    metrics still penalize the wrong purchase. SAVE has no physical quantity.
    """
    options = diagnostics["displayed_options"]
    first = options[0] if options else {"item_counts": {}}
    pred = Counter({int(item): count for item, count in first["item_counts"].items()})
    truth = Counter({item: count for item, count in actual.items() if item != SAVE_ITEM})
    excess = sum(max(count - max(1, truth[item]), 0) for item, count in pred.items())
    missing = sum(max((count - 1) - max(pred[item] - 1, 0), 0) for item, count in truth.items())
    return {"excess_copies": excess, "missing_copies": missing,
            "excess_copy_error": float(excess > 0), "missing_copy_error": float(missing > 0)}


def write_exclusive(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def run_file(run: Path, name: str) -> Path:
    if not name or Path(name).name != name or (run / name).resolve().parent != run.resolve():
        raise ValueError("Evidence and candidate must be named directly in the frozen run directory")
    return run / name


def validate_predeclaration(plan: dict) -> dict:
    declared = predeclared_experiment()
    if canonical_sha(plan.get("decoder_experiment")) != canonical_sha(declared):
        raise ValueError("Run plan must predeclare the exact single additional-copy experiment")
    if plan.get("evaluation_seed") != 16 or plan.get("evaluation_bootstrap") != 400:
        raise ValueError("Additional-copy evaluation requires the shared fixed 400-resample, seed-16 protocol")
    required_code = {"app/predictor.py", "scripts/eval_policy.py", "scripts/eval_copy_policy.py"}
    if not required_code <= (plan.get("code_sha256") or {}).keys():
        raise ValueError("Both real decoders and the shared evaluator must be pinned before training")
    return declared


def validate_baseline_report(report: dict, predictor, manifest: dict, plan_sha: str,
                             run: Path) -> Path:
    expected = {"split": "val", "purpose": "patch_development_1618", "run_plan_sha256": plan_sha,
                "final_test_evaluation": False, "promotion_allowed": False,
                "production_hold_cleared": False}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Step 3 report is not the completed validation comparison for this run")
    if (report.get("artifact_sha256", {}).get("A") != predictor.artifact_sha256
            or report.get("manifest", {}).get("export_fingerprint") != manifest["export_fingerprint"]
            or not report.get("conditional_baseline")
            or report.get("evaluation", {}).get("seed") != 16
            or report["evaluation"].get("bootstrap") != 400
            or report["evaluation"].get("games_requested") != 0
            or report["evaluation"].get("same_export_for_all_policies") is not True
            or report.get("visits", 0) <= 0 or report.get("games", 0) <= 0):
        raise ValueError("Step 3 artifact, export, population or statistical settings changed")
    for tag in ("A", "B"):
        metrics = report.get("metrics", {}).get(tag, {})
        if metrics.get("unaffordable") != 0:
            raise ValueError("Step 3 must have zero unaffordable displayed options")
        for name in PRIMARY:
            if metrics.get(name, {}).get("n", 0) <= 0:
                raise ValueError("Step 3 primary metric population was not measured")
    binding = report.get("paired_outcomes") or {}
    path = Path(binding.get("path", "")).resolve()
    if (path.parent != run.resolve() or binding.get("rows") != report["visits"]
            or not path.is_file() or path.stat().st_size != binding.get("bytes")
            or sha256(path) != binding.get("sha256")):
        raise ValueError("Step 3 paired decision evidence is missing or changed")
    return path


def same_decision(row: dict, actual: Counter, previous: dict) -> None:
    if (previous.get("split") != "val"
            or any(previous.get(key) != row.get(key) for key in ("match_id", "participant_id", "ts"))
            or previous.get("budget") != float(row["gold_est"])
            or previous.get("inventory") != list(row.get("inventory") or [])
            or previous.get("actual_item_counts") != {str(item): count for item, count in sorted(actual.items())}
            or previous.get("next_final_id") != (int(row.get("target_id") or 0) or None)):
        raise ValueError("Decoder comparison must traverse exactly the same frozen step-3 decisions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--baseline-report", required=True)
    args = parser.parse_args()
    from app.dataset_generations import verify_generation, validate_development_run_plan
    from app.conditional_policy import require_explicit_slot_state

    dataset, run = ep.ML_DIR, ep.RUN_DIR
    generation = verify_generation(dataset)
    if generation.get("purpose") != "patch_development_1618":
        raise ValueError("This experiment is scoped exclusively to the qualified 16.18 development benchmark")
    plan, plan_sha = validate_development_run_plan(dataset, run, generation, os.environ["PREFIX_RUN_PLAN"])
    declared = validate_predeclaration(plan)
    artifact = run_file(run, args.artifact)
    if args.artifact != (plan.get("training_stages") or {}).get("graft", {}).get("PREFIX_OUT"):
        raise ValueError("Decoder comparison must use the declared unchanged step-3 graft")
    step3_path = run_file(run, args.baseline_report)
    report_path, protocol_path = run / "decoder_eval_val.json", run / "decoder_protocol.json"
    outcomes_path = run / "decoder_outcomes_val.jsonl"
    if any(path.exists() for path in (report_path, protocol_path, outcomes_path,
                                     outcomes_path.with_suffix(".jsonl.partial"))):
        raise ValueError("The single decoder attempt already exists; do not rerun or overwrite evidence")

    manifest = json.loads(ep.SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    assignments = json.loads((dataset / "split_assignment.json").read_text(encoding="utf-8"))
    models = {tag: Predictor(artifact, additional_copy_policy=policy) for tag, policy in POLICIES.items()}
    original = models["original"]
    provenance = {}
    for tag, predictor in models.items():
        provenance[tag] = ep.verify_artifact_provenance(predictor, manifest, False)
        if (predictor.provenance.get("run_plan_sha256") != plan_sha
                or predictor.provenance.get("training_stage") != "graft"
                or predictor.provenance.get("gold_input") != "prequential-v2"
                or not predictor.config.get("gold_x") or not predictor.has_counts):
            raise ValueError("Decoder experiment requires the bound causal-gold, quantity-trained graft")
        if (predictor.artifact_sha256 != original.artifact_sha256
                or predictor.config != original.config
                or predictor.dragon.signature != original.dragon.signature):
            raise ValueError("Both decoders must have the same unchanged checkpoint, config and static data")

    step3 = json.loads(step3_path.read_text(encoding="utf-8"))
    previous_path = validate_baseline_report(step3, original, manifest, plan_sha, run)
    protocol = {"schema": 1, "declared_at": datetime.now(timezone.utc).isoformat(),
        "experiment": declared, "experiment_sha256": canonical_sha(declared),
        "run_plan_sha256": plan_sha, "artifact": artifact.name,
        "artifact_sha256": original.artifact_sha256, "artifact_config": original.config,
        "artifact_config_sha256": canonical_sha(original.config),
        "export_fingerprint": manifest["export_fingerprint"],
        "generation_manifest_sha256": sha256(dataset / "generation_manifest.json"),
        "step3_report": {"path": str(step3_path), "sha256": sha256(step3_path)},
        "step3_outcomes": step3["paired_outcomes"],
        "policy_sha256": {tag: canonical_sha(additional_copy_policy_spec(policy))
                          for tag, policy in POLICIES.items()},
        "code_sha256": plan["code_sha256"], "final_test_evaluation": False,
        "promotion_allowed": False, "production_hold_cleared": False}
    write_exclusive(protocol_path, protocol)  # Persist exact binding before any validation scoring.

    dragon = original.dragon
    families = {}
    outcomes = {tag: defaultdict(list) for tag in POLICIES}
    unaffordable = {tag: 0 for tag in POLICIES}
    displayed_counts = {tag: 0 for tag in POLICIES}
    scored = 0
    with ep.outcome_capture(outcomes_path) as (write_outcome, outcome_binding), previous_path.open(encoding="utf-8") as previous_rows:
        for row in tp.with_targets(tp.stream_rows(dataset / "visits_val.jsonl", extra_keys=("participant_id",)), dragon):
            if assignments.get(row.get("match_id")) != "val":
                raise ValueError("A non-validation game reached decoder evaluation")
            require_explicit_slot_state(row)
            if row.get("gold_est_version") != "prequential-v2" or row.get("gold_est") is None:
                raise ValueError("Every decision requires the same causal gold budget")
            actual = Counter(int(item) for item in row.get("label_ids") or [] if item)
            if not actual:
                continue
            previous_line = next(previous_rows, None)
            if previous_line is None:
                raise ValueError("Decoder validation exceeds the step-3 population")
            previous = json.loads(previous_line)
            same_decision(row, actual, previous)
            save_kind = row.get("save_kind") or ("no_buy_death" if SAVE_ITEM in actual else
                ("ward_only" if all(item in PINK for item in actual) else "build_spend"))
            target = int(row.get("target_id") or 0)
            if target and target not in families:
                families[target] = components_of(target, dragon) | {target}
            paired = {}
            for tag, predictor in models.items():
                diagnostics = {}
                metrics, over = ep.score_visit(predictor, row, actual, save_kind, row["gold_est"],
                    dragon, families.get(target), diagnostics=diagnostics)
                if tag == "original" and (metrics != previous["policies"]["A"]["metrics"]
                        or diagnostics["displayed_options"] != previous["policies"]["A"]["displayed_options"]):
                    raise ValueError("Unchanged decoder failed exact per-decision parity with step 3")
                metrics.update(copy_errors(actual, diagnostics))
                outcomes[tag][row["match_id"]].append(metrics)
                unaffordable[tag] += len(over)
                displayed_counts[tag] += len(diagnostics["displayed_options"])
                paired[tag] = {**diagnostics, "metrics": metrics, "unaffordable": len(over)}
            write_outcome({"schema_version": 1, "split": "val", "match_id": row["match_id"],
                "participant_id": row.get("participant_id"), "ts": row.get("ts"),
                "budget": float(row["gold_est"]), "actual_item_counts": previous["actual_item_counts"],
                "policies": paired})
            scored += 1
            if scored % 2000 == 0:
                print(f"scored {scored} paired decoder decisions", flush=True)
        if next(previous_rows, None) is not None:
            raise ValueError("Decoder validation omitted step-3 decisions")
    if scored != step3["visits"] or outcome_binding["rows"] != scored or len(outcomes["original"]) != step3["games"]:
        raise ValueError("Decoder comparison did not reproduce the complete step-3 population")

    summaries, differences = {}, {}
    for tag in POLICIES:
        summaries[tag] = {}
        for metric in declared["metrics"]:
            result = ep.summarize(outcomes[tag], metric, 400, 16)
            if metric in COPY_METRICS:
                result["population"] = COPY_METRICS[metric]
                result["total"] = sum(visit[metric] for visits in outcomes[tag].values() for visit in visits)
            summaries[tag][metric] = result
    for metric in declared["metrics"]:
        differences[metric] = ep.paired_diff(outcomes["modified"], outcomes["original"], metric, 400, 16)
        if metric in COPY_METRICS:
            differences[metric]["population"] = COPY_METRICS[metric]
    legal = not any(unaffordable.values())
    primary_win = all(differences[key].get("ci95") and differences[key]["ci95"][0] > 0 for key in PRIMARY)
    copy_not_worse = all(differences[key]["diff"] is not None and differences[key]["diff"] <= 0
                         for key in ("excess_copies", "missing_copies"))
    report = {"schema": 1, "purpose": "patch_development_1618", "split": "val", "visits": scored,
        "games": len(outcomes["original"]), "run_plan_sha256": plan_sha,
        "decoder_protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "artifact_sha256": original.artifact_sha256, "artifact_config_sha256": protocol["artifact_config_sha256"],
        "policy_sha256": protocol["policy_sha256"], "export_fingerprint": manifest["export_fingerprint"],
        "step3_report": protocol["step3_report"], "evaluation": {"bootstrap": 400, "seed": 16,
            "games_requested": 0, "same_export_for_all_policies": True,
            "exact_step3_original_policy_parity": True},
        "artifact_provenance": provenance, "metrics": summaries,
        "paired_diff_modified_minus_original": differences, "paired_outcomes": outcome_binding,
        "purchase_legality": {"passed": legal, "unaffordable": unaffordable,
            "displayed_options_checked": displayed_counts, "checker": "eval_policy.check_displayed_purchases",
            "checks": ["recipe_cost", "wallet", "regular_slots", "ward_stacks", "unique_boots", "purchase_blocks"],
            "scope": "causal reconstructed wallet and audited footwear/role-slot state; no live-wallet claim"},
        "supported_win": bool(legal and primary_win and copy_not_worse),
        "interpretation": "supported win" if legal and primary_win and copy_not_worse else
            ("failed legality gate" if not legal else "no supported win; retain paired effects and copy tradeoff"),
        "final_test_evaluation": False, "promotion_allowed": False, "production_hold_cleared": False,
        "completed_at": datetime.now(timezone.utc).isoformat()}
    if sha256(artifact) != original.artifact_sha256 or sha256(step3_path) != protocol["step3_report"]["sha256"]:
        raise ValueError("Bound checkpoint or step-3 evidence changed during scoring")
    write_exclusive(report_path, report)
    print(f"wrote {report_path}; split=val visits={scored} games={report['games']}; {report['interpretation']}", flush=True)
    if not legal:
        raise SystemExit("Additional-copy comparison failed the zero-unaffordable hard stop")


if __name__ == "__main__":
    main()
