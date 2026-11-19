"""Fixed feature-block WP ablation using derived caches; never reopen raw games."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import setback_experiment as safe
from rich_experiment import ece_bootstrap
from winprob.experiment import CalibratedModel, sample_training_indices
from winprob.features import ALL_FEATURES
from winprob.features_rich import ALL_RICH_FEATURES
from winprob.metrics import paired_comparison, summarize

SCHEMA = "rich-feature-ablation-v1"
ALLOCATION_STEMS = ("std", "q25", "q50", "q75", "max_share", "top2_share")
COMPARISONS = (("total", "original_trajectory"), ("allocation", "original_trajectory"),
               ("geometry", "original_trajectory"), ("combined", "original_trajectory"),
               ("combined", "total"), ("combined", "allocation"), ("combined", "geometry"),
               ("allocation", "total"), ("geometry", "total"))


def feature_arms(names):
    names = tuple(names)
    if len(names) != len(set(names)) or set(names) != set(ALL_RICH_FEATURES):
        raise ValueError("Unexpected or duplicate rich feature schema")
    base = set(ALL_FEATURES)
    total = {"rich_gold_total_sum"}
    allocation = {f"rich_gold_{stem}_{operation}" for stem in ALLOCATION_STEMS for operation in ("diff", "sum")}
    if not (base | total | allocation).issubset(names):
        raise ValueError("Missing declared ablation features")
    geometry = set(names) - base - total - allocation
    arms = {"total": base | total, "allocation": base | total | allocation, "geometry": base | geometry}
    if arms["allocation"] | arms["geometry"] != set(names):
        raise ValueError("Ablation union does not reconstruct combined")
    return {name: [index for index, feature in enumerate(names) if feature in features] for name, features in arms.items()}


def validate_projection(rich, original):
    names = tuple(rich["feature_names"])
    if tuple(original["feature_names"]) != ALL_FEATURES:
        raise ValueError("Original feature ordering changed")
    columns = [names.index(name) for name in ALL_FEATURES]
    if not np.array_equal(rich["x"][:, columns], original["x"]):
        raise ValueError("Rich cache differs on original 19 features")
    for key in ("y", "game", "role", "timestamp_ms", "start_ms", "split_time_ms", "patch"):
        if not np.array_equal(rich[key], original[key]):
            raise ValueError("Rich/original row identity mismatch: " + key)


def verify_rich_run(directory):
    safe.reject_quarantined(directory)
    report_path = directory / "report.json"
    report_hash = safe.verify_sidecar(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("schema") != "rich-state-development-v1" or report.get("stage") != "exploratory_development"
            or report.get("confirmatory_evaluation_done") is not False or report.get("sealed_test_read") is not False):
        raise ValueError("Require completed clean exploratory rich run")
    if json.loads((directory / "status.json").read_text())["stage"] != "completed":
        raise ValueError("Rich run has not completed")
    frozen = dict(report["input_and_protocol_sha256"])
    for path, expected in frozen.items():
        safe.reject_quarantined(path)
        if str(path).endswith(".json.gz"):
            raise ValueError("Ablation must not open raw payload bindings")
        if safe.digest(path) != expected:
            raise ValueError("Rich input/code binding changed: " + path)
    caches = [Path(path) for path, digest in frozen.items()
              if Path(path).name == "states.npz" and digest == report["original_cache_sha256"]]
    originals = [Path(path).parent for path in frozen if Path(path).name == "trajectory.joblib"]
    if len(caches) != 1 or len(originals) != 1:
        raise ValueError("Ambiguous original cache or trajectory artifact binding")
    original_cache = caches[0]
    source, original_bindings = safe.verify_inputs(original_cache, original_cache.parent / "source_manifest.json", originals[0])
    if len(source["games"]) != 6000:
        raise ValueError("Ablation requires the exact 6000 source games")
    frozen.update(original_bindings)
    for artifact in report["artifacts"].values():
        safe.reject_quarantined(artifact["path"])
        if safe.digest(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Rich model artifact changed")
        frozen[str(Path(artifact["path"]).resolve())] = artifact["sha256"]
    for name, digest in (("rich_states.npz", report["rich_cache_sha256"]),
                         ("development_wp_predictions.npz", report["development_wp_predictions_sha256"])):
        if safe.digest(directory / name) != digest:
            raise ValueError("Rich derived cache/prediction hash mismatch: " + name)
        frozen[str((directory / name).resolve())] = digest
    frozen.update({str(report_path.resolve()): report_hash,
                   str(report_path.with_suffix(".sha256").resolve()): safe.digest(report_path.with_suffix(".sha256"))})
    with np.load(directory / "rich_states.npz", allow_pickle=False) as loaded:
        rich = {key: loaded[key] for key in loaded.files}
    with np.load(original_cache, allow_pickle=False) as loaded:
        original = {key: loaded[key] for key in loaded.files}
    if rich["feature_names"].tolist() != report["feature_names"]:
        raise ValueError("Rich report/cache feature ordering differs")
    feature_arms(rich["feature_names"])
    validate_projection(rich, original)
    dev = rich["role"] == "development"
    with np.load(directory / "development_wp_predictions.npz", allow_pickle=False) as saved:
        for key in ("game", "timestamp_ms", "y"):
            if not np.array_equal(saved[key], rich[key][dev]):
                raise ValueError("Frozen prediction population mismatch: " + key)
        predictions = {"original_trajectory": saved["original_trajectory"].copy(), "combined": saved["rich_trajectory"].copy()}
    for probability in predictions.values():
        if not np.isfinite(probability).all() or not ((probability >= 0) & (probability <= 1)).all():
            raise ValueError("Invalid frozen control probability")
    return rich, report, predictions, frozen, (original_cache, original_cache.parent / "source_manifest.json", originals[0])


def comparison_table(y, probabilities, games, resamples):
    return {candidate + "_vs_" + reference: paired_comparison(y, probabilities[reference], probabilities[candidate], games, resamples)
            for candidate, reference in COMPARISONS}


def run(args):
    started = time.monotonic()
    safe.set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    def check():
        safe.check_resources(started, args.deadline_seconds)
    def progress(stage, **extra):
        safe.write_json(args.output / "status.json", {"stage": stage, "elapsed_s": time.monotonic() - started, **extra})
    try:
        check()
        progress("verifying_rich_inputs")
        data, rich_report, probabilities, frozen, original_paths = verify_rich_run(args.rich_run)
        for path in (Path(__file__), ROOT / "docs/rich_ablation_protocol_v1.md", ROOT / "scripts/rich_experiment.py"):
            frozen[str(path.resolve())] = safe.digest(path)
        names = tuple(data["feature_names"])
        arms = feature_arms(names)
        x, y, game, roles = (data[key] for key in ("x", "y", "game", "role"))
        masks = {role: roles == role for role in ("fit", "calibration", "development")}
        for role, mask in masks.items():
            if len(np.unique(game[mask])) < 100 or len(np.unique(y[mask])) != 2:
                raise ValueError("Insufficient population: " + role)
        fit = np.flatnonzero(masks["fit"])
        fit = fit[sample_training_indices(game[fit], data["timestamp_ms"][fit])]
        cal, dev = masks["calibration"], masks["development"]
        if len(fit) != rich_report["wp_fit_sampled_frames"]:
            raise ValueError("Fit-bin sample differs from rich run")
        population = {role: {"games": len(np.unique(game[mask])), "frames": int(mask.sum())} for role, mask in masks.items()}
        if population != rich_report["population"]:
            raise ValueError("Ablation population differs from rich run")
        lock = {"schema": SCHEMA, "stage": "locked_before_new_fitting", "input_and_protocol_sha256": frozen,
                "feature_arms": {name: [names[i] for i in indices] for name, indices in arms.items()},
                "reused_controls": ["original_trajectory", "combined"], "fit_sampled_frames": len(fit),
                "population": population, "comparisons": [list(pair) for pair in COMPARISONS]}
        safe.write_json(args.output / "experiment_lock.json", lock)
        artifacts, raw_predictions = {}, {}
        with threadpool_limits(limits=1):
            for name, indices in arms.items():
                check()
                progress("fitting", model=name)
                model = CalibratedModel("trajectory", indices).fit(x[fit], y[fit], game[fit])
                model.calibrate(x[cal], y[cal], game[cal])
                probabilities[name] = model.predict(x[dev])
                raw_predictions[name] = model.raw(x[dev])
                path = args.output / (name + ".joblib")
                joblib.dump(model, path, compress=3)
                artifacts[name] = {"path": str(path.resolve()), "sha256": safe.digest(path), "selected_columns": indices}
        check()
        progress("evaluating_all_fixed_comparisons")
        methods = {name: summarize(y[dev], p, game[dev], args.resamples) for name, p in probabilities.items()}
        ece, ece_original = ece_bootstrap(y[dev], probabilities, game[dev], args.resamples)
        ece_pairs = {}
        for candidate, reference in COMPARISONS:
            if reference == "original_trajectory":
                ece_pairs[candidate + "_vs_" + reference] = ece_original[candidate + "_vs_original_trajectory"]
            else:
                _, pairs = ece_bootstrap(y[dev], {"original_trajectory": probabilities[reference], candidate: probabilities[candidate]}, game[dev], args.resamples)
                ece_pairs[candidate + "_vs_" + reference] = pairs[candidate + "_vs_original_trajectory"]
        report = {**lock, "stage": "repeated_exploratory_development", "artifacts": artifacts,
                  "causal_claim": False, "confirmatory_evaluation_done": False, "raw_payload_reads": False,
                  "controls_refitted": False, "methods": methods,
                  "raw_new_arm_methods": {name: summarize(y[dev], p, game[dev], args.resamples) for name, p in raw_predictions.items()},
                  "raw_control_predictions": "not available in frozen prediction artifact; not regenerated",
                  "comparisons": comparison_table(y[dev], probabilities, game[dev], args.resamples),
                  "ece20_bootstrap": ece, "paired_ece20_reduction": ece_pairs, "slices": {}}
        slices = {"patch_" + str(patch): data["patch"][dev] == patch for patch in np.unique(data["patch"][dev])}
        for low, high in ((3, 10), (10, 20), (20, 30), (30, 41)):
            slices[f"minutes_{low}_{min(high,40)}"] = (data["timestamp_ms"][dev] >= low * 60000) & (data["timestamp_ms"][dev] < high * 60000)
        for label, keep in slices.items():
            check()
            count = len(np.unique(game[dev][keep]))
            report["slices"][label] = {"games": count, "frames": int(keep.sum()), "measured": count >= 30}
            if count >= 30:
                report["slices"][label]["comparisons"] = comparison_table(y[dev][keep], {name: p[keep] for name, p in probabilities.items()}, game[dev][keep], args.resamples)
        predictions_path = args.output / "development_wp_predictions.npz"
        np.savez_compressed(predictions_path, y=y[dev], game=game[dev], timestamp_ms=data["timestamp_ms"][dev],
                            **probabilities, **{name + "_raw": p for name, p in raw_predictions.items()})
        report["development_wp_predictions_sha256"] = safe.digest(predictions_path)
        for path, expected in frozen.items():
            safe.reject_quarantined(path)
            if safe.digest(path) != expected:
                raise RuntimeError("Input/code changed during ablation: " + path)
        _, final_bindings = safe.verify_inputs(*original_paths)
        if any(frozen.get(path) != digest for path, digest in final_bindings.items()):
            raise RuntimeError("Original input provenance changed during ablation")
        check()
        report["elapsed_s"] = time.monotonic() - started
        report["limitations"] = ["Same exposed development cohort; no confirmatory claim.",
                                  "Feature-block contrasts are conditional on fixed recipes, not causal attribution.",
                                  "Geometry arm also contains declared availability and gold-position coupling."]
        safe.write_json(args.output / "report.json", report)
        (args.output / "report.sha256").write_text(safe.digest(args.output / "report.json") + "  report.json\n", encoding="ascii")
        progress("completed")
        return report
    except Exception as error:
        progress("failed", error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rich-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--deadline-seconds", type=int, default=1200)
    args = parser.parse_args()
    if args.resamples != 2000 or args.deadline_seconds <= 0:
        parser.error("Fixed protocol requires 2000 resamples and a positive deadline")
    args.rich_run, args.output = args.rich_run.resolve(), args.output.resolve()
    report = run(args)
    print(json.dumps({"status": "completed", "report": str(args.output / "report.json"), "elapsed_s": report["elapsed_s"]}))


if __name__ == "__main__":
    main()
