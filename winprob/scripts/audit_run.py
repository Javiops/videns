"""Independent postrun integrity audit; never reopen raw game payloads.

Only admission/split/provenance metadata, derived NPZ arrays, model-file hashes,
and protected source files are read. Performance metric values are not accessed.
No winprob or sibling project modules are imported; no models are unpickled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for thread_option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_option] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = {"split_assignment.json", "preserved_split_assignment.json", "split_plan.json",
               "original_split_assignment.json", "original_split_plan.json"}
ROLE_MAP = {"train": "fit", "calibration": "calibration", "development": "development"}
SHA = re.compile(r"^[0-9a-fA-F]{64}$")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def id_digest(ids):
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def verify_sidecar(path):
    path = Path(path)
    tokens = path.with_suffix(".sha256").read_text(encoding="ascii").split()
    actual = sha_file(path)
    require(tokens == [actual, path.name], f"SHA256 sidecar mismatch: {path}")
    return actual


def same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def selected_games(manifest, limit):
    """Independent reproduction of the frozen role-stratified SHA256 selection."""
    groups = {role: [row for row in manifest["games"] if row["role"] == role] for role in ROLE_MAP}
    total = sum(len(rows) for rows in groups.values())
    require(total > 0 and 1 <= limit <= min(6000, total), "Invalid extraction selection size")
    allocations = {role: limit * len(rows) // total for role, rows in groups.items()}
    remaining = limit - sum(allocations.values())
    for role in sorted(groups, key=lambda role: (-(limit * len(groups[role]) % total), role))[:remaining]:
        allocations[role] += 1
    selected = []
    for role, rows in groups.items():
        ranked = sorted(rows, key=lambda row: hashlib.sha256((manifest["version"] + ":" + row["match_id"]).encode()).digest())
        selected.extend(ranked[:allocations[role]])
    return sorted(selected, key=lambda row: (row["game_creation_ms"], row["match_id"]))


def validate_admission(manifest_path):
    manifest_hash = verify_sidecar(manifest_path)
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "winprob-admission-v1" and manifest.get("state") == "temporal_manifest_ready",
            "Admission manifest is not a ready known schema")
    source_root = Path(manifest["source_data_root"]).resolve(strict=True)
    expected = {row["path"]: row["sha256"] for row in manifest["split_metadata_sources"]}
    require(len(expected) == len(manifest["split_metadata_sources"]), "Duplicate split metadata paths")
    found = set()

    def traversal_error(error):
        raise error

    for directory, subdirs, names in os.walk(source_root, followlinks=False, onerror=traversal_error):
        here = Path(directory)
        if here == source_root:
            subdirs[:] = [name for name in subdirs if name not in {"raw_match_v5", "cache"}]
        for name in subdirs:
            require(not (here / name).is_symlink(), "Linked split directory cannot be independently audited")
        for name in names:
            if name in SPLIT_NAMES:
                path = (here / name).resolve(strict=True)
                require(path.is_relative_to(source_root), "Split metadata path escaped source root")
                found.add(path.relative_to(source_root).as_posix())
    require(found == set(expected), "Source split metadata inventory changed")
    sealed, historical_val = set(), set()
    for relative, expected_hash in expected.items():
        path = source_root / relative
        require(sha_file(path) == expected_hash, f"Source split metadata hash mismatch: {relative}")
        assignment = read_json(path)
        if isinstance(assignment.get("assignment"), dict):
            assignment = assignment["assignment"]
        require(isinstance(assignment, dict) and assignment, "Malformed source assignment metadata")
        require(set(assignment.values()) <= {"train", "val", "test", "excluded", "post_cutoff"},
                "Unrecognized source assignment role")
        sealed.update(mid for mid, role in assignment.items() if role == "test")
        historical_val.update(mid for mid, role in assignment.items() if role == "val")
    build_sealed = set(sealed)
    external_sealed = set()
    external_bindings = manifest.get("external_reservation_sources", [])
    require(external_bindings, "Missing cross-project reservation bindings")
    for binding in external_bindings:
        path = Path(binding["path"])
        require(verify_sidecar(path) == binding["sha256"], "External reservation artifact hash mismatch")
        reservation = read_json(path)
        require(reservation.get("schema") == "winprob-external-reservations-v1", "Unknown external reservation schema")
        ids = reservation["reserved_test_ids"]
        require(len(ids) == len(set(ids)), "Duplicate external reservation IDs")
        require(len(ids) == binding["reserved_test_games"] and id_digest(ids) == binding["reserved_test_ids_sha256"],
                "External reservation count or ID hash mismatch")
        source = Path(reservation["source_metadata_path"])
        require(same_path(source, binding["source_metadata_path"]), "External source metadata path mismatch")
        require(sha_file(source) == reservation["source_metadata_sha256"] == binding["source_metadata_sha256"],
                "External original split metadata hash mismatch")
        if reservation["source_project"] == "macro_coach":
            original = read_json(source)
            require(set(original["test"]) == set(ids), "Macro reservation artifact omits original sealed IDs")
        external_sealed.update(ids)
    sealed.update(external_sealed)
    require(len(sealed) == manifest["denied_test_games"] and id_digest(sealed) == manifest["denied_test_ids_sha256"],
            "Global sealed union count or digest mismatch")
    entries = manifest["games"]
    by_id = {row["match_id"]: row for row in entries}
    require(len(entries) == len(by_id) == manifest["admitted_games"], "Admission game count/duplicates mismatch")
    require(id_digest(by_id) == manifest["admitted_ids_sha256"], "Admission game ID digest mismatch")
    require(not set(by_id).intersection(sealed), "Admitted games intersect a sealed source test")
    require(dict(sorted(Counter(row["role"] for row in entries).items())) == manifest["role_counts"],
            "Admission role counts mismatch")
    prior_reserved = set()
    prior_binding = manifest.get("prior_manifest")
    if prior_binding:
        prior_path = Path(prior_binding["path"])
        require(verify_sidecar(prior_path) == prior_binding["sha256"], "Prior frozen admission hash mismatch")
        prior = read_json(prior_path)  # Admission metadata only, never prior run metrics.
        old_by_id = {row["match_id"]: row for row in prior["games"]}
        require(all(mid in old_by_id and row["role"] == old_by_id[mid]["role"] for mid, row in by_id.items()),
                "Surviving frozen roles changed")
        prior_reserved = {mid for mid, row in old_by_id.items() if row["role"] == "confirmatory_reserved"}
        require(len(prior_reserved) == prior_binding["prior_confirmatory_games"]
                and id_digest(prior_reserved) == prior_binding["prior_confirmatory_ids_sha256"],
                "Prior confirmatory reservation digest mismatch")
    current_reserved = {mid for mid, row in by_id.items() if row["role"] == "confirmatory_reserved"}
    return manifest, manifest_hash, sealed, historical_val, current_reserved | prior_reserved, {
        "source_split_files": len(expected), "build_sealed_ids": len(build_sealed),
        "external_sealed_ids": len(external_sealed), "global_sealed_ids": len(sealed),
        "confirmatory_reserved_ids": len(current_reserved | prior_reserved),
        "admitted_games": len(entries),
    }


def audit_run(run_path, manifest_path, protected_path):
    run = Path(run_path).resolve(strict=True)
    manifest_path = Path(manifest_path).resolve(strict=True)
    result = {"schema": "winprob-run-integrity-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "run_path": str(run), "manifest_path": str(manifest_path), "checks": [],
              "raw_payloads_reopened": False, "models_unpickled": False,
              "performance_metrics_inspected": False, "quarantined_run_metrics_opened": False}
    try:
        manifest, manifest_hash, sealed, historical_val, reserved, admission_counts = validate_admission(manifest_path)
        result["checks"].append({"name": "admission_and_global_reservations", "passed": True, **admission_counts})
        cache_dir = run / "cache"
        metadata_path = cache_dir / "cache_metadata.json"
        metadata_hash = verify_sidecar(metadata_path)
        metadata = read_json(metadata_path)
        require(metadata.get("state") == "complete" and metadata.get("schema") == "winprob-cache-v1", "Cache is not complete")
        require(same_path(metadata["admission_manifest_path"], manifest_path)
                and metadata["admission_manifest_sha256"] == manifest_hash, "Cache admission binding mismatch")
        npz_path, source_path = cache_dir / "states.npz", cache_dir / "source_manifest.json"
        require(same_path(metadata["npz_path"], npz_path) and same_path(metadata["source_manifest_path"], source_path),
                "Cache artifact path escaped expected run")
        require(sha_file(npz_path) == metadata["npz_sha256"], "NPZ cache digest mismatch")
        require(sha_file(source_path) == metadata["source_manifest_sha256"], "Source manifest digest mismatch")
        source = read_json(source_path)
        require(same_path(source["admission_manifest"], manifest_path)
                and source["admission_manifest_sha256"] == manifest_hash, "Source manifest admission mismatch")
        for key in ("test_payload_reads", "confirmatory_payload_reads", "historical_validation_payload_reads"):
            require(source.get(key) is False, f"Source manifest did not deny {key}")
        selected = selected_games(manifest, metadata["selected_games"])
        selected_ids = {row["match_id"] for row in selected}
        require(source["selected_games"] == len(selected) and source["selected_ids_sha256"] == id_digest(selected_ids),
                "Extracted selection differs from frozen role-stratified selection")
        require(not selected_ids.intersection(sealed | historical_val | reserved), "Selected games intersect test/validation/reserved IDs")
        require(all(row["historical_validation"] is False for row in selected), "Selected historical validation game")
        cached = source["games"]
        cached_ids = {row["match_id"] for row in cached}
        exclusions = source["exclusions"]
        excluded_ids = {row["match_id"] for row in exclusions}
        require(len(cached_ids) == len(cached) and len(excluded_ids) == len(exclusions), "Duplicate cached or excluded IDs")
        require(not cached_ids.intersection(excluded_ids) and cached_ids | excluded_ids == selected_ids,
                "Selected population is not exactly cached plus explicitly excluded games")
        require(set(row["reason"] for row in exclusions) <= {"missing_archive", "zero_valid_states"}, "Unknown extraction exclusion")
        require(dict(sorted(Counter(row["reason"] for row in exclusions).items())) == metadata["exclusion_counts"],
                "Exclusion counts mismatch")
        compressed = source["compressed_sources"]
        compressed_ids = {row["match_id"] for row in compressed}
        zero_ids = {row["match_id"] for row in exclusions if row["reason"] == "zero_valid_states"}
        require(len(compressed_ids) == len(compressed) and compressed_ids == cached_ids | zero_ids,
                "Compressed-source declarations mismatch opened population")
        archive_root = Path(manifest["source_data_root"]).resolve() / "raw_match_v5"
        for row in compressed:
            for endpoint in ("match", "timeline"):
                item = row[endpoint]
                expected_path = archive_root / endpoint / (row["match_id"] + ".json.gz")
                require(same_path(item["path"], expected_path) and SHA.fullmatch(item["sha256"])
                        and type(item["bytes"]) is int and item["bytes"] > 0,
                        "Malformed compressed-source hash declaration")
        for name, expected_hash in metadata["code_sha256"].items():
            require(Path(name).name == name, "Unexpected cache code binding path")
            require(sha_file(ROOT / "winprob" / name) == expected_hash, f"Cache code changed: {name}")
        result["checks"].append({"name": "cache_source_admission_hash_chain", "passed": True,
                                 "selected_games": len(selected), "cached_games": len(cached),
                                 "excluded_games": len(exclusions), "compressed_source_declarations": len(compressed),
                                 "selected_test_or_reserved_intersections": 0})
        selected_by_id = {row["match_id"]: row for row in selected}
        with np.load(npz_path, allow_pickle=False) as arrays:
            games, roles = arrays["game"], arrays["role"]
            timestamps, labels = arrays["timestamp_ms"], arrays["y"]
            row_count = len(games)
            for key in ("x", "role", "y", "timestamp_ms", "start_ms", "split_time_ms", "patch"):
                require(len(arrays[key]) == row_count, f"Cache row length mismatch: {key}")
            require(arrays["feature_names"].tolist() == metadata["feature_names"], "Feature schema differs from cache metadata")
            x = arrays["x"]
            require(x.ndim == 2 and x.shape[1] == len(metadata["feature_names"]) and np.isfinite(x).all(), "Invalid feature array")
            del x
            require(np.isin(labels, [0, 1]).all(), "Nonbinary cached labels")
            require(set(np.unique(roles)) <= set(ROLE_MAP.values()), "Forbidden cached role")
            expected_game_indices = np.repeat(np.arange(len(cached)), [row["rows"] for row in cached])
            require(np.array_equal(games, expected_game_indices), "Cached game IDs/row counts/order mismatch")
            require(metadata["cached_games"] == len(cached) and metadata["rows"] == row_count, "Cached population count mismatch")
            offset = 0
            split_times = arrays["split_time_ms"]
            patches = arrays["patch"]
            for index, row in enumerate(cached):
                admitted = selected_by_id[row["match_id"]]
                require(row["game_index"] == index and row["source_role"] == admitted["role"]
                        and row["role"] == ROLE_MAP[admitted["role"]], "Source/cache role mismatch")
                stop = offset + row["rows"]
                require(np.all(roles[offset:stop] == row["role"]), "Game crosses cache roles")
                require(len(np.unique(labels[offset:stop])) == 1, "A game's outcome label changes across frames")
                require(np.all(split_times[offset:stop] == admitted["game_creation_ms"])
                        and np.all(patches[offset:stop] == admitted["patch"]), "Cached game metadata mismatch")
                require(np.all(np.diff(timestamps[offset:stop]) > 0)
                        and np.all((timestamps[offset:stop] >= 180000) & (timestamps[offset:stop] <= 2400000)),
                        "Cached timestamps violate admitted window/order")
                offset = stop
            require(split_times[roles == "fit"].max() < split_times[roles == "calibration"].min()
                    and split_times[roles == "calibration"].max() < split_times[roles == "development"].min(),
                    "Temporal fit/calibration/development ordering violated")
            role_games = dict(sorted(Counter(row["role"] for row in cached).items()))
            role_rows = {role: int(np.sum(roles == role)) for role in ROLE_MAP.values()}
            require(role_games == metadata["role_game_counts"] and role_rows == metadata["role_row_counts"], "Role population metadata mismatch")
            report_path = run / "experiment" / "report.json"
            report = read_json(report_path)
            # Access only provenance and population fields. Never inspect scores or gates.
            require(report["cache_sha256"] == metadata["npz_sha256"] and same_path(report["source_cache"], npz_path),
                    "Experiment cache binding mismatch")
            require(report["cache_metadata_sha256"] == metadata_hash
                    and same_path(report["cache_metadata_path"], metadata_path), "Experiment cache-metadata binding mismatch")
            require(report["features"] == metadata["feature_names"], "Experiment feature schema mismatch")
            for role in ROLE_MAP.values():
                population = report["population"][role]
                require(population["games"] == role_games[role] and population["frames"] == role_rows[role],
                        f"Report population differs from cache: {role}")
            dev = roles == "development"
            pred_path = run / "experiment" / "development_predictions.npz"
            with np.load(pred_path, allow_pickle=False) as predictions:
                require(np.array_equal(predictions["game"], games[dev])
                        and np.array_equal(predictions["timestamp_ms"], timestamps[dev])
                        and np.array_equal(predictions["y"], labels[dev]), "Saved predictions do not match exact development population")
                for name in ("constant", "logistic", "snapshot", "trajectory"):
                    p = predictions[name]
                    require(p.shape == labels[dev].shape and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)),
                            f"Invalid probability array: {name}")
                    require(report["models"][name]["games"] == role_games["development"]
                            and report["models"][name]["frames"] == role_rows["development"], "Model report population mismatch")
            result["checks"].append({"name": "cache_and_prediction_population", "passed": True,
                                     "rows": row_count, "role_game_counts": role_games, "role_row_counts": role_rows,
                                     "prediction_games": role_games["development"], "prediction_rows": role_rows["development"]})
        expected_code_paths = {str(path.relative_to(ROOT)) for path in (ROOT / "winprob").glob("*.py")}
        expected_code_paths.add(str(Path("docs") / "protocol_v1.md"))
        require(set(report["code_sha256"]) == expected_code_paths, "Experiment source-file inventory changed")
        for relative, expected_hash in report["code_sha256"].items():
            path = (ROOT / relative).resolve(strict=True)
            require(path.is_relative_to(ROOT) and sha_file(path) == expected_hash, f"Experiment code changed: {relative}")
        for name in ("logistic", "snapshot", "trajectory"):
            require(sha_file(run / "experiment" / f"{name}.joblib") == report["models"][name]["artifact_sha256"],
                    f"Model artifact hash mismatch: {name}")
        require(read_json(run / "cache" / "status.json")["state"] == "complete", "Cache status incomplete")
        require(read_json(run / "experiment" / "status.json")["stage"] == "completed", "Experiment status incomplete")
        protected = read_json(protected_path)
        require(isinstance(protected, list) and protected, "Protected file baseline is missing/empty")
        for entry in protected:
            require(sha_file(entry["path"]) == entry["sha256"].lower(), f"Protected file changed: {entry['path']}")
        result["checks"].append({"name": "models_code_and_protected_files", "passed": True,
                                 "model_files": 3, "experiment_code_files": len(report["code_sha256"]),
                                 "protected_files": len(protected)})
        require(verify_sidecar(metadata_path) == metadata_hash and verify_sidecar(manifest_path) == manifest_hash,
                "Provenance metadata changed during integrity audit")
        result["status"] = "passed"
        result["limitations"] = [
            "Raw compressed-source digests are verified as bound declarations; raw files are deliberately not reopened/rehashed.",
            "This is an integrity audit, not a review of performance metrics, practical value, or causal validity.",
        ]
    except Exception as error:
        result["status"] = "not_ready" if isinstance(error, FileNotFoundError) else "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protected", type=Path, default=ROOT / "data" / "reports" / "protected_files_before_pilot.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_run(args.run, args.manifest, args.protected)
    output = args.output or args.run / "integrity_report.json"
    forbidden = {args.manifest.resolve(), args.protected.resolve(), (args.run / "experiment" / "report.json").resolve(),
                 (args.run / "cache" / "cache_metadata.json").resolve(), (args.run / "cache" / "source_manifest.json").resolve()}
    require(output.resolve() not in forbidden, "Integrity output must not overwrite source evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "report": str(output.resolve()),
                      "checks_completed": len(report["checks"]), "error": report.get("error")}, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
