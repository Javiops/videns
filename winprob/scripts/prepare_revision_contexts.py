"""Prepare a sealed, outcome-free x19 cache for the fixed revision challenge."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

for option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[option] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from winprob import admission, features
from setback_experiment import (check_resources, digest, reject_quarantined,
                                set_below_normal_priority, verify_admitted_sources,
                                verify_sidecar, write_json)

SCHEMA = "winprob-revision-context-v1"
ROLE_MAP = {"train": "fit", "calibration": "calibration", "development": "development"}
EXPECTED_ROLES = {"fit": 4002, "calibration": 1000, "development": 859}
EXPECTED_REUSE = 5305
EXPECTED_NEW_RAW = 556
EXPECTED_OVERLAPPING_DEVELOPMENT = 303
ARRAY_KEYS = ("x", "game", "role", "timestamp_ms", "start_ms", "split_time_ms", "patch", "feature_names")
REUSED_FIELDS = ("x", "role", "timestamp_ms", "start_ms", "split_time_ms", "patch")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_context_states(match, timeline):
    """Base x19 exactly, without reading match outcomes or final statistics."""
    mid = match.get("metadata", {}).get("matchId")
    require(isinstance(mid, str) and mid and mid == timeline.get("metadata", {}).get("matchId"), "Game identity mismatch")
    info = match["info"]
    start = features._integer(info.get("gameStartTimestamp"))
    version = info.get("gameVersion")
    parsed = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version) if isinstance(version, str) else None
    require(parsed is not None, "Invalid patch version")
    patch = f"{int(parsed.group(1))}.{int(parsed.group(2))}"
    teams = {}
    for player in info.get("participants", []):
        pid, team = features._integer(player.get("participantId")), features._integer(player.get("teamId"))
        require(pid not in teams and team in (100, 200), "Duplicate participant or unknown team")
        teams[pid] = team
    require(set(teams) == set(range(1, 11)) and Counter(teams.values()) == {100: 5, 200: 5}, "Invalid ten-player roster")
    info = timeline["info"]
    require(features._integer(info.get("frameInterval")) == 60000, "Require minute timeline frames")
    raw_frames = info.get("frames")
    require(isinstance(raw_frames, list), "Timeline frames must be a list")
    frames = sorted(((features._integer(frame.get("timestamp")), frame) for frame in raw_frames), key=lambda pair: pair[0])
    require(len({timestamp for timestamp, _ in frames}) == len(frames), "Duplicate frame timestamps")
    cutoff, events = math.inf, []
    for frame_time, frame in frames:
        for event in frame.get("events", []):
            timestamp = features._integer(event.get("timestamp"))
            if event.get("type") == "GAME_END":
                cutoff = min(cutoff, frame_time, timestamp)
            events.append((timestamp, event))
    require(cutoff != math.inf, "Complete timeline requires GAME_END")
    events.sort(key=lambda pair: pair[0])
    totals, event_index, previous, rows = {100: Counter(), 200: Counter()}, 0, None, []
    for timestamp, frame in frames:
        if timestamp >= cutoff or timestamp > 40 * 60000:
            break
        while event_index < len(events) and events[event_index][0] <= timestamp:
            features._credit_event(events[event_index][1], teams, totals)
            event_index += 1
        state = features._frame_diffs(frame, teams)
        if state is None:
            continue
        slope, available = 0.0, float(previous is not None)
        if previous is not None:
            old_time, old_gold = previous
            slope = (state["gold_diff"] - old_gold) / ((timestamp - old_time) / 60000.0)
        previous = timestamp, state["gold_diff"]
        if timestamp < 3 * 60000:
            continue
        x = {"time_min": timestamp / 60000.0, **state}
        x.update({f"{name}_diff": float(totals[100][name] - totals[200][name]) for name in features._EVENT_FEATURES})
        x.update(gold_diff_slope_per_min=slope, gold_diff_slope_available=available)
        require(set(x) == set(features.ALL_FEATURES), "Context feature schema mismatch")
        rows.append({"game_id": mid, "timestamp_ms": timestamp, "game_start_ms": start, "patch": patch, "x": x})
    return rows


def reserved_ids(manifest):
    """Recompute only metadata reservations; no raw file or outcome field."""
    admission.verify_manifest_sources(manifest)
    source_root = Path(manifest["source_data_root"])
    sealed, historical_val = set(), set()
    for binding in manifest["split_metadata_sources"]:
        assignment, observed_hash = admission.read_assignment(source_root / binding["path"])
        require(observed_hash == binding["sha256"], "Split metadata hash changed")
        sealed.update(mid for mid, role in assignment.items() if role == "test")
        historical_val.update(mid for mid, role in assignment.items() if role == "val")
    for binding in manifest.get("external_reservation_sources", []):
        ids, actual = admission.read_external_reservations(Path(binding["path"]))
        require(actual == binding, "External reservations changed")
        sealed.update(ids)
    require(len(sealed) == manifest["denied_test_games"] and admission.id_digest(sealed) == manifest["denied_test_ids_sha256"],
            "Global sealed ID union mismatch")
    confirmation = {row["match_id"] for row in manifest["games"] if row["role"] == "confirmatory_reserved"}
    if manifest.get("prior_manifest"):
        prior, actual = admission.read_prior_manifest(Path(manifest["prior_manifest"]["path"]))
        require(actual == manifest["prior_manifest"], "Prior admission binding changed")
        confirmation.update(row["match_id"] for row in prior["games"] if row["role"] == "confirmatory_reserved")
    return sealed | historical_val | confirmation, {
        "global_sealed_ids": len(sealed), "historical_validation_ids": len(historical_val),
        "confirmatory_reserved_ids": len(confirmation)}


def select_population(manifest, base_source, forbidden, expected_roles=None, expected_reuse=None, expected_new_raw=None):
    expected_roles = EXPECTED_ROLES if expected_roles is None else expected_roles
    expected_reuse = EXPECTED_REUSE if expected_reuse is None else expected_reuse
    expected_new_raw = EXPECTED_NEW_RAW if expected_new_raw is None else expected_new_raw
    entries = {row["match_id"]: row for row in manifest["games"]}
    old = {row["match_id"]: row for row in base_source["games"]}
    require(len(entries) == len(manifest["games"]) and len(old) == len(base_source["games"]), "Duplicate source identity")
    selected = {mid for mid, row in old.items() if row["role"] in ("fit", "calibration")}
    selected.update(mid for mid, row in entries.items() if row["role"] == "development" and row["patch"] == "16.19")
    require(selected and len(selected) <= 6000 and not selected & set(forbidden), "Selected population intersects a forbidden ID or exceeds cap")
    require(selected <= set(entries), "Selected game absent from frozen admission")
    for mid in selected:
        row = entries[mid]
        require(row.get("historical_validation") is False and row["role"] in ROLE_MAP, "Forbidden selected source role")
        if mid in old:
            require(old[mid]["source_role"] == row["role"] and old[mid]["role"] == ROLE_MAP[row["role"]], "Original role changed")
    counts = dict(Counter(ROLE_MAP[entries[mid]["role"]] for mid in selected))
    require(counts == expected_roles, f"Fixed population counts changed: {counts}")
    require(len(selected & set(old)) == expected_reuse and len(selected - set(old)) == expected_new_raw,
            "Fixed reused/new-source population changed")
    return sorted((entries[mid] for mid in selected), key=lambda row: (row["game_creation_ms"], row["match_id"]))


def load_context_arrays(path):
    """Do not access y, even when the original supervised NPZ contains it."""
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in ARRAY_KEYS}


def read_new_payload(path, endpoint, entry, archive, allowed_ids, forbidden_ids, ledger):
    mid = entry["match_id"]
    require(mid in allowed_ids and mid not in forbidden_ids and entry["role"] == "development" and entry["patch"] == "16.19",
            "Raw access attempted before development admission")
    expected = archive / endpoint / (mid + ".json.gz")
    require(path == expected, "Unexpected raw archive path")
    reject_quarantined(path)
    ledger({"event": "read_started", "match_id": mid, "endpoint": endpoint, "path": str(path)})
    require(path.resolve(strict=True).is_relative_to(archive.resolve(strict=True)), "Raw source escaped archive root")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Raw source changed while being read")
    envelope = json.loads(gzip.decompress(raw).decode("utf-8"))
    require(isinstance(envelope, dict) and type(envelope.get("schema")) is int and envelope["schema"] == 1
            and envelope.get("endpoint") == endpoint and envelope.get("match_id") == mid
            and isinstance(envelope.get("payload"), dict)
            and envelope["payload"].get("metadata", {}).get("matchId") == mid, "Raw envelope identity/schema mismatch")
    binding = {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    ledger({"event": "read_complete", "match_id": mid, "endpoint": endpoint, "sha256": binding["sha256"]})
    return envelope["payload"], binding


def run(args):
    started = time.monotonic()
    set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    ledger_count = 0
    def status(stage, **extra):
        write_json(args.output / "status.json", {"stage": stage, "elapsed_s": time.monotonic() - started, **extra})
    def check():
        check_resources(started, args.deadline_seconds)
    def ledger(entry):
        nonlocal ledger_count
        ledger_count += 1
        with (args.output / "access_ledger.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"number": ledger_count, "utc": datetime.now(timezone.utc).isoformat(), **entry}, sort_keys=True) + "\n")
            handle.flush()
    current_id = None
    try:
        check()
        status("verifying_outcome_free_selection")
        for path in (args.admission, args.base_cache):
            reject_quarantined(path)
        manifest_hash = verify_sidecar(args.admission)
        manifest = json.loads(args.admission.read_text(encoding="utf-8"))
        forbidden, denial_counts = reserved_ids(manifest)
        meta_path, base_path, source_path = (args.base_cache / name for name in ("cache_metadata.json", "states.npz", "source_manifest.json"))
        meta_hash = verify_sidecar(meta_path)
        base_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        require(base_meta.get("schema") == "winprob-cache-v1" and base_meta.get("state") == "complete", "Require original complete W1 cache")
        require(Path(base_meta["npz_path"]).resolve() == base_path and Path(base_meta["source_manifest_path"]).resolve() == source_path,
                "Original cache paths do not match supplied directory")
        require(digest(base_path) == base_meta["npz_sha256"] and digest(source_path) == base_meta["source_manifest_sha256"], "Original cache/source digest mismatch")
        source = json.loads(source_path.read_text(encoding="utf-8"))
        require(all(source.get(key) is False for key in ("test_payload_reads", "confirmatory_payload_reads", "historical_validation_payload_reads")),
                "Original source population not verified clean")
        source_admission, source_hash = verify_admitted_sources(source, base_meta)
        require(source_admission == args.admission and source_hash == manifest_hash, "Original cache belongs to another admission")
        selected = select_population(manifest, source, forbidden)
        selected_ids = {row["match_id"] for row in selected}
        old_by_id = {row["match_id"]: row for row in source["games"]}
        old_bindings = {row["match_id"]: row for row in source["compressed_sources"]}
        require(len(old_bindings) == len(source["compressed_sources"]), "Duplicate original source binding")
        base = load_context_arrays(base_path)
        require(tuple(base["feature_names"]) == features.ALL_FEATURES and base["x"].dtype == np.float32
                and base["x"].ndim == 2 and base["x"].shape[1] == len(features.ALL_FEATURES), "Original x19 schema/dtype mismatch")
        require(all(len(base[key]) == len(base["x"]) for key in ARRAY_KEYS if key != "feature_names"), "Original array row lengths mismatch")
        require(np.isfinite(base["x"]).all(), "Original context contains nonfinite values")
        frozen = {str(path.resolve()): digest(path) for path in (
            Path(__file__), ROOT / "docs/revision_context_protocol_v1.md", Path(features.__file__), Path(admission.__file__),
            ROOT / "scripts/setback_experiment.py", args.admission, args.admission.with_suffix(".sha256"),
            meta_path, meta_path.with_suffix(".sha256"), base_path, source_path)}
        require(set(base_meta["code_sha256"]) == {"cache.py", "features.py", "admission.py"}, "Original extraction code binding is incomplete")
        for name, expected in base_meta["code_sha256"].items():
            require(digest(ROOT / "winprob" / name) == expected, "Original extraction code changed: " + name)
        lock = {"schema": SCHEMA, "stage": "locked_before_new_raw_reads", "admission_manifest_sha256": manifest_hash,
                "base_cache_metadata_sha256": meta_hash, "binding_sha256": frozen, "selected_games": len(selected),
                "selected_ids_sha256": admission.id_digest(selected_ids), "expected_role_games": EXPECTED_ROLES,
                "expected_reused_games": EXPECTED_REUSE, "expected_new_raw_games": EXPECTED_NEW_RAW,
                "reservation_counts": denial_counts, "cache_y_accessed": False, "outcome_fields_accessed": False}
        write_json(args.output / "preparation_lock.json", lock)
        archive = (Path(manifest["source_data_root"]) / "raw_match_v5").resolve(strict=True)
        chunks = {key: [] for key in ARRAY_KEYS if key != "feature_names"}
        output_games, compressed_sources, reused_checks = [], [], []
        new_count = 0
        for index, entry in enumerate(selected):
            current_id = entry["match_id"]
            role = ROLE_MAP[entry["role"]]
            if index % 50 == 0:
                check()
                status("preparing_contexts", processed_games=index, selected_games=len(selected), new_raw_games=new_count)
            if current_id in old_by_id:
                old = old_by_id[current_id]
                require(current_id in old_bindings, "Original game has no compressed-source binding")
                indices = np.flatnonzero(base["game"] == old["game_index"])
                require(len(indices) == old["rows"] and set(base["role"][indices].tolist()) == {role}, "Original game row count/role mismatch")
                require(np.all(base["split_time_ms"][indices] == entry["game_creation_ms"])
                        and np.all(base["patch"][indices] == entry["patch"]), "Original game metadata mismatch")
                part = {key: base[key][indices].copy() for key in REUSED_FIELDS}
                reused_checks.append((len(output_games), indices))
                compressed_sources.append(old_bindings[current_id])
                provenance = {"context_source": "original_cache_reuse", "base_game_index": old["game_index"]}
            else:
                require(new_count < EXPECTED_NEW_RAW, "New raw-game budget exceeded")
                if new_count % 50 == 0:
                    require(forbidden == reserved_ids(manifest)[0], "Reservations changed before new raw access")
                match, match_binding = read_new_payload(archive / "match" / (current_id + ".json.gz"), "match", entry, archive, selected_ids, forbidden, ledger)
                timeline, timeline_binding = read_new_payload(archive / "timeline" / (current_id + ".json.gz"), "timeline", entry, archive, selected_ids, forbidden, ledger)
                info = match["info"]
                require(type(info.get("queueId")) is int and info["queueId"] == 420
                        and type(info.get("gameCreation")) is int and info["gameCreation"] == entry["game_creation_ms"], "New source queue/date metadata mismatch")
                rows = extract_context_states(match, timeline)
                del match, timeline
                require(rows and all(row["game_id"] == current_id and row["patch"] == entry["patch"] for row in rows),
                        "New source has no qualifying states or changed identity/patch")
                count = len(rows)
                part = {"x": np.asarray([[row["x"][name] for name in features.ALL_FEATURES] for row in rows], dtype=np.float32),
                        "role": np.full(count, role, dtype="U11"), "timestamp_ms": np.asarray([row["timestamp_ms"] for row in rows], dtype=np.int64),
                        "start_ms": np.asarray([row["game_start_ms"] for row in rows], dtype=np.int64),
                        "split_time_ms": np.full(count, entry["game_creation_ms"], dtype=np.int64), "patch": np.full(count, entry["patch"], dtype="U16")}
                compressed_sources.append({"match_id": current_id, "match": match_binding, "timeline": timeline_binding})
                provenance = {"context_source": "new_outcome_free_extraction"}
                new_count += 1
            count = len(part["x"])
            require(count > 0 and np.isfinite(part["x"]).all() and np.all(np.diff(part["timestamp_ms"]) > 0)
                    and np.all((part["timestamp_ms"] >= 180000) & (part["timestamp_ms"] <= 2400000)), "Invalid complete-frame context block")
            for key in REUSED_FIELDS:
                chunks[key].append(part[key])
            chunks["game"].append(np.full(count, index, dtype=np.int32))
            output_games.append({"game_index": index, "match_id": current_id, "role": role, "source_role": entry["role"],
                                 "patch": entry["patch"], "game_creation_ms": entry["game_creation_ms"], "rows": count, **provenance})
        require(new_count == EXPECTED_NEW_RAW and len(reused_checks) == EXPECTED_REUSE, "Completed source counts differ from lock")
        arrays = {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}
        arrays["feature_names"] = np.asarray(features.ALL_FEATURES)
        require(set(arrays) == set(ARRAY_KEYS) and "y" not in arrays, "Outcome array entered context cache")
        parity_dev = 0
        for new_index, old_indices in reused_checks:
            new_indices = np.flatnonzero(arrays["game"] == new_index)
            for key in REUSED_FIELDS:
                require(np.array_equal(arrays[key][new_indices], base[key][old_indices]), "Reused context parity failed: " + key)
            parity_dev += int(output_games[new_index]["role"] == "development")
        require(parity_dev == EXPECTED_OVERLAPPING_DEVELOPMENT, "Overlapping development context count changed")
        for left, right in (("fit", "calibration"), ("calibration", "development")):
            require(arrays["split_time_ms"][arrays["role"] == left].max() < arrays["split_time_ms"][arrays["role"] == right].min(), "Temporal context roles overlap")
        check()
        require(forbidden == reserved_ids(manifest)[0], "Reservations changed before publication")
        for path, expected in frozen.items():
            require(digest(path) == expected, "Frozen input/code changed: " + path)
        npz_path = args.output / "states.npz"
        np.savez_compressed(npz_path, **arrays)
        with np.load(npz_path, allow_pickle=False) as saved:
            require(set(saved.files) == set(ARRAY_KEYS) and "y" not in saved.files, "Published NPZ array inventory mismatch")
            require(all(np.array_equal(saved[key], arrays[key]) for key in ARRAY_KEYS), "Published context array parity mismatch")
        output_source = {"schema": SCHEMA, "admission_manifest": str(args.admission), "admission_manifest_sha256": manifest_hash,
                         "selected_ids_sha256": admission.id_digest(selected_ids), "selected_games": len(selected),
                         "games": output_games, "compressed_sources": compressed_sources, "exclusions": [],
                         "test_payload_reads": False, "confirmatory_payload_reads": False, "historical_validation_payload_reads": False,
                         "outcome_fields_accessed": False, "cache_y_accessed": False,
                         "source_deserialization_includes_outcome_fields": True}
        output_source_path = args.output / "source_manifest.json"
        write_json(output_source_path, output_source)
        verify_admitted_sources(output_source, {"admission_manifest_path": str(args.admission), "admission_manifest_sha256": manifest_hash})
        metadata_path = args.output / "cache_metadata.json"
        metadata = {**lock, "state": "complete", "stage": "complete", "cache_metadata_path": str(metadata_path),
                    "npz_path": str(npz_path), "npz_sha256": digest(npz_path), "source_manifest_path": str(output_source_path),
                    "source_manifest_sha256": digest(output_source_path), "admission_manifest_path": str(args.admission),
                    "base_cache_metadata_path": str(meta_path), "base_npz_sha256": base_meta["npz_sha256"],
                    "feature_version": features.FEATURE_VERSION, "feature_names": list(features.ALL_FEATURES),
                    "array_names": list(ARRAY_KEYS), "contains_outcome_array": False, "cached_games": len(output_games), "rows": len(arrays["x"]),
                    "role_game_counts": dict(sorted(Counter(row["role"] for row in output_games).items())),
                    "role_row_counts": {role: int(np.sum(arrays["role"] == role)) for role in EXPECTED_ROLES},
                    "reused_games": len(reused_checks), "new_raw_games": new_count, "raw_payloads_opened": 2 * new_count,
                    "overlapping_development_parity_games": parity_dev, "parity_scope": "exact array-copy parity; no independent raw reread of reused games",
                    "access_ledger_sha256": digest(args.output / "access_ledger.jsonl"),
                    "elapsed_s": time.monotonic() - started, "completed_utc": datetime.now(timezone.utc).isoformat()}
        write_json(metadata_path, metadata)
        metadata_path.with_suffix(".sha256").write_text(digest(metadata_path) + "  " + metadata_path.name + "\n", encoding="ascii")
        status("completed", cached_games=len(output_games), rows=metadata["rows"], new_raw_games=new_count)
        return metadata
    except Exception as error:
        status("failed", match_id=current_id, error_type=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 0 < args.deadline_seconds <= 1200:
        parser.error("Context preparation deadline must be within the fixed 20-minute budget")
    for key in ("admission", "base_cache", "output"):
        setattr(args, key, getattr(args, key).resolve())
    result = run(args)
    print(json.dumps({"metadata": result["cache_metadata_path"], "games": result["cached_games"], "rows": result["rows"],
                      "reused_games": result["reused_games"], "new_raw_games": result["new_raw_games"], "contains_y": False}))


if __name__ == "__main__":
    main()
