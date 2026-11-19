"""Bounded standalone raw-archive extraction with manifest-first admission."""
from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import admission, features

CACHE_SCHEMA = "winprob-cache-v1"
ROLE_MAP = {"train": "fit", "calibration": "calibration", "development": "development"}


def _sha_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(admission.canonical_bytes(value))
    os.replace(temporary, path)


def _load_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    sidecar = path.with_suffix(".sha256")
    tokens = sidecar.read_text(encoding="ascii").split()
    actual = admission.digest_bytes(raw)
    if len(tokens) != 2 or tokens[0] != actual or tokens[1] != path.name:
        raise ValueError("Admission manifest SHA256 sidecar mismatch")
    return json.loads(raw), actual


def _read_payload(path: Path, endpoint: str, match_id: str) -> tuple[dict, dict]:
    compressed = path.read_bytes()
    try:
        envelope = json.loads(gzip.decompress(compressed).decode("utf-8"))
    except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid gzip/JSON archive for {endpoint}/{match_id}") from error
    if (not isinstance(envelope, dict) or type(envelope.get("schema")) is not int
            or envelope["schema"] != 1 or envelope.get("endpoint") != endpoint
            or envelope.get("match_id") != match_id or not isinstance(envelope.get("payload"), dict)):
        raise ValueError(f"Invalid archive envelope for {endpoint}/{match_id}")
    payload = envelope["payload"]
    if payload.get("metadata", {}).get("matchId") != match_id:
        raise ValueError(f"Payload match identity mismatch for {endpoint}/{match_id}")
    return payload, {"path": str(path), "bytes": len(compressed),
                     "sha256": admission.digest_bytes(compressed)}


def _validate_match_metadata(payload: dict, entry: dict) -> None:
    info = payload.get("info")
    if not isinstance(info, dict) or type(info.get("queueId")) is not int or info["queueId"] != 420:
        raise ValueError(f"Wrong or absent ranked queue for {entry['match_id']}")
    version = info.get("gameVersion")
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version) if isinstance(version, str) else None
    if match is None or f"{int(match.group(1))}.{int(match.group(2))}" != entry["patch"]:
        raise ValueError(f"Payload patch mismatch for {entry['match_id']}")
    if "gameCreation" in info and (type(info["gameCreation"]) is not int
                                    or info["gameCreation"] != entry["game_creation_ms"]):
        raise ValueError(f"Payload gameCreation mismatch for {entry['match_id']}")


def build_cache(manifest_path: str | Path, output_dir: str | Path, max_games: int = 6000,
                resource_check: Callable[[], object] | None = None) -> dict:
    """Read admitted development payloads one game at a time; seal array artifacts.

    No raw file is opened until its ID has passed the manifest selection.
    Missing files and zero valid frames are explicit exclusions. Malformed or
    mismatched archives fail the run; a partial run never gets success metadata.
    """
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest, manifest_sha256 = _load_manifest(manifest_path)
    selected = admission.extraction_candidates(manifest, max_games=max_games)
    code_paths = (Path(__file__), Path(features.__file__), Path(admission.__file__))
    code_sha256 = {path.name: _sha_file(path) for path in code_paths}
    # Defense in depth before any payload path is even constructed.
    if (len({entry["match_id"] for entry in selected}) != len(selected)
            or any(entry["role"] not in ROLE_MAP or entry.get("historical_validation") is not False
                   or not admission.MATCH_ID.fullmatch(entry["match_id"]) for entry in selected)):
        raise ValueError("Selection contains duplicate, reserved, or historical validation games")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    archive = Path(manifest["source_data_root"]).resolve(strict=True) / "raw_match_v5"
    status_path = output_dir / "status.json"
    status = {"schema": CACHE_SCHEMA, "state": "running", "selected_games": len(selected),
              "processed_games": 0, "cached_games": 0, "rows": 0,
              "manifest_sha256": manifest_sha256, "started_utc": datetime.now(timezone.utc).isoformat()}
    _json(status_path, status)
    chunks = {key: [] for key in ("x", "y", "game", "role", "timestamp_ms", "start_ms", "split_time_ms", "patch")}
    games = []
    exclusions = []
    compressed_sources = []
    try:
        if resource_check is not None:
            resource_check()
        for number, entry in enumerate(selected, start=1):
            mid = entry["match_id"]
            paths = {endpoint: archive / endpoint / (mid + ".json.gz") for endpoint in ("match", "timeline")}
            missing = [endpoint for endpoint, path in paths.items() if not path.is_file()]
            if missing:
                exclusions.append({"match_id": mid, "role": entry["role"],
                                   "reason": "missing_archive", "missing_endpoints": missing})
            else:
                for path in paths.values():
                    if not path.resolve(strict=True).is_relative_to(archive.resolve(strict=True)):
                        raise ValueError(f"Raw archive symlink escaped its root: {path}")
                match_payload, match_source = _read_payload(paths["match"], "match", mid)
                _validate_match_metadata(match_payload, entry)
                timeline_payload, timeline_source = _read_payload(paths["timeline"], "timeline", mid)
                rows = features.extract_game_states(match_payload, timeline_payload)
                compressed_sources.append({"match_id": mid, "match": match_source, "timeline": timeline_source})
                del match_payload, timeline_payload
                if not rows:
                    exclusions.append({"match_id": mid, "role": entry["role"], "reason": "zero_valid_states"})
                else:
                    game_index = len(games)
                    role = ROLE_MAP[entry["role"]]
                    count = len(rows)
                    if any(row["game_id"] != mid or row["patch"] != entry["patch"] for row in rows):
                        raise ValueError(f"Feature rows carry mismatched identity/patch: {mid}")
                    values = np.asarray([[row["x"][name] for name in features.ALL_FEATURES] for row in rows], dtype=np.float32)
                    if not np.isfinite(values).all():
                        raise ValueError(f"Nonfinite float32 features for {mid}")
                    chunks["x"].append(values)
                    chunks["y"].append(np.asarray([row["y"] for row in rows], dtype=np.int8))
                    chunks["game"].append(np.full(count, game_index, dtype=np.int32))
                    chunks["role"].append(np.full(count, role, dtype="U11"))
                    chunks["timestamp_ms"].append(np.asarray([row["timestamp_ms"] for row in rows], dtype=np.int64))
                    chunks["start_ms"].append(np.asarray([row["game_start_ms"] for row in rows], dtype=np.int64))
                    chunks["split_time_ms"].append(np.full(count, entry["game_creation_ms"], dtype=np.int64))
                    chunks["patch"].append(np.full(count, entry["patch"], dtype="U16"))
                    games.append({"game_index": game_index, "match_id": mid, "role": role,
                                  "source_role": entry["role"], "patch": entry["patch"],
                                  "game_creation_ms": entry["game_creation_ms"], "rows": count})
                    status["rows"] += count
                del rows
            status["processed_games"] = number
            status["cached_games"] = len(games)
            if number % 50 == 0 or number == len(selected):
                _json(status_path, status)
                if resource_check is not None:
                    resource_check()
        if not games:
            raise ValueError("No valid feature rows survived extraction")
        # A running upstream admission change must not silently produce success.
        admission.verify_manifest_sources(manifest)
        if {path.name: _sha_file(path) for path in code_paths} != code_sha256:
            raise ValueError("Cache/extractor/admission code changed during extraction")
        if _load_manifest(manifest_path)[1] != manifest_sha256:
            raise ValueError("Admission manifest changed during extraction")
        arrays = {key: np.concatenate(parts, axis=0) for key, parts in chunks.items()}
        del chunks
        arrays["feature_names"] = np.asarray(features.ALL_FEATURES)
        npz_path = output_dir / "states.npz"
        np.savez_compressed(npz_path, **arrays)
        del arrays
        source_manifest = {"schema": CACHE_SCHEMA, "admission_manifest": str(manifest_path),
                           "admission_manifest_sha256": manifest_sha256,
                           "selected_ids_sha256": admission.id_digest(entry["match_id"] for entry in selected),
                           "selected_games": len(selected), "games": games,
                           "compressed_sources": compressed_sources, "exclusions": exclusions,
                           "test_payload_reads": False, "confirmatory_payload_reads": False,
                           "historical_validation_payload_reads": False}
        source_path = output_dir / "source_manifest.json"
        _json(source_path, source_manifest)
        metadata_path = output_dir / "cache_metadata.json"
        metadata = {"schema": CACHE_SCHEMA, "state": "complete", "npz_path": str(npz_path),
                    "npz_sha256": _sha_file(npz_path), "source_manifest_path": str(source_path),
                    "source_manifest_sha256": _sha_file(source_path), "cache_metadata_path": str(metadata_path),
                    "admission_manifest_path": str(manifest_path), "admission_manifest_sha256": manifest_sha256,
                    "code_sha256": code_sha256,
                    "feature_version": features.FEATURE_VERSION, "feature_names": list(features.ALL_FEATURES),
                    "selected_games": len(selected), "cached_games": len(games), "rows": status["rows"],
                    "role_game_counts": dict(sorted(Counter(game["role"] for game in games).items())),
                    "role_row_counts": {role: sum(game["rows"] for game in games if game["role"] == role)
                                        for role in ROLE_MAP.values()},
                    "exclusion_counts": dict(sorted(Counter(row["reason"] for row in exclusions).items())),
                    "completed_utc": datetime.now(timezone.utc).isoformat()}
        _json(metadata_path, metadata)
        metadata_path.with_suffix(".sha256").write_text(_sha_file(metadata_path) + "  " + metadata_path.name + "\n", encoding="ascii")
        status.update({"state": "complete", "exclusion_counts": metadata["exclusion_counts"],
                       "completed_utc": metadata["completed_utc"]})
        _json(status_path, status)
        return metadata
    except BaseException as error:
        status.update({"state": "failed", "error_type": type(error).__name__, "error": str(error),
                       "failed_utc": datetime.now(timezone.utc).isoformat()})
        _json(status_path, status)
        raise
