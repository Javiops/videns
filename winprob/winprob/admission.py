"""Read-only, fail-closed admission from explicit split metadata.

This module owns no build_tracker code, imports no sibling module, and never
opens match/timeline payloads. Test reservations override every allowlist.
Historical validation games can only be development, never fit/calibration.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA = "winprob-admission-v1"
SPLIT_NAMES = frozenset({
    "split_assignment.json", "preserved_split_assignment.json", "split_plan.json",
    "original_split_assignment.json", "original_split_plan.json",
})
SKIP_AT_DATA_ROOT = frozenset({"raw_match_v5", "cache"})
KNOWN_FOLDS = frozenset({"train", "val", "test", "post_cutoff", "excluded"})
MATCH_ID = re.compile(r"[A-Za-z0-9_\-]+\Z")
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,95}\Z")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def id_digest(ids: Iterable[str]) -> str:
    return digest_bytes(("\n".join(sorted(ids)) + "\n").encode())


def _under(path: Path, root: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Path outside source data root: {path}")
    return resolved


def discover_split_files(data_root: Path) -> list[Path]:
    """Enumerate metadata only; refuse symlink escape and traversal errors."""
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Source data root must be a directory")
    found: list[Path] = []

    def onerror(error: OSError) -> None:
        raise error

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False, onerror=onerror):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [name for name in dirnames if name not in SKIP_AT_DATA_ROOT]
        for name in dirnames:
            candidate = here / name
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(root):
                raise ValueError(f"Cannot audit linked split directory: {candidate}")
        dirnames.sort()
        for name in sorted(filenames):
            if name in SPLIT_NAMES:
                found.append(_under(here / name, root))
    if not found:
        raise ValueError("No authoritative split metadata found; refusing empty denylist")
    return sorted(set(found))


def read_assignment(path: Path) -> tuple[dict[str, str], str]:
    """Load supported flat maps or plans with an `assignment` map."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if isinstance(data, dict) and isinstance(data.get("assignment"), dict):
        data = data["assignment"]
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Unrecognized or empty split metadata: {path}")
    for match_id, fold in data.items():
        if not isinstance(match_id, str) or not MATCH_ID.fullmatch(match_id):
            raise ValueError(f"Invalid match ID in {path}")
        if not isinstance(fold, str) or fold not in KNOWN_FOLDS:
            raise ValueError(f"Unknown split fold {fold!r} in {path}; review before admission")
    return data, digest_bytes(raw)


def read_external_reservations(path: Path) -> tuple[set[str], dict]:
    """Validate a versioned reservation export and its original metadata binding."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    sha256 = digest_bytes(raw)
    tokens = path.with_suffix(".sha256").read_text(encoding="ascii").split()
    if len(tokens) != 2 or tokens[0] != sha256 or tokens[1] != path.name:
        raise ValueError(f"External reservation artifact SHA256 sidecar mismatch: {path}")
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema") != "winprob-external-reservations-v1":
        raise ValueError(f"Unknown external reservation schema: {path}")
    ids = payload.get("reserved_test_ids")
    if (not isinstance(ids, list) or not all(isinstance(mid, str) and MATCH_ID.fullmatch(mid) for mid in ids)
            or len(set(ids)) != len(ids) or not isinstance(payload.get("source_project"), str)
            or not payload["source_project"]):
        raise ValueError(f"Invalid external reservation IDs/project: {path}")
    source = Path(payload["source_metadata_path"]).resolve(strict=True)
    source_sha256 = digest_bytes(source.read_bytes())
    if source_sha256 != payload.get("source_metadata_sha256"):
        raise ValueError(f"External reservation source metadata changed: {source}")
    return set(ids), {"path": str(path), "sha256": sha256,
                      "source_project": payload["source_project"],
                      "source_metadata_path": str(source), "source_metadata_sha256": source_sha256,
                      "reserved_test_games": len(ids), "reserved_test_ids_sha256": id_digest(ids)}


def read_prior_manifest(path: Path) -> tuple[dict, dict]:
    """Read a frozen prior assignment; its confirmation reserve must never shrink."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    sha256 = digest_bytes(raw)
    tokens = path.with_suffix(".sha256").read_text(encoding="ascii").split()
    if len(tokens) != 2 or tokens[0] != sha256 or tokens[1] != path.name:
        raise ValueError("Prior manifest SHA256 sidecar mismatch")
    prior = json.loads(raw.decode("utf-8-sig"))
    if prior.get("schema") != SCHEMA or prior.get("state") != "temporal_manifest_ready":
        raise ValueError("Prior manifest must have frozen temporal roles")
    games = prior.get("games")
    if not isinstance(games, list) or len({entry["match_id"] for entry in games}) != len(games):
        raise ValueError("Prior manifest has invalid game identities")
    known_roles = {"train", "calibration", "development", "development_exposed", "confirmatory_reserved"}
    if any(entry.get("role") not in known_roles for entry in games):
        raise ValueError("Prior manifest has unknown roles")
    reserved = {entry["match_id"] for entry in games if entry["role"] == "confirmatory_reserved"}
    return prior, {"path": str(path), "sha256": sha256, "version": prior["version"],
                   "policy": "Preserve every surviving game's prior role; admit no new IDs",
                   "prior_confirmatory_games": len(reserved), "prior_confirmatory_ids_sha256": id_digest(reserved)}


def read_game_metadata(db_path: Path, admitted_ids: Iterable[str]) -> dict[str, dict]:
    """Indexed identity/date/patch/queue lookups only, after test exclusion.

    Does not read win_team, duration, participants, visits, or events. Ordinary
    mode=ro sees the current WAL correctly; immutable=1 is wrong for a live DB.
    """
    ids = sorted(set(admitted_ids))
    if not ids:
        return {}
    result: dict[str, dict] = {}
    uri = Path(db_path).resolve(strict=True).as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=2)) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-1024")
        connection.execute("BEGIN")
        for start in range(0, len(ids), 250):
            chunk = ids[start:start + 250]
            placeholders = ",".join("?" for _ in chunk)
            query = ("SELECT match_id, game_creation, patch, queue_id FROM games "
                     f"WHERE match_id IN ({placeholders})")
            for match_id, created, patch, queue_id in connection.execute(query, chunk):
                if not isinstance(created, int) or created <= 0:
                    raise ValueError(f"Invalid creation timestamp for {match_id}")
                result[match_id] = {"game_creation_ms": created, "patch": patch, "queue_id": queue_id}
        connection.rollback()
    return result


def _assign_temporal_roles(entries: list[dict], calibration_fraction: float,
                           development_fraction: float, confirmatory_fraction: float) -> dict:
    """Split train-eligible IDs temporally; equal timestamps stay together."""
    eligible = sorted((entry for entry in entries if not entry["historical_validation"]),
                      key=lambda entry: (entry["game_creation_ms"], entry["match_id"]))
    train_cut = len(eligible) * (1 - calibration_fraction - development_fraction - confirmatory_fraction)
    calibration_cut = len(eligible) * (1 - development_fraction - confirmatory_fraction)
    development_cut = len(eligible) * (1 - confirmatory_fraction)
    index = 0
    while index < len(eligible):
        end = index + 1
        while end < len(eligible) and eligible[end]["game_creation_ms"] == eligible[index]["game_creation_ms"]:
            end += 1
        role = ("train" if index < train_cut else "calibration" if index < calibration_cut
                else "development" if index < development_cut else "confirmatory_reserved")
        for entry in eligible[index:end]:
            entry["role"] = role
        index = end
    for entry in entries:
        if entry["historical_validation"]:
            entry["role"] = "development_exposed"
    windows = {}
    for role in sorted({entry["role"] for entry in entries}):
        times = [entry["game_creation_ms"] for entry in entries if entry["role"] == role]
        windows[role] = {"min_creation_ms": min(times), "max_creation_ms": max(times), "games": len(times)}
    return windows


def prepare_manifest(data_root: Path, allow_sources: Iterable[Path], version: str,
                     *, db_path: Path | None = None, calibration_fraction: float = 0.15,
                     development_fraction: float = 0.15, confirmatory_fraction: float = 0.10,
                     external_reservations: Iterable[Path] = (), prior_manifest: Path | None = None) -> dict:
    """Construct a complete metadata manifest, without accessing raw payloads."""
    if not VERSION.fullmatch(version):
        raise ValueError("Version must be a safe nonempty versioned artifact name")
    if not (0 < calibration_fraction < 1 and 0 < development_fraction < 1
            and 0 < confirmatory_fraction < 1
            and calibration_fraction + development_fraction + confirmatory_fraction < 1):
        raise ValueError("Held-out fractions must leave a nonempty train fraction")
    root = Path(data_root).resolve(strict=True)
    paths = discover_split_files(root)
    allowed = {_under(Path(path), root) for path in allow_sources}
    if not allowed or not allowed.issubset(set(paths)):
        raise ValueError("Each explicit allow-source must be a discovered authoritative split file")
    reservations: set[str] = set()
    historical_validation: set[str] = set()
    candidates: dict[str, set[str]] = {}
    metadata_sources = []
    for path in paths:
        assignment, sha256 = read_assignment(path)
        relative = path.relative_to(root).as_posix()
        metadata_sources.append({"path": relative, "sha256": sha256,
                                 "explicit_allow_source": path in allowed,
                                 "fold_counts": dict(sorted(Counter(assignment.values()).items()))})
        reservations.update(mid for mid, fold in assignment.items() if fold == "test")
        historical_validation.update(mid for mid, fold in assignment.items() if fold == "val")
        if path in allowed:
            for mid, fold in assignment.items():
                if fold in {"train", "val"}:
                    candidates.setdefault(mid, set()).add(relative + ":" + fold)
    build_reserved_count = len(reservations)
    external_reserved: set[str] = set()
    external_bindings = []
    for path in sorted({Path(path).resolve(strict=True) for path in external_reservations}):
        ids, binding = read_external_reservations(path)
        external_reserved.update(ids)
        external_bindings.append(binding)
    reservations.update(external_reserved)
    safe = set(candidates) - reservations
    if not safe:
        raise ValueError("No candidates remain after sealed-test exclusion")
    metadata = read_game_metadata(db_path, safe) if db_path is not None else {}
    entries = []
    exclusions = {"reserved_test": len(set(candidates) & reservations), "missing_metadata": 0, "wrong_queue": 0}
    for mid in sorted(safe):
        if db_path is not None and mid not in metadata:
            exclusions["missing_metadata"] += 1
            continue
        if db_path is not None and metadata[mid]["queue_id"] != 420:
            exclusions["wrong_queue"] += 1
            continue
        entry = {"match_id": mid, "source_memberships": sorted(candidates[mid]),
                 "historical_validation": mid in historical_validation,
                 "role": "development_exposed" if mid in historical_validation else "pending_temporal_assignment"}
        if mid in metadata:
            entry.update(metadata[mid])
        entries.append(entry)
    if not entries:
        raise ValueError("No admitted games remain after metadata qualification")
    windows = {}
    prior_binding = None
    if prior_manifest is not None:
        if db_path is None:
            raise ValueError("Prior role preservation requires current metadata timestamps")
        prior, prior_binding = read_prior_manifest(prior_manifest)
        prior_games = {entry["match_id"]: entry for entry in prior["games"]}
        for entry in entries:
            old = prior_games.get(entry["match_id"])
            if old is None:
                raise ValueError("New ID absent from prior manifest; cannot silently reassign frozen split")
            for key in ("historical_validation", "game_creation_ms", "patch", "queue_id"):
                if old.get(key) != entry.get(key):
                    raise ValueError(f"Prior game metadata changed: {entry['match_id']} {key}")
            entry["role"] = old["role"]
        for role in sorted({entry["role"] for entry in entries}):
            times = [entry["game_creation_ms"] for entry in entries if entry["role"] == role]
            windows[role] = {"min_creation_ms": min(times), "max_creation_ms": max(times), "games": len(times)}
        entries.sort(key=lambda entry: (entry["game_creation_ms"], entry["match_id"]))
    elif db_path is not None:
        windows = _assign_temporal_roles(entries, calibration_fraction, development_fraction, confirmatory_fraction)
        entries.sort(key=lambda entry: (entry["game_creation_ms"], entry["match_id"]))
    return {
        "schema": SCHEMA, "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "temporal_manifest_ready" if db_path is not None else "metadata_only_dates_pending",
        "source_data_root": str(root), "split_metadata_sources": metadata_sources,
        "external_reservation_sources": external_bindings,
        "prior_manifest": prior_binding,
        "build_tracker_reserved_test_games": build_reserved_count,
        "external_reserved_test_games": len(external_reserved),
        "external_reservations_excluded_games": len(set(candidates) & external_reserved),
        "explicit_allow_sources": sorted(path.relative_to(root).as_posix() for path in allowed),
        "denied_test_games": len(reservations), "denied_test_ids_sha256": id_digest(reservations),
        "historical_validation_games": len(historical_validation),
        "candidate_games": len(candidates), "excluded_games": exclusions,
        "own_temporal_split_candidates": sum(not entry["historical_validation"] for entry in entries),
        "admitted_games": len(entries), "admitted_ids_sha256": id_digest(entry["match_id"] for entry in entries),
        "role_counts": dict(sorted(Counter(entry["role"] for entry in entries).items())),
        "temporal_windows": windows,
        "split_policy": {"unit": "match_id", "order": "game_creation_ms ascending; timestamp ties indivisible",
                         "train_fraction": 1 - calibration_fraction - development_fraction - confirmatory_fraction,
                         "calibration_fraction": calibration_fraction, "development_fraction": development_fraction,
                         "confirmatory_fraction": confirmatory_fraction,
                         "historical_val_policy": "development_exposed only; never fit or calibrate",
                         "future_confirmatory": "Newest train-only 10% is reserved before payload access. Open only once after protocol/model lock; existing sealed tests remain denied.",
                         "confirmatory_games": sum(entry["role"] == "confirmatory_reserved" for entry in entries),
                         "not_player_disjoint": True},
        "database_metadata": {"path": str(Path(db_path).resolve()) if db_path is not None else None,
                              "columns": ["match_id", "game_creation", "patch", "queue_id"] if db_path is not None else [],
                              "mode": "ro", "full_scan": False},
        "test_payload_reads": False, "raw_payload_reads": False, "outcome_reads": False,
        "training_authorized_by_manifest": False,
        "initial_extraction_policy": {"roles": ["train", "calibration", "development"],
                                      "max_games": 6000, "sampling": "SHA256(version + ':' + match_id), within each role",
                                      "confirmatory_payload_reads": False},
        "limitations": ["A new test reservation after this snapshot must be denied before extraction.",
                        "Development-exposed games are diagnostic only; they are not an untouched confirmation cohort.",
                        "Availability/integrity of raw archives has not yet been measured."],
        "games": entries,
    }


def verify_manifest_sources(manifest: dict) -> None:
    """Fail on any new, removed, or changed split file before payload extraction."""
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unknown admission manifest schema")
    root = Path(manifest["source_data_root"]).resolve(strict=True)
    expected = {entry["path"]: entry["sha256"] for entry in manifest["split_metadata_sources"]}
    actual = {path.relative_to(root).as_posix(): digest_bytes(path.read_bytes()) for path in discover_split_files(root)}
    if actual != expected:
        raise ValueError("Split metadata changed; create a new admission manifest before any payload read")
    external_reserved = set()
    for binding in manifest.get("external_reservation_sources", []):
        ids, actual_binding = read_external_reservations(Path(binding["path"]))
        if actual_binding != binding:
            raise ValueError("External reservation artifact changed; create a new admission manifest")
        external_reserved.update(ids)
    if any(entry["match_id"] in external_reserved for entry in manifest["games"]):
        raise ValueError("Admitted game intersects external sealed-test reservation")
    prior_binding = manifest.get("prior_manifest")
    if prior_binding is not None:
        prior, actual_binding = read_prior_manifest(Path(prior_binding["path"]))
        if actual_binding != prior_binding:
            raise ValueError("Prior manifest artifact changed")
        prior_roles = {entry["match_id"]: entry["role"] for entry in prior["games"]}
        if any(prior_roles.get(entry["match_id"]) != entry["role"] for entry in manifest["games"]):
            raise ValueError("A surviving game's prior frozen role changed")


def extraction_candidates(manifest: dict, max_games: int = 6000) -> list[dict]:
    """Deterministically sample safe roles only; preserve each role's fraction."""
    verify_manifest_sources(manifest)
    if manifest.get("state") != "temporal_manifest_ready":
        raise ValueError("Dates and temporal roles must be assigned before extraction")
    if not isinstance(max_games, int) or not 1 <= max_games <= 6000:
        raise ValueError("Initial extraction cap is 1..6000 games")
    roles = ("train", "calibration", "development")
    groups = {role: [entry for entry in manifest["games"] if entry["role"] == role] for role in roles}
    total = sum(len(group) for group in groups.values())
    if not total:
        raise ValueError("No train/calibration/development candidates")
    limit = min(max_games, total)
    allocations = {role: limit * len(groups[role]) // total for role in roles}
    remaining = limit - sum(allocations.values())
    for role in sorted(roles, key=lambda role: (-(limit * len(groups[role]) % total), role))[:remaining]:
        allocations[role] += 1
    selected = []
    for role in roles:
        group = sorted(groups[role], key=lambda entry: digest_bytes((manifest["version"] + ":" + entry["match_id"]).encode()))
        selected.extend(group[:allocations[role]])
    return sorted(selected, key=lambda entry: (entry["game_creation_ms"], entry["match_id"]))


def write_manifest(manifest: dict, output_root: Path) -> tuple[Path, str]:
    """Create a new immutable-by-contract version; never overwrite a manifest."""
    version = manifest.get("version", "")
    if not VERSION.fullmatch(version):
        raise ValueError("Invalid version")
    directory = Path(output_root).resolve() / version
    directory.mkdir(parents=True, exist_ok=False)
    payload = canonical_bytes(manifest)
    path = directory / "admission_manifest.json"
    with path.open("xb") as handle:
        handle.write(payload)
    sha256 = digest_bytes(payload)
    with (directory / "admission_manifest.sha256").open("x", encoding="ascii") as handle:
        handle.write(sha256 + "  admission_manifest.json\n")
    return path, sha256
