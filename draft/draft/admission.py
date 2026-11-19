"""Read-only, fail-closed admission for the draft study.

Metadata only: split files, versioned reservation artifacts and raw archive
file NAMES. No match or timeline payload is opened here. Every reserved test
or confirmation ID of every sibling project is denied before extraction.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "draft-admission-v1"
VIDENS = Path(__file__).resolve().parents[2]
BT_DATA = VIDENS / "build_tracker" / "data"
RAW = BT_DATA / "raw_match_v5"
SPLIT_NAMES = frozenset({
    "split_assignment.json", "preserved_split_assignment.json", "split_plan.json",
    "original_split_assignment.json", "original_split_plan.json",
})
SKIP_AT_DATA_ROOT = frozenset({"raw_match_v5", "cache"})
KNOWN_FOLDS = frozenset({"train", "val", "test", "post_cutoff", "excluded"})
MATCH_ID = re.compile(r"[A-Za-z0-9_\-]+\Z")
MACRO_RESERVATION = VIDENS / "winprob/artifacts/reservations/macro-20260929-v1/reserved_ids.json"
WINPROB_ADMISSION = VIDENS / "winprob/artifacts/admission/w1-admission-20260929-v2/admission_manifest.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def id_digest(ids) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def check_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    tokens = path.with_suffix(".sha256").read_text(encoding="ascii").split()
    if len(tokens) != 2 or tokens[0] != digest or tokens[1] != path.name:
        raise ValueError(f"SHA256 sidecar mismatch: {path}")
    return digest


def build_tracker_test_ids() -> tuple[set[str], list[dict]]:
    """Union of every 'test' fold in build_tracker split metadata."""
    denied: set[str] = set()
    sources = []
    root = BT_DATA.resolve(strict=True)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        if here == root:
            dirnames[:] = [d for d in dirnames if d not in SKIP_AT_DATA_ROOT]
        dirnames.sort()
        for name in sorted(filenames):
            if name not in SPLIT_NAMES:
                continue
            path = here / name
            raw = path.read_bytes()
            data = json.loads(raw.decode("utf-8-sig"))
            if isinstance(data, dict) and isinstance(data.get("assignment"), dict):
                data = data["assignment"]
            if not isinstance(data, dict) or not data:
                raise ValueError(f"Unrecognized split metadata: {path}")
            tests = []
            for mid, fold in data.items():
                if not MATCH_ID.fullmatch(mid) or fold not in KNOWN_FOLDS:
                    raise ValueError(f"Unknown split content in {path}: {mid!r} {fold!r}")
                if fold == "test":
                    tests.append(mid)
            denied.update(tests)
            sources.append({"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest(),
                            "test_ids": len(tests)})
    if not sources:
        raise ValueError("No build_tracker split metadata found; refusing an empty denylist")
    return denied, sources


def macro_reserved_ids() -> tuple[set[str], dict]:
    digest = check_sidecar(MACRO_RESERVATION)
    payload = json.loads(MACRO_RESERVATION.read_text(encoding="utf-8"))
    source = Path(payload["source_metadata_path"])
    if sha256_file(source) != payload["source_metadata_sha256"]:
        raise ValueError("macro_coach split metadata changed since the reservation export")
    ids = set(payload["reserved_test_ids"])
    return ids, {"path": str(MACRO_RESERVATION), "sha256": digest, "ids": len(ids)}


def winprob_confirmatory_ids() -> tuple[set[str], dict]:
    digest = check_sidecar(WINPROB_ADMISSION)
    manifest = json.loads(WINPROB_ADMISSION.read_text(encoding="utf-8"))
    ids = {g["match_id"] for g in manifest["games"] if g["role"] == "confirmatory_reserved"}
    return ids, {"path": str(WINPROB_ADMISSION), "sha256": digest, "ids": len(ids)}


def admit(out: Path) -> dict:
    bt_ids, bt_sources = build_tracker_test_ids()
    macro_ids, macro_meta = macro_reserved_ids()
    wp_ids, wp_meta = winprob_confirmatory_ids()
    denied = bt_ids | macro_ids | wp_ids
    names = sorted(e.name[:-len(".json.gz")] for e in os.scandir(RAW / "match") if e.name.endswith(".json.gz"))
    timelines = {e.name[:-len(".json.gz")] for e in os.scandir(RAW / "timeline") if e.name.endswith(".json.gz")}
    bad = [n for n in names if not MATCH_ID.fullmatch(n)]
    if bad:
        raise ValueError(f"Unexpected raw file names: {bad[:5]}")
    admitted = [n for n in names if n not in denied]
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_payloads_read": False,
        "raw_root": str(RAW),
        "raw_match_files": len(names),
        "denied": {"build_tracker_test": len(bt_ids), "macro_coach_test": len(macro_ids),
                   "winprob_confirmatory": len(wp_ids), "union": len(denied),
                   "union_ids_sha256": id_digest(denied),
                   "denied_present_in_archive": len(set(names) & denied)},
        "sources": {"build_tracker_splits": bt_sources, "macro_coach": macro_meta, "winprob": wp_meta},
        "admitted_games": len(admitted),
        "admitted_with_timeline": sum(1 for n in admitted if n in timelines),
        "admitted_ids_sha256": id_digest(admitted),
        "admitted": admitted,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    out.write_bytes(raw)
    out.with_suffix(".sha256").write_text(f"{hashlib.sha256(raw).hexdigest()} {out.name}\n", encoding="ascii")
    return manifest
