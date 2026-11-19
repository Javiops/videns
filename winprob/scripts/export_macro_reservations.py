"""Export sealed IDs from macro split METADATA; never inspect corpus payloads.

Only the exact sibling data/ml/split.json is opened. No sibling Python module,
database, corpus directory, decision file, match, or timeline is imported/read.
The resulting versioned artifact is the cross-project reservation interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPLIT = ROOT.parent / "macro_coach" / "data" / "ml" / "split.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "reservations" / "macro-20260929-v1"
SCHEMA = "winprob-external-reservations-v1"
MATCH_ID = re.compile(r"^[A-Z0-9]+_[0-9]+$")


def validate_metadata(metadata: dict) -> dict[str, list[str]]:
    if not isinstance(metadata, dict) or metadata.get("provisional"):
        raise ValueError("Require a non-provisional split metadata object")
    if not isinstance(metadata.get("lexicon_v"), str) or not metadata["lexicon_v"]:
        raise ValueError("Missing source lexicon version")
    if not isinstance(metadata.get("features_v"), str) or not metadata["features_v"]:
        raise ValueError("Missing source feature version")
    folds = {}
    seen = set()
    for role in ("train", "val", "test"):
        ids = metadata.get(role)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"Require a nonempty {role} ID list")
        if any(not isinstance(mid, str) or not MATCH_ID.fullmatch(mid) for mid in ids):
            raise ValueError(f"Invalid match identity in {role}")
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError("Duplicate ID within or across split roles")
        seen.update(ids)
        folds[role] = sorted(ids)
    if type(metadata.get("n_games")) is not int or metadata["n_games"] != len(seen):
        raise ValueError("Source n_games differs from unique split IDs")
    return folds


def export_reservations(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    source_bytes = SOURCE_SPLIT.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    metadata = json.loads(source_bytes.decode("utf-8-sig"))
    folds = validate_metadata(metadata)
    artifact = {
        "schema": SCHEMA,
        "version": output_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_project": "macro_coach",
        "source_metadata_path": str(SOURCE_SPLIT.resolve()),
        "source_metadata_sha256": source_hash,
        "source_metadata_type": "frozen_split_ids_only",
        "source_lexicon_v": metadata["lexicon_v"],
        "source_features_v": metadata["features_v"],
        "source_n_games": metadata["n_games"],
        "fold_counts": {role: len(ids) for role, ids in folds.items()},
        "reserved_test_ids": folds["test"],
        "reserved_test_count": len(folds["test"]),
        "scope": "Deny before opening any raw payload in winprob; all experiment roles and review exports.",
        "raw_payloads_read": False,
        "source_split_modified": False,
    }
    if hashlib.sha256(SOURCE_SPLIT.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Source split metadata changed during export")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "reserved_ids.json"
    encoded = (json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    digest = hashlib.sha256(encoded).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = export_reservations(args.output_dir)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"artifact": str(path.resolve()), "reserved_test_count": artifact["reserved_test_count"],
                      "fold_counts": artifact["fold_counts"], "raw_payloads_read": False}, indent=2))


if __name__ == "__main__":
    main()
