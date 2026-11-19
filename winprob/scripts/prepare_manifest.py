"""Prepare a versioned metadata-only admission manifest; never reads payloads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from winprob.admission import prepare_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--allow-source", type=Path, action="append", required=True,
                        help="Explicit split JSON; repeat. Relative paths resolve under source-data-root.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "admission")
    parser.add_argument("--metadata-db", "--db", dest="metadata_db", type=Path,
                        help="Optional indexed read-only creation/patch/queue lookup; omit to defer date extraction.")
    parser.add_argument("--with-times", action="store_true", help="Require --db and assign temporal roles now.")
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--development-fraction", type=float, default=0.15)
    parser.add_argument("--confirmatory-fraction", type=float, default=0.10)
    parser.add_argument("--external-reservations", type=Path, action="append", default=[],
                        help="Versioned external sealed-test reservation artifact; repeat for each project.")
    parser.add_argument("--prior-manifest", type=Path,
                        help="Preserve all surviving prior roles, including the full confirmation reserve.")
    args = parser.parse_args()
    if args.with_times and args.metadata_db is None:
        parser.error("--with-times requires --db")
    sources = [path if path.is_absolute() else args.source_data_root / path for path in args.allow_source]
    manifest = prepare_manifest(args.source_data_root, sources, args.version, db_path=args.metadata_db,
                                calibration_fraction=args.calibration_fraction,
                                development_fraction=args.development_fraction,
                                confirmatory_fraction=args.confirmatory_fraction,
                                external_reservations=args.external_reservations,
                                prior_manifest=args.prior_manifest)
    path, sha256 = write_manifest(manifest, args.output_root)
    print(json.dumps({"path": str(path), "sha256": sha256, "state": manifest["state"],
                      "split_files": len(manifest["split_metadata_sources"]),
                      "candidate_games": manifest["candidate_games"],
                      "admitted_games": manifest["admitted_games"],
                      "denied_test_games": manifest["denied_test_games"],
                      "external_reserved_test_games": manifest["external_reserved_test_games"],
                      "external_reservations_excluded_games": manifest["external_reservations_excluded_games"],
                      "excluded_games": manifest["excluded_games"],
                      "role_counts": manifest["role_counts"], "raw_payload_reads": False}, indent=2))


if __name__ == "__main__":
    main()
