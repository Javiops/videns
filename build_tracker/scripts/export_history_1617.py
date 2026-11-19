"""Export the qualified 16.17 history corpus's admitted rows for pre-training (prior-patch transfer).

Owner's conditions (2026-09-27): frozen snapshot and membership (holdouts and
reserved tests excluded before any payload), reconstruction under 16.17 rules
with every gate, EXACT export of the admitted rows (counts, versions and
exclusion reasons preserved; no extra rows, duplicates or silent drops), and a
manifest + validator used at export and at training. app/, the v4 generation
and M2 are untouched. Row construction is the shared `corpus_rows.iter_games`
(the same row builder every qualified export uses) under the isolated 16.17
runtime; nothing here re-implements its rules.

Order of operations, deliberately: (1) verify the complete build/audit
evidence (`verify_reconstructible_1617_evidence`) — no payload before that;
(2) activate the 16.17 runtime from this clean process; (3) build the rows;
(4) write the manifest; (5) run `history_1617.validate_history` on the result
and delete it if the validator refuses it.

    .venv\\Scripts\\python.exe scripts\\export_history_1617.py [--base ...] [--corpus ...] [--out ...]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from history_1617 import HISTORY_DIR, PURPOSE, ROW_BUILDER_CODE, ROW_VERSION_FIELDS, sha256_file, validate_history
from history_rows_check import validate_rows

BASE = ROOT / "data/acceptance/16.17_history_20260927"
CORPUS = ROOT / "data/corpora/16.17-history-v1"
# The row builder is export-only code (never part of the build/audit protocol), so its identity is DECLARED here:
# the reviewed digests the smoke chain ran with (2026-09-27). A changed builder refuses to export until this
# declaration is updated deliberately (second review, finding: corpus_rows.py outside the protocol gate).
REVIEWED_ROW_BUILDER = {"scripts/corpus_rows.py": "dd4f4f7013b2fd068bb209e7e9fd7eb8206fd5a94acf5e3d1e3eef03d902fecc",
                        "scripts/build_recovered_patch_corpus.py": "0d91247d371f65098b9f4e017722f49422b12d33c4819a296b755562d9bea135",
                        "scripts/train_prefix.py": "e12e5cee905ad026c78e7fc3d1eca6b4ead483ca5c0a58bea31281157776abe2"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=HISTORY_DIR)
    ap.add_argument("--allow-limited", action="store_true", help="accept a snapshot taken with --limit (smoke chains only)")
    args = ap.parse_args()
    started = time.monotonic()
    base, corpus, out = args.base.resolve(), args.corpus.resolve(), args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; a history export is immutable -- use a new directory")

    # ---- 1. complete evidence first, no payload before ----
    from verify_reconstructible_1617_evidence import verify_prerequisite
    verified = verify_prerequisite(ROOT, base=base, corpus=corpus)
    plan, q, audit, manifest, mask = (verified[k] for k in ("source_plan", "original_qualification", "audit", "manifest", "mask"))
    if plan.get("selection_limit") and not args.allow_limited:
        raise SystemExit("the snapshot was limited (smoke); refuse to export it as history without --allow-limited")
    bindings = verified["bindings"]                      # the digests the verifier certified, reused verbatim
    protocol = verified["protocol"]
    print(f"evidence verified: {q['accepted_games']} qualified games, {mask['admitted_rows']} admitted rows "
          f"({time.monotonic() - started:.0f}s)", flush=True)
    # The rows are built by THIS checkout's code: it must be the code that built and audited the corpus.
    drifted = [name for name, digest in protocol["code_sha256"].items() if sha256_file(ROOT / name) != digest]
    if drifted:
        raise SystemExit(f"live code differs from the audited build code; no export: {drifted}")
    drifted = [name for name, digest in REVIEWED_ROW_BUILDER.items() if sha256_file(ROOT / name) != digest]
    if drifted:
        raise SystemExit(f"row-builder code differs from the reviewed declaration; no export: {drifted}")

    # ---- 2. the isolated 16.17 runtime, from this clean process ----
    import reconstructible_1617_runtime as runtime
    declaration = runtime.activate(Path(plan["static_review"]))
    if declaration != manifest["footwear_runtime"]:
        raise SystemExit("active runtime differs from the corpus's declaration")
    from app.ddragon import dragon_for_patch
    dragon = dragon_for_patch("16.17")
    if dragon.version != "16.17.1" or dragon.signature != manifest["static_data_sha256"]:
        raise SystemExit("pinned static data differs from the corpus")
    import corpus_rows
    from corpus_rows import QualifiedCorpus, iter_games, row_builder_digest
    context = QualifiedCorpus(path=corpus, qualification_path=base / "qualification.json",
                              qualification={"accepted_match_ids": sorted(q["accepted_match_ids"])},
                              manifest=manifest, dragon=dragon, decision_admission=mask)
    builder = row_builder_digest()

    # ---- 3. rows, through the shared builder, exactly the admitted ones ----
    pending = out.with_name("." + out.name + ".partial")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    rows_path = pending / "visits_fit.jsonl"
    hasher = hashlib.sha256()
    n_rows, per_game, exported, export_exclusions = 0, {}, [], {}
    versions = {f: Counter() for f in ROW_VERSION_FIELDS}
    t0 = time.monotonic()
    with open(rows_path, "xb") as fh:
        for i, (game, rows, excluded) in enumerate(iter_games(context), 1):
            mid = game["match_id"]
            if excluded:
                export_exclusions[mid] = excluded
                continue
            if len(rows) != mask["games"][mid]["admitted_rows"]:
                raise SystemExit(f"{mid}: {len(rows)} rows built, mask admits {mask['games'][mid]['admitted_rows']}")
            for row in rows:
                payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                fh.write(payload)
                hasher.update(payload)
                for f in ROW_VERSION_FIELDS:
                    versions[f][row.get(f)] += 1
            n_rows += len(rows)
            per_game[mid] = len(rows)
            exported.append(mid)
            if i % 500 == 0:
                print(f"  {i}/{q['accepted_games']} games, {n_rows} rows ({time.monotonic() - t0:.0f}s)", flush=True)
        fh.flush()
        os.fsync(fh.fileno())
    expected_rows = sum(mask["games"][mid]["admitted_rows"] for mid in exported)
    if n_rows != expected_rows:
        raise SystemExit(f"exported {n_rows} rows, the decision mask admits {expected_rows} for the exported games")
    for f in ROW_VERSION_FIELDS:
        if len(versions[f]) != 1:
            raise SystemExit(f"mixed {f} in the export: {dict(versions[f])}")
    exported_sorted = sorted(exported)
    print(f"rows written: {n_rows} from {len(exported)} games; export exclusions {len(export_exclusions)} "
          f"({time.monotonic() - t0:.0f}s)", flush=True)

    # ---- 4. manifest ----
    review_path = Path(plan["static_review"]).resolve()
    hist_manifest = {
        "schema": 1, "patch": "16.17", "purpose": PURPOSE, "not_a_generation": True, "promotion_allowed": False,
        "test_payload_reads": False, "state": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/experiment_prior_patch_transfer_v1.md",
        "corpus": {"path": str(corpus), "manifest_sha256": bindings[str(corpus / "corpus_manifest.json")],
                   "db_sha256": manifest["db_sha256"], "decision_admission_sha256": bindings[str(corpus / "decision_admission.json")],
                   "games": manifest["games"], "reconstruction_version": manifest["reconstruction_version"],
                   "inventory_version": manifest["inventory_version"], "static_data_sha256": manifest["static_data_sha256"]},
        "acceptance": {"path": str(base), "source_plan_sha256": bindings[str(base / "source_plan.json")],
                       "qualification_sha256": bindings[str(base / "qualification.json")],
                       "semantic_audit_sha256": bindings[str(base / "semantic_audit.json")],
                       "static_review_sha256": bindings[str(review_path)],
                       "protocol_sha256": bindings[str(base / "run" / "protocol.json")],
                       "completion_sha256": bindings[str(base / "run" / "completion.json")],
                       "verified_utc": verified["verified_utc"],
                       "snapshot_utc": plan["snapshot_utc"], "database_games_at_snapshot": plan["database_games_at_snapshot"],
                       "selected_sources": plan["selected_sources"], "snapshot_exclusion_counts": plan["exclusion_counts"],
                       "reserved_test_games": len(plan["reserved_test_ids"]),
                       "historical_validation_holdout_games": len(plan.get("historical_validation_holdout_ids", [])),
                       "split_files_scanned_at_snapshot": len(plan["split_metadata_sha256"]),
                       "selection_limit": plan.get("selection_limit"),
                       "reconstruction_exclusion_reason_counts": q.get("source_exclusions"),
                       "accepted_cohorts": q.get("accepted_cohorts")},
        "footwear_runtime": declaration,
        "games": {"qualified": q["accepted_games"], "exported": exported_sorted, "exported_count": len(exported_sorted),
                  "exported_sha256": hashlib.sha256(("\n".join(exported_sorted) + "\n").encode("utf-8")).hexdigest(),
                  "export_exclusions": export_exclusions,
                  "export_exclusion_counts": dict(Counter(export_exclusions.values()))},
        "rows": {"file": "visits_fit.jsonl", "count": n_rows, "bytes": rows_path.stat().st_size, "sha256": hasher.hexdigest(),
                 "admitted_rows_in_mask": mask["admitted_rows"], "boundary_excluded_rows_in_mask": mask["boundary_excluded_rows"],
                 "rows_of_export_excluded_games": mask["admitted_rows"] - expected_rows},
        "versions": {f: next(iter(versions[f])) for f in ROW_VERSION_FIELDS},
        "row_builder_sha256": builder,
        "row_builder_code": {rel: sha256_file(ROOT / rel) for rel in ROW_BUILDER_CODE},
        "build_code_equals_export_code": True,
        "reviewed_row_builder": REVIEWED_ROW_BUILDER,
        "exporter_sha256": sha256_file(Path(__file__)), "validator_sha256": sha256_file(ROOT / "scripts/history_1617.py"),
        "limits": ["pre-training data only: never scored, never served, never promoted",
                   "the row builder is the shared corpus_rows.iter_games under the isolated 16.17 runtime",
                   "the exporting checkout's code equals the completed run's protocol code (checked before any row)",
                   "every export exclusion is recomputed from the frozen corpus by the validator",
                   "reserved tests and historical validation holdouts were excluded at the snapshot, before any payload"],
    }
    (pending / "history_manifest.json").write_text(json.dumps(hist_manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "scripts/history_1617.py", pending / "validator_code.py")
    shutil.copyfile(Path(__file__), pending / "exporter_code.py")
    for path in pending.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    pending.rename(out)

    # ---- 5. the validator must accept what was just written ----
    try:
        checked = validate_rows(out, validate_history(out, scan_rows=True, hash_db=True))
        print(f"validator accepted the export; {checked} sampled rows re-checked against the frozen corpus", flush=True)
    except Exception as exc:                            # noqa: BLE001
        for path in out.rglob("*"):
            if path.is_file():
                path.chmod(0o644)
        shutil.rmtree(out)
        raise SystemExit(f"validator refused the fresh export; deleted it: {exc}")
    print(json.dumps({"history": str(out), "games": len(exported_sorted), "rows": n_rows,
                      "export_exclusions": hist_manifest["games"]["export_exclusion_counts"],
                      "versions": hist_manifest["versions"], "elapsed_seconds": round(time.monotonic() - started, 1)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
