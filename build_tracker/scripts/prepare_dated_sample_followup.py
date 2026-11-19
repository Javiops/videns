"""Plan remaining dated-sample acceptance from metadata and uncovered local sources.

No network, reconstruction, training or final-test payload access. Prior corpus
receipts remain immutable; current code differences are reported, never waived.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import uuid

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = "C:/Users/Javier/build_tracker"
DEFAULT_OUT = ROOT / "data/acceptance/16.18_dated_samples_20260920"
DEFAULT_BASE = ROOT / "data/acceptance/16.18_reconstructible_20260914"
DEFAULT_CORPUS = ROOT / "data/corpora/16.18-reconstructible-v3"
META_FIELDS = ("match_id", "patch", "queue_id", "game_creation", "game_duration")
CLASSES = ("reserved", "qualified", "source_excluded", "uncovered")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed while hashing: " + str(path))
    return digest.hexdigest()


def require_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Missing or invalid SHA256")
    return value


def safe_id(mid):
    if not isinstance(mid, str) or not re.fullmatch(r"(?:KR|EUW1)_[0-9]+", mid):
        raise ValueError("Unscoped match ID: " + str(mid))
    return mid


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relocated_path(value, root):
    """Translate only the documented old project prefix; preserve frozen bytes."""
    root = Path(root).resolve()
    text = str(value).replace("\\", "/")
    if text.casefold().startswith(LEGACY_ROOT.casefold() + "/"):
        path = root / text[len(LEGACY_ROOT) + 1:]
    else:
        path = Path(value)
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("Evidence path escapes the project: " + str(value))
    return path


def check(path, expected, bindings):
    expected = require_digest(expected)
    actual = sha256(path)
    if actual != expected:
        raise ValueError("Evidence hash mismatch: " + str(path))
    bindings[str(Path(path).resolve())] = actual
    return actual


def split_paths(root):
    return sorted({p.resolve() for folder in ("data/ml", "data/dataset_generations")
                   for p in (root / folder).rglob("split_assignment*.json") if p.is_file()})


def reserve_splits(root):
    """Read assignments only, including archived/versioned assignment files."""
    bindings, memberships, reserved = {}, {}, set()
    paths = split_paths(root)
    if not paths:
        raise ValueError("No existing split metadata; cannot establish test reservations")
    for path in paths:
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Split metadata escapes the project")
        digest = sha256(path)
        mapping = load_json(path)
        if not isinstance(mapping, dict):
            raise ValueError("Invalid split assignment mapping")
        for mid, fold in mapping.items():
            safe_id(mid)
            if not isinstance(fold, str) or not fold:
                raise ValueError("Invalid prior fold")
            memberships.setdefault(mid, {})[str(path)] = fold
            if fold == "test":
                reserved.add(mid)
        check(path, digest, bindings)
    return reserved, memberships, bindings


def classify_ids(stored, reserved, selected, accepted, excluded):
    stored, selected, accepted, excluded = map(set, (stored, selected, accepted, excluded))
    if accepted & excluded or accepted | excluded != selected:
        raise ValueError("Prior acceptance does not partition the source plan")
    result = {key: [] for key in CLASSES}
    for mid in sorted(stored):
        key = ("reserved" if mid in reserved else "qualified" if mid in accepted
               else "source_excluded" if mid in excluded else "uncovered")
        result[key].append(mid)
    assert_classification(stored, result)
    return result


def assert_classification(stored, result):
    if set(result) != set(CLASSES):
        raise ValueError("Missing classification category")
    combined = [mid for group in result.values() for mid in group]
    if len(combined) != len(set(combined)) or set(combined) != set(stored):
        raise ValueError("Classification is not complete and disjoint")


def merge_claim(claims, mid, metadata, payload_hashes, sample_name):
    if metadata["match_id"] != mid:
        raise ValueError("Metadata identity mismatch")
    hashes = {kind: require_digest(payload_hashes[kind]) for kind in ("match", "timeline")}
    candidate = {"metadata": metadata, "payload_sha256": hashes}
    if mid in claims:
        prior = claims[mid]
        if any(prior[key] != value for key, value in candidate.items()):
            raise ValueError("Conflicting sample source claims: " + mid)
        prior["samples"].append(sample_name)
    else:
        claims[mid] = {**candidate, "samples": [sample_name]}


def load_sample(root, folder, claims, bindings):
    folder = folder.resolve()
    if not folder.is_relative_to((root / "data/patch_samples").resolve()):
        raise ValueError("Unexpected sample location")
    state_path, report_path = folder / "state.json", folder / "report.json"
    state_digest, report_digest = sha256(state_path), sha256(report_path)
    state, report = load_json(state_path), load_json(report_path)
    if (state.get("patch") != "16.18" or report.get("patch") != "16.18"
            or state.get("discovery_complete") is not True
            or report.get("structural_checks_passed") is not True
            or report.get("checkpoint_sha256") != state_digest):
        raise ValueError("Incomplete or unbound collection: " + folder.name)
    jobs, results = state["jobs"], state["results"]
    if not isinstance(jobs, dict) or not isinstance(results, dict) or jobs.keys() != results.keys():
        raise ValueError("Collection result membership differs from jobs")
    counts = dict(Counter(result["status"] for result in results.values()))
    if counts != report["counts"]:
        raise ValueError("Collection result counts changed")
    stored = {safe_id(mid) for mid, result in results.items() if result["status"] == "stored"}
    cutoff = datetime.fromisoformat(report["cutoff_utc"])
    if (cutoff.tzinfo is None or cutoff.timestamp() != state["end_time"]
            or state["start_time"] >= state["end_time"] or report["games"] != len(stored)
            or report["perspectives"] != 10 * len(stored)):
        raise ValueError("Collection cutoff or population mismatch")
    parent_path = relocated_path(state["cohort_parent"], root)
    check(parent_path, state["cohort_parent_sha256"], bindings)
    parent = load_json(parent_path)
    cohort_keys = ("tracked_pros", "ladders", "pros", "roster_observed_utc")
    if any(state.get(key, {}) != parent.get(key, {}) for key in cohort_keys):
        raise ValueError("Inherited cohort differs from its frozen parent")
    unresolved = sorted(key for key, value in state["pros"].items() if not value.get("puuid"))
    if unresolved != sorted(report["unresolved_pros"]):
        raise ValueError("Unresolved pro outcomes disagree")
    database = folder / "tracker.db"
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        rows = [dict(row) for row in conn.execute(
            "SELECT match_id,patch,queue_id,game_creation,game_duration FROM games ORDER BY match_id")]
    if {r["match_id"] for r in rows} != stored or len(rows) != len(stored):
        raise ValueError("Stored sample DB metadata membership differs from checkpoint")
    for row in rows:
        mid = row["match_id"]
        if (row["patch"] != "16.18" or row["queue_id"] != 420
                or type(row["game_creation"]) is not int or type(row["game_duration"]) is not int
                or row["game_duration"] <= 0 or row["game_creation"] > state["end_time"] * 1000):
            raise ValueError("Invalid dated sample game metadata: " + mid)
        result = results[mid]
        if result.get("perspectives") != 10:
            raise ValueError("Stored result lacks ten perspectives")
        merge_claim(claims, mid, row, {kind: result[kind + "_sha256"]
                                     for kind in ("match", "timeline")}, folder.name)
    check(state_path, state_digest, bindings)
    check(report_path, report_digest, bindings)
    return {"sample": str(folder), "state_sha256": state_digest, "report_sha256": report_digest,
            "database_metadata_only": str(database), "database_metadata_sha256": canonical_sha(rows),
            "stored_games": len(stored), "stored_match_ids": sorted(stored),
            "collection_cutoff_utc": cutoff.astimezone(timezone.utc).isoformat(),
            "original_cutoff_string": report["cutoff_utc"], "start_time": state["start_time"],
            "end_time": state["end_time"], "scope_complete": report["scope_complete"],
            "unresolved_pros": unresolved, "cohort_parent": str(parent_path),
            "cohort_parent_sha256": state["cohort_parent_sha256"],
            "cohort_policy": state["cohort_policy"],
            "cohort_snapshot": {key: state.get(key, {}) for key in cohort_keys}}


def load_prior(root, base, corpus, bindings):
    paths = {"plan": base / "source_plan.json", "qualification": base / "qualification.json",
             "audit": base / "semantic_audit.json", "completion": base / "run/completion.json",
             "status": base / "run/status.json", "protocol": base / "run/protocol.json",
             "manifest": corpus / "corpus_manifest.json", "screening": corpus / "screening_report.json"}
    data = {name: load_json(path) for name, path in paths.items()}
    plan, q, audit, done, manifest = (data[k] for k in
                                    ("plan", "qualification", "audit", "completion", "manifest"))
    if (done != data["status"] or done.get("state") != "complete"
            or done.get("stage") != "complete" or done.get("completed_stages") != ["build", "audit"]
            or done.get("protected_unchanged") is not True or done.get("test_payload_reads") is not False
            or done.get("training_or_export") is not False):
        raise ValueError("Prior build/audit did not complete under the required scope")
    if (plan.get("patch") != "16.18" or plan.get("purpose") != "isolated_reconstruction_qualification"
            or plan.get("test_payload_reads") is not False
            or plan.get("training_or_export_authorized") is not False):
        raise ValueError("Wrong prior source scope")
    if (q.get("status") != "qualified_reconstruction_snapshot" or q.get("patch") != "16.18"
            or q.get("training_clearance") is not False or q.get("export_created") is not False
            or q.get("promotion_allowed") is not False or q.get("test_payload_reads") is not False):
        raise ValueError("Wrong prior qualification scope")
    for actual, expected in ((done["qualification"], paths["qualification"]),
                             (q["corpus"], corpus), (q["semantic_audit"], paths["audit"])):
        if relocated_path(actual, root) != expected.resolve():
            raise ValueError("Prior receipt references different evidence")
    check(paths["qualification"], done["qualification_sha256"], bindings)
    check(paths["manifest"], q["corpus_manifest_sha256"], bindings)
    check(paths["audit"], q["semantic_audit_sha256"], bindings)
    plan_sha = check(paths["plan"], data["protocol"]["source_plan_sha256"], bindings)
    if manifest["source_plan_sha256"] != plan_sha:
        raise ValueError("Corpus and supervisor source plans differ")
    check(corpus / "source_plan.json", plan_sha, bindings)
    check(paths["screening"], manifest["files"]["screening_report.json"]["sha256"], bindings)
    check(corpus / "source_inventory.jsonl", manifest["source_inventory_sha256"], bindings)
    check(base / "audit_code.py", audit["audit_code_sha256"], bindings)
    if (audit.get("passed") is not True or audit.get("hard_failures")
            or audit.get("quarantined_match_ids")
            or audit["corpus_manifest_sha256"] != q["corpus_manifest_sha256"]
            or manifest["db_sha256"] != q["db_sha256"] or audit["db_sha256"] != q["db_sha256"]):
        raise ValueError("Prior audit/database metadata binding did not pass")
    selected = {safe_id(record["match_id"]): record for record in plan["matches"]}
    accepted = set(q["accepted_match_ids"])
    excluded = data["screening"]["excluded_match_ids"]
    if (len(selected) != len(plan["matches"]) or len(selected) != plan["selected_sources"]
            or set(plan["reserved_test_ids"]) & selected.keys()
            or accepted != set(audit["accepted_match_ids"])
            or accepted != set(data["screening"]["accepted_match_ids"])
            or len(accepted) != q["accepted_games"] or len(accepted) != done["accepted_games"]
            or len(accepted) != manifest["games"]):
        raise ValueError("Prior evidence populations differ")
    classify_ids(selected, set(), selected, accepted, excluded)
    if any(not isinstance(reasons, list) or not reasons for reasons in excluded.values()):
        raise ValueError("Prior exclusion has no explicit reasons")
    inventory = {}
    with (corpus / "source_inventory.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            mid = entry["match_id"]
            if mid in inventory:
                raise ValueError("Duplicate accepted source inventory receipt")
            inventory[mid] = entry
    if set(inventory) != accepted:
        raise ValueError("Accepted source inventory membership differs")
    for name, digest in plan["split_metadata_sha256"].items():
        check(relocated_path(name, root), digest, bindings)
    mismatches = {}
    for name, digest in manifest["reconstruction_code_sha256"].items():
        if data["protocol"]["code_sha256"].get(name) != digest:
            raise ValueError("Supervisor and reconstruction code bindings differ")
        check(corpus / "reconstruction_code" / name, digest, bindings)
        check(base / "run/code" / name, digest, bindings)
        current = root / name
        actual = sha256(current) if current.is_file() else None
        if actual != digest:
            mismatches[name] = {"frozen_sha256": digest, "current_sha256": actual}
    review_path = relocated_path(plan["static_review"], root)
    check(review_path, plan["static_review_sha256"], bindings)
    review = load_json(review_path)
    if (review.get("patch") != "16.18" or review.get("ddragon_version") != "16.18.1"
            or review.get("timer_economics_and_physical_rules_compatible") is not True):
        raise ValueError("Prior static mechanics review did not pass")
    for name, digest in review["files"].items():
        path = (review_path.parent / name).resolve()
        if path.parent != review_path.parent.resolve():
            raise ValueError("Static input escapes its folder")
        check(path, digest, bindings)
    for kind in ("item", "champion"):
        name = kind + "-16.18.1.json"
        check(root / "data/cache" / name, manifest["files"]["static/" + name]["sha256"], bindings)
    for path in paths.values():
        bindings[str(path.resolve())] = sha256(path)
    return {"paths": paths, "data": data, "selected": selected, "accepted": accepted,
            "excluded": excluded, "inventory": inventory, "review_path": review_path,
            "review": review, "current_code_mismatches": mismatches}


def covered_receipt(root, mid, claim, category, prior, bindings):
    record = prior["selected"][mid]
    if {key: record[key] for key in META_FIELDS} != claim["metadata"]:
        raise ValueError("Covered source metadata differs from dated sample: " + mid)
    receipt = {"metadata": claim["metadata"], "samples": claim["samples"],
               "sample_payload_sha256": claim["payload_sha256"], "classification": category,
               "prior_files": record["files"], "payloads_opened": False}
    if category == "qualified":
        files = prior["inventory"][mid]["files"]
        for kind in ("match", "timeline"):
            if (files[kind]["payload_sha256"] != claim["payload_sha256"][kind]
                    or files[kind]["sha256"] != record["files"][kind]["sha256"]):
                raise ValueError("Accepted source receipt conflicts with sample: " + mid)
        receipt.update(source_identity="canonical_receipts_match", source_identity_verified=True)
    else:
        for kind in ("match", "timeline"):
            path = root / "data/raw_match_v5" / kind / (mid + ".json.gz")
            binding = record["files"][kind]
            if relocated_path(binding["path"], root) != path.resolve() or path.stat().st_size != binding["bytes"]:
                raise ValueError("Excluded source path or size changed")
            check(path, binding["sha256"], bindings)
        receipt.update(exclusion_reasons=prior["excluded"][mid],
                       source_identity="prior_compressed_sources_match_sample_canonical_link_unverified",
                       source_identity_verified=False,
                       limitation="Prior exclusion is preserved; its canonical identity against the sample "
                                  "is pending because covered payloads were not decompressed.")
    return receipt


def read_uncovered_pair(root, mid, claim, memberships, reserved):
    if mid in reserved or "test" in memberships.get(mid, {}).values():
        raise ValueError("Reserved ID reached raw payload access")
    payloads, files = {}, {}
    for kind in ("match", "timeline"):
        path = root / "data/raw_match_v5" / kind / (safe_id(mid) + ".json.gz")
        digest = sha256(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            envelope = json.load(handle)
        payload = envelope.get("payload")
        if (envelope.get("schema") != 1 or envelope.get("endpoint") != kind
                or envelope.get("match_id") != mid or not isinstance(payload, dict)
                or payload.get("metadata", {}).get("matchId") != mid
                or canonical_sha(payload) != claim["payload_sha256"][kind]):
            raise ValueError("Uncovered archive disagrees with sample source claim: " + mid + "/" + kind)
        if sha256(path) != digest:
            raise ValueError("Uncovered archive changed during validation")
        files[kind] = {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}
        payloads[kind] = payload
    info, timeline = payloads["match"].get("info", {}), payloads["timeline"].get("info", {})
    metadata = claim["metadata"]
    if (".".join(str(info.get("gameVersion", "")).split(".")[:2]) != metadata["patch"]
            or info.get("queueId") != metadata["queue_id"] or info.get("gameCreation") != metadata["game_creation"]
            or info.get("gameDuration") != metadata["game_duration"]):
        raise ValueError("Uncovered source differs from frozen DB metadata: " + mid)
    players = info.get("participants", [])
    identities = {p.get("participantId"): p.get("puuid") for p in players}
    if (len(players) != 10 or set(identities) != set(range(1, 11))
            or len(set(identities.values())) != 10 or not all(identities.values())
            or {p.get("participantId"): p.get("puuid") for p in timeline.get("participants", [])} != identities):
        raise ValueError("Uncovered source participants disagree")
    return {**metadata, "prior_assignments": memberships.get(mid, {}), "files": files,
            "sample_payload_sha256": claim["payload_sha256"], "sample_membership": claim["samples"]}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare(root, output, samples, base, corpus):
    root, output, base, corpus = [Path(p).resolve() for p in (root, output, base, corpus)]
    if not output.is_relative_to(root / "data/acceptance") or output == root / "data/acceptance":
        raise ValueError("Follow-up output must be a dedicated acceptance folder")
    if output.exists():
        raise ValueError("Immutable follow-up already exists; refusing overwrite")
    # Reservation metadata is collected before any raw archive bytes are read.
    reserved, memberships, split_bindings = reserve_splits(root)
    bindings, claims = dict(split_bindings), {}
    samples_meta = [load_sample(root, Path(folder), claims, bindings) for folder in samples]
    if len({sample["sample"] for sample in samples_meta}) != len(samples_meta):
        raise ValueError("Duplicate dated sample")
    prior = load_prior(root, base, corpus, bindings)
    reserved.update(prior["data"]["plan"]["reserved_test_ids"])
    uncovered, receipts = set(), {}
    for sample in samples_meta:
        classification = classify_ids(sample["stored_match_ids"], reserved, prior["selected"],
                                      prior["accepted"], prior["excluded"])
        sample["classification"] = classification
        sample["classification_counts"] = {key: len(ids) for key, ids in classification.items()}
        uncovered.update(classification["uncovered"])
        for category in ("qualified", "source_excluded"):
            for mid in classification[category]:
                if mid not in receipts:
                    receipts[mid] = covered_receipt(root, mid, claims[mid], category, prior, bindings)
    records = [read_uncovered_pair(root, mid, claims[mid], memberships, reserved)
               for mid in sorted(uncovered)]
    # Recheck metadata and directory membership before publishing a frozen plan.
    if split_paths(root) != sorted(Path(path) for path in split_bindings):
        raise ValueError("Split metadata set changed during preparation")
    for path, digest in bindings.items():
        check(path, digest, {})
    now = datetime.now(timezone.utc).isoformat()
    planner_sha = sha256(Path(__file__))
    static_target = output / "uncovered/inputs/static_review.json"
    plan = {"schema": 1, "purpose": "isolated_reconstruction_qualification", "patch": "16.18",
            "queue_id": 420, "snapshot_utc": now, "database_metadata_only": [s["database_metadata_only"] for s in samples_meta],
            "database_games_at_snapshot": len(claims), "selected_sources": len(records), "matches": records,
            "source_selection_exclusions": {mid: "reserved_test_metadata_only" if mid in reserved
                else "covered_by_prior_qualified_receipt" if mid in prior["accepted"] else "covered_by_prior_source_exclusion"
                for mid in sorted(claims.keys() - uncovered)},
            "exclusion_counts": dict(Counter("reserved" if mid in reserved else "covered"
                                           for mid in claims.keys() - uncovered)),
            "reserved_test_ids": sorted(reserved), "split_metadata_sha256": split_bindings,
            "static_review": str(static_target), "static_review_sha256": sha256(prior["review_path"]),
            "test_payload_reads": False, "training_or_export_authorized": False,
            "future_split_policy": "Preserve all prior assignments; acceptance does not authorize a new split or training.",
            "processing_scope": "Uncovered non-test union only; retain prior evidence for covered IDs.",
            "collection_continues": False, "original_collection_cutoffs": {
                Path(s["sample"]).name: s["collection_cutoff_utc"] for s in samples_meta},
            "planner_code_sha256": planner_sha, "input_evidence_sha256": dict(bindings)}
    preflight = {"schema": 1, "purpose": "dated_sample_acceptance_followup", "status": "prepared_not_accepted",
                 "created_utc": now, "source_root": str(root), "samples": samples_meta,
                 "uncovered_non_test_ids": sorted(uncovered), "selected_sources": len(records),
                 "covered_source_receipts": receipts, "prior_evidence": {k: str(v) for k, v in prior["paths"].items()},
                 "input_evidence_sha256": bindings, "planner_code_sha256": planner_sha,
                 "current_vs_prior_code_mismatches": prior["current_code_mismatches"],
                 "covered_evidence_scope": "Metadata/hash chain and frozen code verified; prior multi-GB corpus DB "
                                           "and decision masks were not rehashed or replayed in this preflight.",
                 "test_payload_reads": False, "training_or_export_authorized": False, "promotion_allowed": False,
                 "limits": ["Raw bytes were not opened for reserved IDs or qualified covered IDs.",
                            "Covered exclusions bind compressed prior sources; sample canonical identity remains unverified.",
                            "Uncovered raw identity checks do not establish reconstruction or training acceptance."]}
    pending = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".partial")
    pending.mkdir(parents=True)
    inputs = pending / "uncovered/inputs"
    inputs.mkdir(parents=True)
    for name in ("static_review.json", *prior["review"]["files"]):
        source = prior["review_path"].parent / name
        target = inputs / name
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise ValueError("Static input copy changed")
    write_json(pending / "uncovered/source_plan.json", plan)
    preflight["uncovered_source_plan_sha256"] = sha256(pending / "uncovered/source_plan.json")
    write_json(pending / "preflight.json", preflight)
    if sha256(Path(__file__)) != planner_sha or output.exists():
        raise ValueError("Planner code changed or destination appeared before publication")
    pending.rename(output)
    return preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--sample", type=Path, action="append")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    samples = args.sample or [ROOT / ("data/patch_samples/16.18_" + day) for day in ("20260912", "20260913")]
    result = prepare(ROOT, args.out, samples, args.base, args.corpus)
    print(json.dumps({"output": str(args.out.resolve()), "status": result["status"],
                      "uncovered_non_test_games": result["selected_sources"],
                      "classification_counts": {Path(s["sample"]).name: s["classification_counts"] for s in result["samples"]},
                      "current_code_mismatches": list(result["current_vs_prior_code_mismatches"])}, indent=2))


if __name__ == "__main__":
    main()

