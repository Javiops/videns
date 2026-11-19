"""Actual tracked-pro participation before/after dated acceptance; cached matches only."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import prepare_dated_sample_followup as evidence
BASE = ROOT / "data/acceptance/16.18_dated_samples_20260920"

def pro_names(payload, pros):
    players = payload.get("info", {}).get("participants", [])
    identities = [p.get("puuid") for p in players]
    if len(players) != 10 or len(set(identities)) != 10 or not all(identities):
        raise ValueError("invalid_participant_metadata")
    present = set(identities)
    return {name for name, person in pros.items() if person.get("puuid") in present}

def read_match(root, mid, expected, reserved):
    if mid in reserved:
        raise ValueError("reserved_payload_access_refused")
    path = root / "data/raw_match_v5/match" / (evidence.safe_id(mid) + ".json.gz")
    digest = evidence.sha256(path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        envelope = json.load(handle)
    payload = envelope.get("payload")
    if (envelope.get("schema") != 1 or envelope.get("endpoint") != "match"
            or envelope.get("match_id") != mid or not isinstance(payload, dict)
            or payload.get("metadata", {}).get("matchId") != mid
            or evidence.canonical_sha(payload) != expected):
        raise ValueError("match_identity_or_canonical_hash_mismatch")
    if evidence.sha256(path) != digest:
        raise ValueError("match_changed_during_read")
    return payload, {"path": str(path), "compressed_sha256": digest, "canonical_sha256": expected}

def tally(ids, accepted, quarantined, reserved, observations, pros):
    if (set(accepted) | set(quarantined) | set(reserved)) & set(ids) != set(ids):
        raise ValueError('Unclassified stored source')
    counts = {name: Counter() for name in ("non_test_source", "accepted", "quarantined")}
    unique = Counter()
    unknown = {}
    for mid in sorted(set(ids) - reserved):
        observation = observations[mid]
        if "error" in observation:
            unknown[mid] = observation["error"]
            continue
        group = "accepted" if mid in accepted else "quarantined"
        names = observation["pro_names"]
        counts["non_test_source"].update(names)
        counts[group].update(names)
        unique["non_test_source"] += bool(names)
        unique[group] += bool(names)
    return {"actual_pro_games_in_" + name: dict(value) for name, value in counts.items()} | {
        "unique_games_with_tracked_pro": dict(unique), "unknown_non_test_source_ids": unknown,
        "source_partition": {"stored": len(ids), "reserved_metadata_only": len(set(ids) & reserved),
                             "verified_non_test": len(set(ids) - reserved - unknown.keys()),
                             "unknown_non_test": len(unknown), "accepted_non_test": len(set(ids) & accepted - reserved),
                             "quarantined_non_test": len(set(ids) & quarantined - reserved)},
        "pro_accounts_resolved": sum(bool(p.get("puuid")) for p in pros.values()),
        "unresolved_pros": sorted(name for name, p in pros.items() if not p.get("puuid"))}

def run(base=BASE):
    output = base / "pro_coverage_review.json"
    if output.exists():
        raise ValueError("Immutable pro review already exists; refusing overwrite")
    code_bindings = {str(path): evidence.sha256(path) for path in
                     (Path(__file__), ROOT / "scripts/prepare_dated_sample_followup.py")}
    reserved, _, bindings = evidence.reserve_splits(ROOT)
    split_files = set(bindings)
    done_path = base / "reconciliation_run/completion.json"
    done = evidence.load_json(done_path)
    final_path = base / "final/completion.json"
    final = evidence.load_json(final_path)
    if (done.get("state") != "complete" or done.get("protected_unchanged") is not True
            or done.get("test_payload_reads") is not False or done.get("training_or_export") is not False
            or final.get("status") != "acceptance_reconciliation_complete"
            or final.get("training_export_ready") is not False or final.get("test_payload_reads") is not False):
        raise ValueError("Acceptance supervisor has not completed")
    evidence.check(final_path, done["final_receipt_sha256"], bindings)
    bindings[str(done_path)] = evidence.sha256(done_path)
    samples, claims = {}, {}
    for relative, binding in final["files"].items():
        path = (base / "final" / relative).resolve()
        if not path.is_relative_to((base / "final").resolve()) or path.stat().st_size != binding["bytes"]:
            raise ValueError("Final receipt path or size mismatch")
        evidence.check(path, binding["sha256"], bindings)
    for name in final["sample_counts"]:
        index = evidence.load_json(base / "final" / name / "acceptance_index.json")
        coverage = evidence.load_json(base / "final" / name / "coverage.json")
        folder = evidence.relocated_path(index["sample"], ROOT)
        for file, digest in (("state.json", index["state_sha256"]), ("report.json", index["report_sha256"])):
            evidence.check(folder / file, digest, bindings)
        state = evidence.load_json(folder / "state.json")
        ids = {mid for mid, result in state["results"].items() if result["status"] == "stored"}
        accepted, quarantined, prior_reserved = map(set, (index["accepted_match_ids"], index["quarantined_match_ids"], index["reserved_match_ids"]))
        if accepted & quarantined or accepted & prior_reserved or quarantined & prior_reserved or accepted | quarantined | prior_reserved != ids:
            raise ValueError("Acceptance does not partition stored sample IDs")
        reserved.update(prior_reserved)
        if accepted & reserved:
            raise ValueError("Accepted membership now contains reserved IDs; review reservations first")
        for mid in ids:
            digest = evidence.require_digest(state["results"][mid]["match_sha256"])
            if mid in claims and claims[mid] != digest:
                raise ValueError("Conflicting sample match claims")
            claims[mid] = digest
        samples[name] = (ids, accepted, quarantined, state["pros"], coverage)
    if any(accepted & reserved for _, accepted, _, _, _ in samples.values()):
        raise ValueError("Cross-sample test reservation conflicts with accepted IDs")
    observations, raw_bindings = {}, {}
    for number, mid in enumerate(sorted(claims.keys() - reserved), 1):
        try:
            payload, binding = read_match(ROOT, mid, claims[mid], reserved)
            names = [pro_names(payload, pros) for ids, _, _, pros, _ in samples.values() if mid in ids]
            if any(value != names[0] for value in names):
                raise ValueError("sample_pro_identity_snapshots_conflict")
            observations[mid] = {"pro_names": names[0]}
            raw_bindings[mid] = binding
        except (OSError, ValueError, KeyError, TypeError) as exc:
            observations[mid] = {"error": str(exc) if isinstance(exc, ValueError) else type(exc).__name__}
        if number % 500 == 0:
            print(f"Verified pro source metadata {number}/{len(claims.keys() - reserved)}", flush=True)
    reports = {}
    for name, (ids, accepted, quarantined, pros, coverage) in samples.items():
        report = tally(ids, accepted, quarantined, reserved, observations, pros)
        if report["pro_accounts_resolved"] != coverage["pro_accounts_resolved"] or report["unresolved_pros"] != sorted(coverage["unresolved_pros"]):
            raise ValueError("Tracked-pro scope changed from final acceptance")
        report["original_collection_cutoff_utc"] = coverage["original_collection_cutoff_utc"]
        actual, expected = report["actual_pro_games_in_accepted"], coverage["pro_games"]
        accepted_unknown = set(report["unknown_non_test_source_ids"]) & accepted
        if any(count > expected.get(pro, 0) for pro, count in actual.items()) or not accepted_unknown and actual != expected:
            raise ValueError("Actual accepted pro participation disagrees with final coverage")
        report["accepted_counts_agree_with_final"] = actual == expected if not accepted_unknown else None
        reports[name] = report
    if {str(p) for p in evidence.split_paths(ROOT)} != split_files:
        raise ValueError('Reservation metadata set changed')
    for path, digest in {**bindings, **code_bindings}.items():
        evidence.check(Path(path), digest, {})
    review = {"status": "pro_coverage_incomplete" if any(r["unknown_non_test_source_ids"] for r in reports.values()) else "pro_coverage_verified",
              "purpose": "source_qualification_impact_only", "created_utc": datetime.now(timezone.utc).isoformat(),
              "samples": reports, "input_sha256": bindings, "verified_match_digests": raw_bindings,
              "code_sha256": code_bindings, "test_payload_reads": False, "training_export_ready": False,
              "limits": ["Reserved participation remains unknown; discovery routes never establish participation.",
                         "Unknown or mismatched sources stay explicit. Pro memberships overlap; counts cannot be summed as unique games.",
                         "This is pro-impact QA only; quarantined sources remain excluded and final acceptance views are unchanged."]}
    pending = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".partial")
    evidence.write_json(pending, review)
    os.link(pending, output)
    pending.unlink()
    output.chmod(0o444)
    return review

if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "samples": result["samples"]}, indent=2))

