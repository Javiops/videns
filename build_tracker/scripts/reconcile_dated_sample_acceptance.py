"""Acceptance-only dated views over existing audited corpora; never a training export."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import prepare_dated_sample_followup as source

BASE = ROOT / "data/acceptance/16.18_dated_samples_20260920"
PARENT_BASE = ROOT / "data/acceptance/16.18_reconstructible_20260914"
PARENT_CORPUS = ROOT / "data/corpora/16.18-reconstructible-v3"
CHILD_CORPUS = ROOT / "data/corpora/16.18-dated-samples-uncovered-20260920"
PHYSICAL_BASE = ROOT / "data/acceptance/16.18_development_v3_admission"
STATES = ["board_before", "shop_before", "shop_after"]


def forbid_reserved(selected, reserved):
    overlap = set(selected) & set(reserved)
    if overlap:
        raise ValueError("Reserved IDs in payload-bearing corpus; refusing even full-file hashing: "
                         + ",".join(sorted(overlap)[:5]))


def partition_sample(stored, reserved, accepted, quarantined):
    stored, reserved, accepted = set(stored), set(reserved), set(accepted)
    chosen = accepted & stored - reserved
    failures = {mid: reasons for mid, reasons in quarantined.items() if mid in stored - reserved}
    if chosen & failures.keys():
        raise ValueError("Accepted and quarantined memberships conflict")
    result = {"accepted": sorted(chosen), "reserved": sorted(stored & reserved),
              "quarantined": dict(sorted(failures.items()))}
    flat = result["accepted"] + result["reserved"] + list(result["quarantined"])
    if len(flat) != len(set(flat)) or set(flat) != stored:
        raise ValueError("Dated view does not account for every stored game")
    if any(not isinstance(reasons, list) or not reasons for reasons in failures.values()):
        raise ValueError("Quarantine reason missing")
    return result


def inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("Bound file escapes its corpus")
    return path


def full_corpus_integrity(prior, corpus, reserved, bindings):
    """Check the selected ID guard before opening any corpus DB/mask/raw bytes."""
    forbid_reserved(prior["selected"], reserved)
    manifest = prior["data"]["manifest"]
    source.check(corpus / "tracker.db", manifest["db_sha256"], bindings)
    source.check(corpus / "decision_admission.json", manifest["decision_admission_sha256"], bindings)
    for relative, binding in manifest["files"].items():
        path = inside(corpus, relative)
        if path.stat().st_size != binding["bytes"]:
            raise ValueError("Frozen corpus file size differs: " + relative)
        if str(path) not in bindings:
            source.check(path, binding["sha256"], bindings)
        elif bindings[str(path)] != binding["sha256"]:
            raise ValueError("Conflicting corpus file hash")
    masks = source.load_json(corpus / "decision_admission.json")
    if set(masks["games"]) != prior["accepted"]:
        raise ValueError("Decision mask and accepted corpus membership differ")
    if masks.get("test_payload_reads") is not False:
        raise ValueError("Decision mask lacks sealed-test scope")
    return masks


def bound_physical_metadata(root, bindings):
    """Anchor reusable qualification to the immutable generation's metadata copies."""
    generation = root / "data/dataset_generations/16.18-reconstructible-development-v3"
    coverage_base = PHYSICAL_BASE.with_name(PHYSICAL_BASE.name + "_coverage")
    manifest_path = generation / "generation_manifest.json"
    manifest = source.load_json(manifest_path)
    if (manifest.get("state") != "complete" or manifest.get("patch") != "16.18"
            or manifest.get("purpose") != "patch_development_1618"):
        raise ValueError("Reusable inventory evidence lacks its immutable generation receipt")
    documents = {}
    for name, path in (("qualification.json", PHYSICAL_BASE / "qualification.json"),
                       ("coverage.json", coverage_base / "coverage.json"),
                       ("split_plan.json", coverage_base / "split_plan.json")):
        binding = manifest["files"][name]
        for candidate in (path, generation / name):
            if candidate.stat().st_size != binding["bytes"]:
                raise ValueError("Reusable qualification metadata size changed")
            source.check(candidate, binding["sha256"], bindings)
        documents[name] = source.load_json(path)
    q, coverage, split = (documents[name] for name in ("qualification.json", "coverage.json", "split_plan.json"))
    q_sha = manifest["files"]["qualification.json"]["sha256"]
    if (coverage.get("status") != "coverage_complete" or coverage.get("qualification_sha256") != q_sha
            or split.get("qualification_sha256") != q_sha
            or set(split["assignment"]) != set(q["accepted_match_ids"])
            or "test" in split["assignment"].values()
            or coverage["quality_accepted_games"] != len(q["accepted_match_ids"])
            or coverage["split_plan_sha256"] != manifest["files"]["split_plan.json"]["sha256"]):
        raise ValueError("Reusable qualification, coverage and split membership disagree")
    bindings[str(manifest_path.resolve())] = source.sha256(manifest_path)
    return q


def physical_receipt(root, parent, bindings):
    path = PHYSICAL_BASE / "qualification.json"
    q = bound_physical_metadata(root, bindings)
    if (q.get("status") != "qualified_subset" or q.get("patch") != "16.18"
            or q.get("production_hold_cleared") is not False or q.get("promotion_allowed") is not False
            or q.get("inventory_constraint_states") != STATES
            or q.get("source_qualification_sha256") != source.sha256(parent["paths"]["qualification"])
            or q.get("corpus_manifest_sha256") != source.sha256(parent["paths"]["manifest"])
            or q.get("db_sha256") != parent["data"]["manifest"]["db_sha256"]
            or q.get("decision_admission_sha256") != parent["data"]["manifest"]["decision_admission_sha256"]):
        raise ValueError("Prior inventory qualification is not bound to the parent corpus")
    if source.relocated_path(q["corpus"], root) != PARENT_CORPUS.resolve():
        raise ValueError("Prior physical qualification refers to another corpus")
    source.check(PHYSICAL_BASE / "qualification_code.py", q["qualification_code_sha256"], bindings)
    source.check(PHYSICAL_BASE / "semantic_audit.json", q["semantic_audit_sha256"], bindings)
    if q["semantic_audit_sha256"] != source.sha256(parent["paths"]["audit"]):
        raise ValueError("Physical qualification binds a different semantic audit")
    admission = PHYSICAL_BASE / "benchmark_admission.json"
    source.check(admission, q["benchmark_admission_sha256"], bindings)
    for name, record in source.load_json(admission)["files"].items():
        path_to_check = inside(PHYSICAL_BASE, name)
        if path_to_check.stat().st_size != record["bytes"]:
            raise ValueError("Physical qualification evidence size changed")
        source.check(path_to_check, record["sha256"], bindings)
    accepted = set(q["accepted_match_ids"])
    excluded = set(q["quarantined_match_ids"])
    if (accepted & excluded or accepted | excluded != parent["accepted"]
            or len(accepted) != q["quality_accepted_games"]):
        raise ValueError("Inventory qualification does not partition parent accepted IDs")
    bindings[str(path.resolve())] = source.sha256(path)
    return accepted


def check_selected_raw_receipts(corpus, prior, selected, reserved, bindings):
    forbid_reserved(selected, reserved)
    for mid in sorted(selected):
        for kind in ("match", "timeline"):
            receipt = prior["inventory"][mid]["files"][kind]
            expected = corpus / "raw" / kind / (mid + ".json.gz")
            if inside(corpus, receipt["path"]) != expected.resolve():
                raise ValueError("Frozen raw receipt points outside selected source")
            source.check(expected, receipt["sha256"], bindings)


def inventory_checks(corpus, prior, selected, reserved, bindings):
    if not selected:
        return {}, {"inventory_constraint_states": STATES, "checked_games": 0,
                    "inventory_constraint_checked_events": 0}
    check_selected_raw_receipts(corpus, prior, selected, reserved, bindings)
    from prepare_benchmark_1618 import inventory_qualification
    excluded, counts = inventory_qualification(corpus, prior["data"]["manifest"], set(selected))
    if not set(excluded) <= set(selected):
        raise ValueError("Inventory checks returned a game outside the admitted check set")
    return excluded, {**counts, "checked_games": len(selected),
                      "selected_match_ids": sorted(selected), "excluded_match_ids": excluded}


def new_coverage():
    return {"games": 0, "admitted_decisions": 0, "boundary_excluded_decisions": 0,
            "region_games": Counter(), "participant_role_counts": Counter(),
            "champion_role_participations": Counter(), "decision_role_counts": Counter(),
            "decision_champion_role_counts": Counter(), "label_item_occurrences": Counter(),
            "save_kind_decisions": Counter(), "pro_games": Counter(), "pro_decisions": Counter(),
            "games_with_tracked_pro": 0, "max_target_item_copies": 0,
            "max_copies_of_one_target": 0}


def add_game_coverage(total, game, rows, pros):
    """Count actual participant identities and every target copy; never discovery routes."""
    from app.purchase_multiset import label_counts
    players = game["participants"]
    by_pid = {p["participant_id"]: p for p in players}
    if len(players) != 10 or len(by_pid) != 10:
        raise ValueError("Accepted lobby lacks ten distinct participants")
    names_by_puuid = defaultdict(set)
    for name, record in pros.items():
        if record.get("puuid"):
            names_by_puuid[record["puuid"]].add(name)
    present = set().union(*(names_by_puuid[p["puuid"]] for p in players))
    total["games"] += 1
    total["region_games"][game["match_id"].split("_", 1)[0]] += 1
    total["pro_games"].update(present)
    total["games_with_tracked_pro"] += bool(present)
    total["boundary_excluded_decisions"] += game["boundary_excluded_rows"]
    for player in players:
        role = player["team_position"]
        champion = player.get("champion_name", str(player["champion_id"]))
        total["participant_role_counts"][role] += 1
        total["champion_role_participations"][champion + ":" + role] += 1
    for row in rows:
        player = by_pid[row["participant_id"]]
        if row["role"] != player["team_position"]:
            raise ValueError("Decision role disagrees with participant metadata")
        counts = label_counts(row)
        total["admitted_decisions"] += 1
        total["decision_role_counts"][row["role"]] += 1
        total["decision_champion_role_counts"][row["champion"] + ":" + row["role"]] += 1
        total["label_item_occurrences"].update({str(item): count for item, count in counts.items()})
        total["save_kind_decisions"][row["save_kind"]] += 1
        total["pro_decisions"].update(names_by_puuid[player["puuid"]])
        total["max_target_item_copies"] = max(total["max_target_item_copies"], sum(counts.values()))
        total["max_copies_of_one_target"] = max(total["max_copies_of_one_target"], max(counts.values(), default=0))


def publish(output, documents):
    if output.exists():
        raise ValueError("Immutable acceptance view already exists; refusing overwrite")
    pending = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".partial")
    pending.mkdir(parents=True)
    for relative, document in documents.items():
        if relative == "completion.json":
            continue
        target = inside(pending, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.write_json(target, document)
    if "completion.json" in documents:
        documents["completion.json"]["files"] = {
            path.relative_to(pending).as_posix(): {"sha256": source.sha256(path), "bytes": path.stat().st_size}
            for path in pending.rglob("*") if path.is_file()}
        source.write_json(pending / "completion.json", documents["completion.json"])
    for path in pending.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    pending.rename(output)


def reconcile(base=BASE, output=None, *, stage_only=False):
    base = Path(base).resolve()
    expected_output = base / ("staged_final" if stage_only else "final")
    output = Path(output).resolve() if output else expected_output
    if output != expected_output or output.exists():
        raise ValueError("Use the new immutable final view directory; refusing overwrite")
    reserved, memberships, split_bindings = source.reserve_splits(ROOT)
    bindings = dict(split_bindings)
    preflight_path = base / "preflight.json"
    preflight = source.load_json(preflight_path)
    if (preflight.get("status") != "prepared_not_accepted"
            or preflight.get("test_payload_reads") is not False
            or preflight.get("training_or_export_authorized") is not False
            or preflight.get("promotion_allowed") is not False):
        raise ValueError("Unreviewed or wrong-scope preparation")
    source.check(base / "uncovered/source_plan.json", preflight["uncovered_source_plan_sha256"], bindings)
    bindings[str(preflight_path)] = source.sha256(preflight_path)
    # Metadata plans first: a newly reserved ID blocks full DB/mask hashing too.
    parent_plan = source.load_json(PARENT_BASE / "source_plan.json")
    child_plan = source.load_json(base / "uncovered/source_plan.json")
    reserved.update(parent_plan["reserved_test_ids"])
    reserved.update(child_plan["reserved_test_ids"])
    parent_selected = {record["match_id"] for record in parent_plan["matches"]}
    child_selected = {record["match_id"] for record in child_plan["matches"]}
    forbid_reserved(parent_selected, reserved)
    forbid_reserved(child_selected, reserved)
    if parent_selected & child_selected or child_selected != set(preflight["uncovered_non_test_ids"]):
        raise ValueError("Child scope is not the disjoint uncovered source union")
    for path, digest in preflight["input_evidence_sha256"].items():
        source.check(source.relocated_path(path, ROOT), digest, bindings)
    parent = source.load_prior(ROOT, PARENT_BASE, PARENT_CORPUS, bindings)
    child = source.load_prior(ROOT, base / "uncovered", CHILD_CORPUS, bindings)
    if parent["data"]["manifest"]["footwear_runtime"] != child["data"]["manifest"]["footwear_runtime"]:
        raise ValueError("Parent/child runtimes differ; separate scoped verification is required")
    # Metadata equality check catches sample changes without opening any sample payload.
    claims = {}
    actual_samples = [source.load_sample(ROOT, Path(s["sample"]), claims, bindings) for s in preflight["samples"]]
    for original, actual in zip(preflight["samples"], actual_samples, strict=True):
        for key in actual:
            if actual[key] != original[key]:
                raise ValueError("Prepared sample metadata changed: " + key)
    parent_wanted = set()
    for sample in preflight["samples"]:
        expected = source.classify_ids(sample["stored_match_ids"], reserved, parent["selected"],
                                       parent["accepted"], parent["excluded"])
        if expected != sample["classification"]:
            raise ValueError("Reservations or source classifications changed; prepare a new reviewed view")
        parent_wanted.update(expected["qualified"])
    for mid in parent_wanted:
        receipt = preflight["covered_source_receipts"][mid]
        current = source.covered_receipt(ROOT, mid, claims[mid], "qualified", parent, bindings)
        if current != receipt or receipt.get("source_identity_verified") is not True:
            raise ValueError("Qualified parent canonical receipt changed")
    for mid in child["accepted"]:
        if mid not in claims:
            raise ValueError("Child accepted a game outside both samples")
        receipt = child["inventory"][mid]["files"]
        if any(receipt[kind]["payload_sha256"] != claims[mid]["payload_sha256"][kind]
               for kind in ("match", "timeline")):
            raise ValueError("Child canonical source does not match dated sample")
    print("Verifying frozen parent/child DBs, masks and bound inputs...", flush=True)
    parent_mask = full_corpus_integrity(parent, PARENT_CORPUS, reserved, bindings)
    child_mask = full_corpus_integrity(child, CHILD_CORPUS, reserved, bindings)
    reused = parent_wanted & physical_receipt(ROOT, parent, bindings)
    missing = parent_wanted - reused
    current_code = {str(path): source.sha256(path) for path in
                    [Path(__file__), ROOT / "scripts/prepare_dated_sample_followup.py",
                     ROOT / "scripts/prepare_benchmark_1618.py", ROOT / "scripts/qualify_patch_corpus.py",
                     ROOT / "scripts/corpus_rows.py", ROOT / "app/purchase_multiset.py"]}
    print(f"Checking inventory for {len(missing)} parent games and {len(child['accepted'])} child games; "
          f"reusing {len(reused)} exact-parent checks.", flush=True)
    parent_failures, parent_checks = inventory_checks(PARENT_CORPUS, parent, missing, reserved, bindings)
    child_failures, child_checks = inventory_checks(CHILD_CORPUS, child, child["accepted"], reserved, bindings)
    accepted_by_corpus = {"parent": parent_wanted - parent_failures.keys(),
                          "child": child["accepted"] - child_failures.keys()}
    quarantine = {mid: ["prior_source_exclusion", *reasons,
                       "sample_canonical_link_unverified"] for mid, reasons in parent["excluded"].items()}
    quarantine.update({mid: ["child_source_exclusion", *reasons] for mid, reasons in child["excluded"].items()})
    quarantine.update({mid: ["whole_game_inventory_exclusion", *reasons]
                       for mid, reasons in {**parent_failures, **child_failures}.items()})
    accepted = set.union(*accepted_by_corpus.values())
    if accepted_by_corpus["parent"] & accepted_by_corpus["child"]:
        raise ValueError("Accepted corpus views overlap")
    coverage = {Path(s["sample"]).name: new_coverage() for s in preflight["samples"]}
    views = {Path(s["sample"]).name: partition_sample(s["stored_match_ids"], reserved, accepted, quarantine)
             for s in preflight["samples"]}
    from footwear_runtime import activate_for_manifest
    from types import SimpleNamespace
    for name, corpus, evidence, mask in (("parent", PARENT_CORPUS, parent, parent_mask),
                                         ("child", CHILD_CORPUS, child, child_mask)):
        manifest = evidence["data"]["manifest"]
        activate_for_manifest(manifest, corpus_path=corpus)
        from app.ddragon import dragon_for_patch
        from corpus_rows import iter_games, row_builder_digest
        dragon = dragon_for_patch("16.18")
        if dragon.signature != manifest["static_data_sha256"]:
            raise ValueError("Active static inputs differ from audited corpus")
        for semantic_file in ("app/ddragon.py", "app/shop_econ.py", "scripts/baseline.py"):
            if source.sha256(ROOT / semantic_file) != manifest["reconstruction_code_sha256"][semantic_file]:
                raise ValueError("Coverage row semantics differ from audited corpus")
        builder = row_builder_digest()
        selected = accepted_by_corpus[name]
        forbid_reserved(selected, reserved)
        context = SimpleNamespace(path=corpus, qualification={"accepted_match_ids": sorted(selected)},
                                  manifest=manifest, dragon=dragon, decision_admission=mask)
        seen = set()
        for game, rows, exclusion in iter_games(context):
            mid = game["match_id"]
            if exclusion or mid not in selected or mid in seen:
                raise ValueError("Qualified coverage game omitted or duplicated")
            if len(rows) != mask["games"][mid]["admitted_rows"]:
                raise ValueError("Coverage does not match the bound decision mask")
            seen.add(mid)
            for sample in preflight["samples"]:
                key = Path(sample["sample"]).name
                if mid in views[key]["accepted"]:
                    add_game_coverage(coverage[key], game, rows, sample["cohort_snapshot"]["pros"])
        if seen != selected or row_builder_digest() != builder:
            raise ValueError("Coverage membership or row semantics changed during traversal")
    # Rehash every consumed input, including DB/masks, before publication.
    print("Rechecking input hashes before publishing immutable acceptance views...", flush=True)
    if source.split_paths(ROOT) != sorted(Path(path) for path in split_bindings):
        raise ValueError("Reservation metadata set changed")
    for path, digest in {**bindings, **current_code}.items():
        source.check(Path(path), digest, {})
    timestamp = datetime.now(timezone.utc).isoformat()
    corpus_receipts = {name: {"corpus": str(corpus), "manifest_sha256": source.sha256(evidence["paths"]["manifest"]),
        "db_sha256": evidence["data"]["manifest"]["db_sha256"],
        "decision_admission_sha256": evidence["data"]["manifest"]["decision_admission_sha256"],
        "source_qualification_sha256": source.sha256(evidence["paths"]["qualification"]),
        "semantic_audit_sha256": source.sha256(evidence["paths"]["audit"])}
        for name, corpus, evidence in (("parent", PARENT_CORPUS, parent), ("child", CHILD_CORPUS, child))}
    limits = ["These are immutable audited corpus views/evidence only, not an exportable training database.",
              "Full semantic checks are reused bound parent per-game evidence plus the fresh child audit; "
              "this reconciliation does not rerun the full semantic audit.",
              "Reserved IDs have metadata-only dispositions; no final-test payload was opened.",
              "Previously excluded parent sources remain quarantined; their sample canonical linkage is unverified.",
              "No training split, benchmark admission, model evaluation, promotion or dataset export is created.",
              "Original 16 unresolved pro outcomes remain accepted limitations; pro memberships overlap."]
    documents = {}
    for sample in preflight["samples"]:
        name = Path(sample["sample"]).name
        view = views[name]
        index = {"schema": 1, "status": "immutable_audited_corpus_view", "purpose": "acceptance_evidence_only",
                 "created_utc": timestamp, "sample": sample["sample"], "patch": "16.18",
                 "original_collection_cutoff_utc": sample["collection_cutoff_utc"],
                 "state_sha256": sample["state_sha256"], "report_sha256": sample["report_sha256"],
                 "cohort_parent_sha256": sample["cohort_parent_sha256"],
                 "cohort_snapshot_sha256": source.canonical_sha(sample["cohort_snapshot"]),
                 "accepted_match_ids": view["accepted"], "reserved_match_ids": view["reserved"],
                 "quarantined_match_ids": view["quarantined"],
                 "classification_counts": {key: len(value) for key, value in view.items()},
                 "corpus_membership": {mid: "parent" if mid in accepted_by_corpus["parent"] else "child"
                                       for mid in view["accepted"]},
                 "corpora": corpus_receipts, "inventory_constraint_states": STATES,
                 "inventory_evidence": {"reused_exact_parent_match_ids": sorted(set(view["accepted"]) & reused),
                    "fresh_parent_checks": sorted(set(sample["stored_match_ids"]) & missing),
                    "fresh_child_checks": sorted(set(sample["stored_match_ids"]) & child["accepted"])},
                 "unresolved_pros": sample["unresolved_pros"], "training_export_ready": False,
                 "production_hold_cleared": False, "promotion_allowed": False, "test_payload_reads": False,
                 "limits": limits}
        documents[name + "/acceptance_index.json"] = index
        documents[name + "/coverage.json"] = {"schema": 1, "status": "acceptance_coverage_complete",
            "purpose": "acceptance_evidence_only", "patch": "16.18", **coverage[name],
            "cohort_parent_sha256": sample["cohort_parent_sha256"],
            "original_collection_cutoff_utc": sample["collection_cutoff_utc"],
            "target_encoding": "item-counts-v1", "target_count_unit": "item copies; never a deduplicated set",
            "pro_accounts_resolved": sum(bool(v.get("puuid")) for v in sample["cohort_snapshot"]["pros"].values()),
            "unresolved_pros": sample["unresolved_pros"], "corpora": corpus_receipts,
            "training_export_ready": False, "test_payload_reads": False, "limits": limits}
    documents["completion.json"] = {"schema": 1, "status": "acceptance_reconciliation_complete",
        "purpose": "acceptance_evidence_only", "created_utc": timestamp,
        "sample_counts": {name: {key: len(value) for key, value in view.items()} for name, view in views.items()},
        "input_evidence_sha256": bindings, "reconciliation_code_sha256": current_code,
        "preflight_sha256": source.sha256(preflight_path), "corpora": corpus_receipts,
        "fresh_parent_inventory_checks": parent_checks, "fresh_child_inventory_checks": child_checks,
        "reused_parent_inventory_games": len(reused), "training_export_ready": False,
        "production_hold_cleared": False, "promotion_allowed": False, "test_payload_reads": False,
        "limits": limits}
    documents["completion.json"]["supervisor_completion_required"] = stage_only
    publish(output, documents)
    return documents["completion.json"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--stage-only", action="store_true", help="Publish staged evidence for supervisor verification before final admission")
    args = parser.parse_args()
    result = reconcile(args.base, stage_only=args.stage_only)
    print(json.dumps({"status": result["status"], "sample_counts": result["sample_counts"],
                      "training_export_ready": False}, indent=2))

