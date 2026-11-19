"""Synthetic admission checks only; no real game, timeline or final-test payloads."""
from copy import deepcopy
from contextlib import closing
import gzip
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import prepare_dated_sample_followup as planner


def refused(function, contains=None):
    try:
        function()
    except (ValueError, KeyError) as error:
        if contains is not None:
            assert contains in str(error), str(error)
    else:
        raise AssertionError("Invalid input was accepted")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_reserved_precedence_and_union(root):
    save(root / "data/ml/split_assignment.json", {"KR_1": "train", "KR_2": "val"})
    save(root / "data/dataset_generations/old/split_assignment.json",
         {"KR_1": "test", "KR_2": "train"})
    save(root / "data/ml/archive/split_assignment_v1.json", {"KR_3": "test"})
    reserved, membership, bindings = planner.reserve_splits(root)
    assert reserved == {"KR_1", "KR_3"}
    assert len(bindings) == 3 and set(membership["KR_1"].values()) == {"train", "test"}
    selected, accepted, excluded = {"KR_1", "KR_2", "KR_4"}, {"KR_1", "KR_2"}, {"KR_4"}
    first = planner.classify_ids({"KR_1", "KR_2", "KR_4", "KR_5"},
                                reserved, selected, accepted, excluded)
    second = planner.classify_ids({"KR_3", "KR_5", "KR_6"},
                                 reserved, selected, accepted, excluded)
    assert first == {"reserved": ["KR_1"], "qualified": ["KR_2"],
                     "source_excluded": ["KR_4"], "uncovered": ["KR_5"]}
    assert second["reserved"] == ["KR_3"]
    assert set(first["uncovered"]) | set(second["uncovered"]) == {"KR_5", "KR_6"}
    refused(lambda: planner.classify_ids({"KR_1"}, set(), {"KR_1", "KR_9"}, {"KR_1"}, set()))
    refused(lambda: planner.classify_ids({"KR_1"}, set(), {"KR_1"}, {"KR_1"}, {"KR_1"}))
    omitted = deepcopy(first)
    omitted["uncovered"] = []
    refused(lambda: planner.assert_classification({"KR_1", "KR_2", "KR_4", "KR_5"}, omitted))
    duplicated = deepcopy(first)
    duplicated["uncovered"].append("KR_2")
    refused(lambda: planner.assert_classification({"KR_1", "KR_2", "KR_4", "KR_5"}, duplicated))
    with patch.object(planner, "sha256", side_effect=AssertionError("Reserved raw file was accessed")):
        refused(lambda: planner.read_uncovered_pair(root, "KR_1", {}, membership, reserved), "Reserved")
    return membership, bindings


def test_conflicting_claims():
    metadata = {"match_id": "KR_5", "patch": "16.18", "queue_id": 420,
                "game_creation": 1000, "game_duration": 900}
    hashes = {"match": "a" * 64, "timeline": "b" * 64}
    claims = {}
    planner.merge_claim(claims, "KR_5", metadata, hashes, "first")
    planner.merge_claim(claims, "KR_5", deepcopy(metadata), hashes, "second")
    assert claims["KR_5"]["samples"] == ["first", "second"]
    refused(lambda: planner.merge_claim(claims, "KR_5", {**metadata, "game_duration": 901},
                                        hashes, "third"), "Conflicting")
    refused(lambda: planner.merge_claim(claims, "KR_5", metadata,
                                        {**hashes, "timeline": "c" * 64}, "third"), "Conflicting")
    refused(lambda: planner.merge_claim({}, "KR_5", metadata, {"match": "", "timeline": "b" * 64}, "x"))
    return metadata


def test_qualified_receipt_never_opens_raw(root, metadata):
    claim = {"metadata": metadata, "samples": ["first"], "payload_sha256": {
        "match": "a" * 64, "timeline": "b" * 64}}
    files = {kind: {"sha256": value, "payload_sha256": claim["payload_sha256"][kind]}
             for kind, value in (("match", "c" * 64), ("timeline", "d" * 64))}
    prior = {"selected": {"KR_5": {**metadata, "files": files}},
             "inventory": {"KR_5": {"files": files}}}
    with patch.object(planner, "sha256", side_effect=AssertionError("Covered raw was accessed")):
        result = planner.covered_receipt(root, "KR_5", claim, "qualified", prior, {})
        assert result["source_identity_verified"] and result["payloads_opened"] is False
        wrong = deepcopy(prior)
        wrong["inventory"]["KR_5"]["files"]["match"]["payload_sha256"] = "e" * 64
        refused(lambda: planner.covered_receipt(root, "KR_5", claim, "qualified", wrong, {}), "conflicts")


def test_sample_metadata(root):
    folder = root / "data/patch_samples/synthetic"
    folder.mkdir(parents=True)
    cohort = {"tracked_pros": [{"league": "LCK", "id": "test"}], "ladders": {"kr": [1], "euw": [2]},
              "pros": {"LCK:test": {"puuid": "synthetic-puuid"}}, "roster_observed_utc": {}}
    parent = root / "data/corpora/parent/state.json"
    save(parent, cohort)
    result = {"status": "stored", "perspectives": 10, "match_sha256": "a" * 64, "timeline_sha256": "b" * 64}
    state = {**cohort, "patch": "16.18", "discovery_complete": True,
             "jobs": {"KR_5": {}}, "results": {"KR_5": result},
             "start_time": 1, "end_time": 2000, "cohort_parent": str(parent),
             "cohort_parent_sha256": planner.sha256(parent), "cohort_policy": "same frozen cohort"}
    save(folder / "state.json", state)
    report = {"patch": "16.18", "structural_checks_passed": True,
              "checkpoint_sha256": planner.sha256(folder / "state.json"), "counts": {"stored": 1},
              "cutoff_utc": "1970-01-01T00:33:20+00:00", "games": 1, "perspectives": 10,
              "unresolved_pros": [], "scope_complete": True}
    save(folder / "report.json", report)
    with closing(sqlite3.connect(folder / "tracker.db")) as conn:
        conn.execute("CREATE TABLE games(match_id TEXT,patch TEXT,queue_id INT,game_creation INT,game_duration INT)")
        conn.execute("INSERT INTO games VALUES('KR_5','16.18',420,1000,900)")
        conn.commit()
    claims, bindings = {}, {}
    loaded = planner.load_sample(root, folder, claims, bindings)
    assert loaded["stored_match_ids"] == ["KR_5"] and len(bindings) == 3
    assert loaded["end_time"] == 2000 and claims["KR_5"]["metadata"]["game_duration"] == 900
    state["results"]["KR_5"]["perspectives"] = 9
    save(folder / "state.json", state)
    refused(lambda: planner.load_sample(root, folder, {}, {}), "Incomplete")
    save(folder / "report.json", {**report, "checkpoint_sha256": planner.sha256(folder / "state.json")})
    refused(lambda: planner.load_sample(root, folder, {}, {}), "ten perspectives")


def test_uncovered_pair_and_process_schema(root, metadata, membership, split_bindings):
    identities = [{"participantId": i, "puuid": "synthetic-" + str(i)} for i in range(1, 11)]
    match = {"metadata": {"matchId": "KR_5"}, "info": {"gameVersion": "16.18.1.1",
             "queueId": 420, "gameCreation": metadata["game_creation"], "gameDuration": 900,
             "participants": identities}}
    timeline = {"metadata": {"matchId": "KR_5"}, "info": {"participants": identities}}
    payloads = {"match": match, "timeline": timeline}
    for kind, payload in payloads.items():
        path = root / "data/raw_match_v5" / kind / "KR_5.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"schema": 1, "endpoint": kind, "match_id": "KR_5", "payload": payload}, handle)
    claim = {"metadata": metadata, "samples": ["synthetic"],
             "payload_sha256": {kind: planner.canonical_sha(payload) for kind, payload in payloads.items()}}
    record = planner.read_uncovered_pair(root, "KR_5", claim, membership, {"KR_1", "KR_3"})
    assert record["match_id"] == "KR_5" and set(record["files"]) == {"match", "timeline"}
    wrong = deepcopy(claim)
    wrong["payload_sha256"]["timeline"] = "f" * 64
    refused(lambda: planner.read_uncovered_pair(root, "KR_5", wrong, membership, set()), "disagrees")
    wrong = deepcopy(claim)
    wrong["metadata"]["game_duration"] = 901
    refused(lambda: planner.read_uncovered_pair(root, "KR_5", wrong, membership, set()), "metadata")
    review = root / "inputs/static_review.json"
    save(review, {"synthetic": True})
    plan = {"schema": 1, "patch": "16.18", "purpose": "isolated_reconstruction_qualification",
            "test_payload_reads": False, "training_or_export_authorized": False,
            "static_review": str(review), "static_review_sha256": planner.sha256(review),
            "matches": [record], "selected_sources": 1, "reserved_test_ids": ["KR_1", "KR_3"],
            "split_metadata_sha256": split_bindings}
    path = root / "source_plan.json"
    save(path, plan)
    import process_reconstructible_1618 as consumer
    with patch.object(consumer, "ROOT", root):
        assert consumer.validate_plan(path)["matches"] == [record]
        plan["reserved_test_ids"].append("KR_5")
        save(path, plan)
        refused(lambda: consumer.validate_plan(path), "reserved")


def main():
    with tempfile.TemporaryDirectory(prefix="dated-sample-synthetic-") as directory:
        root = Path(directory).resolve()
        membership, bindings = test_reserved_precedence_and_union(root)
        metadata = test_conflicting_claims()
        test_qualified_receipt_never_opens_raw(root, metadata)
        test_uncovered_pair_and_process_schema(root, metadata, membership, bindings)
        test_sample_metadata(root)
    print("ok dated follow-up: reserved-first metadata, exact partition/union, conflicting claims, "
          "covered receipts without payload reads, uncovered identity, checkpoint/DB agreement, consumer schema")


if __name__ == "__main__":
    main()

