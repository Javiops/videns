"""Synthetic acceptance-view tests; never read actual raw or final-test payloads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import reconcile_dated_sample_acceptance as reconcile


def refused(function, contains=None):
    try:
        function()
    except ValueError as error:
        if contains is not None:
            assert contains in str(error), str(error)
    else:
        raise AssertionError("Invalid input was accepted")


def test_reserved_precedes_all_payloads():
    with patch.object(reconcile.source, "check", side_effect=AssertionError("Payload bytes were opened")):
        refused(lambda: reconcile.full_corpus_integrity(
            {"selected": {"KR_1": {}}, "data": {}}, Path("synthetic"), {"KR_1"}, {}), "Reserved")
        refused(lambda: reconcile.check_selected_raw_receipts(
            Path("synthetic"), {}, {"KR_1"}, {"KR_1"}, {}), "Reserved")
    result = reconcile.partition_sample({"KR_1", "KR_2", "KR_3"}, {"KR_1"},
                                        {"KR_1", "KR_2"}, {"KR_1": ["old_exclusion"], "KR_3": ["unsupported"]})
    assert result == {"accepted": ["KR_2"], "reserved": ["KR_1"], "quarantined": {"KR_3": ["unsupported"]}}


def test_partition_conflicts_and_omissions():
    refused(lambda: reconcile.partition_sample({"KR_1"}, set(), {"KR_1"}, {"KR_1": ["conflict"]}), "conflict")
    refused(lambda: reconcile.partition_sample({"KR_1", "KR_2"}, set(), {"KR_1"}, {}), "every")
    refused(lambda: reconcile.partition_sample({"KR_1"}, set(), set(), {"KR_1": []}), "reason")
    result = reconcile.partition_sample({"KR_1"}, set(), {"KR_1", "KR_9"}, {"KR_10": ["other sample"]})
    assert result["accepted"] == ["KR_1"] and not result["quarantined"]


def test_multisets_and_actual_participant_pro_coverage():
    players = [{"participant_id": n, "puuid": "person-" + str(n),
                "team_position": "TOP", "champion_id": n, "champion_name": "Champion" + str(n)}
               for n in range(1, 11)]
    game = {"match_id": "KR_5", "participants": players, "boundary_excluded_rows": 2,
            "discovery_cohorts": ["pro:NotInTheGame"]}
    rows = [{"participant_id": 2, "champion": "Champion2", "role": "TOP",
             "save_kind": "build_spend", "label_counts": {"1036": 5, "2055": 2},
             "target_encoding": "item-counts-v1"},
            {"participant_id": 2, "champion": "Champion2", "role": "TOP",
             "save_kind": "no_buy_death", "label_counts": {"999999": 1},
             "target_encoding": "item-counts-v1"}]
    pros = {"LCK:Actual": {"puuid": "person-2"}, "LEC:Absent": {"puuid": "absent-person"},
            "LEC:Unresolved": {"puuid": None}}
    original = deepcopy((game, rows, pros))
    coverage = reconcile.new_coverage()
    reconcile.add_game_coverage(coverage, game, rows, pros)
    assert (game, rows, pros) == original, "Coverage mutated source quantities or metadata"
    assert coverage["pro_games"] == {"LCK:Actual": 1}
    assert coverage["pro_decisions"] == {"LCK:Actual": 2}
    assert coverage["games_with_tracked_pro"] == 1
    assert coverage["label_item_occurrences"] == {"1036": 5, "2055": 2, "999999": 1}
    assert coverage["max_target_item_copies"] == 7 and coverage["max_copies_of_one_target"] == 5
    assert coverage["admitted_decisions"] == 2 and coverage["boundary_excluded_decisions"] == 2
    assert coverage["region_games"] == {"KR": 1}
    assert coverage["participant_role_counts"] == {"TOP": 10}
    assert coverage["decision_role_counts"] == {"TOP": 2}
    repeated = deepcopy(rows)
    repeated.reverse()
    reordered = reconcile.new_coverage()
    reconcile.add_game_coverage(reordered, game, repeated, pros)
    assert reordered == coverage, "Target or decision order affected coverage"
    bad = deepcopy(rows)
    bad[0]["role"] = "MIDDLE"
    refused(lambda: reconcile.add_game_coverage(reconcile.new_coverage(), game, bad, pros), "role")


def test_atomic_publication_and_refusal():
    with tempfile.TemporaryDirectory(prefix="dated-reconciliation-synthetic-") as directory:
        root = Path(directory)
        out = root / "final"
        documents = {"sample/acceptance_index.json": {"status": "immutable_audited_corpus_view",
                                                     "training_export_ready": False},
                     "sample/coverage.json": {"games": 1, "training_export_ready": False},
                     "completion.json": {"status": "acceptance_reconciliation_complete",
                                         "training_export_ready": False}}
        reconcile.publish(out, documents)
        receipt = json.loads((out / "completion.json").read_text(encoding="utf-8"))
        assert set(receipt["files"]) == {"sample/acceptance_index.json", "sample/coverage.json"}
        for name, binding in receipt["files"].items():
            assert reconcile.source.sha256(out / name) == binding["sha256"]
        before = reconcile.source.sha256(out / "completion.json")
        refused(lambda: reconcile.publish(out, {"replacement.json": {}}), "overwrite")
        assert reconcile.source.sha256(out / "completion.json") == before
        assert not (out / "replacement.json").exists()
        refused(lambda: reconcile.publish(root / "bad", {"../escape.json": {}}), "escapes")
        assert not (root / "escape.json").exists()




def test_inventory_receipt_is_anchored():
    with tempfile.TemporaryDirectory(prefix="dated-physical-metadata-") as directory:
        root = Path(directory)
        physical = root / "data/acceptance/physical"
        coverage = root / "data/acceptance/physical_coverage"
        generation = root / "data/dataset_generations/16.18-reconstructible-development-v3"
        for folder in (physical, coverage, generation):
            folder.mkdir(parents=True)
        q = {"accepted_match_ids": ["KR_5"]}
        q_path = physical / "qualification.json"
        q_path.write_text(json.dumps(q), encoding="utf-8")
        split = {"qualification_sha256": reconcile.source.sha256(q_path), "assignment": {"KR_5": "train"}}
        split_path = coverage / "split_plan.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        cov = {"status": "coverage_complete", "quality_accepted_games": 1,
               "qualification_sha256": reconcile.source.sha256(q_path),
               "split_plan_sha256": reconcile.source.sha256(split_path)}
        cov_path = coverage / "coverage.json"
        cov_path.write_text(json.dumps(cov), encoding="utf-8")
        files = {}
        for source_path in (q_path, cov_path, split_path):
            target = generation / source_path.name
            target.write_bytes(source_path.read_bytes())
            files[source_path.name] = {"sha256": reconcile.source.sha256(target), "bytes": target.stat().st_size}
        # No test payload is provided: metadata verification must never try to read it.
        files["visits_test.jsonl"] = {"sha256": "0" * 64, "bytes": 999}
        manifest = {"state": "complete", "patch": "16.18", "purpose": "patch_development_1618", "files": files}
        (generation / "generation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(reconcile, "PHYSICAL_BASE", physical):
            assert reconcile.bound_physical_metadata(root, {}) == q
            q_path.write_text(json.dumps({"accepted_match_ids": ["KR_6"]}), encoding="utf-8")
            refused(lambda: reconcile.bound_physical_metadata(root, {}), "hash mismatch")


def main():
    test_reserved_precedes_all_payloads()
    test_partition_conflicts_and_omissions()
    test_multisets_and_actual_participant_pro_coverage()
    test_atomic_publication_and_refusal()
    test_inventory_receipt_is_anchored()
    print("ok dated reconciliation: reserved-first payload guards, complete partitions/conflicts, "
          "actual-pro membership, multiset counts, nonmutating coverage, immutable atomic output")


if __name__ == "__main__":
    main()

