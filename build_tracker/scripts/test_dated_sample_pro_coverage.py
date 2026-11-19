"""Synthetic pro-impact counts and reserved-first cache access; no real source reads."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import report_dated_sample_pro_coverage as review


def refused(function):
    try:
        function()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid source scope was accepted")


with patch.object(review.evidence, "sha256", side_effect=AssertionError("Reserved source opened")):
    refused(lambda: review.read_match(Path("synthetic"), "KR_4", "0" * 64, {"KR_4"}))
players = [{"puuid": "person-" + str(i)} for i in range(1, 11)]
payload = {"metadata": {"matchId": "KR_1"}, "info": {"participants": players}}
pros = {"Actual": {"puuid": "person-2"}, "Alias": {"puuid": "person-2"},
        "Absent": {"puuid": "not-present"}, "Unresolved": {"puuid": None}}
assert review.pro_names(payload, pros) == {"Actual", "Alias"}
observations = {"KR_1": {"pro_names": {"Actual"}}, "KR_2": {"pro_names": {"Actual", "Alias"}},
                "KR_3": {"error": "canonical_hash_mismatch"}}
result = review.tally({"KR_1", "KR_2", "KR_3", "KR_4"}, {"KR_1"}, {"KR_2", "KR_3"},
                      {"KR_4"}, observations, pros)
assert result["actual_pro_games_in_non_test_source"] == {"Actual": 2, "Alias": 1}
assert result["actual_pro_games_in_accepted"] == {"Actual": 1}
assert result["actual_pro_games_in_quarantined"] == {"Actual": 1, "Alias": 1}
assert result["unique_games_with_tracked_pro"] == {"non_test_source": 2, "accepted": 1, "quarantined": 1}
assert result["unknown_non_test_source_ids"] == {"KR_3": "canonical_hash_mismatch"}
partition = result["source_partition"]
assert partition["stored"] == partition["reserved_metadata_only"] + partition["verified_non_test"] + partition["unknown_non_test"]
assert partition["accepted_non_test"] == 1 and partition["quarantined_non_test"] == 2
refused(lambda: review.tally({"KR_99"}, set(), set(), set(), {}, pros))
with tempfile.TemporaryDirectory(prefix="dated-pro-synthetic-") as directory:
    root = Path(directory)
    path = root / "data/raw_match_v5/match/KR_1.json.gz"
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"schema": 1, "endpoint": "match", "match_id": "KR_1", "payload": payload}, handle)
    expected = review.evidence.canonical_sha(payload)
    actual, binding = review.read_match(root, "KR_1", expected, set())
    assert actual == payload and binding["canonical_sha256"] == expected
    refused(lambda: review.read_match(root, "KR_1", "0" * 64, set()))
    base = root / "review"
    base.mkdir()
    (base / "pro_coverage_review.json").write_text('{"immutable":true}', encoding="utf-8")
    with patch.object(review.evidence, "reserve_splits", side_effect=AssertionError("Existing report overwritten")):
        refused(lambda: review.run(base))
print("ok pro impact: reserved-first access, canonical identity, actual participants/aliases, "
      "unknown-source partition, accepted/quarantine counts, immutable output refusal")

