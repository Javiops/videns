"""Minimal tests for the versioned basket probes (scripts/basket_*_probe.py).

Pure-synthetic checks of the load-bearing definitions (basket key, sampling
rule, tie-break, fit/probe boundary), plus one reproduction check against the
frozen evidence sample when it is present on this machine.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from basket_vocabulary_probe import (basket_entries, basket_key, inv_key,
                                     membership_digest, quantile, sampled,
                                     split_fit_probe, top_k)

FROZEN = ROOT / "data/reports/strategy_analysis_20260920"


def test_basket_key_sorts_filters_and_flags_save():
    entries = basket_entries({"3006": 1, "1055": 2, "0": 5})
    assert entries == [(1055, 2), (3006, 1)]
    assert basket_key(entries) == "1055:2,3006:1"
    assert basket_entries({}) == []
    assert any(i == 999999 for i, _ in basket_entries({"999999": 1}))


def test_inv_key_is_a_sorted_multiset():
    assert inv_key([3006, 1055, 1055]) == "1055:2,3006:1"
    assert inv_key([]) == ""


def test_top_k_breaks_ties_by_count_desc_then_key_asc():
    from collections import Counter
    c = Counter({"b": 2, "a": 2, "c": 5})
    assert top_k(c, 2) == ["c", "a"]
    assert top_k(None, 4) == []


def test_membership_digest_encoding():
    # one ID per LF-terminated line, UTF-8 — the binding contract
    import hashlib
    assert membership_digest(["g1", "g2"]) == hashlib.sha256(
        b"g1\ng2\n").hexdigest()


def test_split_boundary_keeps_timestamp_ties_together():
    # 10 games; the floor(0.8*10)=8th game shares creation with the 7th:
    # both must land on the PROBE side (fit uses strict <).
    creation = {f"g{i}": 100 + i for i in range(10)}
    creation["g8"] = creation["g7"]
    plan = {"assignment": {g: "train" for g in creation},
            "game_creation_ms": creation}
    fit, probe, boundary = split_fit_probe(plan)
    assert boundary == creation["g7"]
    assert "g7" in probe and "g8" in probe and "g6" in fit
    assert len(fit) + len(probe) == 10


def test_quantile_uses_floor_index():
    assert quantile([1, 2, 3, 4], 0.5) == 2
    assert quantile([7], 0.9) == 7


def test_sampling_rule_matches_frozen_sample():
    sample = FROZEN / "basket_probe_legality_sample_train.jsonl"
    if not sample.exists():
        return  # frozen evidence not on this machine; synthetic tests stand
    with sample.open(encoding="utf-8") as fh:
        for _ in range(5):
            row = json.loads(fh.readline())["row"]
            assert sampled(row["match_id"], row["participant_id"], row["ts"]), \
                "frozen sample row fails the ported sampling rule"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
