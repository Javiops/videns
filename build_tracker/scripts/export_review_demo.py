"""Build a public Faker review fixture from Riot source and frozen TRAIN only.

Outputs JSON on stdout; callers decide where to publish it. This is a
descriptive nearest-neighbour sample, not a promoted model or optimal advice.
No validation/test payload is opened.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DATA_DIR  # noqa: E402
from app.ddragon import dragon_for_patch  # noqa: E402
from app.review import compile_review  # noqa: E402
from app.shop_econ import try_purchase  # noqa: E402

MATCH_ID = "KR_8386011226"
FAKER_PID = 3
GENERATION = DATA_DIR / "dataset_generations" / "16.18-reconstructible-development-v3"
REPORT_NAME = "review-faker-evidence.json"


def _cohort_rows() -> tuple[list[dict], dict]:
    manifest = json.loads((GENERATION / "split_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("patch") != "16.18" or manifest.get("purpose") != "patch_development_1618"
            or manifest.get("rows_per_split", {}).get("test") != 0
            or manifest.get("export_fingerprint") != "e4ea64319e0566bdfdb775175c9d0fdf71f26b93d07ed4672c42c0d62a0ba1e0"
            or manifest.get("gold_est_versions") != {"prequential-v3": 489156}):
        raise ValueError("Unexpected development generation; refusing to construct review cohort")
    rows = []
    with (GENERATION / "visits_train.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (row.get("champion_id") == 69 and row.get("role") == "MIDDLE"
                    and row.get("patch") == "16.18"
                    and row.get("gold_est_version") == "prequential-v3"
                    and row.get("match_id") != MATCH_ID
                    and row.get("gold_est") is not None):
                rows.append(row)
    return rows, manifest


def _peer_comparison(decision: dict, rows: list[dict], dragon) -> dict:
    gold = decision.get("gold_est")
    level = decision.get("level")
    if gold is None or level is None:
        return {"status": "unavailable", "reason": "No pre-decision gold or level estimate."}
    full_inventory = decision.get("inventory") or []
    held = Counter(item for item in full_inventory if not dragon.classify(item)["skip"])
    # TRAIN inventories omit trinkets/consumables. Match that representation for
    # similarity, but still reserve physical slots occupied by potions.
    omitted = Counter(item for item in full_inventory if dragon.classify(item)["skip"]
                      and "Trinket" not in (dragon.item(item) or {}).get("tags", [])
                      and dragon.gold_block(item)["total"] > 0)
    occupied_omitted_slots = sum(
        (count + max(1, int((dragon.item(item) or {}).get("stacks") or 1)) - 1)
        // max(1, int((dragon.item(item) or {}).get("stacks") or 1))
        for item, count in omitted.items()
    )
    bands = [(180_000, 2, 300), (300_000, 3, 500)]
    eligible = []
    selected_band = None
    for time_band, level_band, gold_band in bands:
        eligible = [row for row in rows
                    if abs(int(row["ts"]) - decision["ts"]) <= time_band
                    and abs(int(row.get("level") or 0) - int(level)) <= level_band
                    and abs(int(row["gold_est"]) - int(gold)) <= gold_band]
        selected_band = {"time_ms": time_band, "level": level_band, "gold_est": gold_band}
        if len(eligible) >= 30:
            break
    if len(eligible) < 30:
        return {"status": "insufficient", "candidate_n": len(eligible), "band": selected_band,
                "reason": "Fewer than 30 same-champion/role shops in the declared state band."}

    def distance(row):
        other = Counter(int(i) for i in row.get("inventory") or [])
        union = sum((held | other).values())
        overlap = sum((held & other).values())
        inventory_gap = 0 if not union else 1 - overlap / union
        return (abs(row["ts"] - decision["ts"]) / 180_000
                + abs(row["level"] - level) / 2
                + abs(row["gold_est"] - gold) / 300
                + inventory_gap * 2)

    nearest = sorted(eligible, key=distance)[:80]
    counts = Counter()
    for row in nearest:
        labels = row.get("label_counts") or {}
        for raw_id, count in labels.items():
            item = int(raw_id)
            if count and item not in {2055, 772043} and not dragon.classify(item)["target_skip"]:
                counts[item] += 1  # visits containing item, not item copies
    legal = []
    for item, count in counts.most_common():
        purchase = try_purchase(item, list(held.elements()), gold, dragon,
                                role=decision.get("role") or "",
                                bot_quest_complete=decision.get("bot_quest_complete"),
                                unmodeled_regular_slots=occupied_omitted_slots)
        if purchase is not None:
            cost, _after = purchase
            legal.append({"item_id": item, "item_name": dragon.item_name(item),
                          "visits": count, "estimated_cost": cost})
        if len(legal) == 3:
            break
    if not legal:
        return {"status": "unavailable", "candidate_n": len(eligible), "band": selected_band,
                "reason": "No observed cohort item fits the estimated budget."}
    return {"status": "descriptive", "n": len(nearest), "candidate_n": len(eligible),
            "band": selected_band, "metric": "visits containing item", "split": "train",
            "report": REPORT_NAME, "top_items": legal,
            "limitations": "Estimated gold; nearest observed purchases, not a validated policy or optimal build."}


def build() -> tuple[dict, dict]:
    review = compile_review(MATCH_ID, "kr", FAKER_PID)
    if review["decisions_status"] != "reconstructed":
        raise ValueError("Faker decisions did not reconstruct")
    decisions = [d for d in review["decisions"] if d["participant_id"] == FAKER_PID]
    rows, manifest = _cohort_rows()
    dragon = dragon_for_patch("16.18")
    for decision in decisions:
        decision["comparison"] = _peer_comparison(decision, rows, dragon)
        for item in decision["comparison"].get("top_items", []):
            review["item_names"][str(item["item_id"])] = item["item_name"]
    review["decisions"] = decisions
    review["shops"] = []  # the event rail uses grouped decision visits, not raw receipts
    review["build_comparison"] = {"status": "descriptive_train_sample", "report": REPORT_NAME}
    review["demo_mode"] = "frozen_public_match"
    report = {
        "schema": 1, "match_id": MATCH_ID, "focus_participant_id": FAKER_PID,
        "metric": "visits containing an item in a nearest state-matched cohort",
        "split": "train", "train_cohort_rows": len(rows),
        "generation": manifest["benchmark_id"], "generation_fingerprint": manifest["export_fingerprint"],
        "generation_purpose": manifest["purpose"], "patch": manifest["patch"],
        "source": "Riot Match-V5 match and timeline; frozen development TRAIN visits",
        "filters": "same champion (Cassiopeia), role (MIDDLE), patch 16.18; time/level/gold-estimate bands shown per decision; 80 nearest maximum",
        "selection_features": ["pre-decision timestamp", "level", "prequential-v3 estimated gold", "held inventory"],
        "display_filter": "Recipe cost, duplicate/boot restrictions and free inventory slots; skipped consumables reserve their physical slots.",
        "excluded_splits": ["validation", "test", "post_cutoff"],
        "limitations": [
            "Development/probe corpus, not a promoted serving policy.",
            "Historical behavior does not establish a correct or optimal purchase.",
            "The target wallet is estimated from pre-decision observations, not exact.",
            "Cohort visits are not independent players; a player may contribute multiple shops.",
        ],
        "decisions": [
            {"ts": d["ts"], "observed_bought": d["bought"],
             "modeled_bought": d["modeled_bought"], "gold_est": d["gold_est"],
             "level": d["level"], "comparison": d["comparison"]}
            for d in decisions
        ],
    }
    return review, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    review, report = build()
    if args.summary:
        print(json.dumps({"match": MATCH_ID, "decisions": len(review["decisions"]),
                          "cohort_rows": report["train_cohort_rows"],
                          "comparisons": [(d["ts"], d["comparison"]["status"],
                                           d["comparison"].get("n"),
                                           [i["item_name"] for i in d["comparison"].get("top_items", [])])
                                          for d in review["decisions"]]}, ensure_ascii=False))
    else:
        raw = json.dumps(review, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        print(json.dumps({"fixture_gzip_base64": base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii"),
                          "fixture_bytes": len(raw), "report": report}, separators=(",", ":"), ensure_ascii=False))
