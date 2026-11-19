"""Report nominal-gold value of unmatched copies in displayed shop baskets.

This reads a completed validation outcome stream; it never runs a model. Gold
weights are frozen Data Dragon full item values, not observed shop spend or
recipe-discounted incremental cost.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random

SAVE_ITEM = 999999
WARD_IDS = {2055, 772043}
COMPONENTS = ("shared_gold", "missing_gold", "extra_gold", "unmatched_gold", "abs_total_gold_gap")
SCOPES = ("all_decisions", "purchase", "no_buy_death", "ward_only", "mixed_ward_purchase")
BOOTSTRAP_REPS = 400
BOOTSTRAP_SEED = 16


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    def reject(value: str) -> None:
        raise ValueError("Nonstandard JSON number: " + value)

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def physical_counts(raw: dict, *, allow_save: bool = False) -> Counter[int]:
    if not isinstance(raw, dict):
        raise ValueError("Basket counts must be a JSON object")
    counts: Counter[int] = Counter()
    for key, count in raw.items():
        if not isinstance(key, str) or not key.isdecimal() or type(count) is not int or count <= 0:
            raise ValueError("Basket contains an invalid item copy count")
        item = int(key)
        if item == SAVE_ITEM:
            if not allow_save:
                raise ValueError("SAVE cannot appear in a purchase basket")
        elif item <= 0:
            raise ValueError("Basket contains an invalid physical item")
        counts[item] = count
    return counts


def nominal_value(counts: Counter[int], prices: dict[int, int]) -> int:
    missing = set(counts) - set(prices)
    if missing:
        raise ValueError("Frozen item catalogue lacks basket IDs: " + str(sorted(missing)))
    return sum(prices[item] * count for item, count in counts.items())


def gold_components(truth: Counter[int], predicted: Counter[int], prices: dict[int, int]) -> dict[str, int]:
    shared = nominal_value(truth & predicted, prices)
    missed = nominal_value(truth - predicted, prices)
    extra = nominal_value(predicted - truth, prices)
    actual_total = shared + missed
    predicted_total = shared + extra
    return {
        "shared_gold": shared,
        "missing_gold": missed,
        "extra_gold": extra,
        "unmatched_gold": missed + extra,
        "abs_total_gold_gap": abs(predicted_total - actual_total),
        "actual_nominal_gold": actual_total,
        "predicted_nominal_gold": predicted_total,
    }


def first_card_counts(policy: dict) -> Counter[int]:
    cards = policy.get("displayed_options")
    if not isinstance(cards, list) or len(cards) > 3:
        raise ValueError("The displayed-policy capture has an invalid card list")
    if not cards:
        return Counter()
    card = cards[0]
    if card.get("kind") not in ("buy", "save"):
        raise ValueError("Unknown lead action kind")
    counts = physical_counts(card.get("item_counts"))
    if card["kind"] == "save" and counts:
        raise ValueError("SAVE card contains physical items")
    sequence = card.get("purchase_sequence")
    if not isinstance(sequence, list) or Counter(item["item_id"] for item in sequence) != counts:
        raise ValueError("Lead card counts disagree with its purchase sequence")
    return counts


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def game_bootstrap(values: dict[str, list[int]], *, reps: int = BOOTSTRAP_REPS,
                   seed: int = BOOTSTRAP_SEED) -> list[float] | None:
    totals = {game: (sum(rows), len(rows)) for game, rows in values.items() if rows}
    games = sorted(totals)
    if not games:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(reps):
        total = count = 0
        for _ in games:
            game_total, game_count = totals[games[rng.randrange(len(games))]]
            total += game_total
            count += game_count
        samples.append(total / count)
    ordered = sorted(samples)
    return [ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))]]


def summarize(rows: dict[str, list[dict]]) -> dict:
    flattened = [row for entries in rows.values() for row in entries]
    if not flattened:
        return {"n": 0, "games": 0, "status": "not_measured"}
    result = {"n": len(flattened), "games": len(rows), "status": "measured", "metrics": {}}
    for name in COMPONENTS:
        per_game = {game: [row[name] for row in entries] for game, entries in rows.items()}
        values = [row[name] for row in flattened]
        record = {"mean": sum(values) / len(values), "ci95_game_bootstrap": game_bootstrap(per_game),
                  "unit": "nominal_gold"}
        if name == "unmatched_gold":
            record.update(median=percentile(values, 0.5), p90=percentile(values, 0.9))
        result["metrics"][name] = record
    shared = sum(row["shared_gold"] for row in flattened)
    actual = sum(row["actual_nominal_gold"] for row in flattened)
    predicted = sum(row["predicted_nominal_gold"] for row in flattened)
    result["nominal_gold_totals"] = {"shared": shared, "actual": actual, "predicted": predicted}
    result["gold_weighted_recall"] = shared / actual if actual else None
    result["gold_weighted_precision"] = shared / predicted if predicted else None
    return result


def compare(candidate: dict[str, list[dict]], control: dict[str, list[dict]]) -> dict:
    if set(candidate) != set(control) or any(len(candidate[game]) != len(control[game]) for game in candidate):
        raise ValueError("Paired gold comparison lost games or decisions")
    result = {}
    for name in ("unmatched_gold", "missing_gold", "extra_gold", "shared_gold"):
        differences = {game: [a[name] - b[name] for a, b in zip(candidate[game], control[game], strict=True)]
                       for game in candidate}
        flattened = [value for entries in differences.values() for value in entries]
        result[name] = {"mean_candidate_minus_control": sum(flattened) / len(flattened),
                        "ci95_game_bootstrap": game_bootstrap(differences), "n": len(flattened),
                        "unit": "nominal_gold"}
    return result


def report_gold(source_report: Path, static_review: Path, output: Path,
                *, candidate: str, control: str) -> dict:
    source_report, static_review, output = source_report.resolve(), static_review.resolve(), output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise ValueError("Preserve the existing gold-distance report or partial attempt")
    report = load_json(source_report)
    source_sha = digest(source_report)
    if report.get("split") != "val" or report.get("final_test_evaluation") is not False:
        raise ValueError("Gold-distance analysis is limited to completed development validation")
    binding = report["paired_outcomes"]
    outcomes = Path(binding["path"]).resolve()
    if outcomes.parent != source_report.parent or "_val" not in outcomes.name or "test" in outcomes.name:
        raise ValueError("Source report does not bind a validation outcome in its own run")
    if outcomes.stat().st_size != binding["bytes"] or digest(outcomes) != binding["sha256"]:
        raise ValueError("Outcome stream differs from the completed evaluation report")
    if candidate == control or {candidate, control} != set(report["metrics"]):
        raise ValueError("Choose the two policy arms from the bound evaluation")
    review = load_json(static_review)
    review_sha = digest(static_review)
    report_patch = report.get("patch") or (report.get("manifest") or {}).get("patch")
    if review.get("patch") != report_patch or report_patch != "16.18":
        raise ValueError("Wrong frozen item patch for this comparison")
    item_path = (static_review.parent / "item-16.18.1.json").resolve()
    if item_path.parent != static_review.parent or digest(item_path) != review["files"][item_path.name]:
        raise ValueError("Frozen item catalogue differs from its static review")
    items = load_json(item_path)["data"]
    prices = {int(key): int(data.get("gold", {}).get("total") or data.get("gold", {}).get("base") or 0)
              for key, data in items.items()}
    if any(price < 0 for price in prices.values()):
        raise ValueError("Negative nominal item gold")
    captures = {arm: {scope: defaultdict(list) for scope in SCOPES} for arm in (control, candidate)}
    seen, games, zero_valued, rows = set(), set(), set(), 0
    with outcomes.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("split") != "val" or set(row.get("policies", {})) != {control, candidate}:
                raise ValueError("Wrong split or policy arms in outcome stream")
            key = (row["match_id"], row["participant_id"], row["ts"])
            if key in seen:
                raise ValueError("Duplicate validation decision")
            seen.add(key)
            games.add(row["match_id"])
            recorded = physical_counts(row["actual_item_counts"], allow_save=True)
            save = row["save_kind"] == "no_buy_death"
            if (save and recorded != Counter({SAVE_ITEM: 1})) or (not save and SAVE_ITEM in recorded):
                raise ValueError("Recorded SAVE label differs from its decision type")
            truth = Counter({item: count for item, count in recorded.items() if item != SAVE_ITEM})
            scopes = ["all_decisions", "no_buy_death" if save else "purchase"]
            if not save:
                if not truth:
                    raise ValueError("Recorded purchase is empty")
                if row["save_kind"] == "ward_only":
                    if not set(truth) <= WARD_IDS:
                        raise ValueError("Ward-only label has a non-ward item")
                    scopes.append("ward_only")
                elif set(truth) & WARD_IDS:
                    scopes.append("mixed_ward_purchase")
            for arm in (control, candidate):
                policy = row["policies"][arm]
                prediction = first_card_counts(policy)
                lead = policy["displayed_options"][0] if policy["displayed_options"] else None
                expected_exact = (lead is not None and lead["kind"] == "save") if save else (
                    lead is not None and lead["kind"] == "buy" and prediction == truth)
                if policy["metrics"]["first_action_exact"] != float(expected_exact):
                    raise ValueError("Recorded exact-action outcome disagrees with the basket")
                zero_valued.update(item for item in truth | prediction if prices.get(item) == 0)
                score = gold_components(truth, prediction, prices)
                for scope in scopes:
                    captures[arm][scope][row["match_id"]].append(score)
            rows += 1
    if rows != report["visits"] or rows != binding["rows"] or len(games) != report["games"]:
        raise ValueError("Gold-distance report population differs from the completed evaluation")
    if (digest(outcomes) != binding["sha256"] or digest(item_path) != review["files"][item_path.name]
            or digest(source_report) != source_sha or digest(static_review) != review_sha):
        raise ValueError("A source changed during gold-distance analysis")
    result = {"schema": 1, "purpose": "gold_weighted_basket_difference",
              "definition": "sum of nominal gold values of missing plus extra physical item copies after multiset intersection",
              "formula": "sum_i price_i * (max(actual_i-predicted_i,0) + max(predicted_i-actual_i,0))",
              "price_basis": "frozen Data Dragon item gold.total, falling back to gold.base; nominal catalogue value, not observed spend or recipe-discounted incremental cost",
              "split": "val", "visits": rows, "games": len(games), "candidate": candidate, "control": control,
              "source_report": {"path": str(source_report), "sha256": source_sha},
              "source_outcomes": binding,
              "static_review": {"path": str(static_review), "sha256": review_sha},
              "static_items": {"path": str(item_path), "sha256": digest(item_path)},
              "zero_nominal_gold_item_ids_seen": sorted(zero_valued),
              "bootstrap": {"games": True, "reps": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
              "arms": {arm: {scope: summarize(captures[arm][scope]) for scope in SCOPES}
                       for arm in (control, candidate)},
              "paired_candidate_minus_control": {scope: compare(captures[candidate][scope], captures[control][scope])
                                                   for scope in SCOPES if captures[control][scope]},
              "test_payload_read": False, "model_loaded": False, "model_scored": False,
              "interpretation_limit": "Nominal item values can differ from actual shop spend because owned components, sales, undos and event timing affect marginal cost. This distance is a basket-overlap diagnostic, not a gold-estimator error or advice quality score."}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if output.exists():
        raise ValueError("Gold-distance report appeared during publication")
    partial.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--static-review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = report_gold(args.source_report, args.static_review, args.output,
                         candidate=args.candidate, control=args.control)
    print(json.dumps({"report": str(args.output.resolve()), "split": result["split"],
                      "visits": result["visits"], "games": result["games"],
                      "purchase_mean_unmatched_gold": {
                          arm: result["arms"][arm]["purchase"]["metrics"]["unmatched_gold"]["mean"]
                          for arm in (args.control, args.candidate)},
                      "paired_purchase_delta": result["paired_candidate_minus_control"]["purchase"]["unmatched_gold"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
