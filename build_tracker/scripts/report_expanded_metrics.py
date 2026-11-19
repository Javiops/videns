"""Expanded descriptive metrics over a completed paired policy evaluation.

Reads the immutable per-visit paired outcomes of a finished displayed-policy
comparison and computes additional diagnostic views WITHOUT re-running any
model: per-slice paired game-bootstrap intervals, head-to-head agreement
partitions, rank-of-truth among displayed cards, quantity over/under detail,
save-card precision per display tier, item-level confusions and per-game
dispersion. Before adding anything, it recomputes every global metric in the
bound parent report from the outcomes and aborts on any mismatch, so the new
report is provably a re-derivation of the same evidence.

This is exploratory development evidence on reused validation. It defines no
win rule, tunes nothing, reads no test payload and cannot support promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUN = ROOT / "data" / "experiments" / "16.18-reconstructible-development-v1"
DEFAULT_OUTCOMES = DEFAULT_RUN / "policy_outcomes_val_candidate_vs_conditional.jsonl"
DEFAULT_REPORT = DEFAULT_RUN / "policy_eval_val_candidate_vs_conditional_policy.json"

PRIMARIES = ("first_action_exact", "action_family_match")
POLICIES = ("A", "B")

SAVE_ITEM = 999999  # sentinel labelling a recorded post-death no-buy

BUDGET_BANDS = ((0, 500, "<500 gold"), (500, 1000, "500-999 gold"),
                (1000, 2000, "1000-1999 gold"), (2000, float("inf"), "2000+ gold"))
TIME_BANDS = ((0, 10, "0-9 minutes"), (10, 20, "10-19 minutes"),
              (20, 30, "20-29 minutes"), (30, float("inf"), "30+ minutes"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def band(value: float, bands) -> str:
    for low, high, name in bands:
        if low <= value < high:
            return name
    return bands[-1][2]


def _ci(samples: list[float]) -> list[float] | None:
    ordered = sorted(samples)
    if not ordered:
        return None
    return [ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))]]


def paired_slice_diff(rows: list[dict], metric: str, reps: int, seed: int) -> dict:
    """Paired A-B game bootstrap on one slice, mirroring eval_policy.paired_diff.

    Games are resampled, visits inside a game stay together, and both policies
    see the identical resample, so the interval describes the difference the
    slice actually shows rather than two overlapping independent intervals.
    """
    per_game: dict[str, dict[str, list[float]]] = {"A": defaultdict(list), "B": defaultdict(list)}
    for row in rows:
        for pol in POLICIES:
            value = row["policies"][pol]["metrics"].get(metric)
            if value is not None:
                per_game[pol][row["match_id"]].append(value)
    n_a = sum(len(v) for v in per_game["A"].values())
    n_b = sum(len(v) for v in per_game["B"].values())
    shared = sorted(set(per_game["A"]) | set(per_game["B"]))
    out = {
        "metric": metric,
        "n_games": len(shared),
        "n_a": n_a,
        "n_b": n_b,
        "a": (sum(sum(v) for v in per_game["A"].values()) / n_a) if n_a else None,
        "b": (sum(sum(v) for v in per_game["B"].values()) / n_b) if n_b else None,
        "diff": None,
        "ci95": None,
        "status": "measured" if n_a and n_b else "not_measured",
    }
    if not (n_a and n_b):
        return out
    out["diff"] = out["a"] - out["b"]
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        ta = tb = ca = cb = 0.0
        for _ in range(len(shared)):
            game = shared[rng.randrange(len(shared))]
            values_a = per_game["A"].get(game, ())
            values_b = per_game["B"].get(game, ())
            ta += sum(values_a); ca += len(values_a)
            tb += sum(values_b); cb += len(values_b)
        if ca and cb:
            diffs.append(ta / ca - tb / cb)
    out["ci95"] = _ci(diffs)
    out["significant"] = bool(out["ci95"] and (out["ci95"][0] > 0 or out["ci95"][1] < 0))
    return out


def multiset(counts: dict) -> Counter:
    return Counter({int(item): int(count) for item, count in (counts or {}).items()})


def lead_option(policy: dict) -> dict | None:
    options = policy["displayed_options"]
    return options[0] if options else None


def truth_rank(policy: dict, actual: Counter, is_purchase: bool) -> str:
    """Position of the recorded action among the displayed cards, or 'absent'."""
    for index, option in enumerate(policy["displayed_options"][:3]):
        if is_purchase:
            if option["kind"] == "buy" and multiset(option["item_counts"]) == actual:
                return str(index + 1)
        elif option["kind"] == "save":
            return str(index + 1)
    return "absent"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "data" / "reports" / "expanded_metrics_16_18_dev_v1")
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=16)
    parser.add_argument("--policy-a", default="A",
                        help="outcome-file policy key reported as A (the candidate)")
    parser.add_argument("--policy-b", default="B",
                        help="outcome-file policy key reported as B (the control)")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows: list[dict] = []
    with args.outcomes.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            # Normalize arbitrary policy names onto the A/B layout used below.
            row["policies"] = {"A": row["policies"][args.policy_a],
                               "B": row["policies"][args.policy_b]}
            rows.append(row)
    report_metric_keys = {"A": args.policy_a, "B": args.policy_b}

    games = sorted({row["match_id"] for row in rows})
    expected_visits = report["visits"]
    expected_games = report["games"]
    if len(rows) != expected_visits or len(games) != expected_games:
        sys.exit(f"outcomes do not match the bound report: {len(rows)} visits / "
                 f"{len(games)} games vs {expected_visits} / {expected_games}")

    # --- Gate: every global metric in the parent report must be reproduced ---
    verification = {}
    for pol in POLICIES:
        for metric, entry in report["metrics"][report_metric_keys[pol]].items():
            if not isinstance(entry, dict) or "value" not in entry:
                continue
            total = n = 0.0
            for row in rows:
                value = row["policies"][pol]["metrics"].get(metric)
                if value is not None:
                    total += value
                    n += row["policies"][pol]["metric_n"].get(metric, 1)
            recomputed = total / n if n else None
            match = (recomputed is not None and entry["value"] is not None
                     and abs(recomputed - entry["value"]) < 1e-9 and int(n) == entry["n"])
            verification[f"{pol}.{metric}"] = {
                "report_value": entry["value"], "report_n": entry["n"],
                "recomputed_value": recomputed, "recomputed_n": int(n), "match": match,
            }
            if not match:
                sys.exit(f"recomputed {pol}.{metric} does not reproduce the bound report: "
                         f"{recomputed} (n={int(n)}) vs {entry['value']} (n={entry['n']})")

    # --- Per-visit derived facts used by every new view ---
    facts = []
    for row in rows:
        actual = multiset(row["actual_item_counts"])
        actual.pop(SAVE_ITEM, None)  # the sentinel is not a physical item
        is_purchase = bool(actual)
        fact = {"row": row, "actual": actual, "is_purchase": is_purchase,
                "slices": {
                    "budget": band(row["budget"], BUDGET_BANDS),
                    "time": band(row["minute"] if "minute" in row else row["ts"] // 60000,
                                 TIME_BANDS),
                    "role": row["role"],
                    "save_kind": row["save_kind"],
                    "truth_basket_quantity": str(sum(actual.values())),
                }}
        for pol in POLICIES:
            policy = row["policies"][pol]
            lead = lead_option(policy)
            lead_counts = multiset(lead["item_counts"]) if lead and lead["kind"] == "buy" else Counter()
            over = sum((lead_counts - actual).values())
            under = sum((actual - lead_counts).values())
            fact[pol] = {
                "rank": truth_rank(policy, actual, is_purchase),
                "lead_kind": lead["kind"] if lead else "none",
                "lead_counts": lead_counts,
                "excess_copies": over,
                "missing_copies": under,
                "same_identity": set(lead_counts) == set(actual) and bool(actual),
                "n_options": len(policy["displayed_options"]),
                "save_positions": [i + 1 for i, o in enumerate(policy["displayed_options"][:3])
                                   if o["kind"] == "save"],
            }
        facts.append(fact)

    # --- 1. Slice metrics with paired CIs (the analysis run had none) ---
    slices = []
    for dimension in ("budget", "time", "role", "save_kind", "truth_basket_quantity"):
        for value in sorted({f["slices"][dimension] for f in facts}):
            member_rows = [f["row"] for f in facts if f["slices"][dimension] == value]
            entry = {"dimension": dimension, "value": value, "visits": len(member_rows),
                     "games": len({r["match_id"] for r in member_rows}), "metrics": {}}
            for metric in PRIMARIES:
                entry["metrics"][metric] = paired_slice_diff(member_rows, metric, args.reps, args.seed)
            slices.append(entry)

    # --- 2. Head-to-head partition on first_action_exact ---
    def h2h(rows_subset: list[dict]) -> dict:
        table = Counter()
        for row in rows_subset:
            a = row["policies"]["A"]["metrics"]["first_action_exact"]
            b = row["policies"]["B"]["metrics"]["first_action_exact"]
            table[("A" if a else "a") + ("B" if b else "b")] += 1
        return {"both_correct": table["AB"], "only_transformer": table["Ab"],
                "only_conditional": table["aB"], "both_wrong": table["ab"],
                "n": sum(table.values())}

    head_to_head = {"global": h2h(rows)}
    for dimension in ("save_kind", "budget", "time"):
        head_to_head[dimension] = {
            value: h2h([f["row"] for f in facts if f["slices"][dimension] == value])
            for value in sorted({f["slices"][dimension] for f in facts})}

    # --- 3. Rank of the recorded action among displayed cards ---
    rank_of_truth = {}
    for pol in POLICIES:
        purchase_ranks = Counter(f[pol]["rank"] for f in facts if f["is_purchase"])
        no_buy_ranks = Counter(f[pol]["rank"] for f in facts if not f["is_purchase"])
        rank_of_truth[pol] = {
            "purchases": {"n": sum(purchase_ranks.values()), **purchase_ranks},
            "no_buys_save_card_position": {"n": sum(no_buy_ranks.values()), **no_buy_ranks},
        }

    # --- 4. Quantity detail: identity right, copies wrong; corpus copy totals ---
    quantity = {}
    for pol in POLICIES:
        purchases = [f for f in facts if f["is_purchase"]]
        same_identity = [f for f in purchases if f[pol]["same_identity"]]
        wrong_copies = [f for f in same_identity if f[pol]["lead_counts"] != f["actual"]]
        quantity[pol] = {
            "purchases_n": len(purchases),
            "lead_same_identity_set": len(same_identity),
            "same_identity_wrong_copies": len(wrong_copies),
            "wrong_copies_overbuy": sum(1 for f in wrong_copies if f[pol]["excess_copies"] > 0),
            "wrong_copies_underbuy": sum(1 for f in wrong_copies if f[pol]["missing_copies"] > 0),
            "total_excess_copies_all_purchases": sum(f[pol]["excess_copies"] for f in purchases),
            "total_missing_copies_all_purchases": sum(f[pol]["missing_copies"] for f in purchases),
        }

    # --- 5. Save card by display tier (HANDOFF flagged this as unmeasured) ---
    save_by_tier = {}
    for pol in POLICIES:
        tiers = {}
        for position in (1, 2, 3):
            shown = [f for f in facts if position in f[pol]["save_positions"]]
            correct = sum(1 for f in shown if f["row"]["save_kind"] == "no_buy_death")
            ward_only = sum(1 for f in shown if f["row"]["save_kind"] == "ward_only")
            tiers[f"position_{position}"] = {
                "shown_n": len(shown),
                "recorded_no_buy_death": correct,
                "recorded_ward_only": ward_only,
                "precision_vs_no_buy_death": correct / len(shown) if shown else None,
            }
        save_by_tier[pol] = tiers

    # --- 6. Item-level views (single-copy lead identities) ---
    item_views = {}
    actual_lead_freq = Counter()
    for f in facts:
        if f["is_purchase"]:
            actual_lead_freq[max(f["actual"], key=lambda i: (f["actual"][i], i))] += 1
    top_items = [item for item, _ in actual_lead_freq.most_common(25)]
    for pol in POLICIES:
        per_item = {}
        for item in top_items:
            containing = [f for f in facts if f["actual"].get(item)]
            hit = sum(1 for f in containing if f[pol]["lead_counts"].get(item, 0) >= f["actual"][item])
            per_item[str(item)] = {"actual_visits": len(containing),
                                   "lead_covers_copies": hit,
                                   "copy_recall": hit / len(containing) if containing else None}
        confusions = Counter()
        for f in facts:
            if f["is_purchase"] and not f["row"]["policies"][pol]["metrics"]["first_action_exact"]:
                if len(f["actual"]) == 1 and len(f[pol]["lead_counts"]) == 1:
                    src = next(iter(f["actual"]))
                    dst = next(iter(f[pol]["lead_counts"]))
                    if src != dst:
                        confusions[f"{src}->{dst}"] += 1
        item_views[pol] = {"top_item_copy_recall": per_item,
                           "top_single_item_confusions": dict(confusions.most_common(25))}

    # --- 7. Per-game dispersion of first_action_exact ---
    dispersion = {}
    for pol in POLICIES:
        per_game = defaultdict(list)
        for row in rows:
            per_game[row["match_id"]].append(row["policies"][pol]["metrics"]["first_action_exact"])
        means = sorted((sum(v) / len(v), len(v), g) for g, v in per_game.items())
        quantile = lambda q: means[int(q * (len(means) - 1))][0]
        dispersion[pol] = {
            "games": len(means),
            "min": means[0][0], "p10": quantile(0.10), "p25": quantile(0.25),
            "median": quantile(0.50), "p75": quantile(0.75), "p90": quantile(0.90),
            "max": means[-1][0],
            "worst_5_games": [{"match_id": g, "visits": n, "first_action_exact": m}
                              for m, n, g in means[:5]],
        }

    # --- 8. Displayed-option statistics ---
    option_stats = {}
    for pol in POLICIES:
        counts = Counter(f[pol]["n_options"] for f in facts)
        option_stats[pol] = {
            "options_shown_distribution": {str(k): counts[k] for k in sorted(counts)},
            "save_card_shown_any_position": sum(1 for f in facts if f[pol]["save_positions"]),
            "save_card_shown_as_lead": sum(1 for f in facts if 1 in f[pol]["save_positions"]),
        }

    out = {
        "schema_version": 1,
        "purpose": "exploratory_expanded_metrics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split": report.get("split") or rows[0]["split"],
        "games": expected_games,
        "visits": expected_visits,
        "policies": {"A": report_metric_keys["A"], "B": report_metric_keys["B"]},
        "provenance": {
            "source_outcomes": str(args.outcomes),
            "source_outcomes_sha256": sha256_file(args.outcomes),
            "source_report": str(args.report),
            "source_report_sha256": sha256_file(args.report),
            "artifact_sha256": report.get("artifact_sha256"),
            "export_fingerprint": (report.get("manifest", {}).get("export_fingerprint")
                                   or report.get("export_fingerprint")),
            "bootstrap": {"reps": args.reps, "seed": args.seed, "resample_unit": "game"},
        },
        "report_reproduction_gate": verification,
        "slices_with_paired_ci": slices,
        "head_to_head_first_action_exact": head_to_head,
        "rank_of_truth_in_displayed_cards": rank_of_truth,
        "quantity_detail": quantity,
        "save_card_by_tier": save_by_tier,
        "item_level": item_views,
        "per_game_dispersion_first_action_exact": dispersion,
        "option_stats": option_stats,
        "limits": [
            "Reused development validation; not final-test, live or release evidence.",
            "All views are descriptive; no threshold, gate or loss may be tuned on them "
            "without a new predeclared experiment on fresh validation.",
            "rank_of_truth is oracle selection headroom only; a missing basket cannot be "
            "rescued by reranking, and rescuing one decision can break another.",
            "save_card_by_tier precision uses recorded no_buy_death as the positive class; "
            "ward_only visits are listed separately, not counted as save hits.",
            "Item confusions cover single-item leads on single-item targets only.",
        ],
        "promotion_allowed": False,
        "final_test_read": False,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "expanded_metrics_val.json"
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"reproduction gate: {len(verification)} metrics matched the bound report")


if __name__ == "__main__":
    main()
