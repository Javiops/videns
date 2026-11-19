"""Split lead-item infeasibility into BUDGET vs LEGALITY, model-free.

`purchase_sequence(...) is None` conflates two causes with opposite remedies:
the wallet was too small under `gold_est`, or the purchase was illegal at the
reconstructed state (slots, ward stacks, boot uniqueness, blocks). The
lead-miss diagnostic could not tell them apart, so its headline
("67.6% infeasible") is not yet attributable.

This audit asks one question per recorded purchase, WITHOUT the model: is the
recorded lead item reachable from the reconstructed state?

  feasible_at_gold_est  — reachable within our causal budget; the mask allows
                          the model to display it, so a miss there is a
                          ranking problem.
  budget_short          — illegal only because of the wallet: reachable when
                          the budget is lifted. The player demonstrably bought
                          it, so this counts our budget UNDERSTATEMENT.
  legality_blocked      — unreachable even with unlimited gold: the
                          reconstructed inventory/slot state disagrees with
                          what the player did. An input-correctness defect,
                          not an economics one.

The recorded shortfall (`cost_at_unlimited_budget - gold_est`) is a LOWER
BOUND on how much our estimate understates the wallet, measured only on rows
where the player bought something we call unaffordable — a label-selected,
biased subset. It is a DIAGNOSTIC of estimator error. It must never become a
feature or an evaluation reference: deriving the wallet from the purchase is
the banned `gold_left + net_spend` pattern (docs/history.md).

TRAIN only. No model, no validation or test payloads, no ledger spend.
Must run through scripts/footwear_runtime.py --entry.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

CHECKPOINT_EVERY = 20000
UNLIMITED = 10_000_000.0
# v2 classification: the lead-only probe alone is unsound — a co-purchased
# combine can free the slot the lead needs (measured: ~10% of the v1
# `legality_blocked` headline was this artifact). The FULL recorded basket is
# consulted before blaming the state or the budget.
CLASSES = ('feasible_at_gold_est', 'budget_short',
           'lead_blocked_full_basket_legal', 'legality_blocked')


def minute_bucket(ts_ms) -> str:
    minute = float(ts_ms or 0) / 60000.0
    if minute < 10:
        return '00-09'
    if minute < 20:
        return '10-19'
    return '20+'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='smoke: stop after N rows')
    args = parser.parse_args()

    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    from app.shop_econ import SAVE_ITEM

    dragon = dragon_for_patch('16.18')
    args.out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    by_minute: dict[str, Counter] = {}
    by_depth: dict[str, Counter] = {}
    shortfalls: list[float] = []
    shortfall_by_depth: dict[str, list[float]] = {}
    examples: dict[str, list] = {name: [] for name in CLASSES}

    scanned = 0
    with (args.dataset / 'visits_train.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            scanned += 1
            if args.limit and scanned > args.limit:
                break
            row = json.loads(line)
            actual = Counter({int(k): int(v) for k, v in (row.get('label_counts') or {}).items()})
            actual.pop(SAVE_ITEM, None)
            if not actual:
                continue
            budget = row.get('gold_est')
            if row.get('gold_est_version') not in ('prequential-v2', 'prequential-v3') or budget is None:
                raise ValueError('TRAIN row without causal gold reached the audit')
            counts['purchases'] += 1
            lead = max(actual, key=lambda item: (actual[item], item))
            depth = str((dragon.classify(lead) or {}).get('depth') or 0)
            action = tuple(sorted(actual.elements()))
            if (purchase_sequence((lead,), row, float(budget), dragon) is not None
                    or purchase_sequence(action, row, float(budget), dragon) is not None):
                name, shortfall = 'feasible_at_gold_est', None
            else:
                lifted = purchase_sequence((lead,), row, UNLIMITED, dragon)
                if lifted is not None:
                    name = 'budget_short'
                    shortfall = float(lifted[0]['cost']) - float(budget)
                    shortfalls.append(shortfall)
                    shortfall_by_depth.setdefault(depth, []).append(shortfall)
                elif purchase_sequence(action, row, UNLIMITED, dragon) is not None:
                    # The state admits the recorded basket; only the lead-only
                    # probe fails (a combine frees the slot). Not a data defect.
                    name, shortfall = 'lead_blocked_full_basket_legal', None
                else:
                    name, shortfall = 'legality_blocked', None
            counts[name] += 1
            by_minute.setdefault(name, Counter())[minute_bucket(row.get('ts'))] += 1
            by_depth.setdefault(name, Counter())[depth] += 1
            if len(examples[name]) < 15:
                examples[name].append({
                    'match_id': row.get('match_id'), 'participant_id': row.get('participant_id'),
                    'ts': row.get('ts'), 'champion': row.get('champion'), 'role': row.get('role'),
                    'gold_est': budget, 'lead': lead, 'lead_name': dragon.item_name(lead),
                    'lead_depth': depth, 'shortfall_gold': shortfall,
                    'inventory': list(row.get('inventory') or []),
                    'label_counts': dict(sorted(actual.items()))})
            if counts['purchases'] % CHECKPOINT_EVERY == 0:
                print(f"audit {scanned} rows: {dict(counts)}", flush=True)

    def quantiles(values: list[float]) -> dict | None:
        if not values:
            return None
        ordered = sorted(values)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        return {'n': len(ordered), 'median': pick(0.5), 'p25': pick(0.25), 'p75': pick(0.75),
                'p90': pick(0.90), 'mean': sum(ordered) / len(ordered),
                'max': ordered[-1], 'min': ordered[0]}

    purchases = counts['purchases']
    report = {'schema': 2, 'purpose': 'lead_feasibility_audit_v2',
        'parent_diagnostic': 'data/reports/lead_miss_diagnostic_v1/lead_miss_diagnostic.json',
        'split': 'train', 'model_used': False, 'rows_scanned': scanned,
        'purchase_decisions': purchases,
        'class_counts': {name: counts[name] for name in CLASSES},
        'class_shares_of_purchases': {name: (counts[name] / purchases if purchases else None)
                                      for name in CLASSES},
        'by_minute': {k: dict(sorted(v.items())) for k, v in by_minute.items()},
        'by_lead_depth': {k: dict(sorted(v.items())) for k, v in by_depth.items()},
        'budget_shortfall_gold': quantiles(shortfalls),
        'budget_shortfall_by_lead_depth': {k: quantiles(v) for k, v in sorted(shortfall_by_depth.items())},
        'shortfall_interpretation': (
            'LOWER BOUND on gold_est understatement, measured only where the player bought a lead '
            'we called unaffordable: a label-selected, biased subset. Diagnostic of estimator error '
            'only. Never a feature and never an evaluation reference (docs/history.md).'),
        'legality_interpretation': (
            'legality_blocked means the reconstructed inventory/slot state forbids a purchase the '
            'player actually made: an input-correctness defect, not an economics one.'),
        'examples': examples, 'dataset': str(args.dataset),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_or_test_reads': False}
    from app.dataset_artifacts import write_json
    out_path = args.out / 'lead_feasibility_audit.json'
    write_json(out_path, report)
    print(json.dumps({'purchases': purchases,
                      'class_counts': report['class_counts'],
                      'shortfall': report['budget_shortfall_gold'],
                      'report': str(out_path)}, default=str), flush=True)


if __name__ == '__main__':
    main()
