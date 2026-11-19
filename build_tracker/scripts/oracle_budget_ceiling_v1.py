"""Oracle-budget ceiling: what would a PERFECT wallet be worth to the policy?

Owner-requested diagnostic (2026-09-15): re-run the displayed policy giving it,
instead of the causal gold estimate, an ORACLE budget — the minimum wallet that
makes the recorded basket affordable, i.e. max(gold_est, exact recursive cost
of what the player actually bought). Everything else identical.

WHAT THIS IS AND IS NOT (pre-registered before the first run):
- The oracle is LABEL-DERIVED (it reads the purchase). It is a diagnostic
  ceiling for "how much is budget error costing us", measured on TRAIN only.
  It must never become a feature, a training input, or an evaluation
  reference — this is the exact pattern class behind the September
  retraction, used here deliberately and only as an upper bound.
- The ceiling is OPTIMISTIC in one more way: the true wallet was >= the
  basket cost, and a larger true wallet would also unlock expensive WRONG
  items; the minimal-sufficient oracle unlocks mostly the right one.
- Paired per visit: the second pass runs only when the oracle differs from
  gold_est, so identical rows contribute zero delta by construction.

Population: every TRAIN purchase decision of the v3 generation. No validation
or test reads, no ledger spend. Must run through footwear_runtime --entry.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

CHECKPOINT_EVERY = 5000
UNLIMITED = 10_000_000.0


def baskets(options):
    return [Counter(int(i['item_id']) for i in (o.get('items') or []))
            for o in options[:3] if o.get('kind') == 'buy']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    from app.predictor import Predictor
    from app.shop_econ import SAVE_ITEM
    from eval_policy import serving_row

    predictor = Predictor(args.artifact)
    if not predictor.config.get('gold_x'):
        raise ValueError('The diagnostic expects the causal-gold candidate configuration')
    dragon = dragon_for_patch('16.18')
    args.out.mkdir(parents=True, exist_ok=True)

    n = Counter()
    strata = {'budget_short': Counter(), 'feasible': Counter()}
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
                raise ValueError('TRAIN row without causal gold reached the diagnostic')
            action = tuple(sorted(actual.elements()))
            lifted = purchase_sequence(action, row, UNLIMITED, dragon)
            if lifted is None:
                n['legality_blocked_skipped'] += 1
                continue
            cost = float(sum(step['cost'] for step in lifted))
            oracle = max(float(budget), cost)
            stratum = 'budget_short' if cost > float(budget) else 'feasible'
            n['purchases'] += 1

            control = baskets(predictor.predict_options(
                serving_row(row, float(budget)), budget_slack=0, allow_save=True))
            c_exact = bool(control) and control[0] == actual
            c_any3 = any(b == actual for b in control)
            if oracle > float(budget):
                oracle_opts = baskets(predictor.predict_options(
                    serving_row(row, oracle), budget_slack=0, allow_save=True))
                o_exact = bool(oracle_opts) and oracle_opts[0] == actual
                o_any3 = any(b == actual for b in oracle_opts)
            else:
                o_exact, o_any3 = c_exact, c_any3

            for name, value in (('control_exact', c_exact), ('oracle_exact', o_exact),
                                ('control_any3', c_any3), ('oracle_any3', o_any3)):
                n[name] += int(value)
                strata[stratum][name] += int(value)
            strata[stratum]['n'] += 1
            if n['purchases'] % CHECKPOINT_EVERY == 0:
                print(f"{n['purchases']} purchases: exact {n['control_exact']}->{n['oracle_exact']} "
                      f"any3 {n['control_any3']}->{n['oracle_any3']}", flush=True)

    def rates(c):
        total = c.get('n', n['purchases'])
        return {k: c[k] / total for k in ('control_exact', 'oracle_exact',
                                          'control_any3', 'oracle_any3')} if total else None

    report = {'schema': 1, 'purpose': 'oracle_budget_ceiling_v1', 'split': 'train',
        'oracle_definition': 'max(gold_est, exact recursive cost of the recorded basket)',
        'label_derived_warning': ('DIAGNOSTIC CEILING ONLY: the oracle reads the label. Never a '
                                  'feature, never a training input, never an evaluation reference.'),
        'optimism_note': ('lower-bound wallet unlocks mostly the right item; the true wallet '
                          'would also unlock expensive wrong ones — the ceiling is optimistic'),
        'artifact': str(args.artifact), 'dataset': str(args.dataset),
        'purchases': n['purchases'], 'legality_blocked_skipped': n['legality_blocked_skipped'],
        'overall': rates(n),
        'by_stratum': {name: rates(c) for name, c in strata.items()},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_or_test_reads': False}
    from app.dataset_artifacts import write_json
    write_json(args.out / 'oracle_budget_ceiling.json', report)
    print(json.dumps({'purchases': n['purchases'], 'overall': report['overall'],
                      'budget_short_n': strata['budget_short'].get('n', 0)}, default=str), flush=True)


if __name__ == '__main__':
    main()
