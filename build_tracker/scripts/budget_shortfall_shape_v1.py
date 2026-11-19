"""Raw shape of the budget shortfall — the companion to gold_error_shape_v1.

TRAIN only. Model-free. No ledger spend.

audit_lead_feasibility_v1 reports five quantiles of the lead-item shortfall.
Two things are needed here that it does not emit:

  * the RAW shortfalls, so "what would a mask slack of delta unmask" can be
    drawn as a curve rather than interpolated between quartiles;
  * the FULL-BASKET shortfall as well as the lead-only one. The oracle-ceiling
    experiment defined its affected mass by the whole recorded basket (~26.7%
    of purchases) while the feasibility audit uses the lead item (13.85%).
    Both are computed here from the same rows so the two headline numbers can
    finally be read on one axis.

Classification calls the same `purchase_sequence` the audit and the displayed
policy use; nothing is re-implemented.

The shortfall is LABEL-SELECTED (only rows where the player bought something we
called unaffordable) and is a LOWER BOUND on the estimator's understatement.
Diagnostic only: deriving a wallet from the purchase is the banned
`gold_left + net_spend` pattern.
"""
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

UNLIMITED = 10_000_000.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--patch', default='16.18')
    ap.add_argument('--static-review', required=True,
                    help='static_review.json of the acceptance run that built this generation')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    # The v3 rows were reconstructed under the opt-in 16.18 economics runtime;
    # purchase_sequence refuses footwear-aware rows unless it is active, and the
    # declaration is checked against the generation manifest so this scan cannot
    # silently price rows under different economics than produced them.
    from reconstructible_1618_runtime import activate, declaration
    manifest = json.loads((args.dataset / 'generation_manifest.json').read_text(encoding='utf-8'))
    activate(Path(args.static_review))
    if declaration() != manifest.get('footwear_runtime'):
        raise ValueError('Runtime declaration does not match the generation manifest')

    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    from app.shop_econ import SAVE_ITEM

    dragon = dragon_for_patch(args.patch)
    args.out.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    lead_short = array('f')       # lead item unaffordable at gold_est
    basket_short = array('f')     # whole recorded basket unaffordable at gold_est
    gold_at_short = array('f')    # the wallet we believed on those rows

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
                raise ValueError('TRAIN row without causal gold reached the shortfall scan')
            counts['purchases'] += 1
            lead = max(actual, key=lambda item: (actual[item], item))
            action = tuple(sorted(actual.elements()))

            lead_ok = purchase_sequence((lead,), row, float(budget), dragon) is not None
            basket_ok = purchase_sequence(action, row, float(budget), dragon) is not None
            if lead_ok or basket_ok:
                counts['feasible_at_gold_est'] += 1
            if not (lead_ok or basket_ok):
                lifted = purchase_sequence((lead,), row, UNLIMITED, dragon)
                if lifted is not None:
                    counts['budget_short_lead'] += 1
                    lead_short.append(float(lifted[0]['cost']) - float(budget))
                    gold_at_short.append(float(budget))

            # Full-basket view, independent of the lead classification: what the
            # whole recorded multiset costs from this state, at any price.
            if not basket_ok:
                lifted_basket = purchase_sequence(action, row, UNLIMITED, dragon)
                if lifted_basket is not None:
                    cost = sum(float(step['cost']) for step in lifted_basket)
                    if cost > float(budget):
                        counts['budget_short_basket'] += 1
                        basket_short.append(cost - float(budget))
                else:
                    counts['basket_illegal_any_budget'] += 1

    import numpy as np
    np.savez_compressed(
        args.out / 'shortfall_raw.npz',
        lead_short=np.frombuffer(lead_short, dtype=np.float32),
        basket_short=np.frombuffer(basket_short, dtype=np.float32),
        gold_at_short=np.frombuffer(gold_at_short, dtype=np.float32))

    meta = {
        'purpose': 'budget_shortfall_shape_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': str(args.dataset),
        'split': 'train',
        'rows_scanned': scanned,
        'counts': dict(counts),
        'definitions': {
            'lead_short': 'lead item unaffordable at gold_est; cost(lead) - gold_est',
            'basket_short': 'whole recorded basket unaffordable at gold_est; cost(basket) - gold_est',
        },
        'footwear_runtime': manifest.get('footwear_runtime'),
        'bounds': 'label-selected lower bound on estimator understatement; never a feature.',
    }
    (args.out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
