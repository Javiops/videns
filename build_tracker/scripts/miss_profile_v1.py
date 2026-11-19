"""Miss profile — when the displayed policy is wrong, HOW wrong and ON WHAT.

TRAIN only. No validation or test read, no ledger spend, no retraining.

`first_action_exact` is binary: it says a visit was missed, never whether we
showed a 400 g component instead of the 3,000 g legendary the player bought, or
a legendary instead of a 75 g control ward. Those are different failures with
different fixes, and the aggregate hides both.

For every scored visit this records:

  * the gold distance between what we showed and what the player bought —
    signed (did we under- or over-spend the recommendation?) and absolute,
    priced by the same recursive recipe walk the shop uses, from this row's
    inventory, so part-owned recipes are discounted exactly as in game;
  * the item class of the recorded lead and of the displayed lead
    (component / completed / boots / pink ward / support item / consumable),
    as a confusion matrix — what we reach for when we are wrong;
  * whether the recorded basket was even affordable at `gold_est`, so the
    ranking failures can be read separately from the wallet failures.

Item classes come from `DataDragon.classify`, not from a local re-derivation.
Costs come from `purchase_sequence`, the same function the displayed policy
uses to price an option.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

UNLIMITED = 10_000_000.0
CLASSES = ('pink_ward', 'support_item', 'consumable', 'boots',
           'completed', 'component', 'basic', 'other')
# Gold-distance buckets, in gold. The first two straddle the measured median
# wallet error (75 g) so a miss inside our own uncertainty is visible as such.
EDGES = (0, 75, 200, 500, 1000, 2000, 10 ** 9)


def primary_class(item_id: int, dragon) -> str:
    """One class per item, from DataDragon's own classification.

    `classify` returns `is_boots` / `is_basic_boots` / `is_completed` /
    `is_component` and the raw tags; the ward, support and consumable
    distinctions live in the tags and the control-ward ids, not in the returned
    flags. `basic` is a no-recipe item that builds into something (Long Sword,
    Cloth Armor) — without it those fall through to `other` and swamp it.
    """
    from app.shop_econ import PINK
    c = dragon.classify(item_id) or {}
    tags = set(c.get('tags') or [])
    if item_id in PINK:
        return 'pink_ward'
    if 'GoldPer' in tags:
        return 'support_item'
    if 'Consumable' in tags:
        return 'consumable'
    if c.get('is_boots') or c.get('is_basic_boots'):
        return 'boots'
    if c.get('is_completed'):
        return 'completed'
    if c.get('is_component'):
        return 'component'
    if not c.get('from'):
        return 'basic'
    return 'other'


def bucket(value: float) -> str:
    for lo, hi in zip(EDGES, EDGES[1:]):
        if lo <= value < hi:
            return f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+'
    return 'unknown'


def quantiles(values):
    if not values:
        return None
    o = sorted(values)
    pick = lambda q: o[min(len(o) - 1, int(q * len(o)))]
    return {'n': len(o), 'p10': pick(0.10), 'p25': pick(0.25), 'median': pick(0.5),
            'p75': pick(0.75), 'p90': pick(0.90), 'mean': sum(o) / len(o)}


# One implementation of the arm grammar, shared with the sweep, so a `quant:`
# arm cannot mean two different budgets in two reports.
from mask_slack_sweep_v1 import parse_arm, load_posterior  # noqa: E402


class SlackProxy:
    """Same injection the sweep uses; see scripts/mask_slack_sweep_v1.py."""

    def __init__(self, inner, slack_fn):
        self.__dict__['_inner'] = inner
        self.__dict__['_slack_fn'] = slack_fn

    def predict_options(self, row, n_options=3, budget_slack=0, allow_save=False):
        return self._inner.predict_options(row, n_options=n_options,
                                           budget_slack=float(self._slack_fn(row)),
                                           allow_save=allow_save)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', default='candidate.pt')
    ap.add_argument('--arms', default='const:0')
    ap.add_argument('--posterior', type=Path,
                    help='posterior.json, required for quant: arms')
    ap.add_argument('--games', type=int, default=0)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    import footwear_runtime as fr
    directory = os.environ.get('PREFIX_ML_DIR')
    if not directory:
        raise SystemExit('PREFIX_ML_DIR must point at the generation under test')
    ml_dir = Path(directory).resolve()
    fr.validate_generation_runtime(ml_dir)
    manifest = json.loads((ml_dir / 'generation_manifest.json').read_text(encoding='utf-8'))
    qualification_path = ml_dir / 'qualification.json'
    qualification = (json.loads(qualification_path.read_text(encoding='utf-8'))
                     if qualification_path.exists() else {})
    review = os.environ.get('PREFIX_RUNTIME_REVIEW')
    runtime = fr.activate_for_manifest(manifest, corpus_path=qualification.get('corpus'),
                                       review_path=Path(review) if review else None)

    import eval_policy as ep
    import train_prefix as tp
    from app.predictor import Predictor
    from app.conditional_policy import purchase_sequence
    from app.shop_econ import SAVE_ITEM

    run_dir = ep.RUN_DIR
    path = ml_dir / 'visits_train.jsonl'
    assignment = json.loads((ml_dir / 'split_assignment.json').read_text(encoding='utf-8'))
    base = Predictor(run_dir / args.artifact)
    dragon = base.dragon
    if args.posterior:
        load_posterior(args.posterior)
    arms = [parse_arm(s.strip()) for s in args.arms.split(',') if s.strip()]

    all_games = sorted({m for m, f in assignment.items() if f == 'train'})
    rng = random.Random(args.seed)
    rng.shuffle(all_games)
    chosen = set(all_games if args.games <= 0 else all_games[:args.games])
    print(f'{len(arms)} arms over {len(chosen)} train games', flush=True)

    def basket_cost(action, row):
        """Recursive recipe cost of a multiset from this row's state, or None."""
        seq = purchase_sequence(tuple(sorted(action)), row, UNLIMITED, dragon)
        if seq is None:
            return None
        return sum(float(step['cost']) for step in seq)

    stats = {name: {
        'visits': 0, 'hits': 0, 'misses': 0,
        'gold_gap_abs': [], 'gold_gap_signed': [],
        'gap_bucket': Counter(), 'confusion': defaultdict(Counter),
        'miss_by_true_class': Counter(), 'visits_by_true_class': Counter(),
        'miss_by_true_class_affordable': Counter(),
        'visits_by_true_class_affordable': Counter(),
        'gap_by_true_class': defaultdict(list),
        'predicted_nothing': 0, 'true_unpriceable': 0,
    } for name, _fn in arms}

    models = [(name, SlackProxy(base, fn)) for name, fn in arms]
    scored = 0
    for row in tp.with_targets(tp.stream_rows(path, extra_keys=('participant_id',)), dragon):
        if row.get('match_id') not in chosen:
            continue
        actual = Counter(int(i) for i in (row.get('label_ids') or []) if i)
        if not actual or SAVE_ITEM in actual:
            continue                      # purchases only; save rows have no basket cost
        budget = float(row.get('gold_est'))
        truth = dict(actual)
        lead = max(truth, key=lambda i: (truth[i], i))
        tclass = primary_class(lead, dragon)
        tcost = basket_cost(list(actual.elements()), row)
        affordable = purchase_sequence(tuple(sorted(actual.elements())), row, budget, dragon) is not None

        for name, proxy in models:
            s = stats[name]
            s['visits'] += 1
            s['visits_by_true_class'][tclass] += 1
            if affordable:
                s['visits_by_true_class_affordable'][tclass] += 1
            options = proxy.predict_options(ep.serving_row(row, budget), budget_slack=0,
                                            allow_save=True)
            first = options[0] if options else {'kind': 'buy', 'items': []}
            pred = Counter(int(i['item_id']) for i in first.get('items') or [])
            if pred == Counter(truth):
                s['hits'] += 1
                continue
            s['misses'] += 1
            s['miss_by_true_class'][tclass] += 1
            if affordable:
                s['miss_by_true_class_affordable'][tclass] += 1
            if not pred:
                s['predicted_nothing'] += 1
                pclass = 'save_or_empty'
            else:
                plead = max(pred, key=lambda i: (pred[i], i))
                pclass = primary_class(plead, dragon)
            s['confusion'][tclass][pclass] += 1
            pcost = sum(float(i['cost']) for i in first.get('items') or [])
            if tcost is None:
                s['true_unpriceable'] += 1
                continue
            gap = pcost - tcost
            s['gold_gap_signed'].append(gap)
            s['gold_gap_abs'].append(abs(gap))
            s['gap_bucket'][bucket(abs(gap))] += 1
            s['gap_by_true_class'][tclass].append(abs(gap))
        scored += 1
        if scored % 2000 == 0:
            print(f'  …{scored} visits', flush=True)

    report = {
        'schema_version': 1, 'purpose': 'miss_profile_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'train', 'selection_only': True, 'promotable': False,
        'ml_dir': str(ml_dir), 'run_dir': str(run_dir),
        'artifact': args.artifact, 'artifact_sha256': base.artifact_sha256,
        'export_fingerprint': manifest.get('export_fingerprint'),
        'footwear_runtime': runtime,
        'games': len(chosen), 'purchase_visits': scored,
        'population': 'purchases only; SAVE rows excluded (no basket cost)',
        'gold_gap_definition': 'cost(displayed first option) - cost(recorded basket), '
                               'both priced by purchase_sequence from this row state',
        'arms': {},
    }
    for name, _p in models:
        s = stats[name]
        report['arms'][name] = {
            'visits': s['visits'], 'hits': s['hits'], 'misses': s['misses'],
            'exact_rate': s['hits'] / s['visits'] if s['visits'] else None,
            'gold_gap_abs': quantiles(s['gold_gap_abs']),
            'gold_gap_signed': quantiles(s['gold_gap_signed']),
            'gap_buckets': dict(s['gap_bucket']),
            'miss_rate_by_true_class': {
                c: {'visits': s['visits_by_true_class'][c],
                    'misses': s['miss_by_true_class'][c],
                    'miss_rate': (s['miss_by_true_class'][c] / s['visits_by_true_class'][c]
                                  if s['visits_by_true_class'][c] else None),
                    'miss_rate_when_affordable': (
                        s['miss_by_true_class_affordable'][c] / s['visits_by_true_class_affordable'][c]
                        if s['visits_by_true_class_affordable'][c] else None),
                    'gold_gap': quantiles(s['gap_by_true_class'][c])}
                for c in CLASSES if s['visits_by_true_class'][c]},
            'confusion_true_to_predicted': {t: dict(p) for t, p in s['confusion'].items()},
            'predicted_nothing': s['predicted_nothing'],
            'true_unpriceable': s['true_unpriceable'],
        }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'miss_profile.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    for name, _p in models:
        a = report['arms'][name]
        print(f"\n=== {name} — exact {a['exact_rate']:.4f} on {a['visits']} purchases ===")
        g = a['gold_gap_abs']
        if g:
            print(f"  |gold gap| on a miss: median {g['median']:.0f}  p25 {g['p25']:.0f}  "
                  f"p75 {g['p75']:.0f}  p90 {g['p90']:.0f}")
            print(f"  signed gap: median {a['gold_gap_signed']['median']:+.0f} "
                  f"(negative = we recommended cheaper than he bought)")
        print(f"  {'true class':<14}{'visits':>9}{'miss rate':>11}{'if affordable':>15}{'med |gap|':>11}")
        for c, v in sorted(a['miss_rate_by_true_class'].items(),
                           key=lambda kv: -kv[1]['visits']):
            aff = v['miss_rate_when_affordable']
            aff_text = f'{aff:.4f}' if aff is not None else 'n/a'
            gap_text = f"{v['gold_gap']['median']:.0f}" if v['gold_gap'] else '-'
            print(f"  {c:<14}{v['visits']:>9}{v['miss_rate']:>11.4f}"
                  f"{aff_text:>15}{gap_text:>11}")
    print(f"\nwrote {args.out / 'miss_profile.json'}")


if __name__ == '__main__':
    main()
