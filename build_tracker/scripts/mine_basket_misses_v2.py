"""Stage 1 of docs/experiment_basket_composition_v1.md: TRAIN-only miss partition.

Runs the actual displayed policy over the v2 TRAIN split with labels stripped
and partitions every purchase decision whose recorded basket is absent from
all displayed cards by mechanism. TRAIN only — this stage never opens the
validation or test payloads and spends nothing from the selection ledger.

Declared measurement limitation: the protocol's `composition_truncated` and
`composition_blocked` classes both require decoder gate traces to separate;
they are reported merged as `composition_truncated_or_blocked` (a shown card
is a strict subset of a recorded basket that has a legal buy-only sequence
within gold_est). Counts are mechanism attribution for choosing the stage-2
intervention, never accuracy claims.

Must run through scripts/footwear_runtime.py --entry so the reviewed 16.18
runtime is active before any consumer import.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

CHECKPOINT_EVERY = 2000
EXAMPLES_PER_CLASS = 25
CLASSES = ('covered', 'lead_item_missing', 'count_truncated',
           'composition_truncated_or_blocked', 'budget_infeasible', 'other')


def classify(actual, options, row, budget, dragon):
    """Mechanism class for one purchase decision, per the frozen protocol."""
    from app.conditional_policy import purchase_sequence
    cards = []
    for option in options[:3]:
        if option.get('kind') != 'buy':
            continue
        cards.append(Counter(int(item['item_id']) for item in option.get('items') or []))
    for rank, card in enumerate(cards, 1):
        if card == actual:
            return 'covered', rank
    lead_item = max(actual, key=lambda item: (actual[item], item))
    if all(lead_item not in card for card in cards):
        return 'lead_item_missing', None
    if any(set(card) == set(actual) for card in cards):
        return 'count_truncated', None
    action = tuple(sorted(actual.elements()))
    feasible = purchase_sequence(action, row, float(budget), dragon) is not None
    if not feasible:
        return 'budget_infeasible', None
    if any(card and not (card - actual) and (actual - card) for card in cards):
        return 'composition_truncated_or_blocked', None
    return 'other', None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    from app.ddragon import dragon_for_patch
    from app.predictor import Predictor
    from app.shop_econ import SAVE_ITEM
    from eval_policy import serving_row

    predictor = Predictor(args.artifact)
    if not predictor.config.get('gold_x'):
        raise ValueError('The diagnostic expects the causal-gold candidate configuration')
    dragon = dragon_for_patch('16.18')
    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / 'progress.json'
    partial_path = args.out / 'per_class_counts_partial.json'
    done_rows = 0
    counts = Counter()
    rank_counts = Counter()
    size_counts = {}
    examples = {name: [] for name in CLASSES}
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding='utf-8'))
        done_rows = state['rows']
        counts = Counter(state['counts'])
        rank_counts = Counter(state['rank_counts'])
        size_counts = {k: Counter(v) for k, v in state['size_counts'].items()}
        examples = state['examples']
        print(f'resuming at row {done_rows}', flush=True)

    train_path = args.dataset / 'visits_train.jsonl'
    scanned = 0
    with train_path.open(encoding='utf-8') as handle:
        for line in handle:
            scanned += 1
            if scanned <= done_rows:
                continue
            row = json.loads(line)
            label_counts = row.get('label_counts') or {}
            actual = Counter({int(k): int(v) for k, v in label_counts.items()})
            actual.pop(SAVE_ITEM, None)
            if not actual:
                counts['no_buy_skipped'] += 1
            else:
                budget = row.get('gold_est')
                if row.get('gold_est_version') not in ('prequential-v2', 'prequential-v3') or budget is None:
                    raise ValueError('TRAIN row without causal gold reached the diagnostic')
                options = predictor.predict_options(serving_row(row, budget),
                                                    budget_slack=0, allow_save=True)
                name, rank = classify(actual, options, row, budget, dragon)
                counts[name] += 1
                size = str(min(sum(actual.values()), 5))
                size_counts.setdefault(name, Counter())[size] += 1
                if rank:
                    rank_counts[str(rank)] += 1
                if name != 'covered' and len(examples[name]) < EXAMPLES_PER_CLASS:
                    examples[name].append({
                        'match_id': row.get('match_id'), 'participant_id': row.get('participant_id'),
                        'ts': row.get('ts'), 'champion': row.get('champion'), 'role': row.get('role'),
                        'budget': budget, 'label_counts': dict(sorted(actual.items())),
                        'displayed': [
                            {'kind': option.get('kind'),
                             'items': sorted(int(item['item_id']) for item in option.get('items') or [])}
                            for option in options[:3]]})
            if scanned % CHECKPOINT_EVERY == 0:
                state = {'rows': scanned, 'counts': dict(counts), 'rank_counts': dict(rank_counts),
                         'size_counts': {k: dict(v) for k, v in size_counts.items()},
                         'examples': examples}
                progress_path.write_text(json.dumps(state), encoding='utf-8')
                partial_path.write_text(json.dumps({'rows': scanned, 'counts': dict(counts)}),
                                        encoding='utf-8')
                print(f'diagnostic {scanned} rows: {dict(counts)}', flush=True)

    purchases = sum(counts[name] for name in CLASSES)
    misses = purchases - counts['covered']
    report = {'schema': 1, 'purpose': 'basket_composition_v1_stage1',
        'protocol': 'docs/experiment_basket_composition_v1.md',
        'split': 'train', 'rows_scanned': scanned, 'no_buy_skipped': counts.pop('no_buy_skipped', 0),
        'purchase_decisions': purchases, 'covered': counts['covered'],
        'covered_rank_counts': dict(rank_counts), 'miss_total': misses,
        'class_counts': {name: counts[name] for name in CLASSES},
        'class_counts_by_basket_size': {k: dict(sorted(v.items())) for k, v in size_counts.items()},
        'class_shares_of_misses': {name: (counts[name] / misses if misses else None)
                                   for name in CLASSES if name != 'covered'},
        'merged_classes_note': ('composition_truncated_or_blocked merges the protocol classes 2 and 3; '
                                'separating them needs decoder gate traces.'),
        'examples': examples, 'artifact': str(args.artifact),
        'dataset': str(args.dataset), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_or_test_reads': False}
    from app.dataset_artifacts import sha256, write_json
    write_json(args.out / 'train_miss_partition.json', report)
    print(json.dumps({'purchases': purchases, 'misses': misses,
                      'class_counts': report['class_counts'],
                      'report': str(args.out / 'train_miss_partition.json')}), flush=True)


if __name__ == '__main__':
    main()
