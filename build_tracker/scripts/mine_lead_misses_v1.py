"""Lead-miss decomposition: TRAIN-only follow-up to the stage-1 miss partition.

The frozen basket-composition diagnostic showed 50.2% of TRAIN misses are
`lead_item_missing` — no displayed card contains the highest-count recorded
item. That class mixes problems with opposite remedies. This script splits it
by mechanism, on the same v2 TRAIN split with the same frozen candidate:

  - `not_in_vocab`        — the recorded lead is not in the artifact's label
                            vocabulary; no trunk change can rank it.
  - `infeasible_at_gold_est` — no legal buy sequence reaches the lead within
                            the causal budget; a ranking fix cannot show it.
  - `feasible_rank_top3`  — the model itself ranks the lead top-3 among legal
                            items, yet no card carries it: decoder/gate loss.
  - `feasible_rank_4_10` / `feasible_rank_11_30` / `feasible_rank_over_30`
                          — pure trunk discrimination shortfall, by depth.

TRAIN only: never opens validation or test payloads, spends nothing from the
selection ledger. Diagnostic attribution, never an accuracy claim.

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
EXAMPLES_PER_CLASS = 15
CLASSES = ('not_in_vocab', 'infeasible_at_gold_est', 'feasible_rank_top3',
           'feasible_rank_4_10', 'feasible_rank_11_30', 'feasible_rank_over_30')


def rank_class(rank: int) -> str:
    if rank <= 3:
        return 'feasible_rank_top3'
    if rank <= 10:
        return 'feasible_rank_4_10'
    if rank <= 30:
        return 'feasible_rank_11_30'
    return 'feasible_rank_over_30'


def minute_bucket(ts_ms) -> str:
    minute = float(ts_ms or 0) / 60000.0
    if minute < 10:
        return '00-09'
    if minute < 20:
        return '10-19'
    return '20+'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
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
    progress_path = args.out / 'progress.json'
    done_rows = 0
    counts = Counter()
    by_minute = {}
    by_size = {}
    by_depth = {}
    examples = {name: [] for name in CLASSES}
    if progress_path.exists():
        state = json.loads(progress_path.read_text(encoding='utf-8'))
        done_rows = state['rows']
        counts = Counter(state['counts'])
        by_minute = {k: Counter(v) for k, v in state['by_minute'].items()}
        by_size = {k: Counter(v) for k, v in state['by_size'].items()}
        by_depth = {k: Counter(v) for k, v in state['by_depth'].items()}
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
            if actual:
                budget = row.get('gold_est')
                if row.get('gold_est_version') not in ('prequential-v2', 'prequential-v3') or budget is None:
                    raise ValueError('TRAIN row without causal gold reached the diagnostic')
                shown = serving_row(row, budget)
                options = predictor.predict_options(shown, budget_slack=0, allow_save=True)
                cards = [Counter(int(item['item_id']) for item in option.get('items') or [])
                         for option in options[:3] if option.get('kind') == 'buy']
                lead = max(actual, key=lambda item: (actual[item], item))
                counts['purchases'] += 1
                if all(lead not in card for card in cards):
                    counts['lead_missing'] += 1
                    idx = predictor.label_index.get(lead)
                    if idx is None:
                        name, rank = 'not_in_vocab', None
                    elif purchase_sequence((lead,), row, float(budget), dragon) is None:
                        name, rank = 'infeasible_at_gold_est', None
                    else:
                        masked, _raw, _want, _save = predictor._probs(shown)
                        rank = 1 + int((masked > masked[idx]).sum())
                        name = rank_class(rank)
                    counts[name] += 1
                    by_minute.setdefault(name, Counter())[minute_bucket(row.get('ts'))] += 1
                    by_size.setdefault(name, Counter())[str(min(sum(actual.values()), 5))] += 1
                    depth = str((dragon.classify(lead) or {}).get('depth') or 0)
                    by_depth.setdefault(name, Counter())[depth] += 1
                    if len(examples[name]) < EXAMPLES_PER_CLASS:
                        examples[name].append({
                            'match_id': row.get('match_id'), 'participant_id': row.get('participant_id'),
                            'ts': row.get('ts'), 'champion': row.get('champion'), 'role': row.get('role'),
                            'budget': budget, 'lead': lead, 'lead_name': dragon.item_name(lead),
                            'lead_rank_among_legal': rank,
                            'label_counts': dict(sorted(actual.items())),
                            'displayed': [
                                {'kind': option.get('kind'),
                                 'items': sorted(int(item['item_id']) for item in option.get('items') or [])}
                                for option in options[:3]]})
            if scanned % CHECKPOINT_EVERY == 0:
                state = {'rows': scanned, 'counts': dict(counts),
                         'by_minute': {k: dict(v) for k, v in by_minute.items()},
                         'by_size': {k: dict(v) for k, v in by_size.items()},
                         'by_depth': {k: dict(v) for k, v in by_depth.items()},
                         'examples': examples}
                progress_path.write_text(json.dumps(state), encoding='utf-8')
                print(f'lead diagnostic {scanned} rows: {dict(counts)}', flush=True)

    misses = counts['lead_missing']
    report = {'schema': 1, 'purpose': 'lead_miss_diagnostic_v1',
        'parent_diagnostic': 'data/experiments/basket-composition-v1/diagnostic/train_miss_partition.json',
        'split': 'train', 'rows_scanned': scanned,
        'purchase_decisions': counts['purchases'], 'lead_missing_total': misses,
        'class_counts': {name: counts[name] for name in CLASSES},
        'class_shares_of_lead_misses': {name: (counts[name] / misses if misses else None)
                                        for name in CLASSES},
        'by_minute': {k: dict(sorted(v.items())) for k, v in by_minute.items()},
        'by_basket_size': {k: dict(sorted(v.items())) for k, v in by_size.items()},
        'by_lead_depth': {k: dict(sorted(v.items())) for k, v in by_depth.items()},
        'rank_population_note': ('rank is the lead item position among LEGAL items by the '
                                 'affordability-masked head; displayed cards additionally pass '
                                 'decoder gates, so feasible_rank_top3 isolates decoder loss.'),
        'examples': examples, 'artifact': str(args.artifact),
        'dataset': str(args.dataset), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_or_test_reads': False}
    from app.dataset_artifacts import write_json
    write_json(args.out / 'lead_miss_diagnostic.json', report)
    print(json.dumps({'purchases': counts['purchases'], 'lead_misses': misses,
                      'class_counts': report['class_counts'],
                      'report': str(args.out / 'lead_miss_diagnostic.json')}), flush=True)


if __name__ == '__main__':
    main()
