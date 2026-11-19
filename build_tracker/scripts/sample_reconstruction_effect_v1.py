"""Phase C: measure the opening-sale fix on a sample BEFORE the full rebuild.

The plan's 2x2 attribution table degenerated to before/after: the prequential
harness rejected every income-term variant under the pre-registered rule, so
prequential-v3's ONLY change is the opening-sale mechanism, which moves gold
(credit) and slots (decision inventory) through one lever.

For N TRAIN games, the SAME raw timelines are reconstructed twice with the
SAME base module in its two versions — OLD from `git show HEAD:app/reconstruct.py`
(the working tree is uncommitted; HEAD still carries the pre-change behaviour)
and NEW from the current source — and both row sets run the same v2-audit
classification (lead-only probe with full-basket fallback). Identical machinery
on both sides; the footwear-variant absolute numbers differ slightly from the
published corpus and are not compared here. TRAIN only, no model, no
validation reads, nothing written into any generation directory.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

UNLIMITED = 10_000_000.0
CLASSES = ('feasible_at_gold_est', 'budget_short',
           'lead_blocked_full_basket_legal', 'legality_blocked')


def classify(row, dragon, purchase_sequence, SAVE_ITEM):
    # Fresh baseline._example rows carry label_ids; the publisher adds
    # label_counts later. Both encode the same multiset.
    actual = Counter(int(i) for i in (row.get('label_ids') or []) if i)
    actual.pop(SAVE_ITEM, None)
    if not actual:
        return None, None
    budget = row.get('gold_est')
    lead = max(actual, key=lambda item: (actual[item], item))
    action = tuple(sorted(actual.elements()))
    if (purchase_sequence((lead,), row, float(budget), dragon) is not None
            or purchase_sequence(action, row, float(budget), dragon) is not None):
        return 'feasible_at_gold_est', None
    lifted = purchase_sequence((lead,), row, UNLIMITED, dragon)
    if lifted is not None:
        return 'budget_short', float(lifted[0]['cost']) - float(budget)
    if purchase_sequence(action, row, UNLIMITED, dragon) is not None:
        return 'lead_blocked_full_basket_legal', None
    return 'legality_blocked', None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--games', type=int, default=200)
    args = parser.parse_args()

    import importlib.util
    import subprocess

    from app.config import RAW_MATCH_V5_DIR
    from app.conditional_policy import purchase_sequence
    from app.ddragon import dragon_for_patch
    from app.riot import RawMatchArchive
    from app.reconstruct import reconstruct_game as reconstruct_new
    from app.shop_econ import SAVE_ITEM
    import baseline

    old_source = subprocess.run(['git', 'show', 'HEAD:app/reconstruct.py'], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout
    if 'opening_sale_gold' in old_source:
        raise ValueError('HEAD already carries the fix; the old side would not be old')
    old_path = args.out / '_old_reconstruct_head.py'
    args.out.mkdir(parents=True, exist_ok=True)
    old_path.write_text(old_source, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('old_reconstruct_head', old_path)
    old_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old_module)
    reconstruct_old = old_module.reconstruct_game

    assignment = json.loads((args.dataset / 'split_assignment.json').read_text(encoding='utf-8'))
    train_ids = sorted(m for m, fold in assignment.items() if fold == 'train')[:args.games]
    archive = RawMatchArchive(RAW_MATCH_V5_DIR)
    dragon = dragon_for_patch('16.18')

    old_counts, new_counts = Counter(), Counter()
    old_short, new_short = [], []
    opening_rows = 0
    flipped = Counter()

    def rows_for(reconstruct, mid, match, timeline):
        rows = {}
        _game, events = reconstruct(match, timeline, dragon)
        for event in events:
            if event.get('type') != 'shop':
                continue
            row = baseline._example({**event, 'match_id': mid}, dragon)
            if not row or row.get('save_kind') == 'no_buy_death':
                continue
            rows[(event.get('participant_id'), event.get('ts'))] = row
        return rows

    processed = 0
    for mid in train_ids:
        match = archive.read('match', mid)
        timeline = archive.read('timeline', mid)
        if not match or not timeline:
            continue
        news = rows_for(reconstruct_new, mid, match, timeline)
        olds = rows_for(reconstruct_old, mid, match, timeline)
        processed += 1
        for key, new_row in news.items():
            name_new, short_new = classify(new_row, dragon, purchase_sequence, SAVE_ITEM)
            if name_new is None:
                continue
            new_counts[name_new] += 1
            if short_new is not None:
                new_short.append(short_new)
            if new_row.get('opening_sale_gold'):
                opening_rows += 1
            old = olds.get(key)
            if old is not None:
                name_old, short_old = classify(old, dragon, purchase_sequence, SAVE_ITEM)
                if name_old is not None:
                    old_counts[name_old] += 1
                    if short_old is not None:
                        old_short.append(short_old)
                    if name_old != name_new:
                        flipped[f'{name_old} -> {name_new}'] += 1

    def quantiles(values):
        if not values:
            return None
        ordered = sorted(values)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
        return {'n': len(ordered), 'median': pick(0.5), 'p90': pick(0.90),
                'mean': sum(ordered) / len(ordered)}

    def shares(counts):
        total = sum(counts[c] for c in CLASSES)
        return {c: {'n': counts[c], 'share': counts[c] / total if total else None} for c in CLASSES}

    report = {'schema': 1, 'purpose': 'sample_reconstruction_effect_v1',
        'plan': 'C:/Users/Javier/.claude/plans/sparkling-bouncing-sparrow.md (Phase C)',
        'note': ('2x2 degenerated to before/after: the harness rejected every income variant, '
                 'so prequential-v3 differs from v2 only by the opening-sale mechanism.'),
        'split': 'train', 'games_requested': args.games, 'games_processed': processed,
        'old_rows_matched': sum(old_counts.values()), 'new_rows': sum(new_counts.values()),
        'rows_with_opening_sale_credit': opening_rows,
        'old': shares(old_counts), 'new': shares(new_counts),
        'old_budget_shortfall': quantiles(old_short), 'new_budget_shortfall': quantiles(new_short),
        'class_transitions_old_to_new': dict(flipped),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_or_test_reads': False}
    from app.dataset_artifacts import write_json
    write_json(args.out / 'sample_effect.json', report)
    print(json.dumps({'games': processed, 'old': {c: old_counts[c] for c in CLASSES},
                      'new': {c: new_counts[c] for c in CLASSES},
                      'transitions': dict(flipped),
                      'opening_rows': opening_rows}, default=str), flush=True)


if __name__ == '__main__':
    main()
