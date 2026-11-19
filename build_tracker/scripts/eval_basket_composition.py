"""Stage-2 paired decoder comparison for docs/experiment_basket_composition_v1.md.

A = the composed-generation policy (frozen addendum version) over the frozen
stage-1 checkpoint; B = the original displayed policy over the SAME checkpoint.
One scored comparison on the v2 validation — a validation-spend event that the
caller must record in data/ml/v2_validation_selection_log.md.

Invariants asserted on every decision: on a SAVE-led original decision the two
policies are identical; on BUY-led decisions every non-lead card is identical;
both arms pass the independent affordability replay with zero violations.
Must run through scripts/footwear_runtime.py --entry.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

PRIMARIES = ('first_action_exact', 'action_family_match')
GUARDRAILS = ('ward_only_recall', 'no_buy_death_recall')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--policy-version', required=True)
    parser.add_argument('--addendum', type=Path, required=True,
                        help='frozen stage-2 addendum document (sha bound into the report)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--seed', type=int, default=16)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1,
                        help='>1 scores a disjoint game shard and writes outcomes only; '
                             'the parallel driver merges shards and computes the report')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError('shard-index must lie inside shard-count')

    import train_prefix as tp
    from app.dataset_artifacts import sha256, write_json
    from app.ddragon import dragon_for_patch
    from app.predictor import Predictor
    from app.shop_econ import SAVE_ITEM, components_of
    from app.conditional_policy import require_explicit_slot_state
    from eval_policy import METRICS, score_visit, summarize, paired_diff, outcome_capture
    from basket_composition_policy import build

    if args.shard_count == 1:
        if args.out.exists():
            raise ValueError('Use a fresh stage-2 report directory')
        args.out.mkdir(parents=True)
    else:
        args.out.mkdir(parents=True, exist_ok=True)  # shards share the driver's directory
    dragon = dragon_for_patch('16.18')
    predictor = Predictor(args.artifact)
    if not predictor.config.get('gold_x'):
        raise ValueError('Stage 2 expects the causal-gold candidate configuration')
    composed = build(args.policy_version, predictor, dragon)
    assignments = json.loads((args.dataset / 'split_assignment.json').read_text(encoding='utf-8'))
    path = args.dataset / 'visits_val.jsonl'

    finals_cache = {}
    def family_of(target_id):
        if target_id not in finals_cache:
            finals_cache[target_id] = components_of(target_id, dragon) | {target_id}
        return finals_cache[target_id]

    models = (('A', f'composed:{args.policy_version}', composed), ('B', 'original', predictor))
    outcomes = {tag: defaultdict(list) for tag, _n, _p in models}
    over_budget = {tag: [] for tag, _n, _p in models}
    scored = 0
    invariant_counts = Counter()
    import hashlib
    def in_shard(match_id):
        if args.shard_count == 1:
            return True
        return int(hashlib.sha256(match_id.encode()).hexdigest()[:8], 16) % args.shard_count == args.shard_index
    outcome_name = ('composition_outcomes_val.jsonl' if args.shard_count == 1
                    else f'composition_outcomes_val_shard{args.shard_index}.jsonl')
    with outcome_capture(args.out / outcome_name) as (write_outcome, outcome_binding):
        for row in tp.with_targets(tp.stream_rows(path, extra_keys=('participant_id',)), dragon):
            if not in_shard(row.get('match_id') or ''):
                continue
            require_explicit_slot_state(row)
            if row.get('gold_est_version') != 'prequential-v2' or row.get('gold_est') is None:
                raise SystemExit('Stage 2 requires causal prequential-v2 gold on every scored row')
            if assignments.get(row.get('match_id')) != 'val':
                raise SystemExit('Non-validation game reached stage-2 evaluation')
            actual = Counter(int(i) for i in (row.get('label_ids') or []) if i)
            if not actual:
                continue
            save_kind = row.get('save_kind') or (
                'no_buy_death' if SAVE_ITEM in actual else 'build_spend')
            target_id = int(row.get('target_id') or 0)
            family = family_of(target_id) if target_id else None
            budget = row.get('gold_est')
            paired_visit = {}
            for tag, name, policy in models:
                diagnostics = {}
                result, over = score_visit(policy, row, actual, save_kind, budget, dragon, family,
                                           diagnostics=diagnostics)
                outcomes[tag][row['match_id']].append(result)
                over_budget[tag].extend(over)
                paired_visit[tag] = {**diagnostics, 'metrics': result,
                    'metric_n': {k: int(v is not None) for k, v in result.items()},
                    'unaffordable': len(over)}
            a_cards = paired_visit['A']['displayed_options']
            b_cards = paired_visit['B']['displayed_options']
            if b_cards and b_cards[0]['kind'] == 'save':
                if a_cards != b_cards:
                    raise SystemExit('Composition changed a SAVE-led decision')
                invariant_counts['save_led_identical'] += 1
            else:
                if a_cards[1:] != b_cards[1:]:
                    raise SystemExit('Composition changed a non-lead card')
                invariant_counts['lead_changed' if a_cards[:1] != b_cards[:1] else 'lead_unchanged'] += 1
            write_outcome({'schema_version': 1, 'split': 'val', 'match_id': row['match_id'],
                'participant_id': row.get('participant_id'), 'ts': row.get('ts'),
                'champion': row.get('champion'), 'role': row.get('role'),
                'save_kind': save_kind, 'minute': int((row.get('ts') or 0) / 60000),
                'budget': float(budget), 'inventory': list(row.get('inventory') or []),
                'actual_item_counts': {str(i): c for i, c in sorted(actual.items())},
                'next_final_id': target_id or None,
                'policies': paired_visit})
            scored += 1
            if scored % 5000 == 0:
                print(f'scored {scored} decisions: {dict(invariant_counts)}', flush=True)

    if args.shard_count > 1:
        from app.dataset_artifacts import sha256 as file_sha, write_json as write_shard
        write_shard(args.out / f'shard{args.shard_index}_summary.json', {
            'schema': 1, 'shard_index': args.shard_index, 'shard_count': args.shard_count,
            'games': sorted(outcomes['B']), 'visits': scored,
            'invariants': dict(invariant_counts),
            'unaffordable': {tag: len(v) for tag, v in over_budget.items()},
            'outcomes': str(args.out / outcome_name),
            'outcomes_sha256': file_sha(args.out / outcome_name),
            'created_utc': datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'shard': args.shard_index, 'games': len(outcomes['B']),
                          'visits': scored, 'invariants': dict(invariant_counts)}), flush=True)
        return

    games = len(outcomes['B'])
    metrics = {tag: {key: summarize(outcomes[tag], key, args.bootstrap, args.seed) for key in METRICS}
               for tag, _n, _p in models}
    paired = {key: paired_diff(outcomes['A'], outcomes['B'], key, args.bootstrap, args.seed)
              for key in METRICS}
    win = all(paired[key]['ci95'] and paired[key]['ci95'][0] > 0 for key in PRIMARIES)
    guardrails_hold = all(not paired[key]['ci95'] or paired[key]['ci95'][1] >= 0 for key in GUARDRAILS)
    report = {'schema': 1, 'purpose': 'basket_composition_v1_stage2',
        'protocol': 'docs/experiment_basket_composition_v1.md',
        'protocol_sha256': sha256(ROOT / 'docs/experiment_basket_composition_v1.md'),
        'addendum': str(args.addendum), 'addendum_sha256': sha256(args.addendum),
        'policy_version': args.policy_version,
        'artifact': str(args.artifact), 'artifact_sha256': sha256(args.artifact),
        'dataset': str(args.dataset),
        'generation_manifest_sha256': sha256(args.dataset / 'generation_manifest.json'),
        'split': 'val', 'games': games, 'visits': scored,
        'invariants': dict(invariant_counts),
        'unaffordable': {tag: len(v) for tag, v in over_budget.items()},
        'metrics': metrics, 'paired_diff_a_minus_b': paired,
        'primary_win': win, 'guardrails_hold': guardrails_hold,
        'verdict': ('supported development win' if win and guardrails_hold
                    else 'rejected under the predeclared rule'),
        'validation_spend_event': True,
        'outcome_binding': outcome_binding,
        'final_test_evaluation': False, 'promotion_allowed': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.out / 'composition_eval_val.json', report)
    print(json.dumps({'verdict': report['verdict'],
        'primaries': {k: {'diff': paired[k]['diff'], 'ci95': paired[k]['ci95']} for k in PRIMARIES},
        'guardrails': {k: {'diff': paired[k]['diff'], 'ci95': paired[k]['ci95']} for k in GUARDRAILS},
        'unaffordable': report['unaffordable'],
        'report': str(args.out / 'composition_eval_val.json')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
