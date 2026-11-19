"""Paired cross-corpus displayed-policy comparison on one frozen validation.

Per docs/experiment_data_scale_v1.md: two graft artifacts, trained on
different corpora under the same fixed recipe, scored by the SAME reused
machinery (eval_policy.score_visit / summarize / paired_diff) on the SAME
frozen 16.18 v2 validation, one visit at a time, game-paired bootstrap.

Static-data governance is predeclared: economy, legality, display and family
resolution for both policies use the 16.18 Data Dragon. A cross-patch artifact
is loaded with the recorded `dragon_override` waiver and is never
serving-eligible. eval_policy's own development gate forbids --compare inside
a generation on purpose; this evaluator exists so that the cross-corpus
exception is explicit, plan-bound and reported, never silent.

Must run through scripts/footwear_runtime.py --entry with PREFIX_ML_DIR set
to the evaluation generation.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

SIZE_BUCKETS = ('1', '2', '3', '4', '5plus')


def size_bucket(n: int) -> str:
    return str(n) if n < 5 else '5plus'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True,
                        help='frozen run_plan.json written by the runner')
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    dataset = Path(plan['dataset'])
    run_dir = Path(plan['run_directory'])
    bootstrap = int(plan['evaluation_bootstrap'])
    seed = int(plan['evaluation_seed'])

    from app.conditional_policy import require_explicit_slot_state
    from app.dataset_artifacts import sha256, write_json
    from app.dataset_generations import verify_generation, validate_run_directory
    from app.ddragon import dragon_for_patch
    from app.predictor import Predictor
    from app.shop_econ import PINK, SAVE_ITEM, components_of
    import train_prefix as tp
    from eval_policy import METRICS, outcome_capture, paired_diff, score_visit, summarize

    generation = verify_generation(dataset)
    validate_run_directory(dataset, run_dir)
    if generation['export_fingerprint'] != plan['export_fingerprint']:
        raise ValueError('Evaluation dataset does not match the frozen plan fingerprint')

    governing = dragon_for_patch('16.18')

    def load(spec: dict, *, override: bool) -> 'Predictor':
        path = Path(spec['artifact'])
        if sha256(path) != spec['artifact_sha256']:
            raise ValueError(f'Artifact bytes changed since the plan froze them: {path}')
        predictor = Predictor(path, dragon_override=governing if override else None)
        prov = predictor.provenance
        if prov.get('gold_input') != 'prequential-v2' or not predictor.config.get('gold_x'):
            raise ValueError(f'{path.name}: cross-corpus comparison requires prequential-v2 gold')
        if prov.get('training_stage') != 'graft':
            raise ValueError(f'{path.name}: only the trained graft candidate is comparable')
        if prov.get('run_plan_sha256') != spec['training_run_plan_sha256']:
            raise ValueError(f'{path.name}: artifact is not bound to its declared training plan')
        return predictor

    arm = load(plan['arm'], override=True)
    control = load(plan['control'], override=False)
    if control.dragon.signature != governing.signature:
        raise ValueError('Control artifact must natively carry the governing 16.18 static data')
    if arm.static_override is None:
        raise ValueError('Cross-corpus arm must record its static-data waiver')

    assignments = json.loads((dataset / 'split_assignment.json').read_text(encoding='utf-8'))
    val_path = dataset / 'visits_val.jsonl'

    finals_cache: dict[int, set[int]] = {}

    def family_of(target_id: int) -> set[int]:
        if target_id not in finals_cache:
            finals_cache[target_id] = components_of(target_id, governing) | {target_id}
        return finals_cache[target_id]

    models = [('A', arm), ('B', control)]
    outcomes = {tag: defaultdict(list) for tag, _p in models}
    over_budget = {tag: [] for tag, _p in models}
    unseen_truth_visits = {tag: 0 for tag, _p in models}
    scored = 0
    games: set[str] = set()
    outcome_path = run_dir / 'data_scale_outcomes_val.jsonl'
    with outcome_capture(outcome_path) as (write_outcome, outcome_binding):
        for row in tp.with_targets(tp.stream_rows(val_path, extra_keys=('participant_id',)), governing):
            require_explicit_slot_state(row)
            if row.get('gold_est_version') != 'prequential-v2' or row.get('gold_est') is None:
                raise ValueError('Validation row without causal prequential-v2 gold')
            if assignments.get(row.get('match_id')) != 'val':
                raise ValueError('Non-validation game reached the paired evaluation')
            actual = Counter(int(i) for i in (row.get('label_ids') or []) if i)
            if not actual:
                continue
            save_kind = row.get('save_kind') or (
                'no_buy_death' if SAVE_ITEM in actual
                else ('ward_only' if all(i in PINK for i in actual) else 'build_spend'))
            target_id = int(row.get('target_id') or 0)
            family = family_of(target_id) if target_id else None
            budget = row['gold_est']
            truth_size = sum(v for k, v in actual.items() if k != SAVE_ITEM)
            paired_visit = {}
            for tag, predictor in models:
                diagnostics: dict = {}
                result, over = score_visit(predictor, row, actual, save_kind, budget,
                                           governing, family, diagnostics=diagnostics)
                bucket = size_bucket(truth_size) if truth_size else None
                for name in SIZE_BUCKETS:
                    result[f'first_basket_exact_size_{name}'] = (
                        result['first_basket_exact_given_purchase'] if bucket == name else None)
                outcomes[tag][row['match_id']].append(result)
                over_budget[tag].extend(over)
                if diagnostics.get('unseen_truth_item_counts'):
                    unseen_truth_visits[tag] += 1
                paired_visit[tag] = {**diagnostics, 'metrics': result, 'unaffordable': len(over)}
            write_outcome({'schema_version': 1, 'split': 'val', 'match_id': row['match_id'],
                'participant_id': row.get('participant_id'), 'ts': row.get('ts'),
                'champion': row.get('champion'), 'role': row.get('role'),
                'save_kind': save_kind, 'basket_size': truth_size,
                'budget': float(budget),
                'actual_item_counts': {str(i): n for i, n in sorted(actual.items())},
                'next_final_id': target_id or None, 'policies': paired_visit})
            games.add(row['match_id'])
            scored += 1
            if scored % 2000 == 0:
                print(f'  …{scored} visits', flush=True)

    metric_keys = list(METRICS) + [f'first_basket_exact_size_{n}' for n in SIZE_BUCKETS]
    metrics = {}
    for tag, _p in models:
        metrics[tag] = {key: summarize(outcomes[tag], key, bootstrap, seed) for key in metric_keys}
        metrics[tag]['unaffordable'] = len(over_budget[tag])
    diffs = {key: paired_diff(outcomes['A'], outcomes['B'], key, bootstrap, seed)
             for key in metric_keys}

    def favors_arm(key: str) -> bool:
        ci = (diffs[key] or {}).get('ci95')
        return bool(ci) and ci[0] > 0
    def guardrail_holds(key: str) -> bool:
        diff = diffs[key] or {}
        ci = diff.get('ci95')
        return diff.get('diff') is None or not ci or ci[1] >= 0

    primary_win = favors_arm('first_action_exact') and favors_arm('action_family_match')
    guardrails = (guardrail_holds('ward_only_recall') and guardrail_holds('no_buy_death_recall')
                  and not over_budget['A'] and not over_budget['B'])
    verdict = ('supported development win' if primary_win and guardrails
               else 'rejected under the predeclared rule')

    report = {
        'schema': 1, 'purpose': 'data_scale_v1_paired_eval',
        'protocol': plan['protocol_document'], 'protocol_sha256': plan['protocol_document_sha256'],
        'plan_sha256': sha256(args.plan),
        'dataset': str(dataset), 'export_fingerprint': plan['export_fingerprint'],
        'generation_manifest_sha256': sha256(dataset / 'generation_manifest.json'),
        'split': 'val', 'games': len(games), 'visits': scored,
        'evaluation': {'bootstrap': bootstrap, 'seed': seed, 'games_requested': 0,
                       'same_split_for_all_policies': True},
        'artifacts': {'A': plan['arm'], 'B': plan['control']},
        'artifact_provenance': {
            tag: {k: p.provenance.get(k) for k in
                  ('status', 'probe', 'export_fingerprint', 'reconstruction_version',
                   'gold_input', 'training_stage', 'run_plan_sha256', 'eval_version', 'patch')}
            for tag, p in models},
        'static_governance': {
            'governing_patch': '16.18', 'governing_signature': governing.signature,
            'arm_static_override': arm.static_override,
            'note': ('economy, legality, display and family resolution use 16.18 static data '
                     'for both policies; the arm features are computed under 16.18 static data, '
                     'a disclosed divergence from its training patch')},
        'vocabulary_bound': {'arm_visits_with_unseen_truth_items': unseen_truth_visits['A'],
                             'control_visits_with_unseen_truth_items': unseen_truth_visits['B']},
        'metrics': metrics, 'paired_diff_a_minus_b': diffs,
        'unaffordable': {'A': len(over_budget['A']), 'B': len(over_budget['B'])},
        'primary_win': primary_win, 'guardrails_hold': guardrails, 'verdict': verdict,
        'paired_outcomes': outcome_binding,
        'final_test_evaluation': False, 'promotion_allowed': False,
        'validation_spend_event': 'one paired comparison on the v2 validation (arm A vs control)',
        'created_utc': datetime.now(timezone.utc).isoformat(),
    }
    out = run_dir / 'report'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'data_scale_eval_val.json'
    if report_path.exists():
        raise ValueError('Policy evidence already exists; use a new run directory')
    json.dumps(report, allow_nan=False)
    write_json(report_path, report)

    print(f'scored {scored} visits from {len(games)} games', flush=True)
    for key in ('first_action_exact', 'action_family_match', 'any_of_three_exact_given_purchase'):
        diff = diffs[key]
        if diff.get('diff') is not None:
            ci = diff.get('ci95')
            print(f"  {key:<38} A-B {diff['diff']:+.4f}  CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]  "
                  f"(A {diff['a']:.4f} vs B {diff['b']:.4f})", flush=True)
    print(f'  unaffordable A={len(over_budget["A"])} B={len(over_budget["B"])} (must be 0)', flush=True)
    print(f'verdict: {verdict}', flush=True)
    print('=' * 72, flush=True)
    print('CITATION — copy this with any number above', flush=True)
    print(f"  A: {Path(plan['arm']['artifact']).name}  sha256={plan['arm']['artifact_sha256'][:12]}  "
          f"static=OVERRIDDEN to 16.18", flush=True)
    print(f"  B: {Path(plan['control']['artifact']).name}  sha256={plan['control']['artifact_sha256'][:12]}", flush=True)
    print(f'  split=val  games={len(games)}  visits={scored}  bootstrap={bootstrap}  seed={seed}', flush=True)
    print(f'  report={report_path}', flush=True)
    print('=' * 72, flush=True)


if __name__ == '__main__':
    main()
