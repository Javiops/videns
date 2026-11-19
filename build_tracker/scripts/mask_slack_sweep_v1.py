"""Mask-slack sweep — phases 1 and 2 of the wallet-lever experiment.

Protocol: docs/experiment_mask_slack_v1.md, FROZEN before the first arm ran.

The question, recorded: with the model held fixed, which decode-time budget
`gold_est + delta` maximises exactly-correct displayed choices, and what does
each delta cost in displayed baskets the player could not actually afford?

TRAIN ONLY. This is a SELECTION run: it may not promote anything, and its
numbers are memorisation-favoured and not comparable with any validation
figure. Refuses any split but train.

Two arm families:
  const:C   delta = C gold, flat (phase 1)
  rate:K    delta = K * dt_seconds, dt from the frame cadence (phase 2)

`dt` is approximated as `ts mod 60000` because the export carries no
`gold_gap.frame_ts`. Measured against the real anchor gap on 60 TRAIN games /
7,350 shop visits: 99.55% of visits land within 1 s of the true gap; the
remaining 0.45% are off by about a full 60 s cadence. Recorded here so the
approximation is never mistaken for the real thing.

Nothing in eval_policy.py is modified — it is pinned by digest in frozen run
plans (scripts/run_development_1618_v3.py) and editing it would invalidate
them. `score_visit` is imported and used verbatim, so every metric is computed
by the same code that produced the Phase D report; the only difference between
arms is the budget the displayed policy may spend. The slack is injected
through a proxy around the predictor because `score_visit` hardcodes
`budget_slack=0`.
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

DT_PROXY_EVIDENCE = {
    'definition': 'dt_seconds = (row.ts mod 60000) / 1000',
    'checked_on': '60 TRAIN games, 7,350 shop visits',
    'within_1s_of_true_anchor_gap': 0.9955,
    'note': 'the 0.45% tail is off by about one 60 s cadence; the export carries '
            'no gold_gap.frame_ts, which is the only exact source.',
}


class SlackProxy:
    """Wraps a Predictor so the displayed policy spends `gold_est + delta(row)`.

    eval_policy.score_visit calls predict_options(..., budget_slack=0). That
    zero is the arm under test, so it is replaced here rather than in
    eval_policy, which stays byte-identical to the version frozen in the run
    plans. Everything else on the predictor is delegated untouched.
    """

    def __init__(self, inner, slack_fn):
        self.__dict__['_inner'] = inner
        self.__dict__['_slack_fn'] = slack_fn
        self.__dict__['applied'] = []

    def predict_options(self, row, n_options=3, budget_slack=0, allow_save=False):
        delta = float(self._slack_fn(row))
        self.applied.append(delta)
        return self._inner.predict_options(row, n_options=n_options,
                                           budget_slack=delta, allow_save=allow_save)

    def __getattr__(self, name):
        return getattr(self._inner, name)


POSTERIOR: dict = {}


def load_posterior(path):
    """The fitted wallet posterior, shared with the trainer so both use one model."""
    global POSTERIOR
    POSTERIOR = json.loads(Path(path).read_text(encoding='utf-8'))
    return POSTERIOR


def dt_seconds(row) -> float:
    return (int(row.get('ts') or 0) % 60000) / 1000.0


def posterior_coefficients(q: float) -> tuple[float, float]:
    """(a_q, k_q) at an arbitrary quantile, linearly interpolated on the grid."""
    grid = sorted((float(key), float(e['a']), float(e['k']))
                  for key, e in (POSTERIOR.get('quantiles') or {}).items())
    if not grid:
        raise SystemExit('posterior carries no quantile grid')
    if q <= grid[0][0]:
        return grid[0][1], grid[0][2]
    if q >= grid[-1][0]:
        return grid[-1][1], grid[-1][2]
    for left, right in zip(grid, grid[1:]):
        if left[0] <= q <= right[0]:
            span = right[0] - left[0]
            w = 0.0 if span <= 0 else (q - left[0]) / span
            return (left[1] + w * (right[1] - left[1]),
                    left[2] + w * (right[2] - left[2]))
    return grid[-1][1], grid[-1][2]


def parse_arm(spec: str):
    kind, _, value = spec.partition(':')
    value = float(value)
    if kind == 'const':
        return spec, (lambda row, c=value: c)
    if kind == 'rate':
        return spec, (lambda row, k=value: k * dt_seconds(row))
    if kind == 'quant':
        # delta = the q-th quantile of the fitted correction at this row's
        # staleness: a_q*dt + k_q*sqrt(dt), drift plus diffusion.
        #
        # Interpolated between grid points rather than looked up by exact key:
        # the trainer's sampler interpolates, so an arm parser that demanded an
        # exact key would silently mean a different budget than the mask the
        # model was trained under (q=0.65 is not even ON the fitted grid).
        if not POSTERIOR:
            raise SystemExit('quant: arms need --posterior')
        a, k = posterior_coefficients(value)
        return spec, (lambda row, a=a, k=k: a * dt_seconds(row) + k * dt_seconds(row) ** 0.5)
    raise SystemExit(f'unknown arm: {spec}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', default='candidate.pt')
    ap.add_argument('--split', default='train', choices=['train'],
                    help='train only; selection never reads validation or test')
    ap.add_argument('--arms', required=True, help='comma-separated, e.g. const:0,const:100,rate:2')
    ap.add_argument('--games', type=int, default=0, help='0 = every TRAIN game')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--bootstrap', type=int, required=True)
    ap.add_argument('--posterior', type=Path,
                    help='posterior.json from gold_posterior_fit_v1, for quant: arms')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    if args.split != 'train':
        raise SystemExit('selection runs on train only')

    # Activate the bound economics runtime BEFORE importing the pipeline, exactly
    # as footwear_runtime.main does for its allowed entries — same functions, same
    # guards. Done here rather than by adding this script to that module's
    # allowlist, because footwear_runtime.py is digest-pinned in frozen run plans.
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
    from app.shop_econ import PINK, SAVE_ITEM, components_of

    run_dir = ep.RUN_DIR
    if ep.ML_DIR != ml_dir:
        raise SystemExit('PREFIX_ML_DIR moved between activation and import')
    path = ml_dir / f'visits_{args.split}.jsonl'
    if not path.exists():
        raise SystemExit(f'missing {path}')
    assignment = json.loads((ml_dir / 'split_assignment.json').read_text(encoding='utf-8'))

    base = Predictor(run_dir / args.artifact)
    dragon = base.dragon
    if args.posterior:
        load_posterior(args.posterior)
    arms = [parse_arm(spec.strip()) for spec in args.arms.split(',') if spec.strip()]
    if not any(name == 'const:0' for name, _fn in arms):
        raise SystemExit('the sweep needs the const:0 reference arm')

    all_games = sorted({mid for mid, fold in assignment.items() if fold == 'train'})
    rng = random.Random(args.seed)
    rng.shuffle(all_games)
    chosen = set(all_games if args.games <= 0 else all_games[:args.games])
    print(f'{len(arms)} arms over {len(chosen)} of {len(all_games)} train games', flush=True)

    models = [(name, SlackProxy(base, fn)) for name, fn in arms]
    outcomes = {name: defaultdict(list) for name, _p in models}
    unaffordable = {name: 0 for name, _p in models}
    unaffordable_gold = {name: [] for name, _p in models}
    finals: dict[int, set[int]] = {}
    scored = 0

    for row in tp.with_targets(tp.stream_rows(path, extra_keys=('participant_id',)), dragon):
        if row.get('match_id') not in chosen:
            continue
        if assignment.get(row.get('match_id')) != 'train':
            raise SystemExit('a non-train game reached the selection sweep')
        if row.get('gold_est') is None or row.get('gold_est_version') not in (
                'prequential-v2', 'prequential-v3'):
            raise SystemExit('row without causal gold reached the sweep')
        actual = Counter(int(i) for i in (row.get('label_ids') or []) if i)
        if not actual:
            continue
        save_kind = row.get('save_kind') or (
            'no_buy_death' if SAVE_ITEM in actual
            else ('ward_only' if all(i in PINK for i in actual) else 'build_spend'))
        target_id = int(row.get('target_id') or 0)
        if target_id and target_id not in finals:
            finals[target_id] = components_of(target_id, dragon) | {target_id}
        family = finals.get(target_id) if target_id else None
        budget = row.get('gold_est')

        for name, proxy in models:
            # `budget` stays the honest wallet: the slack widens what may be
            # DISPLAYED, while over-budget is still judged against gold_est, so
            # the price tag of each arm is measured and not defined away.
            result, over = ep.score_visit(proxy, row, actual, save_kind, float(budget),
                                          dragon, family)
            outcomes[name][row['match_id']].append(result)
            unaffordable[name] += len(over)
            unaffordable_gold[name].extend(o['cost'] - o['budget'] for o in over)
        scored += 1
        if scored % 2000 == 0:
            print(f'  …{scored} visits', flush=True)

    ref = 'const:0'
    report = {
        'schema_version': 1,
        'purpose': 'mask_slack_sweep_v1',
        'protocol': 'docs/experiment_mask_slack_v1.md',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'train',
        'selection_only': True,
        'promotable': False,
        'comparable_with_validation_numbers': False,
        'ml_dir': str(ml_dir),
        'run_dir': str(run_dir),
        'artifact': args.artifact,
        'artifact_sha256': base.artifact_sha256,
        'export_fingerprint': manifest.get('export_fingerprint'),
        'generation': manifest.get('purpose'),
        'footwear_runtime': runtime,
        'games': len(chosen),
        'visits': scored,
        'seed': args.seed,
        'bootstrap': args.bootstrap,
        'dt_proxy': DT_PROXY_EVIDENCE,
        'posterior': {'source': str(args.posterior), 'model': POSTERIOR.get('model')}
                     if POSTERIOR else None,
        'arms': {},
    }
    for name, proxy in models:
        applied = proxy.applied
        arm = {
            'metrics': {key: ep.summarize(outcomes[name], key, args.bootstrap, args.seed)
                        for key in ep.METRICS},
            'unaffordable_displayed_baskets': unaffordable[name],
            'unaffordable_displayed_per_visit': unaffordable[name] / scored if scored else None,
            'unaffordable_excess_gold_mean': (
                sum(unaffordable_gold[name]) / len(unaffordable_gold[name])
                if unaffordable_gold[name] else None),
            'delta_applied_mean': sum(applied) / len(applied) if applied else None,
            'delta_applied_max': max(applied) if applied else None,
        }
        if name != ref:
            arm['paired_vs_const0'] = {
                key: ep.paired_diff(outcomes[name], outcomes[ref], key,
                                    args.bootstrap, args.seed)
                for key in ep.METRICS}
        report['arms'][name] = arm

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'mask_slack_sweep.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    print()
    print(f'{"arm":<12} {"exact":>8} {"basket":>8} {"any3":>8} {"unaff/vis":>10} {"mean d":>8}')
    for name, _p in models:
        a = report['arms'][name]
        f = lambda k: (a['metrics'][k]['value'] or 0.0)
        print(f'{name:<12} {f("first_action_exact"):8.4f} '
              f'{f("first_basket_exact_given_purchase"):8.4f} '
              f'{f("any_of_three_exact_given_purchase"):8.4f} '
              f'{a["unaffordable_displayed_per_visit"]:10.4f} '
              f'{a["delta_applied_mean"]:8.1f}')
    print(f'\nwrote {args.out / "mask_slack_sweep.json"}')


if __name__ == '__main__':
    main()
