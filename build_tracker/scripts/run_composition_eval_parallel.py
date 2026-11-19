"""Parallel driver for the stage-2 paired evaluation: 4 disjoint game shards.

Pure throughput change over the serial evaluator: identical rows, policies,
invariants and statistics. Each shard scores its games and writes outcomes;
this driver waits for all four, verifies disjoint complete coverage of the
validation games, rebuilds the per-game outcome structure and computes the
same summaries and paired game-bootstrap intervals (seed 16, 400 resamples)
the serial path would have produced.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

SHARDS = 4
PRIMARIES = ('first_action_exact', 'action_family_match')
GUARDRAILS = ('ward_only_recall', 'no_buy_death_recall')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--policy-version', required=True)
    parser.add_argument('--addendum', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=400)
    parser.add_argument('--seed', type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a fresh stage-2 report directory')
    args.out.mkdir(parents=True)

    import os
    children = []
    for index in range(SHARDS):
        log = (args.out / f'shard{index}.log').open('w', encoding='utf-8')
        env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '3'}
        # No CREATE_SUSPENDED here: that flag belongs to supervisors that pair
        # it with a job-object attach+resume. Children inherit this process's
        # job (and its memory ceiling) automatically.
        child = subprocess.Popen([sys.executable, 'scripts/footwear_runtime.py', '--entry',
            'scripts/eval_basket_composition.py', '--',
            '--dataset', str(args.dataset), '--artifact', str(args.artifact),
            '--policy-version', args.policy_version, '--addendum', str(args.addendum),
            '--out', str(args.out), '--shard-index', str(index), '--shard-count', str(SHARDS)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((index, child, log))
    failures = []
    for index, child, log in children:
        code = child.wait()
        log.close()
        if code:
            failures.append((index, code))
    if failures:
        raise RuntimeError(f'shards failed: {failures}; inspect shard logs')

    from app.dataset_artifacts import sha256, write_json
    from eval_policy import METRICS, summarize, paired_diff

    assignments = json.loads((args.dataset / 'split_assignment.json').read_text(encoding='utf-8'))
    val_games = {mid for mid, fold in assignments.items() if fold == 'val'}
    outcomes = {'A': defaultdict(list), 'B': defaultdict(list)}
    invariants = Counter()
    unaffordable = Counter()
    visits = 0
    shard_bindings = {}
    seen_games = set()
    for index in range(SHARDS):
        summary = json.loads((args.out / f'shard{index}_summary.json').read_text(encoding='utf-8'))
        shard_games = set(summary['games'])
        if shard_games & seen_games:
            raise ValueError('Shards overlap; the paired merge would double-count games')
        seen_games |= shard_games
        invariants.update(summary['invariants'])
        unaffordable.update(summary['unaffordable'])
        outcome_path = Path(summary['outcomes'])
        if sha256(outcome_path) != summary['outcomes_sha256']:
            raise ValueError(f'Shard {index} outcomes changed after its summary was written')
        shard_bindings[outcome_path.name] = summary['outcomes_sha256']
        with outcome_path.open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                for tag in ('A', 'B'):
                    outcomes[tag][row['match_id']].append(row['policies'][tag]['metrics'])
                visits += 1
    if seen_games != val_games:
        raise ValueError(f'Shards cover {len(seen_games)} games but validation has {len(val_games)}')

    games = len(outcomes['B'])
    metrics = {tag: {key: summarize(outcomes[tag], key, args.bootstrap, args.seed) for key in METRICS}
               for tag in ('A', 'B')}
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
        'split': 'val', 'games': games, 'visits': visits,
        'execution': f'{SHARDS} disjoint game shards; statistics computed once at merge',
        'shard_outcomes_sha256': shard_bindings,
        'invariants': dict(invariants),
        'unaffordable': dict(unaffordable),
        'metrics': metrics, 'paired_diff_a_minus_b': paired,
        'primary_win': win, 'guardrails_hold': guardrails_hold,
        'verdict': ('supported development win' if win and guardrails_hold
                    else 'rejected under the predeclared rule'),
        'validation_spend_event': True,
        'final_test_evaluation': False, 'promotion_allowed': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(args.out / 'composition_eval_val.json', report)
    print(json.dumps({'verdict': report['verdict'],
        'primaries': {k: {'diff': paired[k]['diff'], 'ci95': paired[k]['ci95']} for k in PRIMARIES},
        'guardrails': {k: {'diff': paired[k]['diff'], 'ci95': paired[k]['ci95']} for k in GUARDRAILS},
        'unaffordable': dict(unaffordable),
        'report': str(args.out / 'composition_eval_val.json')}, indent=1), flush=True)


if __name__ == '__main__':
    main()
