"""Stage 2 of the basket-composition experiment: one paired decoder comparison.

Requires: stage-1 completion evidence, a frozen dated addendum naming exactly
one composition policy, and the stage-1 candidate checkpoint. Runs the full
assert suite, then the single paired evaluation on the v2 validation, then
appends the validation-spend ledger line. One comparison, no sweeps, no
promotion, no test reads.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json
from run_patch_development import WindowsMemoryJob

DATASET = ROOT / 'data/dataset_generations/16.18-reconstructible-development-v2'
STAGE1 = ROOT / 'data/experiments/basket-composition-v1'
RUN = STAGE1 / 'stage2'
ADDENDUM = ROOT / 'docs/experiment_basket_composition_v1_addendum.md'
LEDGER = ROOT / 'data/ml/v2_validation_selection_log.md'
WALL_SECONDS = 28800


def main():
    completion = json.loads((STAGE1 / 'completion.json').read_text(encoding='utf-8'))
    if completion.get('state') != 'complete':
        raise ValueError('Stage 1 did not complete cleanly')
    if not ADDENDUM.is_file():
        raise ValueError('Stage 2 requires the frozen dated addendum')
    addendum_text = ADDENDUM.read_text(encoding='utf-8')
    policy_line = [l for l in addendum_text.splitlines() if l.startswith('frozen_policy:')]
    if len(policy_line) != 1:
        raise ValueError('The addendum must declare exactly one frozen_policy: line')
    policy = policy_line[0].split(':', 1)[1].strip()
    if RUN.exists():
        raise ValueError('Use a fresh stage-2 run directory')
    RUN.mkdir()
    artifact = STAGE1 / 'candidate.pt'
    if sha256(artifact) != completion['candidate_sha256']:
        raise ValueError('Stage-1 candidate changed after completion')
    plan = {'schema': 1, 'purpose': 'basket_composition_v1_stage2',
        'policy_version': policy, 'addendum': str(ADDENDUM), 'addendum_sha256': sha256(ADDENDUM),
        'artifact_sha256': completion['candidate_sha256'],
        'stage1_completion_sha256': sha256(STAGE1 / 'completion.json'),
        'diagnostic_sha256': completion['diagnostic_sha256'],
        'dataset': str(DATASET),
        'generation_manifest_sha256': sha256(DATASET / 'generation_manifest.json'),
        'evaluation_seed': 16, 'evaluation_bootstrap': 400,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(RUN / 'stage2_plan.json', plan)

    start = time.monotonic()
    completed, current = [], 'preflight'
    job = WindowsMemoryJob(12_000_000_000)

    def status(state, **extra):
        write_json(RUN / 'status.json', {'state': state, 'stage': current,
            'completed_stages': completed, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - start, **extra})

    def budget():
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError(f'{WALL_SECONDS}s stage-2 budget reached')
        if shutil.disk_usage(ROOT).free < 40e9:
            raise RuntimeError('40 GB disk reserve reached')

    env = {k: v for k, v in os.environ.items() if not k.startswith('PREFIX_')}
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HTTP_PROXY='http://127.0.0.1:9',
               HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    # The generation binding reaches ONLY the eval stage: leaking PREFIX_* into
    # the assert suite reroutes trainer tests into the development path.
    eval_env = {'PREFIX_ML_DIR': str(DATASET), 'PREFIX_RUN_DIR': str(RUN)}
    stages = [
        ('tests', [sys.executable, 'scripts/run_tests.py'], {}),
        ('eval', [sys.executable, 'scripts/run_composition_eval_parallel.py',
                  '--dataset', str(DATASET), '--artifact', str(artifact),
                  '--policy-version', policy, '--addendum', str(ADDENDUM),
                  '--out', str(RUN / 'report'), '--bootstrap', '400', '--seed', '16'],
         eval_env),
    ]
    try:
        for name, arguments, extra_env in stages:
            current = name
            budget()
            status('running')
            with (RUN / (name + '.log')).open('a', encoding='utf-8') as log:
                child = subprocess.Popen(arguments, cwd=ROOT, env={**env, **extra_env},
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=4)
                job.attach(child)
                while True:
                    try:
                        code = child.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        budget()
                        status('running')
            if code:
                raise RuntimeError(f'{name} failed with exit {code}; inspect {name}.log')
            completed.append(name)
            status('running')
        report_path = RUN / 'report/composition_eval_val.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        with LEDGER.open('a', encoding='utf-8') as ledger:
            ledger.write(f"| {datetime.now(timezone.utc).date()} | basket-composition-v1 stage 2 "
                         f"({policy}) | {completion['candidate_sha256'][:12]} composed-vs-original | "
                         f"{report_path} |\n")
        current = 'complete'
        status('complete', verdict=report['verdict'], report=str(report_path),
               report_sha256=sha256(report_path), validation_spend_recorded=True,
               promotion_allowed=False)
        shutil.copy2(RUN / 'status.json', RUN / 'completion.json')
    except Exception as exc:
        job.stop()
        status('failed', reason=str(exc))
        raise
    finally:
        job.close()


if __name__ == '__main__':
    main()
