"""Train a candidate whose wallet is a RANGE, then measure it on TRAIN.

Protocol: docs/experiment_gold_posterior_training_v1.md, frozen before this ran.

Stages: tests -> trunk -> graft -> slack sweep -> miss profile.

It STOPS before the validation comparison. The plan declares that evaluation
because the development validator requires it declared, but this generation's
window already carries ONE scored comparison (Phase D), and spending a second is
the owner's call, not a side effect of a training run.

Nothing here re-exports, re-reconstructs or promotes. The only deliberate
difference from the Phase D candidate is how the wallet is treated:
PREFIX_GOLD_POSTERIOR draws it per row from the fitted posterior and masks at
that posterior's q-th quantile instead of a flat GOLD_DRIFT.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json

GENERATION = '16.18-reconstructible-development-v3'
NAME = '16.18-gold-posterior-v1'
DATASET = ROOT / 'data/dataset_generations' / GENERATION
RUN = ROOT / 'data/experiments' / NAME
POSTERIOR = ROOT / 'data/reports/gold_posterior_v1/posterior.json'
REVIEW = ROOT / 'data/acceptance/16.18_reconstructible_20260914/inputs/static_review.json'
PROTOCOL = ROOT / 'docs/experiment_gold_posterior_training_v1.md'
MASK_Q = '0.65'
WALL_SECONDS = 20 * 3600

# Identical to the Phase D candidate except the wallet treatment, so the
# comparison isolates one change.
FIXED = {'PREFIX_DMODEL': '128', 'PREFIX_LAYERS': '4', 'PREFIX_FF': '256', 'PREFIX_HEADS': '8',
         'PREFIX_COSINE': '1', 'PREFIX_EXTRAS': '1', 'PREFIX_HISTORY': '0',
         'PREFIX_GOLDEST': '1', 'PREFIX_GOLDX': '1', 'PREFIX_RUNES': '1',
         'PREFIX_POSW': '8', 'PREFIX_POSW_SCHED': '8:24', 'PREFIX_POSW_CUSTOM': '0',
         'PREFIX_TBLEND': '0', 'PREFIX_SPLICE': '0',
         'PREFIX_GOLD_POSTERIOR': str(POSTERIOR), 'PREFIX_GOLD_MASK_Q': MASK_Q}
TRAINING = {
    'trunk': {'PREFIX_OUT': 'trunk.pt', 'PREFIX_EPOCHS': '16', 'PREFIX_SAVEW': '0',
              'PREFIX_TARGETW': '0', 'PREFIX_INIT_FROM': '', 'PREFIX_FREEZE': '0'},
    'graft': {'PREFIX_OUT': 'candidate.pt', 'PREFIX_EPOCHS': '6', 'PREFIX_SAVEW': '0.5',
              'PREFIX_TARGETW': '0.25', 'PREFIX_INIT_FROM': 'trunk.pt', 'PREFIX_FREEZE': '1'},
}
CODE_NAMES = tuple(sorted({
    *(f'app/{p.name}' for p in (ROOT / 'app').glob('*.py')),
    'scripts/train_prefix.py', 'scripts/eval_policy.py', 'scripts/footwear_runtime.py',
    'scripts/mask_slack_sweep_v1.py', 'scripts/miss_profile_v1.py',
    'scripts/gold_posterior_fit_v1.py', 'scripts/run_gold_posterior_training_v1.py'}))


def main() -> None:
    if RUN.exists():
        raise ValueError(f'Use a fresh isolated run directory: {RUN} exists')
    for path in (DATASET / 'generation_manifest.json', POSTERIOR, REVIEW, PROTOCOL):
        if not path.is_file():
            raise ValueError(f'missing required input: {path}')
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 60:
        raise RuntimeError(f'{free:.1f} GB free; training needs headroom for tensors and artifacts')

    generation = json.loads((DATASET / 'generation_manifest.json').read_text(encoding='utf-8'))
    RUN.mkdir(parents=True)
    # The bound recipe: exactly the keys current_training_environment() resolves.
    # PREFIX_RUNTIME_REVIEW and PREFIX_STAGE are process wiring, not recipe, and
    # including them would fail the plan's environment equality check.
    fixed = {**FIXED, 'PREFIX_ML_DIR': str(DATASET), 'PREFIX_RUN_DIR': str(RUN)}
    plan = {
        'schema': 1,
        'purpose': generation['purpose'],
        'patch': '16.18',
        'dataset': str(DATASET),
        'run_directory': str(RUN),
        'export_fingerprint': generation['export_fingerprint'],
        'generation_manifest_sha256': sha256(DATASET / 'generation_manifest.json'),
        'final_test_evaluation': False,
        'promotion_allowed': False,
        'production_hold_cleared': False,
        'owner_authorization': ('owner instructed the uncertain-wallet retrain on 2026-09-17: '
                                '"train this model with the uncertainty that the gold is in an '
                                'estimated range", "i want to retrain the model then run my metrics"'),
        'protocol_document': str(PROTOCOL.relative_to(ROOT)).replace('\\', '/'),
        'protocol_document_sha256': sha256(PROTOCOL),
        'question': ('Does drawing the training wallet from the measured gold posterior, and '
                     'masking at its q=0.65 quantile instead of a flat GOLD_DRIFT, produce a '
                     'candidate that beats the point-estimate candidate and depends less on '
                     'decode-time slack?'),
        'stopping_rule': ('Stop after graft and the TRAIN-side metrics. The paired validation '
                          'comparison is NOT run by this chain: this window already carries one '
                          'scored comparison and spending a second is an explicit owner decision.'),
        'primary_metrics': ['first_action_exact', 'action_family_match'],
        'populations': ['All scored TRAIN decisions for first_action_exact',
                        'TRAIN purchases whose next completed final is known for action_family_match'],
        'evaluation': {'split': 'val', 'games': 0, 'compare_conditional': True},
        'evaluation_seed': 16,
        'evaluation_bootstrap': 400,
        'resource_budget': {'max_wall_seconds': WALL_SECONDS, 'max_ram_gb': 12,
                            'minimum_free_disk_gb': 40, 'max_train_processes': 1},
        'gold_posterior': {'source': str(POSTERIOR.relative_to(ROOT)).replace('\\', '/'),
                           'sha256': sha256(POSTERIOR), 'mask_quantile': float(MASK_Q),
                           'labels_used': False},
        'fixed_environment': fixed,
        'training_stages': dict(TRAINING),
        'code_sha256': {name: sha256(ROOT / name) for name in CODE_NAMES},
        'protected_sha256': {str(p): (sha256(p) if p.is_file() else None) for p in
                             [ROOT / 'data/ml/prefix_model.pt',
                              ROOT / 'data/ml/deployment_manifest.json',
                              ROOT / 'data/ml/pipeline_hold.json',
                              DATASET / 'generation_manifest.json']},
    }
    plan_path = RUN / 'run_plan.json'
    write_json(plan_path, plan)
    plan_sha = sha256(plan_path)

    from app.dataset_generations import validate_development_run_plan
    validate_development_run_plan(DATASET, RUN, generation, plan_path)
    print(f'run plan validated: {plan_path}', flush=True)

    completed: list[str] = []
    current = 'preflight'
    start = time.monotonic()

    def status(state: str, **extra) -> None:
        write_json(RUN / 'status.json', {
            'state': state, 'stage': current, 'completed_stages': list(completed),
            'plan_sha256': plan_sha, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - start, **extra})

    def unchanged() -> None:
        if sha256(plan_path) != plan_sha:
            raise ValueError('Run plan changed mid-run')
        for name, digest in plan['code_sha256'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Code changed mid-run: {name}')

    def budget() -> None:
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError('Wall-time budget reached')
        if shutil.disk_usage(ROOT).free < plan['resource_budget']['minimum_free_disk_gb'] * 1e9:
            raise RuntimeError('Disk reserve reached')

    sweep_out = ROOT / 'data/reports/mask_slack_sweep_posterior_v1'
    miss_out = ROOT / 'data/reports/miss_profile_posterior_v1'
    stages = [
        ('tests', [sys.executable, 'scripts/run_tests.py'], {}),
        ('trunk', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/train_prefix.py'],
         {**fixed, **TRAINING['trunk'], 'PREFIX_STAGE': 'trunk',
          'PREFIX_RUN_PLAN': str(plan_path), 'PREFIX_RUNTIME_REVIEW': str(REVIEW)}),
        ('graft', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/train_prefix.py'],
         {**fixed, **TRAINING['graft'], 'PREFIX_STAGE': 'graft',
          'PREFIX_RUN_PLAN': str(plan_path), 'PREFIX_RUNTIME_REVIEW': str(REVIEW)}),
        # The wallet is uncertain at EVALUATION too, not only in training. A model
        # masked at the posterior's q=0.65 during training and then decoded at a
        # flat const:0 would be scored in a regime it was not trained for — the
        # exact train/decode mismatch this experiment exists to remove. So the
        # arms are quantiles of the same posterior, with const:0 and const:500
        # kept only as anchors against last night's table.
        ('slack_sweep', [sys.executable, 'scripts/mask_slack_sweep_v1.py',
                         '--artifact', 'candidate.pt',
                         '--arms', 'const:0,quant:0.50,quant:0.55,quant:0.60,quant:0.65,'
                                   'quant:0.70,quant:0.75,quant:0.80,quant:0.90,const:500',
                         '--posterior', str(POSTERIOR),
                         '--games', '500', '--seed', '16', '--bootstrap', '400',
                         '--out', str(sweep_out)],
         {**fixed, 'PREFIX_RUNTIME_REVIEW': str(REVIEW)}),
        # Both: the model's own operating rule, and the honest-wallet baseline
        # that makes it comparable with the point candidate's profile.
        ('miss_profile', [sys.executable, 'scripts/miss_profile_v1.py',
                          '--artifact', 'candidate.pt', '--arms', 'const:0,quant:0.65',
                          '--posterior', str(POSTERIOR),
                          '--games', '500', '--seed', '16', '--out', str(miss_out)],
         {**fixed, 'PREFIX_RUNTIME_REVIEW': str(REVIEW)}),
    ]

    try:
        for name, command, extra_env in stages:
            current = name
            unchanged()
            budget()
            status('running')
            env = {k: v for k, v in os.environ.items() if not k.startswith('PREFIX_')}
            env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', **extra_env)
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            with (RUN / f'{name}_{stamp}.log').open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT)
                while True:
                    try:
                        code = process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        budget()
                        status('running')
            if code:
                raise RuntimeError(f'{name} failed with exit {code}')
            completed.append(name)
            status('running')
            print(f'{name} complete', flush=True)
        current = 'done'
        status('completed', completed_stages=completed)
        print('chain complete; validation comparison deliberately NOT run', flush=True)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
