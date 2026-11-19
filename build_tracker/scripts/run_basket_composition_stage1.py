"""Stage 1 of the basket-composition experiment: control training + miss partition.

Per docs/experiment_basket_composition_v1.md. Stages: full assert suite, frozen
run plan, fixed 16-epoch trunk, six-epoch frozen-trunk graft (the stage-2
control candidate), then the TRAIN-only miss diagnostic. No validation or test
payloads are opened; the v2 validation spend ledger is untouched by this run.
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
from app.dataset_generations import verify_generation, validate_development_run_plan
from run_patch_development import WindowsMemoryJob

DATASET = ROOT / 'data/dataset_generations/16.18-reconstructible-development-v2'
RUN = ROOT / 'data/experiments/basket-composition-v1'
WALL_SECONDS = 28800

FIXED = {'PREFIX_DMODEL': '128', 'PREFIX_LAYERS': '4', 'PREFIX_FF': '256', 'PREFIX_HEADS': '8',
         'PREFIX_COSINE': '1', 'PREFIX_EXTRAS': '1', 'PREFIX_HISTORY': '0',
         'PREFIX_GOLDEST': '1', 'PREFIX_GOLDX': '1', 'PREFIX_RUNES': '1',
         'PREFIX_POSW': '8', 'PREFIX_POSW_SCHED': '8:24', 'PREFIX_POSW_CUSTOM': '0',
         'PREFIX_TBLEND': '0', 'PREFIX_SPLICE': '0'}
TRAINING = {
    'trunk': {'PREFIX_OUT': 'trunk.pt', 'PREFIX_EPOCHS': '16', 'PREFIX_SAVEW': '0',
              'PREFIX_TARGETW': '0', 'PREFIX_INIT_FROM': '', 'PREFIX_FREEZE': '0'},
    'graft': {'PREFIX_OUT': 'candidate.pt', 'PREFIX_EPOCHS': '6', 'PREFIX_SAVEW': '0.5',
              'PREFIX_TARGETW': '0.25', 'PREFIX_INIT_FROM': 'trunk.pt', 'PREFIX_FREEZE': '1'},
}
CODE_NAMES = tuple(sorted({*(f'app/{p.name}' for p in (ROOT / 'app').glob('*.py')),
    'scripts/train_prefix.py', 'scripts/eval_policy.py', 'scripts/footwear_runtime.py',
    'scripts/mine_basket_misses_v2.py', 'scripts/corpus_rows.py', 'scripts/baseline.py',
    'scripts/benchmark_1618_runtime.py', 'scripts/reconstructible_1618_runtime.py',
    'scripts/run_basket_composition_stage1.py'}))


def main():
    generation = verify_generation(DATASET)
    if RUN.exists():
        raise ValueError('Use a fresh isolated run directory')
    RUN.mkdir(parents=True)
    code = {name: sha256(ROOT / name) for name in CODE_NAMES}
    protected_paths = [ROOT / 'data/ml' / n for n in
                       ('prefix_model.pt', 'deployment_manifest.json', 'pipeline_hold.json',
                        'v2_validation_selection_log.md')]
    protected = {str(p): sha256(p) if p.is_file() else None for p in protected_paths}
    fixed = {**FIXED, 'PREFIX_ML_DIR': str(DATASET), 'PREFIX_RUN_DIR': str(RUN)}
    plan = {'schema': 1, 'purpose': 'patch_development_1618', 'patch': '16.18',
        'dataset': str(DATASET), 'run_directory': str(RUN), 'code_root': str(ROOT),
        'benchmark_id': '16.18-reconstructible-development-v2', 'eval_version': 7,
        'export_fingerprint': generation['export_fingerprint'],
        'generation_manifest_sha256': sha256(DATASET / 'generation_manifest.json'),
        'owner_authorization': 'owner instructed stages 1 and 2 on 2026-09-13',
        'protocol_document': 'docs/experiment_basket_composition_v1.md',
        'protocol_document_sha256': sha256(ROOT / 'docs/experiment_basket_composition_v1.md'),
        'question': ('Can changing how candidate baskets are generated raise displayed-policy '
                     'imitation on multi-item decisions without degrading save/ward behavior?'),
        'primary_metrics': ['first_action_exact', 'action_family_match'],
        'populations': ['All scored validation decisions for first_action_exact',
                        'Validation purchases with a known next final for action_family_match'],
        'stopping_rule': ('Stage 1: one fixed trunk+graft training and one TRAIN-only miss '
                          'partition; stop on any failed gate. Stage 2 needs its own frozen '
                          'addendum and exactly one paired decoder comparison. No sweeps, no '
                          'promotion, no test reads.'),
        'evaluation': {'split': 'val', 'games': 0, 'compare_conditional': True},
        'evaluation_seed': 16, 'evaluation_bootstrap': 400,
        'resource_budget': {'max_wall_seconds': WALL_SECONDS, 'max_train_processes': 1,
                            'max_ram_gb': 10, 'minimum_free_disk_gb': 40},
        'fixed_environment': fixed, 'training_stages': TRAINING,
        'code_sha256': code, 'protected_before': protected,
        'final_test_evaluation': False, 'promotion_allowed': False,
        'production_hold_cleared': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    plan_path = RUN / 'run_plan.json'
    write_json(plan_path, plan)
    validate_development_run_plan(DATASET, RUN, generation, plan_path)
    for name in CODE_NAMES:
        target = RUN / 'code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)

    start = time.monotonic()
    completed, current = [], 'preflight'
    job = WindowsMemoryJob(10_000_000_000)

    def status(state, **extra):
        write_json(RUN / 'status.json', {'state': state, 'stage': current,
            'completed_stages': completed, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - start,
            'plan_sha256': sha256(plan_path), **extra})

    def unchanged():
        for name, digest in code.items():
            if sha256(ROOT / name) != digest:
                raise ValueError('Frozen stage-1 code changed: ' + name)
        for name, digest in protected.items():
            path = Path(name)
            if (sha256(path) if path.is_file() else None) != digest:
                raise ValueError('Protected artifact changed: ' + name)

    def budget():
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError(f'{WALL_SECONDS}s stage-1 budget reached')
        if shutil.disk_usage(ROOT).free < 40e9:
            raise RuntimeError('40 GB disk reserve reached')

    base_env = {k: v for k, v in os.environ.items() if not k.startswith('PREFIX_')}
    base_env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HTTP_PROXY='http://127.0.0.1:9',
                    HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    stages = [
        ('tests', [sys.executable, 'scripts/run_tests.py'], {}),
        ('trunk', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/train_prefix.py'],
         {**fixed, **TRAINING['trunk'], 'PREFIX_STAGE': 'trunk', 'PREFIX_RUN_PLAN': str(plan_path)}),
        ('graft', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/train_prefix.py'],
         {**fixed, **TRAINING['graft'], 'PREFIX_STAGE': 'graft', 'PREFIX_RUN_PLAN': str(plan_path)}),
        ('mine', [sys.executable, 'scripts/footwear_runtime.py', '--entry',
                  'scripts/mine_basket_misses_v2.py', '--',
                  '--dataset', str(DATASET), '--artifact', str(RUN / 'candidate.pt'),
                  '--out', str(RUN / 'diagnostic')],
         {**fixed, 'PREFIX_RUN_PLAN': str(plan_path)}),
    ]
    try:
        for name, arguments, extra_env in stages:
            current = name
            unchanged()
            budget()
            status('running')
            with (RUN / (name + '.log')).open('a', encoding='utf-8') as log:
                child = subprocess.Popen(arguments, cwd=ROOT, env={**base_env, **extra_env},
                                         stdout=log, stderr=subprocess.STDOUT, creationflags=4)
                job.attach(child)
                while True:
                    try:
                        code_result = child.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        budget()
                        status('running')
            if code_result:
                raise RuntimeError(f'{name} failed with exit {code_result}; inspect {name}.log')
            unchanged()
            completed.append(name)
            status('running')
        partition = json.loads((RUN / 'diagnostic/train_miss_partition.json').read_text(encoding='utf-8'))
        current = 'complete'
        status('complete', candidate_sha256=sha256(RUN / 'candidate.pt'),
               diagnostic=str(RUN / 'diagnostic/train_miss_partition.json'),
               diagnostic_sha256=sha256(RUN / 'diagnostic/train_miss_partition.json'),
               class_counts=partition['class_counts'],
               validation_or_test_reads=False, promotion_allowed=False)
        shutil.copy2(RUN / 'status.json', RUN / 'completion.json')
    except Exception as exc:
        job.stop()
        status('failed', reason=str(exc))
        raise
    finally:
        job.close()


if __name__ == '__main__':
    main()
