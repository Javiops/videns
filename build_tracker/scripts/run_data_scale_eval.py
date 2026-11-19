"""Arm A of docs/experiment_data_scale_v1.md: one paired cross-corpus eval.

Stages: full assert suite, frozen run plan (artifact bytes, training-plan
digests, code digests, protected serving files), ONE paired evaluation of the
16.17-trained candidate against the 16.18-only control on the frozen v2
validation, then the validation-spend ledger line and completion record.

No training happens here — both artifacts already exist under their own
frozen plans. No test reads, no promotion, no re-tuning on any outcome.
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
from app.dataset_generations import verify_generation, validate_run_directory
from run_patch_development import WindowsMemoryJob

DATASET = ROOT / 'data/dataset_generations/16.18-reconstructible-development-v2'
RUN = ROOT / 'data/experiments/data-scale-v1-arm-1617'
ARM_DIR = ROOT / 'data/experiments/16.17-development-v2'
CONTROL_DIR = ROOT / 'data/experiments/basket-composition-v1'
LEDGER = ROOT / 'data/ml/v2_validation_selection_log.md'
WALL_SECONDS = 28800

CODE_NAMES = tuple(sorted({*(f'app/{p.name}' for p in (ROOT / 'app').glob('*.py')),
    'scripts/train_prefix.py', 'scripts/eval_policy.py', 'scripts/eval_data_scale_v1.py',
    'scripts/footwear_runtime.py', 'scripts/corpus_rows.py', 'scripts/baseline.py',
    'scripts/benchmark_1618_runtime.py', 'scripts/reconstructible_1618_runtime.py',
    'scripts/run_data_scale_eval.py'}))


def main():
    generation = verify_generation(DATASET)
    if RUN.exists():
        raise ValueError('Use a fresh isolated run directory')
    validate_run_directory(DATASET, RUN)
    RUN.mkdir(parents=True)
    code = {name: sha256(ROOT / name) for name in CODE_NAMES}
    protected_paths = [ROOT / 'data/ml' / n for n in
                       ('prefix_model.pt', 'deployment_manifest.json', 'pipeline_hold.json')]
    protected_paths += [ARM_DIR / 'candidate.pt', CONTROL_DIR / 'candidate.pt']
    protected = {str(p): sha256(p) if p.is_file() else None for p in protected_paths}
    plan = {'schema': 1, 'purpose': 'data_scale_v1_paired_eval', 'patch': '16.18',
        'dataset': str(DATASET), 'run_directory': str(RUN), 'code_root': str(ROOT),
        'export_fingerprint': generation['export_fingerprint'],
        'generation_manifest_sha256': sha256(DATASET / 'generation_manifest.json'),
        'owner_authorization': ('owner authorized the two-arm data-scale comparison on '
                                '2026-09-14 ~00:30 (16.17-solo + joint; joint pending its addendum)'),
        'protocol_document': 'docs/experiment_data_scale_v1.md',
        'protocol_document_sha256': sha256(ROOT / 'docs/experiment_data_scale_v1.md'),
        'question': ('Does training the fixed V2 recipe on more adjacent-patch (16.17) data produce '
                     'better displayed-policy imitation on the frozen 16.18 v2 validation than the '
                     'same recipe trained on 16.18 data alone?'),
        'primary_metrics': ['first_action_exact', 'action_family_match'],
        'populations': ['All scored validation decisions for first_action_exact',
                        'Validation purchases with a known next final for action_family_match'],
        'stopping_rule': ('Exactly one paired evaluation; a loss or guardrail failure is a retained '
                          'negative result; no re-tuning on this validation; arm B needs its own '
                          'frozen addendum. No sweeps, no promotion, no test reads.'),
        'evaluation_seed': 16, 'evaluation_bootstrap': 400,
        'arm': {'artifact': str(ARM_DIR / 'candidate.pt'),
                'artifact_sha256': sha256(ARM_DIR / 'candidate.pt'),
                'training_run_plan_sha256': sha256(ARM_DIR / 'run_plan.json'),
                'training_dataset': str(ROOT / 'data/dataset_generations/16.17-development-v2'),
                'training_export_fingerprint': json.loads((ROOT /
                    'data/dataset_generations/16.17-development-v2/generation_manifest.json'
                    ).read_text(encoding='utf-8'))['export_fingerprint']},
        'control': {'artifact': str(CONTROL_DIR / 'candidate.pt'),
                    'artifact_sha256': sha256(CONTROL_DIR / 'candidate.pt'),
                    'training_run_plan_sha256': sha256(CONTROL_DIR / 'run_plan.json'),
                    'training_dataset': str(DATASET),
                    'training_export_fingerprint': generation['export_fingerprint']},
        'resource_budget': {'max_wall_seconds': WALL_SECONDS, 'max_train_processes': 0,
                            'max_ram_gb': 10, 'minimum_free_disk_gb': 40},
        'code_sha256': code, 'protected_before': protected,
        'ledger_before_sha256': sha256(LEDGER) if LEDGER.is_file() else None,
        'final_test_evaluation': False, 'promotion_allowed': False,
        'production_hold_cleared': False,
        'created_utc': datetime.now(timezone.utc).isoformat()}
    plan_path = RUN / 'run_plan.json'
    write_json(plan_path, plan)
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
                raise ValueError('Frozen evaluation code changed: ' + name)
        for name, digest in protected.items():
            path = Path(name)
            if (sha256(path) if path.is_file() else None) != digest:
                raise ValueError('Protected artifact changed: ' + name)

    def budget():
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError(f'{WALL_SECONDS}s budget reached')
        if shutil.disk_usage(ROOT).free < 40e9:
            raise RuntimeError('40 GB disk reserve reached')

    base_env = {k: v for k, v in os.environ.items() if not k.startswith('PREFIX_')}
    base_env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HTTP_PROXY='http://127.0.0.1:9',
                    HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    stages = [
        ('tests', [sys.executable, 'scripts/run_tests.py'], {}),
        ('eval', [sys.executable, 'scripts/footwear_runtime.py', '--entry',
                  'scripts/eval_data_scale_v1.py', '--', '--plan', str(plan_path)],
         {'PREFIX_ML_DIR': str(DATASET), 'PREFIX_RUN_DIR': str(RUN)}),
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
        report_path = RUN / 'report/data_scale_eval_val.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        current = 'ledger'
        with LEDGER.open('a', encoding='utf-8') as ledger:
            ledger.write(f"| {datetime.now(timezone.utc).date()} | data-scale-v1 arm A (16.17-solo) "
                         f"| {plan['arm']['artifact_sha256'][:12]} vs {plan['control']['artifact_sha256'][:12]} "
                         f"| {report_path} |\n")
        completed.append('ledger')
        current = 'complete'
        status('complete', report=str(report_path), report_sha256=sha256(report_path),
               verdict=report.get('verdict'), primary_win=report.get('primary_win'),
               guardrails_hold=report.get('guardrails_hold'),
               validation_spend_recorded=True, promotion_allowed=False)
        shutil.copy2(RUN / 'status.json', RUN / 'completion.json')
    except Exception as exc:
        job.stop()
        status('failed', reason=str(exc))
        raise
    finally:
        job.close()


if __name__ == '__main__':
    main()
