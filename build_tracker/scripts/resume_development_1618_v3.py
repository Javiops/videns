"""Resume Phase D after the 2026-09-15 01:42 external termination.

The chain died mid-`audit` with exit 1073807364 (DBG_TERMINATE_PROCESS — the
process was killed from outside, not a pipeline fault). What survived is the
expensive half: `process_reconstructible_1618 build` COMPLETED, writing
data/corpora/16.18-reconstructible-v3 (6,541 sources, 631,454 admitted rows,
15 GB) with its frozen manifest and mask. Only the semantic audit was
interrupted, and it writes nothing until it finishes, so the acceptance
directory holds no partial audit state.

This runner therefore re-runs the audit alone against the existing corpus and
then continues the original chain. It reuses the ALREADY FROZEN run plan
(data/experiments/16.18-reconstructible-development-v3/run_plan.json) and
re-verifies every code and protected digest recorded in it, so the same
contract governs the resumed run.

The recon evidence (run/status.json + run/completion.json) is written exactly
as run_reconstructible_1618.py writes it on success — build and audit did
both complete, across two invocations — and records that split explicitly in
`resumed_after_external_termination`. The interrupted status file is
preserved beside it as status_failed_audit_terminated.json.

No test reads, no promotion, no cross-export comparison.
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

NAME = '16.18-reconstructible-development-v3'
EVAL_VERSION = 7
ACCEPT = ROOT / 'data/acceptance/16.18_reconstructible_20260914'
CORPUS = ROOT / 'data/corpora/16.18-reconstructible-v3'
PRIOR_BASE = ROOT / 'data/acceptance/16.18_reconstructible_20260913'
PRIOR_GEN = ROOT / 'data/dataset_generations/16.18-reconstructible-development-v2'
PREPARED = ROOT / 'data/acceptance/16.18_development_v3_admission'
COVERAGE = ROOT / 'data/acceptance/16.18_development_v3_admission_coverage'
DATASET = ROOT / 'data/dataset_generations' / NAME
RUN = ROOT / 'data/experiments' / NAME
RECON_RUN = ACCEPT / 'run'
WALL_SECONDS = 14 * 3600


def main():
    plan_path = RUN / 'run_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    for path in (PREPARED, COVERAGE, DATASET):
        if path.exists():
            raise ValueError(f'Downstream output already exists: {path}')
    # Two resume points. When the corpus is gone the whole reconstruction runs
    # again under its own supervisor (which writes legitimate build+audit
    # evidence itself); when only the audit was lost, the audit alone re-runs
    # against the surviving corpus and this script writes that evidence.
    rebuild_recon = not (CORPUS / 'corpus_manifest.json').is_file()
    recon_status = None
    if not rebuild_recon:
        recon_status = json.loads((RECON_RUN / 'status.json').read_text(encoding='utf-8'))
        if recon_status.get('completed_stages') != ['build']:
            raise ValueError(f'Unexpected recon state to resume from: {recon_status.get("completed_stages")}')
    elif RECON_RUN.exists():
        raise ValueError('A full reconstruction needs a fresh run directory; quarantine the old one')

    code = plan['code_sha256']
    protected = plan['protected_before']

    def unchanged():
        for name, digest in code.items():
            if sha256(ROOT / name) != digest:
                raise ValueError('Frozen Phase D code changed since the plan froze: ' + name)
        for name, digest in protected.items():
            path = Path(name)
            if (sha256(path) if path.is_file() else None) != digest:
                raise ValueError('Protected artifact changed: ' + name)

    unchanged()
    fixed = plan['fixed_environment']
    start = time.monotonic()
    completed = ['prepare'] if rebuild_recon else ['prepare', 'recon_build']
    current = 'preflight'
    job = WindowsMemoryJob(10_000_000_000)

    def status(state, **extra):
        write_json(RUN / 'status.json', {'state': state, 'stage': current,
            'completed_stages': completed, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - start,
            'plan_sha256': sha256(plan_path),
            'resumed_after_external_termination': '2026-09-15T01:42 local, audit killed with 1073807364',
            **extra})

    def budget():
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError(f'{WALL_SECONDS}s resume budget reached')
        if shutil.disk_usage(ROOT).free < 40e9:
            raise RuntimeError('40 GB disk reserve reached')

    base_env = {k: v for k, v in os.environ.items() if not k.startswith('PREFIX_')}
    base_env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HTTP_PROXY='http://127.0.0.1:9',
                    HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    stages = [
        ('recon', [sys.executable, 'scripts/run_reconstructible_1618.py',
                   '--base', str(ACCEPT), '--corpus', str(CORPUS),
                   '--max-wall-seconds', str(9 * 3600)], {}) if rebuild_recon else
        ('audit', [sys.executable, 'scripts/process_reconstructible_1618.py', 'audit',
                   '--source-plan', str(ACCEPT / 'source_plan.json'), '--corpus', str(CORPUS),
                   '--out', str(ACCEPT / 'semantic_audit.json')], {}),
        ('verify', [sys.executable, 'scripts/verify_reconstructible_1618_evidence.py',
                    '--source-root', str(ROOT), '--base', str(ACCEPT), '--corpus', str(CORPUS),
                    '--out', str(RUN / 'prerequisite_verification.json')], {}),
        ('admission', [sys.executable, 'scripts/prepare_development_1618_v2.py',
                       '--source-root', str(ROOT), '--base', str(ACCEPT), '--corpus', str(CORPUS),
                       '--out', str(PREPARED), '--benchmark-id', NAME,
                       '--eval-version', str(EVAL_VERSION),
                       '--prior-base', str(PRIOR_BASE),
                       '--prior-generation', str(PRIOR_GEN)], {}),
        ('coverage', [sys.executable, 'scripts/benchmark_1618_entry.py', '--corpus', str(CORPUS),
                      '--entry', 'report_corpus_coverage', '--',
                      '--qualification', str(PREPARED / 'qualification.json'),
                      '--out', str(COVERAGE), '--benchmark-id', NAME,
                      '--eval-version', str(EVAL_VERSION),
                      '--purpose', 'patch_development_1618',
                      '--split-assignment', str(PREPARED / 'preserved_split_assignment.json'),
                      '--split-metadata', str(PREPARED / 'split_metadata.json')], {}),
        ('export', [sys.executable, 'scripts/benchmark_1618_entry.py', '--corpus', str(CORPUS),
                    '--entry', 'publish_patch_dataset', '--',
                    '--coverage', str(COVERAGE), '--name', NAME], {}),
        ('tests', [sys.executable, 'scripts/run_tests.py'], {}),
        ('trunk', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/train_prefix.py'],
         {**fixed, **plan['training_stages']['trunk'], 'PREFIX_STAGE': 'trunk',
          'PREFIX_RUN_PLAN': str(plan_path)}),
        ('graft', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/train_prefix.py'],
         {**fixed, **plan['training_stages']['graft'], 'PREFIX_STAGE': 'graft',
          'PREFIX_RUN_PLAN': str(plan_path)}),
        ('policy', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/eval_policy.py',
                    '--', '--artifact', 'candidate.pt', '--split', 'val', '--games', '0',
                    '--compare-conditional', '--bootstrap', '400', '--seed', '16'],
         {**fixed, 'PREFIX_RUN_PLAN': str(plan_path)}),
    ]
    try:
        for name, arguments, extra_env in stages:
            current = name
            unchanged()
            budget()
            status('running')
            with (RUN / (name + '_resume.log')).open('a', encoding='utf-8') as log:
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
                raise RuntimeError(f'{name} failed with exit {code_result}; inspect {name}_resume.log')
            unchanged()
            completed.append(name)
            if name == 'audit':
                # Both recon stages have now completed, across two invocations.
                # Write the evidence exactly as run_reconstructible_1618.py does
                # on success (the verifier requires status == completion with
                # completed_stages ['build','audit']), preserving the killed run.
                qualification = json.loads((ACCEPT / 'qualification.json').read_text(encoding='utf-8'))
                if qualification.get('status') != 'qualified_reconstruction_snapshot':
                    raise ValueError('Qualification missing after the resumed audit')
                shutil.copy2(RECON_RUN / 'status.json',
                             RECON_RUN / 'status_failed_audit_terminated.json')
                recon_complete = {
                    'state': 'complete', 'stage': 'complete',
                    'completed_stages': ['build', 'audit'],
                    'time_utc': datetime.now(timezone.utc).isoformat(),
                    'elapsed_seconds': recon_status['elapsed_seconds'] + (time.monotonic() - start),
                    'peak_child_job_memory_bytes': max(
                        recon_status.get('peak_child_job_memory_bytes') or 0, job.peak_bytes()),
                    'selected_sources': recon_status['selected_sources'],
                    'accepted_games': qualification['accepted_games'],
                    'accepted_cohorts': qualification['accepted_cohorts'],
                    'qualification': str(ACCEPT / 'qualification.json'),
                    'qualification_sha256': sha256(ACCEPT / 'qualification.json'),
                    'protected_unchanged': True, 'training_or_export': False,
                    'test_payload_reads': False,
                    'resumed_after_external_termination': (
                        'build completed 2026-09-15T00:08-01:4x local; audit was killed at 01:42 '
                        'with 1073807364 and re-ran to completion under scripts/resume_development_1618_v3.py'),
                }
                write_json(RECON_RUN / 'status.json', recon_complete)
                shutil.copy2(RECON_RUN / 'status.json', RECON_RUN / 'completion.json')
            if name == 'export':
                generation = json.loads((DATASET / 'generation_manifest.json')
                                        .read_text(encoding='utf-8'))
                plan['export_fingerprint'] = generation['export_fingerprint']
                plan['generation_manifest_sha256'] = sha256(DATASET / 'generation_manifest.json')
                write_json(plan_path, plan)
            status('running')
        report_path = next(iter(RUN.glob('policy_eval_val_*.json')), None)
        current = 'complete'
        status('complete',
               candidate_sha256=sha256(RUN / 'candidate.pt'),
               export_fingerprint=plan.get('export_fingerprint'),
               policy_report=str(report_path) if report_path else None,
               policy_report_sha256=sha256(report_path) if report_path else None,
               validation_or_test_reads='one declared conditional-baseline evaluation on val',
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
