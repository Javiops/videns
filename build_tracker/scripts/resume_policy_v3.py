"""Re-run ONLY the Phase D policy evaluation after the 05:36 Windows Update reboot.

Everything through `graft` completed and is preserved (trunk.pt, candidate.pt,
the published generation). The evaluation has no mid-run checkpoint: it was
88.6% through (170,000 of 191,916 validation rows) when TrustedInstaller
forced a restart, so it starts over. Its interrupted outputs — the partial
paired-visit stream and the fitted conditional baseline — are quarantined as
*_killed_by_winupdate_0536.*, and the evaluator re-fits the baseline from the
same frozen TRAIN split (deterministic).

Reuses the already frozen run plan and re-verifies every code and protected
digest recorded in it, so the same contract governs. One evaluation, on
validation, against this generation's own conditional lookup. No test reads,
no promotion, no cross-export comparison.
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
DATASET = ROOT / 'data/dataset_generations' / NAME
RUN = ROOT / 'data/experiments' / NAME
WALL_SECONDS = 6 * 3600


def main():
    plan_path = RUN / 'run_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if not (RUN / 'candidate.pt').is_file():
        raise ValueError('No graft candidate to evaluate')
    if next(iter(RUN.glob('policy_eval_val_*.json')), None):
        raise ValueError('A policy report already exists; refusing to score twice')
    for stale in ('conditional_policy.json',
                  'policy_outcomes_val_candidate_vs_conditional.jsonl',
                  'policy_outcomes_val_candidate_vs_conditional.jsonl.partial'):
        if (RUN / stale).exists():
            raise ValueError(f'Quarantine the interrupted evidence first: {stale}')
    generation = json.loads((DATASET / 'generation_manifest.json').read_text(encoding='utf-8'))
    if plan.get('export_fingerprint') != generation['export_fingerprint']:
        raise ValueError('The frozen plan is not bound to the published generation')

    code, protected = plan['code_sha256'], plan['protected_before']

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
    completed = ['prepare', 'recon', 'verify', 'admission', 'coverage', 'export',
                 'tests', 'trunk', 'graft']
    current = 'policy'
    job = WindowsMemoryJob(10_000_000_000)

    def status(state, **extra):
        write_json(RUN / 'status.json', {'state': state, 'stage': current,
            'completed_stages': completed, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - start,
            'plan_sha256': sha256(plan_path),
            'interrupted_runs': ['2026-09-15T01:42 audit killed (1073807364)',
                                 '2026-09-15T05:36 policy killed by a Windows Update reboot at 88.6%'],
            **extra})

    env = {k: v for k, v in os.environ.items() if not k.startswith('PREFIX_')}
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HTTP_PROXY='http://127.0.0.1:9',
               HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='',
               **fixed, PREFIX_RUN_PLAN=str(plan_path))
    arguments = [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/eval_policy.py',
                 '--', '--artifact', 'candidate.pt', '--split', 'val', '--games', '0',
                 '--compare-conditional', '--bootstrap', '400', '--seed', '16']
    try:
        status('running')
        with (RUN / 'policy_rerun.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen(arguments, cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT, creationflags=4)
            job.attach(child)
            while True:
                try:
                    code_result = child.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() - start > WALL_SECONDS:
                        raise RuntimeError('policy rerun budget reached')
                    status('running')
        if code_result:
            raise RuntimeError(f'policy failed with exit {code_result}; inspect policy_rerun.log')
        unchanged()
        completed.append('policy')
        report_path = next(iter(RUN.glob('policy_eval_val_*.json')), None)
        current = 'complete'
        status('complete',
               candidate_sha256=sha256(RUN / 'candidate.pt'),
               export_fingerprint=plan['export_fingerprint'],
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
