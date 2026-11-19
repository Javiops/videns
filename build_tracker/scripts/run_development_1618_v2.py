"""Durable admission, coverage and immutable export for the v2 development benchmark.

Runs only after the v2 reconstruction/qualification job has completed. Stages:
prerequisite verification, forward-validation admission, coverage, immutable
export. No Riot requests, no training, no test payloads, no promotion.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json
from app.dataset_generations import verify_generation
from run_patch_development import WindowsMemoryJob

NAME = '16.18-reconstructible-development-v2'
EVAL_VERSION = 7
BASE = ROOT / 'data/acceptance/16.18_reconstructible_20260913'
CORPUS = ROOT / 'data/corpora/16.18-reconstructible-v2'
PREPARED = ROOT / 'data/acceptance/16.18_development_v2_admission'
COVERAGE = ROOT / 'data/acceptance/16.18_development_v2_admission_coverage'
DATASET = ROOT / 'data/dataset_generations' / NAME
RUN = BASE / 'run_admission'
WALL_SECONDS = 5400


def main():
    completion = BASE / 'run/completion.json'
    if not completion.is_file():
        raise ValueError('The v2 reconstruction job has not completed')
    if json.loads(completion.read_text(encoding='utf-8')).get('state') != 'complete':
        raise ValueError('The v2 reconstruction job did not complete cleanly')
    if RUN.exists() or PREPARED.exists() or COVERAGE.exists() or DATASET.exists():
        raise ValueError('Use fresh immutable admission outputs')
    RUN.mkdir()
    start = time.monotonic()
    completed, current = [], 'preflight'
    job = WindowsMemoryJob(6_000_000_000)

    def status(state, **extra):
        write_json(RUN / 'status.json', {'state': state, 'stage': current,
            'completed_stages': completed, 'time_utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds': time.monotonic() - start, **extra})

    def budget():
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError(f'{WALL_SECONDS}s admission budget reached')
        if shutil.disk_usage(ROOT).free < 40e9:
            raise RuntimeError('40 GB disk reserve reached')

    env = {k: v for k, v in __import__('os').environ.items() if not k.startswith('PREFIX_')}
    env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HTTP_PROXY='http://127.0.0.1:9',
               HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    stages = [
        ('verify', [sys.executable, 'scripts/verify_reconstructible_1618_evidence.py',
                    '--source-root', str(ROOT), '--base', str(BASE), '--corpus', str(CORPUS),
                    '--out', str(RUN / 'prerequisite_verification.json')]),
        ('admission', [sys.executable, 'scripts/prepare_development_1618_v2.py',
                       '--source-root', str(ROOT), '--base', str(BASE), '--corpus', str(CORPUS),
                       '--out', str(PREPARED), '--benchmark-id', NAME,
                       '--eval-version', str(EVAL_VERSION)]),
        ('coverage', [sys.executable, 'scripts/benchmark_1618_entry.py', '--corpus', str(CORPUS),
                      '--entry', 'report_corpus_coverage', '--',
                      '--qualification', str(PREPARED / 'qualification.json'),
                      '--out', str(COVERAGE), '--benchmark-id', NAME,
                      '--eval-version', str(EVAL_VERSION),
                      '--purpose', 'patch_development_1618',
                      '--split-assignment', str(PREPARED / 'preserved_split_assignment.json'),
                      '--split-metadata', str(PREPARED / 'split_metadata.json')]),
        ('export', [sys.executable, 'scripts/benchmark_1618_entry.py', '--corpus', str(CORPUS),
                    '--entry', 'publish_patch_dataset', '--',
                    '--coverage', str(COVERAGE), '--name', NAME]),
    ]
    try:
        for name, arguments in stages:
            current = name
            budget()
            status('running')
            with (RUN / (name + '.log')).open('x', encoding='utf-8') as log:
                child = subprocess.Popen(arguments, cwd=ROOT, env=env,
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
        generation = verify_generation(DATASET)
        current = 'complete'
        status('complete', dataset=str(DATASET),
               export_fingerprint=generation['export_fingerprint'],
               generation_manifest_sha256=sha256(DATASET / 'generation_manifest.json'),
               qualification_sha256=sha256(PREPARED / 'qualification.json'),
               training_or_promotion=False, test_payload_reads=False)
        shutil.copy2(RUN / 'status.json', RUN / 'completion.json')
    except Exception as exc:
        job.stop()
        status('failed', reason=str(exc))
        raise
    finally:
        job.close()


if __name__ == '__main__':
    main()
