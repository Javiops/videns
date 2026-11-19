"""Phase D of the leak-safe inputs plan: ONE rebuild, ONE candidate, ONE eval.

Owner-authorized 2026-09-14 ("haz D"). Chain, all on FRESH immutable paths:

  1. prepare  — freeze a new dated source plan + static review over every
                16.18 game now in the DB (includes tonight's pull).
  2. recon    — build + audit + qualify the corpus with the opening-sale /
                prequential-v3 reconstruction (scripts/run_reconstructible_1618.py).
  3. verify / admission / coverage / export — forward-validation-v1 rolled one
                step: boundary = the 20260913 snapshot instant, the v2
                validation window (2 ledger comparisons spent) retires from
                every fold, newer qualified games become the fresh validation.
  4. tests / trunk / graft — the fixed V2 recipe on the new generation.
  5. policy   — eval_policy --compare-conditional, all validation games,
                bootstrap 400 seed 16 (the only development evaluation the
                validators permit).

No test reads, no promotion, no cross-export comparisons; the candidate's
numbers live only against this generation's own conditional baseline.
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
WALL_SECONDS = 16 * 3600

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
    'scripts/corpus_rows.py', 'scripts/baseline.py', 'scripts/benchmark_1618_runtime.py',
    'scripts/reconstructible_1618_runtime.py', 'scripts/prepare_reconstructible_1618.py',
    'scripts/run_reconstructible_1618.py', 'scripts/process_reconstructible_1618.py',
    'scripts/prepare_development_1618_v2.py', 'scripts/benchmark_1618_entry.py',
    'scripts/verify_reconstructible_1618_evidence.py', 'scripts/run_development_1618_v3.py'}))


def main():
    for path in (ACCEPT, CORPUS, PREPARED, COVERAGE, DATASET, RUN):
        if path.exists():
            raise ValueError(f'Use fresh immutable outputs: {path} exists')
    if not (PRIOR_BASE / 'source_plan.json').is_file():
        raise ValueError('Prior snapshot evidence is missing')
    ACCEPT.mkdir(parents=True)
    RUN.mkdir(parents=True)
    # The static review's one network fetch (the pinned runesReforged CDN
    # file) is already frozen byte-for-byte in the 20260912 evidence and the
    # prepare step skips the fetch when the file exists. Seeding it keeps the
    # ENTIRE chain inside the proxy blackhole — the first launch died here
    # with a ConnectError against the blackhole.
    seed = ROOT / 'data/acceptance/16.18_reconstructible_20260912/inputs/runesReforged-16.18.1.json'
    if seed.is_file():
        (ACCEPT / 'inputs').mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed, ACCEPT / 'inputs' / seed.name)
    code = {name: sha256(ROOT / name) for name in CODE_NAMES}
    protected_paths = [ROOT / 'data/ml' / n for n in
                       ('prefix_model.pt', 'deployment_manifest.json', 'pipeline_hold.json',
                        'v2_validation_selection_log.md')]
    protected_paths += [PRIOR_BASE / 'source_plan.json', PRIOR_GEN / 'generation_manifest.json']
    protected = {str(p): sha256(p) if p.is_file() else None for p in protected_paths}
    fixed = {**FIXED, 'PREFIX_ML_DIR': str(DATASET), 'PREFIX_RUN_DIR': str(RUN)}
    plan = {'schema': 1, 'purpose': 'patch_development_1618', 'patch': '16.18',
        'dataset': str(DATASET), 'run_directory': str(RUN), 'code_root': str(ROOT),
        'benchmark_id': NAME, 'eval_version': EVAL_VERSION,
        'owner_authorization': 'owner instructed Phase D and the window daily pull on 2026-09-14 (~20:40)',
        'protocol_document': 'docs/experiment_gold_prequential_v3.md',
        'protocol_document_sha256': sha256(ROOT / 'docs/experiment_gold_prequential_v3.md'),
        'plan_document': 'C:/Users/Javier/.claude/plans/sparkling-bouncing-sparrow.md',
        'question': ('Does the corpus rebuilt with corrected inputs (prequential-v3 opening-sale '
                     'credit + decision inventory) support a fixed-recipe candidate that beats the '
                     'conditional lookup on its own fresh forward validation?'),
        'primary_metrics': ['first_action_exact', 'action_family_match'],
        'populations': ['All scored validation decisions for first_action_exact',
                        'Validation purchases with a known next final for action_family_match'],
        'stopping_rule': ('One reconstruction, one admission, one fixed trunk+graft training and '
                          'exactly one paired conditional-baseline evaluation; stop on any failed '
                          'gate; no sweeps, no promotion, no test reads, no cross-export claims.'),
        # export_fingerprint / generation_manifest_sha256 are stamped by the
        # plan-finalize step right after the export stage publishes the
        # generation; the training/eval validators require them to match it.
        'export_fingerprint': None, 'generation_manifest_sha256': None,
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
                raise ValueError('Frozen Phase D code changed: ' + name)
        for name, digest in protected.items():
            path = Path(name)
            if (sha256(path) if path.is_file() else None) != digest:
                raise ValueError('Protected artifact changed: ' + name)

    def budget():
        if time.monotonic() - start > WALL_SECONDS:
            raise RuntimeError(f'{WALL_SECONDS}s Phase D budget reached')
        if shutil.disk_usage(ROOT).free < 40e9:
            raise RuntimeError('40 GB disk reserve reached')

    base_env = {k: v for k, v in os.environ.items() if not k.startswith('PREFIX_')}
    base_env.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', HTTP_PROXY='http://127.0.0.1:9',
                    HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9', NO_PROXY='')
    stages = [
        ('prepare', [sys.executable, 'scripts/prepare_reconstructible_1618.py',
                     '--base', str(ACCEPT)], {}),
        ('recon', [sys.executable, 'scripts/run_reconstructible_1618.py',
                   '--base', str(ACCEPT), '--corpus', str(CORPUS),
                   '--max-wall-seconds', str(9 * 3600)], {}),
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
         {**fixed, **TRAINING['trunk'], 'PREFIX_STAGE': 'trunk', 'PREFIX_RUN_PLAN': str(plan_path)}),
        ('graft', [sys.executable, 'scripts/footwear_runtime.py', '--entry', 'scripts/train_prefix.py'],
         {**fixed, **TRAINING['graft'], 'PREFIX_STAGE': 'graft', 'PREFIX_RUN_PLAN': str(plan_path)}),
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
            if name == 'export':
                # Finalize the plan with the published generation's identity;
                # the trunk/graft/policy validators bind to these fields.
                generation = json.loads((DATASET / 'generation_manifest.json')
                                        .read_text(encoding='utf-8'))
                plan['export_fingerprint'] = generation['export_fingerprint']
                plan['generation_manifest_sha256'] = sha256(DATASET / 'generation_manifest.json')
                write_json(plan_path, plan)
            status('running')
        report_path = next(iter((RUN).glob('policy_eval_val_*.json')), None)
        current = 'complete'
        status('complete',
               candidate_sha256=sha256(RUN / 'candidate.pt'),
               export_fingerprint=json.loads((DATASET / 'generation_manifest.json')
                                             .read_text(encoding='utf-8'))['export_fingerprint'],
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
