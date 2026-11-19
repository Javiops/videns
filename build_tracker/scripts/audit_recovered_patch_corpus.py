"""Full, disjoint parallel semantic audit of the recovered 16.17 corpus.

Only the aggregate can qualify. Parent preflight verifies the immutable DB and
all files once; every worker still replays and independently checks every raw
source and every persisted row in its exact assigned games.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import canonical_sha, sha256, write_json

VERSION = 'footwear-timer-slots-v1'


def validate_preflight(corpus, path):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if (receipt.get('schema') != 1 or receipt.get('scope') != 'recovered-corpus-audit-preflight'
        or Path(receipt.get('corpus', '')).resolve() != corpus.resolve()
        or receipt.get('corpus_manifest_sha256') != sha256(corpus / 'corpus_manifest.json')
        or receipt.get('audit_code_sha256') != sha256(ROOT / 'scripts/audit_patch_corpus.py')
        or receipt.get('recovered_audit_code_sha256') != sha256(Path(__file__))
        or receipt.get('decision_admission_sha256') != sha256(corpus / 'decision_admission.json')
        or receipt.get('sqlite_integrity') is not True or receipt.get('foreign_keys') is not True):
        raise ValueError('Recovered audit preflight is incomplete or changed')
    manifest = json.loads((corpus / 'corpus_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('recovery_version') != VERSION or manifest.get('db_sha256') != receipt.get('db_sha256'):
        raise ValueError('Preflight does not bind this recovered database')
    for name, record in receipt['file_stats'].items():
        stat = (corpus / name).stat()
        if {'bytes':stat.st_size, 'mtime_ns':stat.st_mtime_ns} != record:
            raise ValueError(f'Frozen audit input changed after parent preflight: {name}')
    for name, digest in manifest['reconstruction_code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Recovered audit code changed: {name}')
    shards = receipt.get('shards')
    if not isinstance(shards, list) or not 1 <= len(shards) <= 4 or any(not isinstance(s, list) or not s for s in shards):
        raise ValueError('Invalid recovered audit shard plan')
    ids = [mid for shard in shards for mid in shard]
    if len(ids) != len(set(ids)) or sorted(ids) != receipt.get('all_match_ids') or len(ids) != manifest['games']:
        raise ValueError('Recovered audit shards are incomplete or overlap')
    return receipt


def prepare(corpus, folder, workers):
    from footwear_runtime import activate, declaration
    activate()
    from app.ddragon import dragon_for_patch
    from build_recovered_patch_corpus import read_selection, decision_key
    manifest_path = corpus / 'corpus_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('patch') != '16.17' or manifest.get('recovery_version') != VERSION
        or manifest.get('footwear_runtime') != declaration() or manifest.get('purpose') != 'patch_development'):
        raise ValueError('Recovered audit requires explicit isolated 16.17 runtime provenance')
    for name, binding in manifest['files'].items():
        if sha256(corpus / name) != binding['sha256']:
            raise ValueError(f'Frozen corpus file changed: {name}')
    if manifest['files']['tracker.db']['sha256'] != manifest['db_sha256']:
        raise ValueError('Database manifest binding is inconsistent')
    if dragon_for_patch('16.17').signature != manifest['static_data_sha256']:
        raise ValueError('Recovered static data binding changed')
    for name, digest in manifest['reconstruction_code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Recovered code differs from frozen builder: {name}')
    admission_path = corpus / 'decision_admission.json'
    admission = json.loads(admission_path.read_text(encoding='utf-8'))
    if sha256(admission_path) != manifest['decision_admission_sha256']:
        raise ValueError('Frozen decision admission changed')
    inputs = json.loads((corpus / 'recovery_inputs.json').read_text(encoding='utf-8'))
    if inputs != admission.get('source_bindings'):
        raise ValueError('Recovery inputs disagree with the admission ledger')
    for name, binding in inputs.items():
        if sha256(Path(binding['path'])) != binding['sha256']:
            raise ValueError(f'Frozen recovery selection input changed: {name}')
    plan, prior, summary, recovered, records = read_selection(
        Path(inputs['source_plan']['path']), Path(inputs['prior_qualification']['path']), Path(inputs['recovery_summary']['path']))
    if sha256(corpus / 'source_plan.json') != inputs['source_plan']['sha256']:
        raise ValueError('Copied corpus plan differs from the reviewed source plan')
    selected = {r['match_id']:r for r in records}
    with closing(sqlite3.connect((corpus / 'tracker.db').as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
        integrity = [r[0] for r in conn.execute('PRAGMA integrity_check')] == ['ok']
        foreign = not list(conn.execute('PRAGMA foreign_key_check'))
        game_ids = {r[0] for r in conn.execute('SELECT match_id FROM games')}
    if (not integrity or not foreign or game_ids != admission['games'].keys()
        or len(game_ids) != manifest['games'] or not game_ids <= selected.keys()):
        raise ValueError('Corrupted or incompletely accounted recovered corpus')
    # Structural mask validation uses a temporary expected qualification chain.
    # This grants no clearance; fresh worker checks and aggregate remain required.
    from app.dataset_generations import validate_recovered_evidence
    mask_sha = manifest['decision_admission_sha256']
    evidence = {'recovery_version':VERSION,'semantic_audit_sha256':'pending',
        'decision_admission_sha256':mask_sha,'accepted_match_ids':sorted(game_ids),
        'footwear_qualification':{'version':VERSION,'passed':True,'semantic_audit_sha256':'pending','decision_admission_sha256':mask_sha}}
    validate_recovered_evidence(evidence, {'decision_admission_sha256':mask_sha}, manifest, admission, mask_sha)
    for mid, mask in admission['games'].items():
        record = selected[mid]
        if (mask['split'] != record['split']
            or mask['raw_sources_sha256'] != {k:v['sha256'] for k,v in record['files'].items()}):
            raise ValueError(f'Decision mask changed source identity or split: {mid}')
        if mid in recovered:
            if (mask['source_cohort'] != 'recovered_footwear'
                or [decision_key(v) for v in mask['admitted_decision_ids']] !=
                   [decision_key(v) for v in recovered[mid]['candidate_decision_ids']]):
                raise ValueError(f'Admission differs from the exact original recovery: {mid}')
        elif mask['source_cohort'] != 'prior_qualified' or mid not in prior['accepted_match_ids']:
            raise ValueError(f'Unqualified source cohort: {mid}')
    ids = sorted(game_ids)
    if type(workers) is not int or not 1 <= workers <= min(4, len(ids)):
        raise ValueError('Recovered audit supports one to four nonempty disjoint workers')
    receipt = {'schema':1,'scope':'recovered-corpus-audit-preflight','corpus':str(corpus),
        'corpus_manifest_sha256':sha256(manifest_path),'db_sha256':manifest['db_sha256'],
        'decision_admission_sha256':mask_sha,'audit_code_sha256':sha256(ROOT / 'scripts/audit_patch_corpus.py'),
        'recovered_audit_code_sha256':sha256(Path(__file__)), 'sqlite_integrity':True,'foreign_keys':True,
        'all_match_ids':ids,'shards':[ids[i::workers] for i in range(workers)],
        'source_selection_verified':True, 'final_test_payload_opened':False,
        'file_stats':{name:{'bytes':(corpus/name).stat().st_size,'mtime_ns':(corpus/name).stat().st_mtime_ns}
                      for name in (*manifest['files'], 'corpus_manifest.json')}}
    path = folder / 'preflight.json'
    write_json(path, receipt)
    path.chmod(0o444)
    validate_preflight(corpus, path)
    return path


class RecoveredChecks:
    def __init__(self, admission):
        self.admission = admission

    def __call__(self, match, timeline, stored, fresh, debug, dragon):
        from build_recovered_patch_corpus import admission_for_game
        from build_patch_development_corpus import replay_exclusions
        from magical_footwear import _rune_selected
        from validate_footwear_timer_results import validate_timer_result
        mid = match['metadata']['matchId']
        mask = self.admission['games'][mid]
        qa = validate_timer_result(match, timeline, stored, debug, dragon)
        yield qa['passed'], 'footwear_independent_source_qa', qa.get('issues', [])[:8]
        yield not debug.get('footwear_inventory_failures'), 'footwear_full_physical_replay', debug.get('footwear_inventory_failures', [])[:8]
        invalid = [state for state in debug.get('footwear_state', []) if state.get('invalid')]
        yield not invalid, 'footwear_timer_state_valid', invalid[:8]
        issues = replay_exclusions(match, timeline, stored, debug, dragon)
        yield not issues, 'footwear_all_inventory_states', sorted(issues)
        selected = {p['participantId']:_rune_selected(p) for p in match['info']['participants']}
        yield (any(selected.values()) == (mask['source_cohort']=='recovered_footwear')), 'footwear_source_cohort', None
        expected = {'candidate_decision_ids':mask['admitted_decision_ids'],'candidate_rows':mask['admitted_rows']}
        try:
            recalculated = admission_for_game(fresh, debug, dragon, expected if mask['source_cohort']=='recovered_footwear' else None)
            equal = all(recalculated[k] == mask[k] for k in recalculated)
        except ValueError as exc:
            equal = False
            recalculated = {'error':str(exc)}
        yield equal, 'footwear_exact_decision_admission', None if equal else recalculated
        grant_times = {grant['participant_id']:grant['timestamp'] for grant in debug.get('footwear_grants', [])}
        for event in stored:
            anchor = event['ts'] - 15000 if event.get('is_save') else event['ts']
            for state in (event, *event['board']):
                pid = state['participant_id']
                granted = grant_times.get(pid, float('inf')) < anchor
                valid = (state.get('footwear_state_version') == VERSION
                    and type(state.get('magical_footwear_selected')) is bool
                    and state['magical_footwear_selected'] is selected[pid]
                    and type(state.get('footwear_granted')) is bool and state['footwear_granted'] is granted)
                yield valid, 'footwear_strict_predecision_state', None if valid else {
                    'event_index':event['event_index'],'participant_id':pid,'anchor_ts':anchor,
                    'grant_ts':grant_times.get(pid),'expected_selected':selected[pid],'expected_granted':granted}


def worker(corpus, preflight, index, out):
    from footwear_runtime import activate
    activate()
    validate_preflight(corpus, preflight)
    from audit_patch_corpus import audit
    admission = json.loads((corpus / 'decision_admission.json').read_text(encoding='utf-8'))
    passed = audit(corpus, out, recovered_checker=RecoveredChecks(admission), preflight=preflight, shard_index=index)
    validate_preflight(corpus, preflight)
    return passed


def aggregate(corpus, preflight, paths, out, elapsed):
    if out.exists(): raise ValueError('Immutable recovered audit already exists')
    receipt = validate_preflight(corpus, preflight)
    if len(paths) != len(receipt['shards']): raise ValueError('Missing recovered audit workers')
    seen, accepted, excluded = set(), set(), {}
    counts, diagnostics, failures = Counter(), Counter(), Counter()
    examples, reports, gaps = defaultdict(list), [], []
    for index, path in enumerate(paths):
        report = json.loads(path.read_text(encoding='utf-8'))
        assigned = set(receipt['shards'][index])
        accounted = set(report['accepted_match_ids']) | report['quarantined_match_ids'].keys()
        if (report.get('audit_scope') != 'recovered_shard' or report.get('shard_index') != index
            or report.get('training_clearance') is not False or report.get('preflight_sha256') != sha256(preflight)
            or report.get('corpus_manifest_sha256') != receipt['corpus_manifest_sha256']
            or report.get('db_sha256') != receipt['db_sha256'] or report.get('counts', {}).get('games') != len(assigned)
            or report.get('audit_code_sha256') != receipt['audit_code_sha256']
            or report.get('recovered_audit_code_sha256') != receipt['recovered_audit_code_sha256']
            or sha256(path.parent/'audit_code.py') != receipt['audit_code_sha256']
            or set(report.get('shard_match_ids', [])) != assigned or accounted != assigned or seen & accounted
            or set(report['accepted_match_ids']) & report['quarantined_match_ids'].keys()):
            raise ValueError('Audit worker did not certify its exact complete disjoint shard')
        seen.update(accounted)
        accepted.update(report['accepted_match_ids'])
        excluded.update(report['quarantined_match_ids'])
        counts.update(report['counts'])
        diagnostics.update(report['diagnostics'])
        failures.update(report['hard_failures'])
        for key, values in report['examples'].items(): examples[key].extend(values[:max(0,12-len(examples[key]))])
        if 'budget_shortfall_diagnostic' in report: gaps.append({'shard':index, **report['budget_shortfall_diagnostic']})
        reports.append({'path':str(path),'sha256':sha256(path),'games':len(assigned),'passed':report['passed']})
    if seen != set(receipt['all_match_ids']): raise ValueError('Aggregate audit omitted source games')
    passed = not failures and bool(accepted) and all(r['passed'] for r in reports)
    final = {'schema':2,'audit_scope':'complete_recovered_corpus','corpus':str(corpus),
        'corpus_manifest_sha256':receipt['corpus_manifest_sha256'],'db_sha256':receipt['db_sha256'],
        'audit_code_sha256':receipt['audit_code_sha256'],'recovered_audit_code_sha256':receipt['recovered_audit_code_sha256'],
        'slot_state_version':'role-slots-v1','recovery_version':VERSION,
        'decision_admission_sha256':receipt['decision_admission_sha256'],'preflight_sha256':sha256(preflight),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':elapsed,
        'counts':dict(counts),'hard_failures':dict(failures),'diagnostics':dict(diagnostics),'examples':dict(examples),
        'passed':passed,'training_clearance':passed,'accepted_match_ids':sorted(accepted),'quarantined_match_ids':excluded,
        'shard_reports':reports,'final_test_payload_opened':False,'source_selection_verified':True,
        'clearance_scope':'exact admitted decision IDs in accepted games after separate footwear qualification',
        'budget_shortfall_by_shard':gaps,
        'limits':['All existing per-game semantic checks ran on every stored event, including omitted boundary decisions.',
            'Delivery follows the source-consistent timer/first-room hypothesis; exact server ticks are not independently observed.',
            'The one-second exclusion is a sensitivity guard, not a verified universal delivery bound.',
            'Postgame inventories and rune summaries are QA anchors only; full-outcome perturbation must leave every replayed input unchanged.',
            'No test examples are materialized, read, trained on or evaluated.']}
    write_json(out, final)
    shutil.copy2(ROOT/'scripts/audit_patch_corpus.py', out.parent/'audit_code.py')
    shutil.copy2(Path(__file__), out.parent/'recovered_audit_code.py')
    print(json.dumps({'report':str(out),'games':counts['games'],'shops':counts['shop_events'],
        'passed':passed,'hard_failures':dict(failures),'elapsed_seconds':elapsed},indent=2),flush=True)
    return passed


def run(corpus, out, *, workers=4, max_seconds=10800):
    if out.exists(): raise ValueError('Immutable recovered audit already exists')
    folder = out.parent / (out.stem + '_workers')
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    preflight = prepare(corpus, folder, workers)
    game_count = len(json.loads(preflight.read_text(encoding='utf-8'))['all_match_ids'])
    paths = []
    for index in range(workers):
        destination = folder/f'shard_{index}'
        destination.mkdir()
        paths.append(destination/'semantic_audit.json')
    processes, streams = [], []
    try:
        for index, path in enumerate(paths):
            log = (path.parent/'worker.log').open('x',encoding='utf-8')
            streams.append(log)
            command = [sys.executable,str(Path(__file__).resolve()),'--corpus',str(corpus),'--out',str(path),
                       '--preflight',str(preflight),'--shard-index',str(index)]
            process = subprocess.Popen(command,cwd=ROOT,env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1'},
                stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000 if os.name=='nt' else 0)
            processes.append(process)
        while True:
            codes = [p.poll() for p in processes]
            elapsed = time.monotonic()-started
            if any(code not in (None,0,2) for code in codes):
                raise RuntimeError(f'Recovered audit worker crashed: {codes}; inspect {folder}')
            if elapsed > max_seconds: raise RuntimeError('Recovered semantic audit exceeded wall budget')
            write_json(folder/'status.json',{'state':'running','worker_exit_codes':codes,'elapsed_seconds':elapsed,
                'preflight_sha256':sha256(preflight),'games':game_count})
            if all(code is not None for code in codes): break
            time.sleep(5)
        passed = aggregate(corpus,preflight,paths,out,time.monotonic()-started)
        write_json(folder/'status.json',{'state':'complete','passed':passed,'report_sha256':sha256(out),
            'elapsed_seconds':time.monotonic()-started,'worker_exit_codes':[p.returncode for p in processes]})
        return passed
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        for stream in streams: stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-seconds',type=int,default=10800)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--shard-index',type=int)
    args = parser.parse_args()
    if args.preflight:
        if args.shard_index is None: parser.error('--preflight requires --shard-index')
        passed = worker(args.corpus.resolve(),args.preflight.resolve(),args.shard_index,args.out.resolve())
    else:
        if args.shard_index is not None: parser.error('--shard-index requires --preflight')
        passed = run(args.corpus.resolve(),args.out.resolve(),workers=args.workers,max_seconds=args.max_seconds)
    raise SystemExit(0 if passed else 2)
