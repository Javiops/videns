"""Resumable, archive-only recovery of missing non-test 16.17 sources.

Does not reconstruct, write the game database, assign folds, export, or train.
Use the supervise entry only from the one-shot Windows scheduled task.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import sha256, write_json
from app.riot import RawMatchArchive, RiotClient, RiotError

DEFAULT_RUN = ROOT / 'data/source_refetch/16.17-missing-20260911'
PREVIEW = ROOT / 'data/acceptance/16.17_development_20260911/source_inventory_preview.json'
ARCHIVE = ROOT / 'data/raw_match_v5'
CODE = ('scripts/refetch_missing_sources.py', 'scripts/run_missing_source_refetch.ps1',
        'scripts/run_patch_development.py', 'app/riot.py', 'app/collector_lock.py',
        'app/config.py', 'app/dataset_artifacts.py', 'scripts/collect_patch_sample.py')
RECORD_FIELDS = ('match_id', 'patch', 'queue_id', 'legacy_split', 'game_creation',
                 'game_duration', 'prior_probe_train_assignment', 'prior_probe_val_assignment')
RESULT_STATUSES = {'raw_pair_recovered', 'unavailable_404', 'invalid_existing_archive',
                   'source_metadata_mismatch', 'source_validation_failed'}


def select_sources(preview):
    missing = [r for r in preview['games_metadata']
               if r['patch'] == '16.17' and not r['raw_pair_exists']]
    if len({r['match_id'] for r in missing}) != len(missing):
        raise ValueError('Duplicate missing-source metadata')
    test = [r for r in missing if r['legacy_split'] == 'test']
    selected = [r for r in missing if r['legacy_split'] != 'test']
    if any(r['queue_id'] != 420 or r['legacy_split'] not in ('train', 'val', 'unassigned')
           or not r['match_id'].startswith(('KR_', 'EUW1_')) for r in selected):
        raise ValueError('Unexpected patch, queue, route or prior split')
    priority = {'train':0, 'val':1, 'unassigned':2}
    selected.sort(key=lambda r:(priority[r['legacy_split']], -r['game_creation'], r['match_id']))
    return selected, len(test), len(missing)


def prepare(directory):
    if directory.exists():
        raise ValueError('Refetch directory exists; resume its immutable plan')
    preview = json.loads(PREVIEW.read_text(encoding='utf-8'))
    sources, test_count, missing_count = select_sources(preview)
    records = []
    endpoints = Counter()
    for source in sources:
        record = {k:source[k] for k in RECORD_FIELDS}
        record['existing_files'] = {}
        for kind in ('match', 'timeline'):
            path = ARCHIVE / kind / (RawMatchArchive._safe_match_id(record['match_id']) + '.json.gz')
            record['existing_files'][kind] = sha256(path) if path.exists() else None
            if not path.exists(): endpoints[kind] += 1
        records.append(record)
    protected = [ROOT / 'data/ml' / n for n in ('prefix_model.pt', 'deployment_manifest.json', 'pipeline_hold.json')]
    protected += [ROOT/'data/experiments/16.17-development-v2/protocol.json',
                  ROOT/'data/acceptance/16.17_development_20260911/source_plan.json',
                  ROOT/'data/acceptance/16.17_development_20260911/split_assignment.json',
                  ROOT/'data/acceptance/16.17_development_20260911/split_metadata.json']
    plan = {'schema':1, 'purpose':'missing_source_archive_only', 'patch':'16.17',
        'created_utc':datetime.now(timezone.utc).isoformat(), 'preview':str(PREVIEW),
        'preview_sha256':sha256(PREVIEW), 'archive':str(ARCHIVE),
        'missing_games_in_original_inventory':missing_count, 'reserved_test_games_not_requested':test_count,
        'selected_games':len(records), 'selected_by_legacy_split':dict(Counter(r['legacy_split'] for r in records)),
        'missing_endpoints_at_prepare':dict(endpoints), 'records':records,
        'code_sha256':{name:sha256(ROOT/name) for name in CODE},
        'protected_sha256':{str(p):sha256(p) if p.exists() else None for p in protected},
        'current_training_integration':False, 'reconstruction_or_training_clearance':False,
        'test_payload_reads':False, 'preserve_original_split_and_exposure':True,
        'limits':['New raw pairs require reconstruction, semantic qualification and a separately frozen future dataset.',
                  'Legacy validation previously used in probe training is not restored as untouched validation.',
                  'Short games remain subject to the minimum-duration gate.']}
    directory.mkdir(parents=True)
    (directory/'results').mkdir()
    for name in CODE:
        target = directory/'code'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    write_json(directory/'plan.json', plan)
    (directory/'plan.json').chmod(0o444)
    print(json.dumps({k:plan[k] for k in ('selected_games','selected_by_legacy_split',
        'missing_endpoints_at_prepare','reserved_test_games_not_requested')},indent=2),flush=True)
    return plan


def read_plan(directory):
    plan = json.loads((directory/'plan.json').read_text(encoding='utf-8'))
    if (plan.get('schema') != 1 or plan.get('purpose') != 'missing_source_archive_only'
        or plan.get('patch') != '16.17' or plan.get('test_payload_reads') is not False
        or plan.get('current_training_integration') is not False
        or Path(plan['archive']).resolve() != ARCHIVE.resolve()
        or sha256(Path(plan['preview'])) != plan['preview_sha256']):
        raise ValueError('Invalid immutable archive-only plan')
    expected, heldout, count = select_sources(json.loads(Path(plan['preview']).read_text(encoding='utf-8')))
    actual_ids = [r['match_id'] for r in plan['records']]
    if (actual_ids != [r['match_id'] for r in expected] or len(actual_ids) != plan['selected_games']
        or heldout != plan['reserved_test_games_not_requested'] or count != plan['missing_games_in_original_inventory']):
        raise ValueError('Missing-source membership changed or includes test games')
    for record, original in zip(plan['records'], expected, strict=True):
        if (set(record) != set(RECORD_FIELDS) | {'existing_files'}
            or any(record[k] != original[k] for k in RECORD_FIELDS)
            or set(record['existing_files']) != {'match', 'timeline'}
            or any(d is not None and (not isinstance(d, str) or len(d) != 64
                    or any(c not in '0123456789abcdef' for c in d))
                   for d in record['existing_files'].values())):
            raise ValueError('Source metadata or split exposure changed')
    assert_unchanged(plan)
    return plan


def assert_unchanged(plan):
    if set(plan['code_sha256']) != set(CODE):
        raise ValueError('Incomplete frozen refetch code bindings')
    for name, digest in plan['code_sha256'].items():
        if sha256(ROOT/name) != digest: raise ValueError('Frozen refetch code changed: '+name)
    for name, digest in plan['protected_sha256'].items():
        path = Path(name)
        if (sha256(path) if path.exists() else None) != digest:
            raise ValueError('Protected training/serving input changed: '+name)


class AppendOnlyArchive(RawMatchArchive):
    def write(self, endpoint, match_id, payload):
        if endpoint not in ('match', 'timeline'):
            raise ValueError('Unexpected archive endpoint')
        if payload.get('metadata', {}).get('matchId') != match_id:
            raise ValueError('Returned payload identity does not match requested game')
        path = self._path(endpoint, match_id)
        if path.exists():
            raise ValueError('Existing archive entry requires separate repair; refusing overwrite')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        envelope = {'schema':1, 'endpoint':endpoint, 'match_id':match_id,
                    'retrieved_at_utc':datetime.now(timezone.utc).isoformat(), 'payload':payload}
        try:
            with temporary.open('xb') as raw:
                with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6) as compressed:
                    compressed.write(json.dumps(envelope, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))
                raw.flush()
                os.fsync(raw.fileno())
            # A hard link publishes complete bytes and fails if another writer
            # has created the destination. Unlike replace(), it never overwrites.
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise ValueError('Archive entry appeared during fetch; refusing overwrite') from None
        finally:
            temporary.unlink(missing_ok=True)


def validate_result(result, record, plan_sha, archive):
    if (result.get('plan_sha256') != plan_sha or result.get('match_id') != record['match_id']
        or result.get('legacy_split') != record['legacy_split']
        or result.get('status') not in RESULT_STATUSES):
        raise ValueError('Result belongs to a different refetch plan or has an invalid status')
    if result['status'] == 'raw_pair_recovered':
        if set(result.get('files', {})) != {'match', 'timeline'}:
            raise ValueError('Recovered result lacks both source bindings')
        for kind, binding in result['files'].items():
            expected = archive._path(kind, record['match_id'])
            if (Path(binding['path']).resolve() != expected.resolve()
                or sha256(expected) != binding['sha256'] or expected.stat().st_size != binding['bytes']):
                raise ValueError('Previously recovered archive changed')


def fetch_record(record, client, archive):
    mid = record['match_id']
    if record['legacy_split'] == 'test':
        raise ValueError('Test source payload reads are forbidden in this refetch')
    route = 'asia' if mid.startswith('KR_') else 'europe'
    for kind, digest in record['existing_files'].items():
        path = archive._path(kind, mid)
        if digest is not None and (not path.exists() or sha256(path) != digest):
            raise ValueError('Pre-existing source changed: '+mid+'/'+kind)
        if path.exists() and archive.read(kind, mid) is None:
            return {'status':'invalid_existing_archive', 'endpoint':kind}
    match = client.match(route, mid)
    if match is None: return {'status':'unavailable_404', 'endpoint':'match'}
    info = match.get('info', {})
    version = '.'.join(str(info.get('gameVersion', '')).split('.')[:2])
    if (match.get('metadata', {}).get('matchId') != mid or version != '16.17' or info.get('queueId') != 420
        or info.get('gameCreation') != record['game_creation'] or info.get('gameDuration') != record['game_duration']):
        return {'status':'source_metadata_mismatch', 'endpoint':'match'}
    timeline = client.timeline(route, mid)
    if timeline is None: return {'status':'unavailable_404', 'endpoint':'timeline'}
    from collect_patch_sample import validate_pair
    validate_pair(mid, match, timeline, '16.17')
    files = {kind:{'path':str(archive._path(kind,mid)), 'sha256':sha256(archive._path(kind,mid)),
                   'bytes':archive._path(kind,mid).stat().st_size} for kind in ('match','timeline')}
    return {'status':'raw_pair_recovered', 'files':files,
            'structural_source_checks_passed':True, 'reconstruction_or_training_clearance':False}


def worker(directory):
    from app.collector_lock import CollectorLease
    import httpx
    plan = read_plan(directory)
    plan_sha = sha256(directory/'plan.json')
    counts, done = Counter(), {}
    archive = AppendOnlyArchive(ARCHIVE)
    for record in plan['records']:
        path = directory/'results'/(record['match_id']+'.json')
        if path.exists():
            result = json.loads(path.read_text(encoding='utf-8'))
            validate_result(result, record, plan_sha, archive)
            done[record['match_id']] = result
            counts[result['status']] += 1
    started = time.monotonic()
    def status(state, **extra):
        write_json(directory/'status.json', {'state':state, 'plan_sha256':plan_sha,
            'selected_games':len(plan['records']), 'processed_games':len(done), 'counts':dict(counts),
            'elapsed_seconds_this_attempt':time.monotonic()-started,
            'time_utc':datetime.now(timezone.utc).isoformat(), 'current_training_integration':False,
            'test_payload_reads':False, **extra})
    lease = None
    client = None
    current = None
    try:
        while lease is None:
            try: lease = CollectorLease()
            except RuntimeError:
                status('waiting_for_collector_lease')
                time.sleep(30)
        status('rate_limit_cooldown')
        time.sleep(125)
        client = RiotClient(raw_cache_enabled=True)
        client._raw_archive = archive
        for record in plan['records']:
            current = record['match_id']
            if current in done: continue
            if sha256(directory/'plan.json') != plan_sha: raise ValueError('Refetch plan changed')
            if shutil.disk_usage(ROOT).free < 40e9: raise RuntimeError('40 GB disk reserve reached')
            status('fetching', current_match_id=current, legacy_split=record['legacy_split'])
            try:
                result = fetch_record(record, client, archive)
            except (ValueError, KeyError, TypeError) as exc:
                result = {'status':'source_validation_failed', 'reason':str(exc)}
            result.update(match_id=current, legacy_split=record['legacy_split'], plan_sha256=plan_sha,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(directory/'results'/(current+'.json'),result)
            done[current] = result
            counts[result['status']] += 1
            if len(done)%25==0 or len(done)<=3:
                print(f'Refetch {len(done)}/{len(plan["records"])}: {dict(counts)}',flush=True)
                assert_unchanged(plan)
            status('fetching', last_match_id=current)
        assert_unchanged(plan)
        status('complete', protected_inputs_unchanged=True)
        shutil.copy2(directory/'status.json',directory/'completion.json')
    except (RiotError, httpx.HTTPError) as exc:
        # Never include HTTP exception strings: request details may be sensitive.
        code = getattr(exc,'status',None)
        reason = 'riot_authentication_rejected' if code in (401,403) else 'network_or_api_failure'
        status('blocked', reason=reason, http_status=code, current_match_id=current)
        print(f'Refetch stopped: {reason}; completed game checkpoints preserved.',flush=True)
        return 2
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, reason=str(exc), current_match_id=current)
        raise
    finally:
        if client: client.close()
        if lease: lease.__exit__()
    return 0


def collection_budget_seconds(local):
    cutoff = local.replace(hour=4,minute=15,second=0,microsecond=0)
    if cutoff <= local < local.replace(hour=4,minute=30,second=0,microsecond=0):
        raise ValueError('Daily collection guard is active; refetch must wait')
    if cutoff <= local: cutoff += timedelta(days=1)
    # Leave room for the 04:30 daily collector even after retry stalls.
    seconds = min(5*3600, (cutoff-local).total_seconds())
    if seconds < 180: raise ValueError('Too close to daily collection; refetch must wait')
    return seconds


def supervise(directory):
    from run_patch_development import WindowsMemoryJob
    read_plan(directory)
    seconds = collection_budget_seconds(datetime.now().astimezone())
    job = WindowsMemoryJob(1_000_000_000)
    started = time.monotonic()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    try:
        with (directory/f'worker_{stamp}.log').open('x',encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--out',str(directory)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x00000004)
            job.attach(process)
            while True:
                try: return process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if time.monotonic()-started >= seconds:
                        job.stop()
                        process.wait(timeout=30)
                        status_path=directory/'status.json'
                        status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
                        status.update(state='paused_at_collection_deadline', resumable=True,
                            time_utc=datetime.now(timezone.utc).isoformat())
                        write_json(status_path,status)
                        return 3
    finally:
        job.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','supervise'))
    parser.add_argument('--out',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args()
    directory=args.out.resolve()
    if not directory.is_relative_to((ROOT/'data/source_refetch').resolve()):
        parser.error('Refetch evidence must remain under data/source_refetch')
    if args.command=='prepare': prepare(directory)
    else: raise SystemExit(worker(directory) if args.command=='run' else supervise(directory))
