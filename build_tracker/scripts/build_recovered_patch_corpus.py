"""Build the frozen 16.17 prior-plus-footwear development corpus.

All events remain in the forensic database. A separate, content-bound decision
admission ledger excludes only the frozen grant-boundary rows. This is a
screening gate: fresh independent audit and qualification still come next.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import canonical_sha, sha256, write_json, publish_directory

RECOVERY_VERSION = 'footwear-timer-slots-v1'
DECISION_FIELDS = ('event_index', 'participant_id', 'ts')


def decision_id(event):
    value = {key: event[key] for key in DECISION_FIELDS}
    if (any(type(v) is not int for v in value.values())
        or value['event_index'] < 0 or not 1 <= value['participant_id'] <= 10 or value['ts'] < 0):
        raise ValueError('Invalid decision identity')
    return value


def decision_key(event):
    value = decision_id(event)
    return tuple(value[key] for key in DECISION_FIELDS)


def read_selection(plan_path, prior_path, summary_path):
    """Metadata only: reject reassignment or test inclusion before raw reads."""
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    report_path = Path(summary['source_report']).resolve()
    if (plan.get('patch') != '16.17' or plan.get('queue_id') != 420
        or prior.get('status') != 'qualified_subset' or prior.get('patch') != '16.17'
        or summary.get('status') != 'recovery_measurement_complete'
        or summary.get('final_test_payload_opened') is not False
        or summary.get('training_admission_performed') is not False
        or sha256(prior_path) != summary['previous_qualification_sha256']
        or sha256(report_path) != summary['source_report_sha256']):
        raise ValueError('Unbound recovery selection or unauthorized scope')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('patch') != '16.17' or report.get('source_plan_sha256') != sha256(plan_path)
        or report.get('timing_guard_ms') != 1000 or report.get('final_test_payload_opened') is not False
        or report.get('original_test_explicitly_excluded') is not True):
        raise ValueError('Recovery evidence differs from frozen 16.17 source plan')
    records = {r['match_id']: r for r in plan['matches']}
    if not records or len(records) != len(plan['matches']):
        raise ValueError('Empty or duplicate source plan')
    if any(r.get('patch') != '16.17' or r.get('split') not in ('train', 'val')
           or r.get('legacy_split') == 'test' or set(r.get('files', {})) != {'match', 'timeline'}
           for r in records.values()):
        raise ValueError('Source plan includes unscoped or reserved-test games')
    assignment_path = plan_path.parent / 'split_assignment.json'
    metadata_path = plan_path.parent / 'split_metadata.json'
    if (sha256(assignment_path) != plan['split_assignment_sha256']
        or sha256(metadata_path) != plan['split_metadata_sha256']
        or json.loads(assignment_path.read_text(encoding='utf-8')) != {mid:r['split'] for mid,r in records.items()}
        or json.loads(metadata_path.read_text(encoding='utf-8')) != plan['split_metadata']):
        raise ValueError('Frozen split evidence changed')
    from app.dataset_generations import validate_development_split_metadata
    validate_development_split_metadata(plan['split_metadata'])
    old_ids = set(prior['accepted_match_ids'])
    new_ids = set(summary['recovered_match_ids'])
    recovered = {r['match_id']: r for r in report['games'] if r.get('ordinary_replay_explains_game') is True}
    if (len(old_ids) != prior['quality_accepted_games'] or len(new_ids) != summary['additional_source_consistent_games']
        or old_ids & new_ids or new_ids != recovered.keys() or not old_ids | new_ids <= records.keys()):
        raise ValueError('Recovered/prior game sets are not the exact disjoint frozen union')
    for mid, recovery in recovered.items():
        ids = [decision_key(v) for v in recovery['candidate_decision_ids']]
        if (len(set(ids)) != len(ids) or len(ids) != recovery['candidate_rows']
            or recovery['split'] != records[mid]['split'] or recovery.get('status') != 'source_consistent'
            or recovery.get('failures') or not recovery.get('independent_qa', {}).get('passed')
            or recovery.get('raw_sources_sha256') != {k:v['sha256'] for k,v in records[mid]['files'].items()}):
            raise ValueError(f'Invalid recovered evidence: {mid}')
    return plan, prior, summary, recovered, [records[mid] for mid in sorted(old_ids | new_ids)]


def admission_for_game(events, debug, dragon, recovered=None):
    import baseline
    from validate_footwear_timer_results import grant_boundary_reason
    admitted, excluded = [], []
    for event in events:
        if event.get('training_eligible') is not True:
            raise ValueError('Admission cannot clear a source-quality exclusion')
        if baseline._example(event, dragon) is None:
            continue
        reason = grant_boundary_reason(event, debug.get('footwear_grants', []), guard_ms=1000)
        if reason is None:
            admitted.append(decision_id(event))
        else:
            excluded.append({**decision_id(event), 'reason': reason})
    if recovered is not None and (
        [decision_key(v) for v in admitted] != [decision_key(v) for v in recovered['candidate_decision_ids']]
        or len(admitted) != recovered['candidate_rows']):
        raise ValueError('Fresh replay does not reproduce the exact frozen recovery decisions')
    if recovered is None and (excluded or debug.get('footwear_grants')):
        raise ValueError('Prior no-footwear game unexpectedly requires a footwear mask')
    if not admitted:
        raise ValueError('No admissible model decisions')
    return {'source_cohort': 'recovered_footwear' if recovered is not None else 'prior_qualified',
            'admitted_decision_ids': admitted, 'boundary_exclusions': excluded,
            'admitted_rows': len(admitted), 'boundary_excluded_rows': len(excluded),
            'modeled_rows_before_boundary_guard': len(admitted) + len(excluded)}


def build(plan_path, prior_path, summary_path, destination):
    if destination.exists():
        raise ValueError('Immutable corpus already exists')
    plan, prior, summary, recovered, sources = read_selection(plan_path, prior_path, summary_path)
    from footwear_runtime import activate
    runtime = activate()
    # All semantics must be imported after installing this isolated runtime.
    from app import db
    from app.config import CACHE_DIR
    from app.ddragon import dragon_for_patch
    from app.ladder import _player_from_match, routing_for_match_id
    from app.reconstruct import reconstruct_game, perspective_from_game, RECONSTRUCTION_VERSION, SLOT_STATE_VERSION
    from app.riot import RawMatchArchive
    from build_patch_development_corpus import source_exclusions, replay_exclusions, CODE as BASE_CODE
    from collect_patch_sample import validate_pair
    from validate_footwear_timer_results import validate_timer_result
    code = sorted(set(BASE_CODE) | {p.relative_to(ROOT).as_posix() for p in (ROOT / 'app').glob('*.py')} | {'scripts/build_recovered_patch_corpus.py', 'scripts/footwear_runtime.py',
        'scripts/audit_recovered_patch_corpus.py',
        'app/footwear_reconstruction.py', 'scripts/footwear_timer.py', 'scripts/footwear_timer_inventory.py',
        'scripts/footwear_economics.py', 'scripts/magical_footwear.py', 'scripts/validate_footwear_timer_results.py',
        'scripts/replay_footwear_windows.py', 'scripts/corpus_rows.py', 'app/dataset_generations.py'})
    code_hashes = {name:sha256(ROOT / name) for name in code}
    bindings = {'source_plan': {'path':str(plan_path), 'sha256':sha256(plan_path)},
        'prior_qualification': {'path':str(prior_path), 'sha256':sha256(prior_path)},
        'recovery_summary': {'path':str(summary_path), 'sha256':sha256(summary_path)},
        'recovery_source_report': {'path':summary['source_report'], 'sha256':summary['source_report_sha256']}}
    dragon = dragon_for_patch('16.17')
    pending = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir(parents=True)
    for name in code:
        target = pending / 'reconstruction_code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    for name in ('source_plan.json', 'split_assignment.json', 'split_metadata.json'):
        shutil.copy2(plan_path if name == 'source_plan.json' else plan_path.parent / name, pending / name)
    shutil.copy2(prior_path, pending / 'prior_qualification.json')
    shutil.copy2(summary_path, pending / 'recovery_summary.json')
    write_json(pending / 'recovery_inputs.json', bindings)
    (pending / 'static').mkdir()
    for kind in ('item', 'champion'):
        name = f'{kind}-{dragon.version}.json'
        shutil.copy2(CACHE_DIR / name, pending / 'static' / name)
    # Preserve pro identity coverage metadata from the original corpus too.
    old_corpus = Path(prior['corpus'])
    old_manifest_path = old_corpus / 'corpus_manifest.json'
    if sha256(old_manifest_path) != prior['corpus_manifest_sha256']:
        raise ValueError('Prior corpus metadata changed')
    old_manifest = json.loads(old_manifest_path.read_text(encoding='utf-8'))
    state_path = old_corpus / 'state.json'
    if sha256(state_path) != old_manifest['files']['state.json']['sha256']:
        raise ValueError('Prior account-identity evidence changed')
    state = json.loads(state_path.read_text(encoding='utf-8'))
    state.update(recovery_version=RECOVERY_VERSION, purpose='patch_development')
    write_json(pending / 'state.json', state)
    archive = RawMatchArchive(ROOT / 'data/raw_match_v5')
    old_db = db.DB_PATH
    accepted, exclusions, masks = [], {}, {}
    split_counts = Counter()
    try:
        db.DB_PATH = pending / 'tracker.db'
        db.init_db()
        with (pending / 'source_inventory.jsonl').open('x', encoding='utf-8', newline='\n') as inventory:
            for index, record in enumerate(sources, 1):
                mid, raw = record['match_id'], {}
                for kind, binding in record['files'].items():
                    source = (ROOT / binding['path']).resolve()
                    expected = (ROOT / 'data/raw_match_v5' / kind / f'{mid}.json.gz').resolve()
                    if source != expected or sha256(source) != binding['sha256']:
                        raise ValueError(f'Frozen raw source changed: {mid}/{kind}')
                    raw[kind] = archive.read(kind, mid)
                match, timeline = raw['match'], raw['timeline']
                validate_pair(mid, match, timeline, '16.17')
                if (match['info']['gameCreation'] != record['game_creation']
                    or match['info']['gameDuration'] != record['game_duration']):
                    raise ValueError(f'Frozen game metadata changed: {mid}')
                reasons = source_exclusions(match, timeline) - {'unobserved_magical_footwear_grant'}
                if not reasons:
                    debug = {}
                    game, events = reconstruct_game(match, timeline, dragon, debug)
                    reasons.update(replay_exclusions(match, timeline, events, debug, dragon))
                    qa = validate_timer_result(match, timeline, events, debug, dragon)
                    if not qa['passed']: reasons.add('footwear_source_qa')
                    if debug.get('footwear_inventory_failures'): reasons.add('footwear_physical_inventory')
                if reasons:
                    exclusions[mid] = sorted(reasons)
                    continue
                # A mask mismatch is a run failure, never silent new selection.
                mask = admission_for_game(events, debug, dragon, recovered.get(mid))
                masks[mid] = {**mask, 'split':record['split'], 'raw_sources_sha256':{k:v['sha256'] for k,v in record['files'].items()}}
                region, platform = routing_for_match_id(mid)
                for player in match['info']['participants']:
                    db.upsert_player(_player_from_match(player, platform, region))
                entries = [perspective_from_game(game, events, p['puuid']) for p in match['info']['participants']]
                db.insert_reconstruction(game, events, entries)
                copied = {'match_id':mid, 'cohorts':['local_archived_16.17', record['split'], mask['source_cohort']], 'files':{}}
                for kind, binding in record['files'].items():
                    relative = Path('raw') / kind / f'{mid}.json.gz'
                    target = pending / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / binding['path'], target)
                    if sha256(target) != binding['sha256']: raise ValueError('Raw copy changed')
                    copied['files'][kind] = {'path':relative.as_posix(), 'sha256':binding['sha256'], 'payload_sha256':canonical_sha(raw[kind])}
                inventory.write(json.dumps(copied, sort_keys=True) + '\n')
                accepted.append(mid)
                split_counts[record['split']] += 1
                if index % 100 == 0:
                    print(f'Rebuilt {index}/{len(sources)}; stored {len(accepted)}; excluded {len(exclusions)}', flush=True)
        with db.db() as conn:
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            conn.execute('PRAGMA journal_mode=DELETE')
            if [r[0] for r in conn.execute('PRAGMA integrity_check')] != ['ok']:
                raise ValueError('Rebuilt database integrity failed')
    finally:
        db.DB_PATH = old_db
    if not accepted or set(split_counts) != {'train', 'val'}:
        raise ValueError('Rebuilt corpus lacks both required development folds')
    if (any(sha256(ROOT / name) != digest for name,digest in code_hashes.items())
        or any(sha256(Path(v['path'])) != v['sha256'] for v in bindings.values())
        or any(sha256(plan_path.parent / name) != plan[name.removesuffix('.json') + '_sha256']
               for name in ('split_assignment.json', 'split_metadata.json'))):
        raise ValueError('Bound source, split, or reconstruction code changed during rebuild')
    admission = {'schema':1, 'recovery_version':RECOVERY_VERSION, 'patch':'16.17', 'timing_guard_ms':1000,
        'timing_guard_is_verified_delivery_bound':False, 'purpose':'patch_development',
        'source_bindings':bindings, 'games':masks, 'games_count':len(masks),
        'admitted_rows':sum(v['admitted_rows'] for v in masks.values()),
        'boundary_excluded_rows':sum(v['boundary_excluded_rows'] for v in masks.values()),
        'final_test_payload_opened':False}
    write_json(pending / 'decision_admission.json', admission)
    report = {'selected_sources':len(sources), 'stored_games':len(accepted), 'stored_games_by_split':dict(split_counts),
        'accepted_match_ids':accepted, 'excluded_match_ids':exclusions, 'excluded_games':len(exclusions),
        'exclusion_reason_counts':dict(Counter(r for reasons in exclusions.values() for r in reasons)),
        'admitted_rows':admission['admitted_rows'], 'boundary_excluded_rows':admission['boundary_excluded_rows'],
        'passed_screen_only':True, 'training_clearance':False}
    write_json(pending / 'screening_report.json', report)
    files = {p.relative_to(pending).as_posix():{'sha256':sha256(p), 'bytes':p.stat().st_size}
        for p in pending.rglob('*') if p.is_file() and 'raw' not in p.relative_to(pending).parts}
    manifest = {'schema':2, 'status':'screened_reconstructed_corpus_pending_full_audit',
        'created_utc':datetime.now(timezone.utc).isoformat(), 'patch':'16.17', 'games':len(accepted),
        'selected_source_games':len(sources), 'purpose':'patch_development', 'source_plan_sha256':sha256(plan_path),
        'static_data_sha256':dragon.signature, 'ddragon_version':dragon.version,
        'reconstruction_version':RECONSTRUCTION_VERSION, 'inventory_version':runtime['inventory_version'],
        'label_source_version':'purchase-net-v1', 'slot_state_version':SLOT_STATE_VERSION,
        'recovery_version':RECOVERY_VERSION, 'footwear_runtime':runtime,
        'decision_admission_sha256':sha256(pending / 'decision_admission.json'),
        'collection_code_sha256':code_hashes, 'reconstruction_code_sha256':code_hashes,
        'source_inventory_sha256':sha256(pending / 'source_inventory.jsonl'),
        'db_sha256':sha256(pending / 'tracker.db'), 'files':files,
        'unresolved_pros':old_manifest['unresolved_pros'],
        'training_acceptance':'pending full semantic audit and footwear qualification',
        'production_hold_cleared':False}
    write_json(pending / 'corpus_manifest.json', manifest)
    for path in pending.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    publish_directory(pending, destination)
    print(json.dumps({'corpus':str(destination), **{k:report[k] for k in ('stored_games', 'stored_games_by_split',
          'admitted_rows', 'boundary_excluded_rows', 'excluded_games', 'training_clearance')}}, indent=2), flush=True)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan', type=Path, required=True)
    parser.add_argument('--prior-qualification', type=Path, required=True)
    parser.add_argument('--recovery-summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.source_plan.resolve(), args.prior_qualification.resolve(), args.recovery_summary.resolve(), args.out.resolve())
