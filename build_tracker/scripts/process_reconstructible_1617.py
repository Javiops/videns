"""Screen and independently audit an isolated 16.17 HISTORY source snapshot (derived from the 16.18 processor, 2026-09-27; same gates).

Retain only fully explained whole games, including strict Viego cases. Every
raw event remains in admitted games; ambiguous footwear inputs receive a
separate decision mask. This does not create a split/export or train a model.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import canonical_sha,sha256,write_json,publish_directory
from reconstructible_1617_runtime import activate,declaration,variants,STATE_VERSION

BASE=ROOT/'data/acceptance/16.17_history_20260927'
CORPUS=ROOT/'data/corpora/16.17-history-v1'
RECOVERY_VERSION='footwear-viego-16.17-v1'


def validate_plan(path):
    plan=json.loads(path.read_text(encoding='utf-8'))
    if (plan.get('patch')!='16.17' or plan.get('purpose')!='isolated_reconstruction_qualification'
        or plan.get('test_payload_reads') is not False or plan.get('training_or_export_authorized') is not False
        or sha256(Path(plan['static_review']))!=plan['static_review_sha256']):
        raise ValueError('Invalid patch-scoped processing plan')
    records=plan['matches'];ids=[r['match_id'] for r in records]
    if len(ids)!=len(set(ids)) or len(ids)!=plan['selected_sources'] or set(ids)&set(plan['reserved_test_ids']):
        raise ValueError('Source selection includes duplicate or reserved test games')
    if set(ids)&set(plan['historical_validation_holdout_ids']):
        raise ValueError('Source selection includes historical validation holdouts')
    for record in records:
        if (record['patch']!='16.17' or record['queue_id']!=420 or set(record['files'])!={'match','timeline'}
            or {'test','val'}&set(record['prior_assignments'].values())): raise ValueError('Unscoped source record')
        for kind,binding in record['files'].items():
            if Path(binding['path']).resolve()!=(ROOT/'data/raw_match_v5'/kind/(record['match_id']+'.json.gz')).resolve():
                raise ValueError('Unexpected raw source path')
    for name,digest in plan['split_metadata_sha256'].items():
        if sha256(Path(name))!=digest: raise ValueError('Test reservation metadata changed')
    from history_1617 import split_files,holdout_memberships
    _,live_reserved,live_holdouts=holdout_memberships(split_files(ROOT,exclude_dirs=[path.resolve().parent]))
    if set(ids)&(live_reserved|live_holdouts): raise ValueError('A split file holds a selected source as test or val')
    return plan


def decision_id(event):
    return {k:event[k] for k in ('event_index','participant_id','ts')}


def decision_mask(events,debug,dragon):
    import baseline
    from validate_footwear_timer_results import grant_boundary_reason
    admitted,excluded=[],[]
    for event in events:
        if event.get('training_eligible') is not True: raise ValueError('Source-ineligible event cannot be masked into admission')
        if baseline._example(event,dragon) is None: continue
        reason=grant_boundary_reason(event,debug.get('footwear_grants',[]),guard_ms=1000)
        if reason: excluded.append({**decision_id(event),'reason':reason})
        else: admitted.append(decision_id(event))
    if not admitted: raise ValueError('No admissible model decisions')
    return {'admitted_decision_ids':admitted,'boundary_exclusions':excluded,
            'admitted_rows':len(admitted),'boundary_excluded_rows':len(excluded)}


def prefix_check(match,timeline,events,dragon):
    from app.reconstruct import reconstruct_game
    duration=match['info']['gameDuration']*1000
    cutoff=int(duration*0.6)
    shortened=deepcopy(timeline)
    shortened['info']['frames']=[f for f in shortened['info']['frames'] if f['timestamp']<=cutoff]
    for frame in shortened['info']['frames']:
        frame['events']=[e for e in frame.get('events',[]) if e['timestamp']<=cutoff]
    replayed=reconstruct_game(match,shortened,dragon)[1]
    fields=('participant_id','ts','is_save','inventory_before','board','gold_left','gold_est',
        'gold_est_version','gold_gap','level','total_gold','cs','kills','deaths','assists','score',
        'slot_state_version','bot_quest_complete','footwear_state_version','magical_footwear_selected','footwear_granted',
        'decision_inventory','opening_sales','opening_sale_gold')
    def prior(rows):
        return {(e['participant_id'],e['ts'],bool(e.get('is_save'))):{k:e.get(k) for k in fields}
            for e in rows if e['ts_end']<cutoff-120000}
    original,perturbed=prior(events),prior(replayed)
    equal=(original.keys()==perturbed.keys() and all(
        canonical_sha(value)==canonical_sha(perturbed[key]) for key,value in original.items()))
    return {'passed':bool(original) and equal,'cutoff_ms':cutoff,'checked_decisions':len(original)}


def game_checks(match,timeline,events,debug,dragon,*,check_prefix=True):
    from build_patch_development_corpus import source_exclusions,replay_exclusions
    from validate_footwear_timer_results import validate_timer_result
    from viego_source_gate_1617 import audit_viego_source
    reasons=source_exclusions(match,timeline)-{'unmodeled_viego_inventory','unobserved_magical_footwear_grant'}
    reasons.update(replay_exclusions(match,timeline,events,debug,dragon))
    footwear=validate_timer_result(match,timeline,events,debug,dragon)
    if not footwear['passed']: reasons.add('footwear_source_qa')
    if debug.get('footwear_inventory_failures'): reasons.add('physical_inventory_replay')
    if any(s.get('invalid') for s in debug.get('footwear_state',[])): reasons.add('invalid_footwear_timer_state')
    viego=None
    if any(p['championId']==234 for p in match['info']['participants']):
        viego=audit_viego_source(match,timeline,dragon,replay_events=events,debug=debug,footwear_qa=footwear)
        if not viego['passed']: reasons.add('unexplained_viego_source_state')
    prefix=prefix_check(match,timeline,events,dragon) if check_prefix and not reasons else None
    if prefix is not None and not prefix['passed']: reasons.add('future_prefix_input_dependency')
    return reasons,{'footwear':footwear,'viego':viego,'prefix':prefix}


def build(plan_path,destination):
    if destination.exists(): raise ValueError('Immutable target corpus already exists')
    plan=validate_plan(plan_path)
    runtime=activate(plan['static_review'])
    from app import db
    from app.ddragon import dragon_for_patch
    from app.reconstruct import reconstruct_game,perspective_from_game,RECONSTRUCTION_VERSION,SLOT_STATE_VERSION
    from app.ladder import _player_from_match,routing_for_match_id
    from app.riot import RawMatchArchive
    from collect_patch_sample import validate_pair
    from build_patch_development_corpus import source_exclusions
    dragon=dragon_for_patch('16.17')
    names={p.relative_to(ROOT).as_posix() for p in (ROOT/'app').glob('*.py')}
    names.update('scripts/'+name for name in ('process_reconstructible_1617.py','prepare_reconstructible_1617.py',
        'reconstructible_1617_runtime.py','viego_source_gate_1617.py','audit_patch_corpus.py','qualify_patch_corpus.py',
        'build_patch_development_corpus.py','baseline.py','collect_patch_sample.py','footwear_economics.py','footwear_timer.py',
        'footwear_timer_inventory.py','magical_footwear.py','validate_footwear_timer_results.py','replay_footwear_windows.py'))
    code={name:sha256(ROOT/name) for name in sorted(names)}
    pending=destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex+'.partial')
    pending.mkdir(parents=True)
    for name in code:
        target=pending/'reconstruction_code'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    for name,text in variants().items():
        target=pending/'reconstruction_variants'/(name+'.py');target.parent.mkdir(exist_ok=True)
        target.write_text(text,encoding='utf-8',newline='\n')
    shutil.copy2(plan_path,pending/'source_plan.json')
    shutil.copytree(Path(plan['static_review']).parent,pending/'static_review')
    (pending/'static').mkdir()
    for kind in ('item','champion'):
        shutil.copy2(ROOT/'data/cache'/(kind+'-16.17.1.json'),pending/'static'/(kind+'-16.17.1.json'))
    parent_state=ROOT/'data/corpora/16.17-development-v2/state.json'
    state=json.loads(parent_state.read_text(encoding='utf-8'))
    state.update(purpose='isolated_reconstruction_qualification',cohort_policy='Main database metadata snapshot; this is not a complete scheduled ladder sample.')
    write_json(pending/'state.json',state)
    archive=RawMatchArchive(ROOT/'data/raw_match_v5')
    accepted=[];excluded={};masks={};cohorts=Counter();old_db=db.DB_PATH
    source_file=(pending/'source_inventory.jsonl').open('x',encoding='utf-8')
    try:
        db.DB_PATH=pending/'tracker.db';db.init_db()
        for index,record in enumerate(plan['matches'],1):
            mid=record['match_id'];raw={}
            for kind,binding in record['files'].items():
                if sha256(Path(binding['path']))!=binding['sha256']: raise ValueError('Frozen source changed: '+mid)
                raw[kind]=archive.read(kind,mid)
            match,timeline=raw['match'],raw['timeline']
            reasons=set();details={}
            try:
                validate_pair(mid,match,timeline,'16.17')
                if (match['info']['gameCreation']!=record['game_creation'] or match['info']['gameDuration']!=record['game_duration']):
                    raise ValueError('Game metadata differs from snapshot')
                reasons=source_exclusions(match,timeline)-{'unmodeled_viego_inventory','unobserved_magical_footwear_grant'}
                if not reasons:
                    debug={};game,events=reconstruct_game(match,timeline,dragon,debug)
                    reasons,details=game_checks(match,timeline,events,debug,dragon)
                    if not reasons: mask=decision_mask(events,debug,dragon)
            except (ValueError,KeyError,TypeError) as exc:
                reasons.add('unsupported_or_invalid_source');details={'error':str(exc)}
            if reasons:
                excluded[mid]=sorted(reasons)
                write_json(pending/'excluded_evidence'/(mid+'.json'),{'match_id':mid,'reasons':sorted(reasons),'checks':details})
            else:
                has_viego=details['viego'] is not None
                has_footwear=any(p.get('selected') for p in debug['footwear_state'])
                cohort=('viego' if has_viego else 'no_viego')+('_footwear' if has_footwear else '_no_footwear')
                cohorts[cohort]+=1
                masks[mid]={**mask,'cohort':cohort,'prior_assignments':record['prior_assignments'],
                    'raw_sources_sha256':{k:v['sha256'] for k,v in record['files'].items()}}
                write_json(pending/'accepted_evidence'/(mid+'.json'),{'match_id':mid,'cohort':cohort,'checks':details})
                route,platform=routing_for_match_id(mid)
                for participant in match['info']['participants']:
                    db.upsert_player(_player_from_match(participant,platform,route))
                perspectives=[perspective_from_game(game,events,p['puuid']) for p in match['info']['participants']]
                db.insert_reconstruction(game,events,perspectives)
                source={'match_id':mid,'files':{},'cohorts':[cohort]}
                for kind,binding in record['files'].items():
                    relative=Path('raw')/kind/(mid+'.json.gz');target=pending/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(binding['path'],target)
                    if sha256(target)!=binding['sha256']: raise ValueError('Frozen raw copy changed')
                    source['files'][kind]={'path':relative.as_posix(),'sha256':binding['sha256'],'payload_sha256':canonical_sha(raw[kind])}
                source_file.write(json.dumps(source,sort_keys=True)+'\n');accepted.append(mid)
            if index%50==0:
                print(f'Processed {index}/{len(plan["matches"])}; accepted {len(accepted)}; excluded {len(excluded)}; cohorts {dict(cohorts)}',flush=True)
        source_file.close()
        with db.db() as conn:
            conn.execute('PRAGMA wal_checkpoint(TRUNCATE)');conn.execute('PRAGMA journal_mode=DELETE')
            if [r[0] for r in conn.execute('PRAGMA integrity_check')]!=['ok']: raise ValueError('Rebuilt DB integrity failed')
    finally:
        source_file.close();db.DB_PATH=old_db
    if not accepted: raise ValueError('No source games passed screening')
    validate_plan(plan_path)
    if any(sha256(ROOT/name)!=digest for name,digest in code.items()): raise ValueError('Processing code changed')
    admission={'schema':1,'patch':'16.17','recovery_version':RECOVERY_VERSION,'timing_guard_ms':1000,
        'games':masks,'games_count':len(masks),'admitted_rows':sum(v['admitted_rows'] for v in masks.values()),
        'boundary_excluded_rows':sum(v['boundary_excluded_rows'] for v in masks.values()),'test_payload_reads':False}
    write_json(pending/'decision_admission.json',admission)
    report={'source_snapshot_games':plan['database_games_at_snapshot'],'selected_sources':len(plan['matches']),
        'accepted_games':len(accepted),'accepted_match_ids':accepted,'excluded_games':len(excluded),'excluded_match_ids':excluded,
        'exclusion_reason_counts':dict(Counter(r for reasons in excluded.values() for r in reasons)),
        'accepted_cohorts':dict(cohorts),'admitted_rows':admission['admitted_rows'],
        'boundary_excluded_rows':admission['boundary_excluded_rows'],'training_clearance':False,'pending_full_semantic_audit':True}
    write_json(pending/'screening_report.json',report)
    files={p.relative_to(pending).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
        for p in pending.rglob('*') if p.is_file() and 'raw' not in p.relative_to(pending).parts}
    manifest={'schema':2,'status':'screened_pending_full_semantic_audit','patch':'16.17','purpose':'isolated_reconstruction_qualification',
        'games':len(accepted),'ddragon_version':dragon.version,'static_data_sha256':dragon.signature,
        'reconstruction_version':RECONSTRUCTION_VERSION,'inventory_version':runtime['inventory_version'],
        'slot_state_version':SLOT_STATE_VERSION,'label_source_version':'purchase-net-v1','recovery_version':RECOVERY_VERSION,
        'footwear_runtime':runtime,'source_plan_sha256':sha256(plan_path),'decision_admission_sha256':sha256(pending/'decision_admission.json'),
        'collection_code_sha256':code,'reconstruction_code_sha256':code,'db_sha256':sha256(pending/'tracker.db'),
        'source_inventory_sha256':sha256(pending/'source_inventory.jsonl'),'files':files,'test_payload_reads':False,
        'unresolved_pros':sorted(k for k,v in state.get('pros',{}).items() if not v.get('puuid')),'production_hold_cleared':False}
    write_json(pending/'corpus_manifest.json',manifest)
    for path in pending.rglob('*'):
        if path.is_file(): path.chmod(0o444)
    publish_directory(pending,destination)
    print(json.dumps({k:report[k] for k in ('accepted_games','excluded_games','accepted_cohorts','admitted_rows','boundary_excluded_rows')},indent=2),flush=True)


class FullChecks:
    def __init__(self,admission): self.admission=admission
    def __call__(self,match,timeline,stored,fresh,debug,dragon):
        mid=match['metadata']['matchId']
        reasons,details=game_checks(match,timeline,stored,debug,dragon)
        yield not reasons,'reconstructible_whole_game_source_checks',{'reasons':sorted(reasons),
            'viego_failures':(details['viego'] or {}).get('failures',[])[:8]}
        actual=decision_mask(fresh,debug,dragon);expected=self.admission['games'][mid]
        yield all(actual[k]==expected[k] for k in actual),'reconstructible_exact_admission_mask',None
        from magical_footwear import _rune_selected
        selected={p['participantId']:_rune_selected(p) for p in match['info']['participants']}
        grants={g['participant_id']:g['timestamp'] for g in debug['footwear_grants']}
        for event in stored:
            anchor=event['ts']-15000 if event.get('is_save') else event['ts']
            for row in (event,*event['board']):
                pid=row['participant_id'];expected_grant=grants.get(pid,float('inf'))<anchor
                good=(row.get('footwear_state_version')==STATE_VERSION
                    and type(row.get('magical_footwear_selected')) is bool and row['magical_footwear_selected'] is selected[pid]
                    and type(row.get('footwear_granted')) is bool and row['footwear_granted'] is expected_grant)
                yield good,'reconstructible_causal_footwear_state',None if good else {'participant_id':pid,'anchor':anchor}


def audit_and_qualify(plan_path,corpus,out):
    plan=validate_plan(plan_path);activate(plan['static_review'])
    manifest=json.loads((corpus/'corpus_manifest.json').read_text(encoding='utf-8'))
    if manifest['source_plan_sha256']!=sha256(plan_path) or manifest['footwear_runtime']!=declaration():
        raise ValueError('Corpus runtime or source plan changed')
    for name,digest in manifest['reconstruction_code_sha256'].items():
        if sha256(ROOT/name)!=digest: raise ValueError('Frozen reconstruction code changed')
    mask_path=corpus/'decision_admission.json';admission=json.loads(mask_path.read_text(encoding='utf-8'))
    if sha256(mask_path)!=manifest['decision_admission_sha256']: raise ValueError('Frozen mask changed')
    from audit_patch_corpus import audit
    if not audit(corpus,out,recovered_checker=FullChecks(admission)): raise ValueError('Full semantic audit failed')
    report=json.loads(out.read_text(encoding='utf-8'))
    screening=json.loads((corpus/'screening_report.json').read_text(encoding='utf-8'))
    if set(report['accepted_match_ids'])!=set(screening['accepted_match_ids']) or report['quarantined_match_ids']:
        raise ValueError('Screening and independent qualification disagree')
    qualification={'schema':1,'status':'qualified_reconstruction_snapshot','patch':'16.17','corpus':str(corpus),
        'corpus_manifest_sha256':sha256(corpus/'corpus_manifest.json'),'db_sha256':manifest['db_sha256'],
        'semantic_audit':str(out),'semantic_audit_sha256':sha256(out),'decision_admission_sha256':sha256(mask_path),
        'accepted_match_ids':report['accepted_match_ids'],'accepted_games':len(report['accepted_match_ids']),
        'accepted_cohorts':screening['accepted_cohorts'],'source_exclusions':screening['exclusion_reason_counts'],
        'admitted_rows':admission['admitted_rows'],'boundary_excluded_rows':admission['boundary_excluded_rows'],
        'test_payload_reads':False,'training_clearance':False,'export_created':False,'promotion_allowed':False,
        'limits':['Footwear delivery remains a causal inference with a 1000 ms sensitivity guard.',
            'Viego is admitted only with a fully explained independent source ledger and all-board semantic checks.',
            'No held-out test source payloads were read. Prior fold metadata is preserved for a future benchmark review.',
            'This frozen snapshot does not include games collected after its metadata cutoff.']}
    write_json(out.parent/'qualification.json',qualification)
    print(json.dumps({k:qualification[k] for k in ('status','accepted_games','accepted_cohorts','admitted_rows')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('build','audit'))
    parser.add_argument('--source-plan',type=Path,default=BASE/'source_plan.json')
    parser.add_argument('--corpus',type=Path,default=CORPUS)
    parser.add_argument('--out',type=Path,default=BASE/'semantic_audit.json')
    args=parser.parse_args()
    if args.stage=='build': build(args.source_plan.resolve(),args.corpus.resolve())
    else: audit_and_qualify(args.source_plan.resolve(),args.corpus.resolve(),args.out.resolve())
