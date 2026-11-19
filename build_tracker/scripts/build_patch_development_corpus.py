"""Build a separate screened 16.17 database from a frozen local source plan.

Every selected source is accounted for. Unsupported or unresolved games remain
in the raw archive and exclusion ledger, never in this database. A separate
full semantic audit and qualification are still mandatory before export.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app import db
from app.config import CACHE_DIR
from app.dataset_artifacts import canonical_sha, sha256, write_json, publish_directory
from app.ddragon import dragon_for_patch
from app.ladder import _player_from_match, routing_for_match_id
from app.reconstruct import RECONSTRUCTION_VERSION, SLOT_STATE_VERSION, reconstruct_game, perspective_from_game
from app.riot import RawMatchArchive
from app.shop_econ import payload_ids
from audit_patch_corpus import terminal_differences, slot_state_checks, quest_completion_times
from qualify_patch_corpus import inventory_issues, inventory_shape
from baseline import _unmodeled_regular_slots
from collect_patch_sample import validate_pair

CODE=('scripts/build_patch_development_corpus.py','scripts/collect_patch_sample.py','scripts/audit_patch_corpus.py',
      'scripts/qualify_patch_corpus.py','scripts/baseline.py','app/reconstruct.py','app/inventory.py',
      'app/shop_econ.py','app/db.py','app/config.py','app/ddragon.py','app/riot.py','app/ladder.py','app/dataset_artifacts.py')


def source_exclusions(match,timeline):
    reasons=set()
    players=match['info']['participants']
    if any(p['championId']==234 for p in players): reasons.add('unmodeled_viego_inventory')
    if any(s.get('perk')==8304 for p in players for style in p.get('perks',{}).get('styles',[]) for s in style.get('selections',[])):
        reasons.add('unobserved_magical_footwear_grant')
    if match['info']['gameDuration']<600: reasons.add('duration_below_600s')
    if any(e.get('type')=='ITEM_UNDO' and not e.get('beforeId') and not e.get('afterId')
           for f in timeline['info']['frames'] for e in f.get('events',[])):
        reasons.add('undo_without_item_ids')
    return reasons


def replay_exclusions(match,timeline,events,debug,dragon):
    reasons=set()
    if terminal_differences(match,debug['final_inventories'],dragon):
        reasons.add('unreconciled_terminal_inventory')
    quests=quest_completion_times(timeline,1202)
    for e in events:
        if any(not good for good,_rule,_detail in slot_state_checks(e,quests)):
            reasons.add('invalid_causal_slot_state')
        anchor=e['ts']-15000 if e.get('is_save') else e['ts']
        owner=next(m for m in e['board'] if m['participant_id']==e['participant_id'])
        states=[(m,m['items'],anchor,False) for m in e['board']]
        states.extend([(owner,e['inventory_before'],anchor,False),(owner,e['inventory_after'],e['ts_end'],True)])
        for member,items,when,inclusive in states:
            quest=quests.get(member['participant_id'],float('inf'))
            completed=quest<=when if inclusive else quest<when
            reasons.update(inventory_issues(items,member['team_position'],completed,dragon))
            if inventory_shape(items,member['team_position'],completed,dragon)['regular_slots']+_unmodeled_regular_slots(items,dragon)>6:
                reasons.add('total_regular_slot_overflow')
        if Counter(payload_ids(e.get('label_bought',[])))-Counter(payload_ids(e['inventory_after'])):
            reasons.add('purchase_label_not_retained')
    if not events: reasons.add('no_reconstructed_visits')
    return reasons


def build(plan_path,destination):
    if destination.exists(): raise ValueError('Immutable corpus already exists')
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['patch']!='16.17' or plan['queue_id']!=420:
        raise ValueError('This authorized development scope is patch16.17 ranked solo only')
    sources=plan['matches']
    if len({r['match_id'] for r in sources})!=len(sources) or not sources:
        raise ValueError('Source plan is empty or duplicates games')
    if any(r['split'] not in ('train','val') or r['patch']!='16.17' for r in sources):
        raise ValueError('Source scope includes another patch or a reserved test game')
    if any(set(r['files'])!={'match','timeline'} for r in sources):
        raise ValueError('Every planned game needs exactly the match and timeline sources')
    # Both fold files are part of the frozen selection. Never silently copy a
    # changed assignment into an otherwise correctly source-bound corpus.
    split_files={}
    for name in ('split_assignment.json','split_metadata.json'):
        path=plan_path.parent/name
        if sha256(path)!=plan[name.removesuffix('.json')+'_sha256']:
            raise ValueError(f'Frozen split evidence changed: {name}')
        value=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(value,dict):
            raise ValueError(f'Invalid split evidence: {name}')
        split_files[name]=value
    if split_files['split_assignment.json']!={r['match_id']:r['split'] for r in sources}:
        raise ValueError('Source-plan folds disagree with frozen split assignment')
    if split_files['split_metadata.json']!=plan['split_metadata']:
        raise ValueError('Source-plan metadata disagrees with frozen split metadata')
    plan_sha=sha256(plan_path)
    code_hashes={name:sha256(ROOT/name) for name in CODE}
    dragon=dragon_for_patch('16.17')
    pending=destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}.partial')
    pending.mkdir(parents=True)
    for name in CODE:
        target=pending/'reconstruction_code'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    shutil.copy2(plan_path,pending/'source_plan.json')
    for name in ('split_assignment.json','split_metadata.json'):
        shutil.copy2(plan_path.parent/name,pending/name)
    (pending/'static').mkdir()
    for kind in ('item','champion'):
        name=f'{kind}-{dragon.version}.json'
        shutil.copy2(CACHE_DIR/name,pending/'static'/name)
    pro_path=ROOT/'data/pros_resolved.json'
    pro_records=json.loads(pro_path.read_text(encoding='utf-8')) if pro_path.exists() else []
    pros={f"{r.get('league','unknown')}:{r.get('id',i)}":r for i,r in enumerate(pro_records)}
    write_json(pending/'state.json',{'patch':'16.17','pros':pros,
               'pro_label_scope':'currently resolved tracked-account identities; not retrospective roster/rank membership',
               'pro_resolution_sha256':sha256(pro_path) if pro_path.exists() else None,
               'source_plan_sha256':plan_sha,'purpose':'patch_development','discovery_complete':True})
    db.DB_PATH=pending/'tracker.db'
    db.init_db()
    archive=RawMatchArchive(ROOT/'data/raw_match_v5')
    exclusions,accepted={},[]
    accepted_splits=Counter()
    inventory_path=pending/'source_inventory.jsonl'
    with inventory_path.open('x',encoding='utf-8',newline='\n') as inventory:
        for index,record in enumerate(sources,1):
            mid=record['match_id']
            payloads={}
            for kind,binding in record['files'].items():
                path=(ROOT/binding['path']).resolve()
                expected=(ROOT/'data/raw_match_v5'/kind/f'{mid}.json.gz').resolve()
                if path!=expected:
                    raise ValueError(f'Planned file differs from archive source: {mid}/{kind}')
                if sha256(path)!=binding['sha256']: raise ValueError(f'Frozen planned source changed: {mid}/{kind}')
                payloads[kind]=archive.read(kind,mid)
            match,timeline=payloads['match'],payloads['timeline']
            try:
                validate_pair(mid,match,timeline,'16.17')
                if match['info']['gameCreation']!=record['game_creation'] or match['info']['gameDuration']!=record['game_duration']:
                    raise ValueError('planned metadata changed')
            except (ValueError,KeyError,TypeError) as exc:
                exclusions[mid]=['invalid_raw_source:'+str(exc)]
                continue
            reasons=source_exclusions(match,timeline)
            if not reasons:
                debug={}
                game,events=reconstruct_game(match,timeline,dragon,debug)
                reasons.update(replay_exclusions(match,timeline,events,debug,dragon))
            if reasons:
                exclusions[mid]=sorted(reasons)
            else:
                region,platform=routing_for_match_id(mid)
                for p in match['info']['participants']: db.upsert_player(_player_from_match(p,platform,region))
                entries=[perspective_from_game(game,events,p['puuid']) for p in match['info']['participants']]
                db.insert_reconstruction(game,events,entries)
                source={'match_id':mid,'cohorts':['local_archived_16.17',record['split']], 'files':{}}
                for kind,binding in record['files'].items():
                    relative=Path('raw')/kind/f'{mid}.json.gz'
                    target=pending/relative
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(ROOT/binding['path'],target)
                    if sha256(target)!=binding['sha256']: raise ValueError('Raw source copy changed')
                    source['files'][kind]={'path':relative.as_posix(),'sha256':binding['sha256'],
                                          'payload_sha256':canonical_sha(payloads[kind])}
                inventory.write(json.dumps(source,sort_keys=True)+'\n')
                accepted.append(mid)
                accepted_splits[record['split']]+=1
            if index%250==0:
                print(f'Screened {index}/{len(sources)} sources; stored {len(accepted)}; excluded {len(exclusions)}',flush=True)
    if not accepted or set(accepted_splits)!= {'train','val'}:
        raise ValueError('Screened corpus does not support both development folds')
    with db.db() as conn:
        conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        conn.execute('PRAGMA journal_mode=DELETE')
        if [r[0] for r in conn.execute('PRAGMA integrity_check')]!=['ok']: raise ValueError('DB integrity failed')
    if (sha256(plan_path)!=plan_sha
        or any(sha256(plan_path.parent/name)!=plan[name.removesuffix('.json')+'_sha256']
               for name in split_files)
        or any(sha256(ROOT/name)!=digest for name,digest in code_hashes.items())):
        raise ValueError('Source plan or reconstruction code changed during build')
    report={'source_plan_sha256':plan_sha,'selected_sources':len(sources),'stored_games':len(accepted),
            'stored_games_by_split':dict(accepted_splits),'excluded_games':len(exclusions),
            'exclusion_reason_counts':dict(Counter(reason for reasons in exclusions.values() for reason in reasons)),
            'excluded_match_ids':exclusions,'accepted_match_ids':accepted,'passed_screen_only':True,
            'training_clearance':False,'pro_coverage_scope':'resolved account identities only; coverage calculated after qualification'}
    write_json(pending/'screening_report.json',report)
    files={p.relative_to(pending).as_posix():{'sha256':sha256(p),'bytes':p.stat().st_size}
           for p in pending.rglob('*') if p.is_file() and 'raw' not in p.relative_to(pending).parts}
    manifest={'schema':1,'status':'screened_reconstructed_corpus_pending_full_audit',
              'created_utc':datetime.now(timezone.utc).isoformat(),'patch':'16.17','games':len(accepted),
              'selected_source_games':len(sources),'purpose':'patch_development',
              'source_plan_sha256':plan_sha,'static_data_sha256':dragon.signature,'ddragon_version':dragon.version,
              'reconstruction_version':RECONSTRUCTION_VERSION,'inventory_version':'causal-quests-v1',
              'label_source_version':'purchase-net-v1','slot_state_version':SLOT_STATE_VERSION,
              'collection_code_sha256':code_hashes,'reconstruction_code_sha256':code_hashes,
              'source_inventory_sha256':sha256(inventory_path),'db_sha256':sha256(pending/'tracker.db'),'files':files,
              'unresolved_pros':[name for name,r in pros.items() if not r.get('puuid')],
              'training_acceptance':'pending full semantic audit and qualification','production_hold_cleared':False}
    write_json(pending/'corpus_manifest.json',manifest)
    for p in pending.rglob('*'):
        if p.is_file(): p.chmod(0o444)
    publish_directory(pending,destination)
    print(json.dumps({'corpus':str(destination),'source_games':len(sources),'stored_games':len(accepted),
                      'stored_games_by_split':dict(accepted_splits),'excluded_games':len(exclusions),
                      'exclusion_reason_counts':report['exclusion_reason_counts'],'training_clearance':False},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    build(a.source_plan.resolve(),a.out.resolve())
