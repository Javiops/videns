"""Stream complete catalogue-supported trajectories from the original4002FIT games.

One source-owned native runtime per invocation (16.17 or16.18). No calibration
or development game payload is opened. Published records retain the v1 node
contract; the v2 manifest states conditional-population exclusions explicitly.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import export_winprob_trajectories_v1 as producer
import export_winprob_revision_contexts_v2 as native

ROOT=producer.ROOT
SCHEMA='winprob-fit-trajectories-v2'
BASE_CACHE=ROOT.parent/'winprob/data/runs/w1-pilot-20260929-v3/cache'
CATALOGUE=ROOT/'data/exports/winprob-action-catalogue/full-action-catalogue-20260930-v1/catalogue.json'
FORBIDDEN_STATE={'win','win_team','winner','outcome','game_duration','source_target_id','label_id','label_ids','label_counts',
                 'decision','save_kind','gold_arrival_true','role','team_position'}


def require(condition,message):
    if not condition:raise ValueError(message)


def fit_scope(manifest,base_source,denied):
    admitted={row['match_id']:row for row in manifest['games']}
    fits=[row for row in base_source['games'] if row['role']=='fit']
    ids={row['match_id'] for row in fits}
    require(len(fits)==len(ids)==4002,'Require the exact original4002FIT identities')
    require(not ids&set(denied),'Original FIT universe intersects a reserved identity')
    for row in fits:
        entry=admitted.get(row['match_id'])
        require(entry is not None and entry['role']=='train' and entry.get('historical_validation') is False
                and row['source_role']=='train' and row['patch']==entry['patch']
                and row['game_creation_ms']==entry['game_creation_ms'],'Original FIT identity/role/date changed')
        require(row['patch'] in ('16.17','16.18'),'Unexpected patch in original FIT universe')
    return fits


def source_scope(base_cache,manifest,denied,current_admission_hash):
    base_cache=Path(base_cache).resolve(strict=True)
    require(not any((parent/'QUARANTINED.md').exists() for parent in (base_cache,*base_cache.parents)),'Quarantined original FIT source')
    metadata,meta_hash=producer.load_bound(base_cache/'cache_metadata.json')
    require(metadata.get('schema')=='winprob-cache-v1' and metadata.get('state')=='complete','Expected complete original W1 metadata')
    source_path=base_cache/'source_manifest.json'
    require(Path(metadata['source_manifest_path']).resolve()==source_path
            and producer.file_hash(source_path)==metadata['source_manifest_sha256'],'Original source membership binding changed')
    source=producer.load_json(source_path)
    require(all(source.get(key) is False for key in ('test_payload_reads','confirmatory_payload_reads','historical_validation_payload_reads')),
            'Original FIT source is not clean')
    require(source['admission_manifest_sha256']==metadata['admission_manifest_sha256']==current_admission_hash,
            'Original cache admission binding differs from the current reservation snapshot')
    fits=fit_scope(manifest,source,denied)
    bindings={str(base_cache/'cache_metadata.json'):meta_hash,str(source_path):metadata['source_manifest_sha256']}
    # Deliberately do not open states.npz: it also contains cal/dev labels.
    return fits,bindings


def action_map(path):
    catalogue,digest=producer.load_bound(path)
    require(catalogue.get('schema')=='winprob-full-action-catalogue-v1'
            and catalogue.get('full_action_count')==13166 and catalogue.get('group_count')==9333
            and catalogue.get('save_action_id')==0 and catalogue.get('source_save_sentinel')==producer.SAVE_SENTINEL,
            'Unknown full-action catalogue contract')
    actions=catalogue['full_actions'];kinds=catalogue['full_action_kinds'];mapping=catalogue['full_to_group']
    require(len(actions)==len(kinds)==len(mapping)==13166 and actions[0]==[]
            and kinds[0]=='observed_no_buy_death','Full action axes/SAVE identity changed')
    index={producer.canonical(action):i for i,action in enumerate(actions)}
    require(len(index)==len(actions),'Duplicate full action')
    for file,expected in catalogue['source_files_sha256'].items():
        require(producer.file_hash(file)==expected,'Catalogue source metadata changed')
    return catalogue,digest,index


def lookup_action(action,catalogue,index):
    counts=action.get('label_counts');kind=action.get('kind')
    require(action.get('encoding')=='item-counts-v1' and isinstance(counts,dict) and counts,'Missing complete retained action')
    if kind=='observed_no_buy_death':
        require(counts=={str(producer.SAVE_SENTINEL):1},'SAVE semantics differ from source sentinel')
        return 0
    require(kind=='retained_net_purchase' and str(producer.SAVE_SENTINEL) not in counts,'Invalid full purchase semantic tag')
    require(all(isinstance(key,str) and key.isdecimal() and str(int(key))==key and int(key)>0
                and type(value) is int and value>0 for key,value in counts.items()),'Invalid full purchase counts')
    values=[[int(item),count] for item,count in sorted(counts.items(),key=lambda pair:int(pair[0]))]
    found=index.get(producer.canonical(values))
    if found is not None:require(catalogue['full_action_kinds'][found]=='retained_net_purchase','Purchase mapped to nonpurchase action')
    return found


def assert_prestate(value):
    if isinstance(value,dict):
        require(not set(value)&FORBIDDEN_STATE,'Label/future/role entered IQL prestate')
        for child in value.values():assert_prestate(child)
    elif isinstance(value,list):
        for child in value:assert_prestate(child)


def validate_chain(record):
    require(record.get('schema')==producer.SCHEMA and record.get('whole_player_game_eligible') is True
            and not record.get('ineligibility_reasons'),'Only complete eligible source chains may be validated')
    decisions=record['decisions'];terminal=record['terminal']
    require(decisions and terminal['kind']=='terminal' and terminal['done'] is True and terminal['action'] is None
            and terminal['prestate'] is None and terminal['next_ref'] is None and type(terminal['reward']) is int
            and terminal['reward'] in (0,1),'Invalid terminal node')
    nodes=decisions+[terminal];ids=[node['node_id'] for node in nodes]
    prefix=f'{record["match_id"]}:{record["participant_id"]}:'
    require(len(ids)==len(set(ids)) and all(isinstance(identity,str) and identity.startswith(prefix) for identity in ids),
            'Duplicate/cross-player decision reference')
    arrival_rewards=[]
    for node,nxt in zip(decisions,nodes[1:]):
        require(node['kind']=='decision' and node['done'] is False and node['reward']==0
                and node['next_ref']==nxt['node_id'],'Gap or reward-contaminated chain')
        require(type(node['timestamp_ms']) is int and type(nxt['timestamp_ms']) is int
                and node['timestamp_ms']<nxt['timestamp_ms'] and node['visit_end_ms']<=nxt['timestamp_ms'],
                'Non-increasing or overlapping transition')
        assert_prestate(node['prestate']);arrival_rewards.append(nxt['reward'])
    require(sum(arrival_rewards)==terminal['reward'] and arrival_rewards[-1]==terminal['reward'],
            'Terminal reward does not reach the last action exactly once')
    return len(decisions)


def encode_record(record,catalogue,catalogue_hash,index,game_binding_index):
    """Whole-chain selection only; no skipped, mapped-away, or stitched action."""
    if not record['whole_player_game_eligible']:
        return None,{'scope':'player_game','trajectory_id':record['trajectory_id'],'match_id':record['match_id'],
                     'participant_id':record['participant_id'],'role':'fit','patch':record['patch'],
                     'reasons':list(record['ineligibility_reasons'])}
    try:validate_chain(record)
    except ValueError as error:
        return None,{'scope':'player_game','trajectory_id':record['trajectory_id'],'match_id':record['match_id'],
                     'participant_id':record['participant_id'],'role':'fit','patch':record['patch'],
                     'reasons':['source_chain_validation_failed'],'detail':str(error)}
    action_ids=[lookup_action(node['action'],catalogue,index) for node in record['decisions']]
    unsupported=[node['node_id'] for node,action_id in zip(record['decisions'],action_ids,strict=True) if action_id is None]
    if unsupported:
        return None,{'scope':'player_game','trajectory_id':record['trajectory_id'],'match_id':record['match_id'],
                     'participant_id':record['participant_id'],'role':'fit','patch':record['patch'],
                     'reasons':['complete_action_outside_frozen_catalogue'],'unsupported_node_ids':unsupported}
    for node,action_id in zip(record['decisions'],action_ids,strict=True):
        node['action']['full_action_id']=action_id
        node['action']['main_group_id']=catalogue['full_to_group'][action_id]
        node.pop('reference_compatibility',None)
    record.update(role='fit',consumer_role='fit',source_role='qualified_history_fit' if record['patch']=='16.17' else 'v4_train',
                  full_action_catalogue_sha256=catalogue_hash,
                  source_quality={'game_binding_index':game_binding_index,'indexed_rows_sha256':record['source_quality']['indexed_rows_sha256']})
    return record,None


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Immutable FIT trajectory output already exists')
    require(args.patch in ('16.17','16.18'),'Only original FIT patches are authorized')
    require((args.full and args.limit==0) or (not args.full and 1<=args.limit<=10),'Use≤10 smoke or separately reviewed --full --limit0')
    manifest,tests,validation,confirm,bindings=producer.validate_admission(args.admission)
    denied=tests|validation|confirm
    fits,fit_bindings=source_scope(args.base_cache,manifest,denied,bindings[str(Path(args.admission).resolve())]);bindings.update(fit_bindings)
    full_fit_ids={row['match_id'] for row in fits}
    patch_fits=[row for row in fits if row['patch']==args.patch]
    selected=patch_fits if args.full else patch_fits[:args.limit]
    wanted={row['match_id'] for row in selected}
    require(wanted and not wanted&denied,'Requested FIT patch has no safe source games')
    catalogue,catalogue_hash,catalogue_index=action_map(args.catalogue)
    bindings[str(Path(args.catalogue).resolve())]=catalogue_hash
    source=native.native_metadata(args.patch,wanted);bindings.update(source['bindings'])
    require(not source['approved']&denied and source['approved']<=full_fit_ids,'Qualified source escaped exact FIT universe')
    active,baseline,reconstruct_game,dragon,decision_mask=native.activate_native(source)
    suffix=args.patch.replace('.','')
    code_paths=(Path(__file__),Path(producer.__file__),Path(native.__file__),ROOT/'scripts/baseline.py',
                ROOT/f'scripts/reconstructible_{suffix}_runtime.py',ROOT/f'scripts/process_reconstructible_{suffix}.py',
                ROOT/'scripts/footwear_runtime.py',ROOT/'app/footwear_reconstruction.py',ROOT/'scripts/footwear_economics.py',
                ROOT/'scripts/footwear_timer.py',ROOT/'scripts/footwear_timer_inventory.py')
    code={str(path.resolve()):producer.file_hash(path) for path in code_paths}
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n')
    selection={'schema':SCHEMA,'full_mode':bool(args.full),'patch':args.patch,'role':'fit','original_fit_games':4002,
               'original_fit_ids_sha256':producer.id_hash(full_fit_ids),'original_fit_game_ids':sorted(full_fit_ids),
               'original_fit_patch_counts':dict(Counter(row['patch'] for row in fits)),
               'requested_game_ids':[row['match_id'] for row in selected],'requested_patch_fit_games':len(patch_fits),
               'qualified_game_ids':sorted(source['approved']),'qualification_exclusions':source['excluded'],
               'global_tests_denied':len(tests),'historical_validation_denied':len(validation),'confirmation_denied':len(confirm),
               'input_metadata_sha256':bindings,'source_code_sha256':code,'catalogue_sha256':catalogue_hash}
    (output/'selection.json').write_text(json.dumps(selection,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    status('selected_before_payloads',requested_games=len(wanted),qualified_games=len(source['approved']))
    db_path=source['corpus_path']/'tracker.db';initial=db_path.stat()
    counters=Counter();reasons=Counter();game_bindings=[];queried=[];eligible_games=set();uncompressed_hash=hashlib.sha256()
    rows_path=output/'trajectories.jsonl.gz';exclusions_path=output/'exclusions.jsonl'
    deadline=1800 if args.full else 150
    try:
        with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=2)) as conn, \
                rows_path.open('xb') as binary, gzip.GzipFile(filename='',mode='wb',fileobj=binary,compresslevel=1,mtime=0) as compressed, \
                exclusions_path.open('x',encoding='utf-8',newline='\n') as exclusions:
            conn.execute('PRAGMA query_only=ON');conn.execute('PRAGMA cache_size=-2048')
            reader=producer.ApprovedGameReader(conn,source['approved'],denied) if source['approved'] else None
            for number,entry in enumerate(selected,1):
                require(time.monotonic()-started<=deadline,'FIT trajectory deadline exceeded')
                mid=entry['match_id']
                if mid not in source['approved']:
                    reason=source['excluded'][mid];reasons[reason]+=1;counters['qualification_excluded_games']+=1
                    exclusions.write(producer.canonical({'scope':'game','match_id':mid,'role':'fit','patch':args.patch,'reasons':[reason]}).decode()+'\n')
                    continue
                if number%50==1:producer.validate_admission(args.admission)
                game,events=reader.game(mid);queried.append(mid)
                require(game['patch']==args.patch and game['queue_id']==420 and game['game_creation']==entry['game_creation_ms'],
                        'FIT game metadata differs from selected source')
                raw,raw_bindings=producer.read_raw_pair(mid,source['plans'][mid],source['approved'],denied)
                debug={};replayed_game,replayed_events=reconstruct_game(raw['match'],raw['timeline'],dragon,debug)
                producer.verify_replay(game,events,replayed_game,replayed_events)
                fresh=decision_mask(replayed_events,debug,dragon);frozen=source['masks'][mid]
                require(all(fresh[key]==frozen[key] for key in ('admitted_decision_ids','boundary_exclusions','admitted_rows','boundary_excluded_rows')),
                        'FIT decision mask no longer reproduces')
                terminal_ms=producer.terminal_from_timeline(raw['timeline'])
                binding={'match_id':mid,'raw_sources':raw_bindings,'selected_mask_sha256':producer.value_hash(frozen),
                         'indexed_rows_sha256':producer.value_hash({'game':game,'events':events}),
                         'raw_replay_matches_indexed_rows':True,'admitted_mask_reproduced':True}
                binding_index=len(game_bindings);game_bindings.append(binding)
                records=producer.trajectories_for_game(game,events,frozen,lambda event:baseline._example(event,dragon),terminal_ms,binding)
                for record in records:
                    counters['considered_player_games']+=1
                    exported,exclusion=encode_record(record,catalogue,catalogue_hash,catalogue_index,binding_index)
                    if exclusion is not None:
                        exclusions.write(producer.canonical(exclusion).decode()+'\n');counters['excluded_player_games']+=1
                        reasons.update(exclusion['reasons']);continue
                    count=validate_chain(exported)
                    require(all(node['action']['full_action_id']==lookup_action(node['action'],catalogue,catalogue_index)
                                for node in exported['decisions']),'Exported action ID changed')
                    line=producer.canonical(exported)+b'\n';compressed.write(line);uncompressed_hash.update(line)
                    counters['eligible_player_games']+=1;counters['decision_nodes']+=count;counters['terminal_transitions']+=1
                    eligible_games.add(mid)
                del raw,replayed_game,replayed_events,game,events,records,debug
                if number%25==0 or number==len(selected):
                    compressed.flush();exclusions.flush();status('exporting',processed_games=number,queried_games=len(queried),**dict(counters))
        require(db_path.stat().st_size==initial.st_size and db_path.stat().st_mtime_ns==initial.st_mtime_ns,'FIT corpus changed during indexed read')
        producer.validate_admission(args.admission)
        require(all(producer.file_hash(path)==expected for path,expected in {**bindings,**code}.items()),'FIT input/code changed during export')
        files={name:producer.file_hash(output/name) for name in ('trajectories.jsonl.gz','exclusions.jsonl','selection.json')}
        for name,digest in files.items():(output/(name+'.sha256')).write_text(digest+'  '+name+'\n',encoding='ascii')
        result={**selection,'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'rows_schema':producer.SCHEMA,'rows_file':'trajectories.jsonl.gz','rows_compression':'gzip','gzip_compresslevel':1,'gzip_mtime':0,
                'rows_sha256':files['trajectories.jsonl.gz'],'uncompressed_rows_sha256':uncompressed_hash.hexdigest(),'files':files,
                'trajectory_count':counters['eligible_player_games'],'eligible_player_games':counters['eligible_player_games'],
                'eligible_games':len(eligible_games),'eligible_game_ids':sorted(eligible_games),'queried_game_ids':queried,
                'counts':dict(counters),'exclusion_reason_counts':dict(reasons),'game_bindings':game_bindings,'runtime':active,
                'source_database':str(db_path),'qualified_database_sha256_reference':source['db_reference_sha256'],
                'whole_database_scanned':False,'only_complete_catalogue_supported_player_chains':True,
                'terminal_reward_reaches_last_action_once':True,'full_action_catalogue_path':str(Path(args.catalogue).resolve()),
                'full_action_catalogue_sha256':catalogue_hash,'prestate_whitelist':list(producer.STATE_FIELDS),
                'other_player_whitelist':list(producer.OTHER_FIELDS),'reference_compatibility_exported':False,
                'privileged_fields_excluded_from_actor':['total_gold','others[].gold'],
                'inferred_role_fields_absent_from_prestate':True,'outcomes_scope':'Only original FIT game terminal labels; no outcome-based selection.',
                'calibration_game_payload_reads':False,'development_game_payload_reads':False,'test_payload_reads':False,
                'confirmation_payload_reads':False,'mixed_npz_or_training_export_read':False,'policy_loaded':False,'training_performed':False,
                'conditional_population':'Native-qualified, complete observed modeled player chains with every full action in the frozen catalogue.',
                'combined_training_gate':'Combine both patches and independently require at least1000 distinct eligible FIT games before fitting.',
                'limitations':['Complete-chain/catalogue qualification conditions on observed source coverage and future action support; not a general-policy value estimand.',
                               'No unobserved manual no-buy, primitive transaction, propensity or causal-outcome identification claim.',
                               'No cal/development labels are exported or consulted.'],
                'elapsed_s':time.monotonic()-started}
        report=output/'manifest.json';report.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        report.with_suffix('.sha256').write_text(producer.file_hash(report)+'  manifest.json\n',encoding='ascii')
        status('complete',eligible_games=len(eligible_games),**dict(counters));return result
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-cache',type=Path,default=BASE_CACHE)
    parser.add_argument('--admission',type=Path,default=producer.ADMISSION)
    parser.add_argument('--catalogue',type=Path,default=CATALOGUE)
    parser.add_argument('--patch',choices=('16.17','16.18'),required=True)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args)
    print(json.dumps({key:report[key] for key in ('patch','trajectory_count','eligible_games','counts','exclusion_reason_counts','rows_sha256','elapsed_s')},indent=2))


if __name__=='__main__':main()
