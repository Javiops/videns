"""Freeze metadata-only ordinary selection, then separately export chosen contexts.

The select command cannot open SQLite or raw games. The contexts command
consumes an already published859-game selection, never draws replacements,
and writes model inputs and retained-purchase targets to separate files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import export_winprob_trajectories_v1 as producer
import export_winprob_revision_contexts_v2 as native

ROOT=producer.ROOT
PROTOCOL=ROOT.parent/'winprob/docs/bc20_ordinary_eval_protocol_v1.md'
CATALOGUE=ROOT/'data/exports/winprob-action-catalogue/full-action-catalogue-20260930-v1/catalogue.json'
STATIC=ROOT.parent/'winprob/artifacts/boots-static/20260930-v1/boots_static.json'
SELECTION_SCHEMA='winprob-ordinary-selection-v1'
CONTEXT_SCHEMA='winprob-ordinary-contexts-v1'
FOCAL_PREFIX='bc20-ordinary-v1/focal/'
VISIT_PREFIX='bc20-ordinary-v1/visit/'
WARDS={2055,772043}


def require(condition,message):
    if not condition:raise ValueError(message)


def write_json(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sidecar(path):
    path=Path(path);value=producer.file_hash(path)
    path.with_suffix('.sha256').write_text(value+'  '+path.name+'\n',encoding='ascii')
    return value


def focal_player(mid):
    return 1+int.from_bytes(hashlib.sha256((FOCAL_PREFIX+mid).encode()).digest()[:8],'big')%10


def choose_decision(mid,mask):
    pid=focal_player(mid);entries=mask['admitted_decision_ids']
    keys=[producer.decision_key(entry) for entry in entries]
    require(len(set(keys))==len(keys)==mask['admitted_rows'],'Duplicate/missing admitted decision metadata')
    excluded={producer.decision_key(entry) for entry in mask['boundary_exclusions']}
    require(len(excluded)==mask['boundary_excluded_rows'] and not set(keys)&excluded,'Overlapping native decision mask')
    eligible=[entry for entry in entries if entry['participant_id']==pid]
    def rank(entry):
        text=VISIT_PREFIX+mid+'|'+str(pid)+'|'+str(entry['event_index'])+'|'+str(entry['ts'])
        return hashlib.sha256(text.encode()).digest(),entry['ts'],entry['event_index']
    picked=min(eligible,key=rank) if eligible else None
    decision={'event_index':picked['event_index'],'pid':pid,'ts':picked['ts']} if picked else None
    identity=f'{mid}|{pid}|{decision["event_index"]}|{decision["ts"]}' if decision else f'{mid}|{pid}|none'
    return {'match_id':mid,'pid':pid,'sample_id':identity,'role':'development','patch':'16.19',
            'status':'selected' if decision else 'no_admitted_decision','selected_decision':decision,
            'admitted_count':len(eligible),'selected_decision_hash':rank(picked)[0].hex() if picked else None,
            'focal_hash_first8':hashlib.sha256((FOCAL_PREFIX+mid).encode()).digest()[:8].hex()}


def code_bindings():
    names=(Path(__file__),Path(producer.__file__),Path(native.__file__),ROOT/'scripts/baseline.py',
           ROOT/'scripts/footwear_runtime.py',ROOT/'scripts/reconstructible_1619_runtime.py',
           ROOT/'scripts/process_reconstructible_1619.py',ROOT/'app/footwear_reconstruction.py',
           ROOT/'scripts/footwear_economics.py',ROOT/'scripts/footwear_timer_inventory.py',ROOT/'scripts/footwear_timer.py')
    return {str(path.resolve()):producer.file_hash(path) for path in names}


def sampling_metadata(admission_path,protocol_path):
    manifest,tests,validation,confirmation,bindings=producer.validate_admission(admission_path)
    wanted={entry['match_id'] for entry in manifest['games'] if entry['role']=='development' and entry['patch']=='16.19'}
    require(len(wanted)==859 and not wanted&(tests|validation|confirmation),'Ordinary cohort is not the exact safe859development games')
    source=native.native_metadata('16.19',wanted)
    require(source['approved']==wanted,'The frozen859-game universe is not fully present in native qualification metadata')
    bindings.update(source['bindings'])
    protocol_path=Path(protocol_path).resolve(strict=True)
    protocol=protocol_path.read_text(encoding='utf-8')
    require(FOCAL_PREFIX in protocol and VISIT_PREFIX in protocol,'Sampling salts differ from the approved protocol')
    bindings[str(protocol_path)]=producer.file_hash(protocol_path)
    bindings.update(code_bindings())
    entries=[{'ordinal':ordinal,**choose_decision(mid,source['masks'][mid])} for ordinal,mid in enumerate(sorted(wanted))]
    return entries,source,bindings,{'global_test_count':len(tests),'historical_validation_count':len(validation),
                                  'confirmation_count':len(confirmation),'denied_ids_sha256':producer.id_hash(tests|validation|confirmation)},tests|validation|confirmation


def export_selection(admission_path,protocol_path,output):
    output=Path(output).resolve();require(not output.exists(),'Immutable ordinary selection already exists')
    entries,source,bindings,reservations,_=sampling_metadata(admission_path,protocol_path)
    require(all(producer.file_hash(path)==digest for path,digest in bindings.items()),'Selection metadata/code changed')
    payload={'schema':SELECTION_SCHEMA,'state':'complete','version':output.name,'created_utc':datetime.now(timezone.utc).isoformat(),
             'selection_count':859,'admitted_ids':[entry['match_id'] for entry in entries],
             'admitted_ids_sha256':producer.id_hash(entry['match_id'] for entry in entries),'entries':entries,
             'decision_bearing_games':sum(entry['status']=='selected' for entry in entries),
             'status_counts':dict(Counter(entry['status'] for entry in entries)),
             'focal_player_counts':dict(sorted(Counter(entry['pid'] for entry in entries).items())),
             'admission_path':str(Path(admission_path).resolve()),'admission_sha256':producer.file_hash(admission_path),
             'protocol_path':str(Path(protocol_path).resolve()),'protocol_sha256':producer.file_hash(protocol_path),
             'bindings':bindings,'reservation_counts':reservations,'native_runtime':source['corpus']['footwear_runtime'],
             'sampling':{'focal_prefix':FOCAL_PREFIX,'focal_digest_bytes':8,'focal_byte_order':'big','focal_modulus':10,
                         'visit_prefix':VISIT_PREFIX,'visit_digest_bytes':32,'visit_tie_break':['ts','event_index'],
                         'ledger_order':'match_id ascending','replacement_allowed':False},
             'selection_before_payloads':True,'raw_payload_reads':False,'sqlite_opened':False,
             'catalogue_or_policy_used_for_selection':False,'outcomes_exported':False,'outcomes_analyzed':False}
    output.mkdir(parents=True,exist_ok=False);path=output/'selection.json';write_json(path,payload);sidecar(path)
    return path,payload


def load_selection(path):
    path=Path(path).resolve(strict=True);selection,digest=producer.load_bound(path)
    require(selection.get('schema')==SELECTION_SCHEMA and selection.get('state')=='complete' and selection.get('selection_count')==859
            and selection.get('selection_before_payloads') is True and selection.get('raw_payload_reads') is False
            and selection.get('sqlite_opened') is False and selection.get('catalogue_or_policy_used_for_selection') is False,
            'Unknown or non-blind ordinary selection artifact')
    for source,expected in selection['bindings'].items():require(producer.file_hash(source)==expected,'Frozen selection input/code changed: '+source)
    entries,native_source,bindings,reservations,denied=sampling_metadata(selection['admission_path'],selection['protocol_path'])
    require(entries==selection['entries'] and [entry['match_id'] for entry in entries]==selection['admitted_ids']
            and reservations==selection['reservation_counts'],'Ordinary selection changed; no replacement is permitted')
    require(producer.file_hash(selection['admission_path'])==selection['admission_sha256']
            and producer.file_hash(selection['protocol_path'])==selection['protocol_sha256'],'Selection admission/protocol binding changed')
    bindings.update({str(path):digest,str(path.with_suffix('.sha256')):producer.file_hash(path.with_suffix('.sha256'))})
    return selection,native_source,bindings,denied


def target_maps(path):
    catalogue,digest=producer.load_bound(path)
    require(catalogue.get('schema')=='winprob-full-action-catalogue-v1' and catalogue.get('full_action_count')==13166
            and catalogue.get('group_count')==9333,'Wrong fixed target catalogue')
    for source,expected in catalogue['source_files_sha256'].items():require(producer.file_hash(source)==expected,'Target catalogue source changed')
    full={producer.canonical(action):i for i,action in enumerate(catalogue['full_actions'])}
    groups={producer.canonical(group):i for i,group in enumerate(catalogue['group_main'])}
    require(len(full)==13166 and len(groups)==9333 and catalogue['full_actions'][0]==[] and catalogue['group_main'][0]==[],
            'Target catalogue has duplicate/invalid empty action')
    return catalogue,digest,full,groups


def target_record(row,full,groups):
    counts=producer.counts_of(row)
    if counts=={str(producer.SAVE_SENTINEL):1}:
        kind='observed_no_buy_death';physical=[];main=[]
    else:
        physical=[[int(item),count] for item,count in sorted(counts.items(),key=lambda pair:int(pair[0]))]
        main=[entry for entry in physical if entry[0] not in WARDS]
        wards=any(item in WARDS for item,count in physical)
        kind='ward_only_purchase' if not main else 'mixed_purchase_with_wards' if wards else 'other_purchase'
    fid=full.get(producer.canonical(physical));gid=groups.get(producer.canonical(main))
    return {'encoding':'item-counts-v1','kind':kind,
            'action_kind':'observed_no_buy_death' if kind=='observed_no_buy_death' else 'retained_net_purchase',
            'label_counts':counts,'physical_label_counts':{str(item):count for item,count in physical},
            'main_label_counts':{str(item):count for item,count in main},
            'ward_label_counts':{str(item):count for item,count in physical if item in WARDS},
            'full_action_id':fid,'main_group_id':gid,'full_catalogue_supported':fid is not None,'main_catalogue_supported':gid is not None}


def project_selected(entry,events,players,projector,attach_gold,debug,full,groups):
    selected=entry['selected_decision']
    expected=producer.decision_key({'event_index':selected['event_index'],'participant_id':selected['pid'],'ts':selected['ts']})
    found=[event for event in events if producer.decision_key(event)==expected]
    require(len(found)==1,'Selected immutable decision identity did not reproduce')
    event=found[0];pid=entry['pid']
    require(event.get('type')=='shop' and event.get('training_eligible') is True and not event.get('source_quality_issues')
            and event['puuid']==players[pid]['puuid'],'Selected decision source provenance failed')
    bound={**event,'match_id':entry['match_id'],'perks':players[pid]};row=projector(bound)
    require(row is not None,'Selected admitted decision no longer has its original retained target')
    target=target_record(row,full,groups)
    try:prestate=producer.project_state(row);projection=producer.reference_compatibility(row,prestate)['source_projection']
    except (ValueError,KeyError,TypeError) as error:
        return None,target,{'reason':'unavailable_predecision_state','detail':str(error)}
    # Preserve source-estimator semantics, including its separately documented
    # opening-sale adjustment; all frame/event series arguments are truncated.
    cut=event['ts'];prefix=deepcopy(event);prefix.pop('gold_est',None);prefix.pop('gold_gap',None)
    frame_series={pid:[value for value in series if value[0]<cut] for pid,series in debug['frame_series'].items()}
    gold_events={pid:[value for value in series if value[0]<cut] for pid,series in debug['gold_events'].items()}
    attach_gold([prefix],frame_series,gold_events)
    require(prefix.get('gold_est')==event.get('gold_est') and prefix.get('gold_est_version')==event.get('gold_est_version')
            and prefix.get('gold_gap')==event.get('gold_gap'),'Selected wallet estimate failed prefix invariance')
    altered=deepcopy(bound)
    altered.update(is_save=not bool(event.get('is_save')),label_bought=[{'item_id':1036}],bought=[{'item_id':1036}],
                   inventory_after=[],gold=10**9,ts_end=event['ts_end']+999999,source_target_id=987654321,win='mutation-only')
    altered['perks']={**altered['perks'],'win':'mutation-only','kills':9999,'deaths':9999,'assists':9999}
    changed=projector(altered)
    require(changed is not None,'Synthetic target mutation unexpectedly removed projection')
    changed_state=producer.project_state(changed)
    require(changed_state==prestate and producer.reference_compatibility(changed,changed_state)['source_projection']==projection,
            'Selected prediction context depends on target/future fields')
    context={'prestate':prestate,'source_projection':projection,'projection_sha256':producer.value_hash(projection),
             'context_timestamp_ms':event['ts'],'source_visit_end_ms':event['ts_end'],
             'source_event_sha256':producer.value_hash(event),'gold_est_version':event['gold_est_version'],
             'checks':{'selected_identity_matches':True,'prefix_wallet_reproduced':True,'target_future_projection_mutation_equal':True},
             'reference_role_warning':'M2 compatibility projection retains legacy final-match inferred roles; public actor prestate excludes them.',
             'prestate_cutoff_semantics':'Qualified source decision context, including only opening sales before first purchase; source ts is visit start. Observed no-buy uses the source frozen-window convention.'}
    return context,target,None


def context_export(selection_path,output,catalogue_path=CATALOGUE,static_export_path=STATIC,full_mode=False,limit=10):
    started=time.monotonic();output=Path(output).resolve();require(not output.exists(),'Immutable ordinary context export already exists')
    require((full_mode and limit==0) or (not full_mode and 1<=limit<=10),'Use≤10 preselected rows for smoke or reviewed full limit0')
    selection,source,bindings,denied=load_selection(selection_path)
    # Target/static metadata is deliberately first opened AFTER selection is frozen.
    catalogue,catalogue_hash,full_map,group_map=target_maps(catalogue_path)
    static,static_hash=producer.load_bound(static_export_path)
    require(static.get('schema')=='winprob-boots-static-v1' and set(static['patches'])=={'16.17','16.18','16.19'},'Wrong fixed public static union')
    for patch in static['patches'].values():
        for record in patch['source_files']:require(producer.file_hash(record['path'])==record['sha256'],'Public static metadata changed')
    bindings.update({str(Path(catalogue_path).resolve()):catalogue_hash,str(Path(static_export_path).resolve()):static_hash})
    active,baseline,reconstruct_game,dragon,decision_mask=native.activate_native(source)
    attach_gold=reconstruct_game.__globals__['_attach_gold_estimates']
    attempts=[entry['sample_id'] for entry in selection['entries'] if entry['status']=='selected']
    attempts=set(attempts if full_mode else attempts[:limit])
    allowed={entry['match_id'] for entry in selection['entries'] if entry['sample_id'] in attempts}
    require(not allowed&denied and allowed<=source['approved'],'Selected context scope escaped safe native IDs')
    output.mkdir(parents=True,exist_ok=False)
    (output/'selection.json').write_bytes(Path(selection_path).read_bytes());sidecar(output/'selection.json')
    def status(stage,**extra):write_json(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**extra})
    status('locked_before_payload_reads',selection_count=859,attempted_contexts=len(attempts),smoke=not full_mode)
    tensor_row=0;queried=[];counts=Counter();unavailable=Counter();targets_count=Counter();deadline=1200 if full_mode else 150
    db_path=source['corpus_path']/'tracker.db';initial=db_path.stat()
    try:
        with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=2)) as conn, \
                (output/'contexts.jsonl').open('x',encoding='utf-8',newline='\n') as contexts, \
                (output/'targets.jsonl').open('x',encoding='utf-8',newline='\n') as targets:
            conn.execute('PRAGMA query_only=ON');conn.execute('PRAGMA cache_size=-2048')
            reader=producer.ApprovedGameReader(conn,allowed,denied) if allowed else None
            for ordinal,entry in enumerate(selection['entries']):
                require(time.monotonic()-started<=deadline,'Ordinary context export deadline exceeded')
                identity={key:entry[key] for key in ('ordinal','sample_id','match_id','role','patch','selected_decision')}
                identity['participant_id']=entry['pid']
                base={'schema':'winprob-ordinary-context-v1',**identity,'tensor_row':None,'prestate':None,'source_projection':None,
                      'projection_sha256':None,'status':'unavailable','unavailable_reason':None}
                target_base={'schema':'winprob-ordinary-target-v1',**identity,'status':'unavailable','target':None}
                if entry['status']=='no_admitted_decision':
                    base.update(status='no_admitted_decision',unavailable_reason='no_admitted_decision');target_base['status']='no_admitted_decision'
                elif entry['sample_id'] not in attempts:
                    base['unavailable_reason']='not_processed_smoke'
                else:
                    mid=entry['match_id']
                    if len(queried)%50==0:producer.validate_admission(selection['admission_path'])
                    plan=source['plans'][mid]
                    missing=[endpoint for endpoint,binding in plan['files'].items() if not Path(binding['path']).is_file()]
                    if missing:
                        base.update(unavailable_reason='missing_archived_source',missing_endpoints=missing)
                    else:
                        game,events=reader.game(mid);queried.append(mid)
                        raw,raw_hashes=producer.read_raw_pair(mid,plan,allowed,denied)
                        debug={};replayed_game,replayed_events=reconstruct_game(raw['match'],raw['timeline'],dragon,debug)
                        producer.verify_replay(game,events,replayed_game,replayed_events)
                        fresh=decision_mask(replayed_events,debug,dragon);frozen=source['masks'][mid]
                        require(all(fresh[key]==frozen[key] for key in ('admitted_decision_ids','boundary_exclusions','admitted_rows','boundary_excluded_rows')),
                                'Selected native decision mask did not reproduce')
                        players={player['participant_id']:player for player in producer.load_participants(game)}
                        projected,target,error=project_selected(entry,events,players,lambda event:baseline._example(event,dragon),
                                                               attach_gold,debug,full_map,group_map)
                        target_base.update(status='available',target=target);targets_count[target['kind']]+=1
                        if error is not None:base.update(unavailable_reason=error['reason'],detail=error['detail'])
                        else:
                            base.update(status='available',tensor_row=tensor_row,**projected)
                            base['source_receipts']={'raw_sources':raw_hashes,'selected_mask_sha256':producer.value_hash(frozen),
                                'raw_replay_matches_indexed_rows':True,'admitted_mask_reproduced':True}
                            tensor_row+=1
                        del raw,events,replayed_events,replayed_game,game,debug
                counts[base['status']]+=1
                if base['unavailable_reason']:unavailable[base['unavailable_reason']]+=1
                contexts.write(producer.canonical(base).decode()+'\n');targets.write(producer.canonical(target_base).decode()+'\n')
                if ordinal%25==0 or ordinal==858:
                    contexts.flush();targets.flush();status('exporting',ledger_games=ordinal+1,available_contexts=tensor_row,**dict(counts))
        require(db_path.stat().st_size==initial.st_size and db_path.stat().st_mtime_ns==initial.st_mtime_ns,'Native corpus changed during indexed read')
        repeated,_,_,_=load_selection(selection_path);require(repeated==selection,'Selection changed during source export')
        require(all(producer.file_hash(path)==digest for path,digest in bindings.items()),'Source/protocol/static bindings changed')
        files={name:producer.file_hash(output/name) for name in ('selection.json','contexts.jsonl','targets.jsonl')}
        for name in files:sidecar(output/name)
        result={'schema':CONTEXT_SCHEMA,'state':'complete','version':output.name,'created_utc':datetime.now(timezone.utc).isoformat(),
                'smoke':not full_mode,'full_cohort_complete':bool(full_mode),'selection_count':859,'admitted_ids':selection['admitted_ids'],
                'decision_bearing_games':selection['decision_bearing_games'],'available_contexts':tensor_row,'attempted_contexts':len(attempts),
                'status_counts':dict(counts),'unavailable_reasons':dict(unavailable),'target_kind_counts':dict(targets_count),
                'files':files,'admission_path':selection['admission_path'],'admission_sha256':selection['admission_sha256'],
                'protocol_path':selection['protocol_path'],'protocol_sha256':selection['protocol_sha256'],
                'static_export_path':str(Path(static_export_path).resolve()),'static_export_sha256':static_hash,
                'catalogue_path':str(Path(catalogue_path).resolve()),'catalogue_sha256':catalogue_hash,'bindings':bindings,
                'native_runtime':active,'queried_game_ids':queried,'all_rows_in_selection_order':True,'targets_separate_from_contexts':True,
                'test_payload_reads':False,'confirmation_payload_reads':False,'outcomes_exported':False,'outcomes_analyzed':False,
                'source_replay_internally_reads_final_match_fields':True,'training_performed':False,'policy_loaded':False,
                'scope':'Previously exposed development ordinary-decision endpoint; no new confirmation or win target.',
                'unavailable_primary_accounting':'Selected decision-bearing unavailable rows remain zero-credit for all arms; no alternate visit/player.',
                'elapsed_s':time.monotonic()-started}
        path=output/'manifest.json';write_json(path,result);sidecar(path);status('complete',available_contexts=tensor_row,**dict(counts))
        return result
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    select=sub.add_parser('select');select.add_argument('--admission',type=Path,default=producer.ADMISSION)
    select.add_argument('--protocol',type=Path,default=PROTOCOL);select.add_argument('--output',type=Path,required=True)
    contexts=sub.add_parser('contexts');contexts.add_argument('--selection',type=Path,required=True)
    contexts.add_argument('--output',type=Path,required=True);contexts.add_argument('--catalogue',type=Path,default=CATALOGUE)
    contexts.add_argument('--static-export',type=Path,default=STATIC);contexts.add_argument('--full',action='store_true')
    contexts.add_argument('--limit',type=int,default=10)
    args=parser.parse_args()
    if args.stage=='select':
        path,result=export_selection(args.admission,args.protocol,args.output)
        print(json.dumps({'selection':str(path),'sha256':producer.file_hash(path),'count':result['selection_count'],
                          'decision_bearing_games':result['decision_bearing_games'],'status_counts':result['status_counts'],'raw_payload_reads':False},indent=2))
    else:
        result=context_export(args.selection,args.output,args.catalogue,args.static_export,args.full,args.limit)
        print(json.dumps({key:result[key] for key in ('smoke','selection_count','available_contexts','status_counts','unavailable_reasons','elapsed_s')},indent=2))


if __name__=='__main__':main()
