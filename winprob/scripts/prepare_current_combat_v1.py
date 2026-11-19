"""Exact-frame observed-combat features on the frozen clean6000-game scope."""
from collections import Counter,defaultdict
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import setback_experiment as safe
from rich_experiment import read_frozen_payload
from winprob import admission
from winprob.iql_dataset_contract import current_reservations
from winprob.features_current_combat_v1 import (VERSION,FEATURES,HEALTH_FEATURES,STAT_SCALES,TAGS,
    roster_identity,frame_features,exact_frame_map,evaluation_end_time)

PROTOCOL=ROOT/'docs/current_combat_protocol_v1.md'
RICH=ROOT/'data/runs/rich-20260929-v1/experiment'
RICH_SHA='65414f41331d4a82601ef38d9a95ff4b465bc90a758629730683593a93e729cb'
STATIC=ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'
STATIC_SHA='5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0'


def need(value,message):
    if not value:raise ValueError(message)


def write_json(path,value):
    safe.write_json(path,value)
    Path(path).with_suffix('.sha256').write_text(safe.digest(path)+' '+Path(path).name+'\n',encoding='ascii')


def run(output):
    started=time.monotonic();output=Path(output).resolve();need(not output.exists(),'Refuse combat cache overwrite')
    safe.set_below_normal_priority();bindings={}
    def guard():
        need(time.monotonic()-started<900,'Combat preparation900second ceiling')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            value=Memory();value.length=ctypes.sizeof(value)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) and value.available>=.75*1024**3,'Combat preparation RAM floor0.75GiB')
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);safe.reject_quarantined(path);digest=safe.digest(path) if digest is None else digest
        need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting source/code digest')
        need(safe.digest(path)==digest,'Source/code changed');bindings[str(path)]=digest
    guard()
    for path in (Path(__file__),PROTOCOL,ROOT/'winprob/features_current_combat_v1.py',ROOT/'scripts/rich_experiment.py',
                 ROOT/'scripts/setback_experiment.py',ROOT/'winprob/admission.py',ROOT/'winprob/iql_dataset_contract.py'):
        bind(path)
    original=ROOT/'data/runs/w1-pilot-20260929-v3'
    source,old=safe.verify_inputs(original/'cache/states.npz',original/'cache/source_manifest.json',original/'experiment')
    for path,digest in old.items():bind(path,digest)
    manifest=json.loads(Path(source['admission_manifest']).read_text());ids={row['match_id'] for row in source['games']}
    need(len(ids)==6000 and not ids&current_reservations(manifest),'Reserved or duplicate game in combat scope')
    bind(RICH/'rich_states.npz',RICH_SHA);bind(STATIC,STATIC_SHA)
    static=json.loads(STATIC.read_text())['patches'];need(set(static)=={'16.17','16.18','16.19'},'Native static patch scope differs')
    for world in static.values():
        need({tag for tags in world['champion_tags'].values() for tag in tags}==set(TAGS),'Public tag vocabulary changed')
        for source_file in world['source_files']:bind(source_file['path'],source_file['sha256'])
    with np.load(RICH/'rich_states.npz',allow_pickle=False) as archive:
        data={key:archive[key] for key in ('x','game','role','timestamp_ms','patch','feature_names')}
    n=len(data['game']);need(n==138831 and data['x'].shape==(n,61),'Original frame/feature population differs')
    names=data['feature_names'].tolist();gold_diff=names.index('gold_diff');gold_sum=names.index('rich_gold_total_sum')
    games=source['games'];need([row['game_index'] for row in games]==list(range(6000)),'Canonical source game order differs')
    compressed={row['match_id']:row for row in source['compressed_sources']}
    need(len(compressed)==6000 and set(compressed)==ids,'Compressed source mapping differs')
    output.mkdir(parents=True,exist_ok=False)
    def status(stage,**extra):write_json(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**extra})
    write_json(output/'input_lock.json',{'published_before_raw_access':True,'feature_version':VERSION,'feature_names':list(FEATURES),
        'health_feature_names':list(HEALTH_FEATURES),'games':games,'frames':n,'bindings_sha256':bindings,
        'raw_scope':'exact clean W1v3 compressed_sources only','labels_will_be_extracted':False})
    extra=np.empty((n,len(FEATURES)),dtype=np.float32);covered=np.zeros(n,dtype=bool);ends=np.full(n,-1,dtype=np.int64)
    quality=defaultdict(Counter);dynamics=defaultdict(Counter);end_counts=defaultdict(Counter);source_coverage=[]
    source_root=Path(manifest['source_data_root']).resolve(strict=True)
    try:
        with (output/'access_ledger.jsonl').open('x',encoding='utf-8') as ledger:
            for number,game in enumerate(games,1):
                if number%25==1:guard();status('extracting_exact_current_frames',processed_games=number-1,total_games=6000)
                mid=game['match_id'];payloads={}
                for endpoint in ('match','timeline'):
                    binding=compressed[mid][endpoint]
                    ledger.write(json.dumps({'match_id':mid,'role':game['role'],'endpoint':endpoint,**binding})+'\n');ledger.flush()
                    payloads[endpoint]=read_frozen_payload(binding,endpoint,mid,source_root)
                info=payloads['match']['info']
                need(info['queueId']==420 and info['gameCreation']==game['game_creation_ms'] and
                     '.'.join(str(info['gameVersion']).split('.')[:2])==game['patch'],'Raw match metadata differs')
                roster=roster_identity(payloads['match']);frames=exact_frame_map(payloads['timeline'])
                row_indices=np.flatnonzero(data['game']==game['game_index']);clock=data['timestamp_ms'][row_indices]
                need(len(clock)==game['rows'] and len(set(map(int,clock)))==len(clock) and
                     all(int(t) in frames for t in clock) and set(data['role'][row_indices])=={game['role']} and
                     set(data['patch'][row_indices])=={game['patch']},'Current frame identity/population differs')
                ending=evaluation_end_time(payloads['timeline'],int(clock.max()));ends[row_indices]=ending
                end_counts[game['role']]['known' if ending>=0 else 'unknown']+=1
                histories={pid:set() for pid,team,champ in roster}
                for index,t in zip(row_indices,clock,strict=True):
                    value,q,gold,changes=frame_features(roster,frames[int(t)],static[game['patch']]['champion_tags'])
                    need(np.float32(gold[0]-gold[1])==data['x'][index,gold_diff] and
                         np.float32(gold[0]+gold[1])==data['x'][index,gold_sum],'Current gold differs from immutable rich frame')
                    extra[index]=value;covered[index]=True;quality[game['role']].update(q)
                    for pid,pair in changes.items():
                        if all(v is not None for v in pair):histories[pid].add(pair)
                for values in histories.values():
                    dynamics[game['role']]['participant_series']+=1
                    dynamics[game['role']]['varying_health_or_maximum']+=int(len(values)>1)
                source_coverage.append({'game_index':game['game_index'],'match_id':mid,'role':game['role'],'patch':game['patch'],
                    'frames':len(clock),'end_time_known':ending>=0})
                del payloads,info,frames
        need(covered.all() and np.isfinite(extra).all(),'Missing/nonfinite current features')
        checks={'original_population_complete':True,'native_tag_and_gold_parity':True};quality_report={}
        for role in ('fit','calibration','development'):
            q=quality[role];d=dynamics[role];ratios={}
            for field,threshold in [('health_observed',.99),('wallet_observed',.99),('health_ratio_observed',.99)]+[
                                  ('stat_'+stat+'_observed',.95) for stat in STAT_SCALES]:
                ratios[field]=q[field]/q['player_frames'];checks[role+'_'+field]=ratios[field]>=threshold
            dynamic_fraction=d['varying_health_or_maximum']/d['participant_series']
            checks[role+'_health_dynamic']=dynamic_fraction>=.9
            quality_report[role]={'counts':dict(q),'observed_fractions':ratios,'dynamics':dict(d),'varying_fraction':dynamic_fraction}
        np.savez_compressed(output/'current_features.npz',x_extra=extra,game=data['game'],role=data['role'],timestamp_ms=data['timestamp_ms'],
                            patch=data['patch'],feature_names=np.asarray(FEATURES),evaluation_only_end_ms=ends)
        for path,digest in bindings.items():need(safe.digest(path)==digest,'Input/code changed before publication')
        admission.verify_manifest_sources(manifest);need(not ids&current_reservations(manifest),'Reservation changed during preparation');guard()
        result={'schema':'current-observed-combat-cache-v1','state':'complete','feature_version':VERSION,'training_authorized':all(checks.values()),
            'feature_names':list(FEATURES),'health_feature_names':list(HEALTH_FEATURES),'current_feature_count':len(FEATURES),
            'frames':n,'games':6000,'source_coverage':source_coverage,'quality':quality_report,'quality_gates':checks,
            'evaluation_end_time_coverage':{role:dict(value) for role,value in end_counts.items()},
            'bindings_sha256':bindings,'feature_file':'current_features.npz','feature_sha256':safe.digest(output/'current_features.npz'),
            'input_lock_sha256':safe.digest(output/'input_lock.json'),'access_ledger_sha256':safe.digest(output/'access_ledger.jsonl'),
            'rich_cache_path':str(RICH/'rich_states.npz'),'rich_cache_sha256':RICH_SHA,'static_sha256':STATIC_SHA,
            'raw_sources_read':12000,'source_json_deserialization_includes_outcomes':True,'outcome_fields_indexed':False,
            'cache_y_array_accessed':False,'end_timing_evaluation_only':True,'source_db_reads':False,'source_code_imported':False,
            'sealed_or_confirmation_payload_reads':False,'models_fitted':False,'elapsed_s':time.monotonic()-started,
            'purpose':'observational retrospective WP only; no live-safety or counterfactual item-use claim',
            'limits':['Current recorded frame statistics may have timing/representation limitations; no subsecond causality guarantee.',
                      'Health-positive and net gold outflow are transparent proxies, not complete alive state or inventory value.',
                      'Champion tags are immutable public classes, not inferred player roles.']}
        write_json(output/'manifest.json',result);status('complete',training_authorized=result['training_authorized'])
        return result
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    result=run(parser.parse_args().output)
    print(json.dumps({'games':result['games'],'frames':result['frames'],'training_authorized':result['training_authorized'],
                      'quality':result['quality'],'end_time_coverage':result['evaluation_end_time_coverage'],'elapsed_s':result['elapsed_s']},indent=2))
