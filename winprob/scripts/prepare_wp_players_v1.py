"""Stream joint player tokens for exactly the audited full-scale WP rows."""
from collections import Counter
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
from winprob.cache import _validate_match_metadata
from winprob.iql_dataset_contract import current_reservations
from winprob.wp_scale_scope_v1 import checked_json,verify_scope
from winprob.wp_scale_models_v1 import validate_shard
from winprob.features_current_combat_v1 import roster_identity,exact_frame_map,frame_features
from winprob.features_rich import ALL_RICH_FEATURES
from winprob.features_wp_players_v1 import FIELDS,frame_tokens

PARENT=ROOT/'data/runs/wp-scale-cache-20260930-v1/cache/manifest.json'
PROTOCOL=ROOT/'docs/wp_player_relations_protocol_v1.md'
STATIC=ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'


def need(value,message):
    if not value:raise ValueError(message)


def write(path,value):
    safe.write_json(path,value);Path(path).with_suffix('.sha256').write_text(safe.digest(path)+' '+Path(path).name+'\n',encoding='ascii')


def run(output):
    started=time.monotonic();output=Path(output).resolve();need(not output.exists(),'Refuse player-cache overwrite')
    safe.set_below_normal_priority();bindings={}
    def guard():
        need(time.monotonic()-started<1800,'Player extraction1800second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            m=Memory();m.length=ctypes.sizeof(m)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) and m.available>=.75*1024**3,'Player extraction RAM floor0.75GiB')
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);safe.reject_quarantined(path);actual=safe.digest(path)
        need(digest is None or actual==digest,'Changed bound input');bindings[str(path)]=actual
    code=[Path(__file__),PROTOCOL,ROOT/'winprob/features_wp_players_v1.py',ROOT/'winprob/features_current_combat_v1.py',
        ROOT/'winprob/features_rich.py',ROOT/'winprob/features.py',ROOT/'winprob/wp_scale_scope_v1.py',ROOT/'winprob/wp_scale_models_v1.py',
        ROOT/'winprob/cache.py',ROOT/'winprob/admission.py',ROOT/'winprob/iql_dataset_contract.py',ROOT/'winprob/experiment.py',
        ROOT/'winprob/metrics.py',ROOT/'scripts/setback_experiment.py',ROOT/'scripts/rich_experiment.py']
    for path in code:bind(path)
    parent,parent_sha=checked_json(PARENT)
    need(parent_sha=='7a6852da971246e67051b0ad7622b27b1ff20fd7ce8cb43d9fd76ccf41d89c14','Wrong scale cache')
    need(parent['training_authorized'] and all(parent['gates'].values()) and parent['qualified_games']==17195 and parent['frames']==396667
         and not parent['exclusions'],'Incomplete fixed full-pool scope')
    bind(PARENT,parent_sha)
    audit_path=ROOT/'data/reports/wp_scale_cache_audit_v1.json';audit,audit_sha=checked_json(audit_path)
    need(audit['state']=='passed' and audit['cache_manifest_sha256']==parent_sha and audit['fit_samples']==64422,'Unverified parent cache')
    bind(audit_path,audit_sha)
    scale_report=ROOT/'data/runs/wp-scale-fit-20260930-v2/experiment/report.json'
    bind(scale_report,'6e10781c05ce85d706ef87f723c36bdabebbc75613624193bd2144a8fa2fe014')
    scale_audit=ROOT/'data/reports/wp_scale_result_audit_v2.json'
    bind(scale_audit,'388d0e963a00aba6f96ec7d19bf28198f0ed7ebfa87ddf800bd65ae39ce29c7f')
    scope,scope_sha=checked_json(parent['scope_path']);need(scope_sha==parent['scope_sha256'],'Scope changed')
    admitted,pilot,requested=verify_scope(scope);requested={r['match_id']:r for r in requested}
    for path,digest in {**scope['bindings_sha256'],**parent['bindings_sha256']}.items():bind(path,digest)
    sources,source_sha=checked_json(parent['source_bindings_path']);need(source_sha==parent['source_bindings_sha256'],'Sources changed')
    bind(parent['source_bindings_path'],source_sha);source={r['match_id']:r for r in sources['compressed_sources']}
    need(not sources['missing_archives'] and len(source)==17195 and set(source)==set(requested),'Source scope differs')
    need(not set(source)&current_reservations(admitted),'Reserved source scope')
    bind(STATIC,'5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0')
    static=json.loads(STATIC.read_text())['patches'];root=Path(admitted['source_data_root'])
    guard();output.mkdir(parents=True,exist_ok=False)
    def status(stage,**extra):write(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**extra})
    write(output/'input_lock.json',{'published_before_raw_reads':True,'parent_manifest_sha256':parent_sha,
        'bindings_sha256':bindings,'feature_names':list(FIELDS),'games':17195,'frames':396667,
        'training_authorized_before_quality_checks':False,'raw_scope':'exact parent qualified game IDs only'})
    path=output/'players.npy';matrix=None;entries=[];offset=0;quality={r:Counter() for r in ('fit','calibration','development')}
    try:
        matrix=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(396667,10,35))
        with (output/'access_ledger.jsonl').open('x',encoding='utf-8') as ledger:
            for gid,row in enumerate(parent['games']):
                if gid%50==0:guard();matrix.flush();status('streaming_exact_frames',games=gid,total_games=17195,frames=offset,total_frames=396667)
                need(row['game_index']==gid and row['match_id'] in requested,'Parent identity/order changed')
                mid=row['match_id'];scope_row=requested[mid]
                for key in ('role','source_role','patch','game_creation_ms','cohort','scope_index'):need(row[key]==scope_row[key],'Parent scope metadata differs')
                shard=(PARENT.parent/row['file']).resolve(strict=True)
                need(shard.is_relative_to(PARENT.parent) and safe.digest(shard)==row['sha256'],'Parent shard changed')
                with np.load(shard,allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
                validate_shard(a,row);payload={}
                for endpoint in ('match','timeline'):
                    binding=source[mid][endpoint]
                    ledger.write(json.dumps({'match_id':mid,'role':row['role'],'game_index':gid,'endpoint':endpoint,**binding})+'\n');ledger.flush()
                    payload[endpoint]=read_frozen_payload(binding,endpoint,mid,root)
                match,timeline=payload['match'],payload['timeline'];_validate_match_metadata(match,scope_row)
                roster=roster_identity(match);frames=exact_frame_map(timeline);tags=static[row['patch']]['champion_tags']
                n=row['rows'];values=np.empty((n,10,35),dtype=np.float32)
                for j,t in enumerate(a['timestamp_ms']):
                    need(int(t) in frames,'Missing admitted exact frame');frame=frames[int(t)]
                    tokens,counts=frame_tokens(roster,frame,tags);values[j]=tokens;quality[row['role']].update(counts)
                    old,_,_,_=frame_features(roster,frame,tags)
                    need(np.array_equal(old,a['x_combat'][j]),'Overlapping112 combat features differ from frozen parent')
                    b=float(tokens[:5,2].sum());r=float(tokens[5:,2].sum())
                    need(a['x_rich'][j,ALL_RICH_FEATURES.index('rich_position_count_diff')]==b-r
                         and a['x_rich'][j,ALL_RICH_FEATURES.index('rich_position_count_sum')]==b+r,'Position availability differs from frozen parent')
                matrix[offset:offset+n]=values
                entry={k:v for k,v in row.items() if k not in ('file','sha256')}
                entry.update(offset=offset,parent_shard_file=row['file'],parent_shard_sha256=row['sha256']);entries.append(entry);offset+=n
                del payload,match,timeline,frames,values,a
        need(offset==396667 and len(entries)==17195,'Incomplete player cache');matrix.flush();del matrix;matrix=None;guard()
        for path_,digest in bindings.items():need(safe.digest(path_)==digest,'Bound input/code changed during extraction')
        verify_scope(scope);need(not set(source)&current_reservations(admitted),'Reservation changed during extraction')
        position={role:q['position_observed']/q['player_frames'] for role,q in quality.items()}
        result={'schema':'wp-joint-player-cache-v1','state':'complete','training_authorized':True,'games':entries,
            'game_count':17195,'frames':396667,'feature_names':list(FIELDS),'tensor_file':'players.npy','tensor_shape':[396667,10,35],
            'tensor_dtype':'float32','tensor_sha256':safe.digest(path),'parent_manifest_path':str(PARENT),'parent_manifest_sha256':parent_sha,
            'bindings_sha256':bindings,'source_manifest_path':str(Path(parent['source_bindings_path']).resolve()),'source_manifest_sha256':source_sha,
            'scope_path':parent['scope_path'],'scope_sha256':scope_sha,'quality':{k:dict(v) for k,v in quality.items()},
            'position_observed_fraction':position,'all_parent_combat_frames_reproduced_exactly':396667,
            'fit_sampled_frames':64422,'raw_payload_reads':34390,'access_ledger_sha256':safe.digest(output/'access_ledger.jsonl'),
            'input_lock_sha256':safe.digest(output/'input_lock.json'),'outcome_fields_indexed_for_tokens':False,'models_fitted':False,
            'sealed_or_confirmation_payload_reads':False,'elapsed_s':time.monotonic()-started,
            'limits':['All source frames retain the parent qualification; missing token values are masked, never row filters.',
                      'Position/health resource combinations are observational OMNI, not causal action values or live-safe.',
                      'All2861development games have previous exposure; no untouched or confirmation claim.']}
        write(output/'manifest.json',result);status('complete',games=17195,frames=396667,tensor_bytes=path.stat().st_size);return result
    except Exception as error:
        if matrix is not None:matrix.flush()
        status('failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    result=run(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ('state','game_count','frames','position_observed_fraction','elapsed_s')},indent=2))
