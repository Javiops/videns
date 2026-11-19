"""Stream a larger admitted WP cohort into immutable per-game feature shards."""
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
from winprob.wp_scale_scope_v1 import checked_json,verify_scope
from winprob.wp_scale_cache_v1 import convert_game,verify_pilot_arrays
from winprob.features_rich import ALL_RICH_FEATURES
from winprob.features_current_combat_v1 import FEATURES,STAT_SCALES
from winprob.experiment import sample_training_indices

PROTOCOL=ROOT/'docs/wp_scale_protocol_v1.md'
SOURCES=ROOT/'data/runs/wp-scale-bind-20260930-v1/bindings/sources.json'
RICH=ROOT/'data/runs/rich-20260929-v1/experiment/rich_states.npz'
COMBAT=ROOT/'data/runs/current-combat-cache-20260930-v1/cache/manifest.json'
STATIC=ROOT/'artifacts/boots-static/20260930-v1/boots_static.json'


def need(value,message):
    if not value:raise ValueError(message)


def write(path,value):
    safe.write_json(path,value);Path(path).with_suffix('.sha256').write_text(safe.digest(path)+' '+Path(path).name+'\n',encoding='ascii')


def run(output):
    started=time.monotonic();output=Path(output).resolve();need(not output.exists(),'Refuse scale-cache overwrite')
    safe.set_below_normal_priority();bindings={}
    def guard():
        need(time.monotonic()-started<1800,'Scale extraction1800second ceiling')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            value=Memory();value.length=ctypes.sizeof(value)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) and value.available>=.75*1024**3,'Scale extraction RAM floor0.75GiB')
    def bind(path,digest=None):
        path=Path(path).resolve(strict=True);safe.reject_quarantined(path);digest=safe.digest(path) if digest is None else digest
        need(str(path) not in bindings or bindings[str(path)]==digest,'Conflicting bound digest')
        need(safe.digest(path)==digest,'Bound input/code changed');bindings[str(path)]=digest
    guard()
    code=[Path(__file__),PROTOCOL,ROOT/'winprob/wp_scale_scope_v1.py',ROOT/'winprob/wp_scale_cache_v1.py',ROOT/'winprob/features.py',
          ROOT/'winprob/features_rich.py',ROOT/'winprob/features_current_combat_v1.py',ROOT/'winprob/cache.py',ROOT/'winprob/admission.py',
          ROOT/'winprob/iql_dataset_contract.py',ROOT/'winprob/experiment.py',ROOT/'winprob/metrics.py',
          ROOT/'scripts/rich_experiment.py',ROOT/'scripts/setback_experiment.py']
    for path in code:bind(path)
    sources,source_sha=checked_json(SOURCES);need(source_sha=='056e0e4bd9933eefb7507240c0742d290bee414e74e49159986b507e6b117f5c','Wrong bound source set')
    scope,scope_sha=checked_json(sources['scope_path']);need(scope_sha==sources['scope_sha256'],'Changed source scope')
    manifest,pilot,requested=verify_scope(scope)
    for path,digest in scope['bindings_sha256'].items():bind(path,digest)
    bind(SOURCES,source_sha);bind(sources['scope_path'],scope_sha)
    available={row['match_id']:row for row in sources['compressed_sources']};missing={row['match_id']:row for row in sources['missing_archives']}
    need(not(set(available)&set(missing)) and set(available)|set(missing)=={row['match_id'] for row in requested},'Source availability ledger differs')
    combat,csha=checked_json(COMBAT);need(csha=='117f7ad81732ce9758cbf088037a57ff680a858331d76f8da2e863ef1236bfed','Wrong pilot combat cache')
    bind(COMBAT,csha);bind(RICH,'65414f41331d4a82601ef38d9a95ff4b465bc90a758629730683593a93e729cb')
    bind(STATIC,'5e3fcf7ebb18bd2a26769d636a40a40b8a8242a382ba3d470115b03b005544d0')
    extra_path=COMBAT.parent/combat['feature_file'];bind(extra_path,combat['feature_sha256'])
    for path,digest in combat['bindings_sha256'].items():bind(path,digest)
    static=json.loads(STATIC.read_text())['patches']
    with np.load(RICH,allow_pickle=False) as loaded:old={name:loaded[name] for name in loaded.files}
    with np.load(extra_path,allow_pickle=False) as loaded:extra={name:loaded[name] for name in loaded.files}
    for key in ('game','role','patch','timestamp_ms'):need(np.array_equal(old[key],extra[key]),'Pilot old/combat row alignment differs')
    unique,starts,counts=np.unique(old['game'],return_index=True,return_counts=True)
    need(np.array_equal(unique,np.arange(6000)) and np.array_equal(old['game'],np.repeat(unique,counts)),'Pilot frames are not contiguous by canonical game')
    output.mkdir(parents=True,exist_ok=False);(output/'games').mkdir()
    def status(stage,**extra_status):write(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**extra_status})
    write(output/'input_lock.json',{'published_before_new_payload_deserialization':True,'scope_sha256':scope_sha,'sources_sha256':source_sha,
        'bindings_sha256':bindings,'requested_games':17195,'unchanged_pilot_games':6000,'rich_feature_names':list(ALL_RICH_FEATURES),
        'combat_feature_names':list(FEATURES),'raw_scope':'new admitted complement only; pilot rows reused exactly'})
    entries=[];exclusions=[];quality={role:Counter(value['counts']) for role,value in combat['quality'].items()}
    dynamics={role:Counter(value['dynamics']) for role,value in combat['quality'].items()};end_counts=defaultdict(Counter)
    archive_root=Path(manifest['source_data_root'])
    def save(entry,arrays,start_ms,ending,pilot_flag):
        gid=len(entries);file=f'games/{gid:05d}.npz';path=output/file
        selected=sample_training_indices(np.full(len(arrays['y']),gid,dtype=np.int32),arrays['timestamp_ms']) if entry['role']=='fit' else np.empty(0,dtype=np.int64)
        with path.open('xb') as handle:np.savez_compressed(handle,**arrays,fit_selected_local=selected)
        if pilot_flag:
            with np.load(path,allow_pickle=False) as saved:verify_pilot_arrays(saved,arrays['x_rich'],arrays['x_combat'],arrays['y'],arrays['timestamp_ms'])
        record={'game_index':gid,'scope_index':entry['scope_index'],'match_id':entry['match_id'],'role':entry['role'],
            'source_role':entry['source_role'],'patch':entry['patch'],'game_creation_ms':entry['game_creation_ms'],'game_start_ms':start_ms,
            'cohort':entry['cohort'],'rows':len(arrays['y']),'fit_sampled_frames':len(selected),
            'file':file,'sha256':safe.digest(path),'evaluation_only_end_ms':ending}
        entries.append(record);end_counts[entry['role']]['known' if ending>=0 else 'unknown']+=1
    try:
        with (output/'access_ledger.jsonl').open('x',encoding='utf-8') as ledger:
            for position,entry in enumerate(requested):
                if position%50==0:guard();status('streaming_games',processed=position,total=17195,cached=len(entries),excluded=len(exclusions))
                mid=entry['match_id']
                if entry['cohort']=='pilot':
                    gid=entry['pilot_game_index'];need(gid==len(entries),'Pilot game index changed')
                    lo=int(starts[gid]);hi=lo+int(counts[gid]);slice_=slice(lo,hi)
                    need(set(old['role'][slice_])=={entry['role']} and set(old['patch'][slice_])=={entry['patch']},'Pilot role/patch mismatch')
                    arrays={'x_rich':old['x'][slice_],'x_combat':extra['x_extra'][slice_],'y':old['y'][slice_],'timestamp_ms':old['timestamp_ms'][slice_]}
                    need(len(set(extra['evaluation_only_end_ms'][slice_].tolist()))==1,'Pilot end timing differs within game')
                    save(entry,arrays,int(old['start_ms'][lo]),int(extra['evaluation_only_end_ms'][lo]),True)
                    continue
                if mid in missing:exclusions.append({**missing[mid],'scope_index':entry['scope_index'],'cohort':'complement'});continue
                payload={}
                for endpoint in ('match','timeline'):
                    binding=available[mid][endpoint]
                    ledger.write(json.dumps({'match_id':mid,'source_role':entry['source_role'],'role':entry['role'],'endpoint':endpoint,**binding})+'\n');ledger.flush()
                    payload[endpoint]=read_frozen_payload(binding,endpoint,mid,archive_root)
                converted=convert_game(payload['match'],payload['timeline'],entry,static[entry['patch']]);del payload
                if converted is None:
                    exclusions.append({'match_id':mid,'role':entry['role'],'scope_index':entry['scope_index'],'cohort':'complement','reason':'zero_valid_states'})
                    continue
                save(entry,converted['arrays'],converted['game_start_ms'],converted['evaluation_only_end_ms'],False)
                quality[entry['role']].update(converted['quality']);dynamics[entry['role']].update(converted['dynamics']);del converted
        need(len(entries)>=6000 and all(row['cohort']=='pilot' and row['game_index']==i and row['match_id']==pilot['games'][i]['match_id'] for i,row in enumerate(entries[:6000])),
             'Original pilot identity prefix changed')
        need(sum(row['fit_sampled_frames'] for row in entries[:6000])==22554,'Original pilot sampled FIT count differs')
        role_counts=Counter(row['role'] for row in entries);complement_counts=Counter(row['role'] for row in entries if row['cohort']=='complement')
        checks={'all_original_pilot_games_and_values_preserved':True,'minimum11000_fit':role_counts['fit']>=11000,
                'minimum2700_calibration':role_counts['calibration']>=2700,'minimum1700_complement_development':complement_counts['development']>=1700}
        quality_report={}
        for role in ('fit','calibration','development'):
            q=quality[role];d=dynamics[role];ratios={}
            for field,threshold in [('health_observed',.99),('wallet_observed',.99),('health_ratio_observed',.99)]+[('stat_'+key+'_observed',.95) for key in STAT_SCALES]:
                ratios[field]=q[field]/q['player_frames'];checks[role+'_'+field]=ratios[field]>=threshold
            changing=d['varying_health_or_maximum']/d['participant_series'];checks[role+'_dynamic_health']=changing>=.9
            quality_report[role]={'counts':dict(q),'fractions':ratios,'dynamics':dict(d),'changing_fraction':changing}
        for path,digest in bindings.items():need(safe.digest(path)==digest,'Source/code changed during extraction')
        verify_scope(scope);need(not {row['match_id'] for row in entries}&current_reservations(manifest),'Reservation changed during extraction');guard()
        result={'schema':'wp-full-scale-feature-cache-v1','state':'complete','training_authorized':all(checks.values()),'gates':checks,
            'games':entries,'requested_games':17195,'qualified_games':len(entries),'frames':sum(row['rows'] for row in entries),
            'role_game_counts':dict(role_counts),'complement_role_game_counts':dict(complement_counts),'exclusions':exclusions,
            'rich_feature_names':list(ALL_RICH_FEATURES),'combat_feature_names':list(FEATURES),'quality':quality_report,
            'end_time_coverage':{role:dict(value) for role,value in end_counts.items()},'source_bindings_path':str(SOURCES),
            'source_bindings_sha256':source_sha,'scope_path':str(Path(sources['scope_path']).resolve()),
            'scope_sha256':scope_sha,'bindings_sha256':bindings,
            'admission_path':scope['admission_path'],'admission_sha256':scope['admission_sha256'],
            'access_ledger_sha256':safe.digest(output/'access_ledger.jsonl'),'input_lock_sha256':safe.digest(output/'input_lock.json'),
            'pilot_features_reused_without_raw_reopening':True,'new_payloads_deserialized':2*sum(row['cohort']=='complement' for row in requested if row['match_id'] in available),
            'outcomes_extracted_for_supervised_roles':True,'confirmation_or_sealed_payload_reads':False,'models_fitted':False,
            'elapsed_s':time.monotonic()-started,'limits':['Larger admitted development study; no confirmation or practical-benefit claim.',
                'Rich/combat definitions and original pilot rows unchanged; new qualifying games appended without replacement.',
                'Current combat inputs are observational OMNI, not action-neutral or live-safe.']}
        write(output/'manifest.json',result);status('complete',training_authorized=result['training_authorized'],games=len(entries),frames=result['frames'])
        return result
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    result=run(parser.parse_args().output)
    print(json.dumps({key:result[key] for key in ('qualified_games','frames','role_game_counts','complement_role_game_counts','gates','elapsed_s')},indent=2))
