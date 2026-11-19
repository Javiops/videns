"""Verify the complete scale cache, pilot parity and sampled raw replay before fit."""
from collections import Counter,defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from winprob.wp_scale_scope_v1 import checked_json,verify_scope
from winprob.iql_dataset_contract import file_hash,current_reservations
from winprob.wp_scale_cache_v1 import convert_game
from rich_experiment import read_frozen_payload

CACHE=ROOT/'data/runs/wp-scale-cache-20260930-v1/cache'
OUT=ROOT/'data/reports/wp_scale_cache_audit_v1.json'


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')
    Path(path).with_suffix('.sha256').write_text(file_hash(path)+' '+Path(path).name+'\n',encoding='ascii')


def run():
    started=time.monotonic();need(not OUT.exists(),'Refuse audit overwrite')
    def guard():
        need(time.monotonic()-started<900,'Cache audit900second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            m=Memory();m.length=ctypes.sizeof(m)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) and m.available>=.75*1024**3,'Audit RAM floor')
    need(read(CACHE.parent/'launcher_status.json')['state']=='completed','Cache producer must be terminal')
    cache,csha=checked_json(CACHE/'manifest.json');scope,ssha=checked_json(cache['scope_path'])
    need(cache['state']=='complete' and cache['training_authorized'] and all(cache['gates'].values()),'Unqualified cache')
    need(ssha==cache['scope_sha256'],'Scope digest differs');admitted,pilot,requested=verify_scope(scope)
    sources,source_sha=checked_json(cache['source_bindings_path']);need(source_sha==cache['source_bindings_sha256'],'Sources changed')
    for path,digest in cache['bindings_sha256'].items():need(file_hash(path)==digest,'Changed cache binding')
    req={r['match_id']:r for r in requested};entries=cache['games'];ids={r['match_id'] for r in entries};exc=cache['exclusions']
    excluded={r['match_id'] for r in exc}
    need(len(ids)==len(entries) and len(excluded)==len(exc) and not ids&excluded and ids|excluded==set(req),'Scope accounting differs')
    need(len(req)==cache['requested_games']==17195 and len(entries)==cache['qualified_games'],'Denominators differ')
    need(not ids&current_reservations(admitted),'Reserved cache identity')
    available={r['match_id']:r for r in sources['compressed_sources']};missing={r['match_id'] for r in sources['missing_archives']}
    expected_access={(r['match_id'],endpoint) for r in requested if r['cohort']=='complement' and r['match_id'] in available
                     for endpoint in ('match','timeline')}
    access=[]
    need(file_hash(CACHE/'access_ledger.jsonl')==cache['access_ledger_sha256'],'Payload ledger changed')
    for line in (CACHE/'access_ledger.jsonl').read_text().splitlines():
        item=json.loads(line);mid=item['match_id'];endpoint=item['endpoint'];access.append((mid,endpoint))
        need({k:item[k] for k in ('path','bytes','sha256')}==available[mid][endpoint],'Payload binding mismatch')
        need(item['role']==req[mid]['role'] and item['source_role']==req[mid]['source_role'],'Payload role mismatch')
    need(len(access)==len(set(access)) and set(access)==expected_access and len(access)==cache['new_payloads_deserialized'],'Payload access scope differs')
    for row in exc:
        need(row['match_id'] in missing or row['reason']=='zero_valid_states','Unexpected exclusion')
        need(row['cohort']=='complement' and row['scope_index']==req[row['match_id']]['scope_index'],'Invalid excluded identity')
    with np.load(ROOT/'data/runs/rich-20260929-v1/experiment/rich_states.npz',allow_pickle=False) as saved:old={k:saved[k] for k in saved.files}
    oldcombat=read(ROOT/'data/runs/current-combat-cache-20260930-v1/cache/manifest.json')
    with np.load(ROOT/'data/runs/current-combat-cache-20260930-v1/cache'/oldcombat['feature_file'],allow_pickle=False) as saved:
        combat=saved['x_extra'];ends=saved['evaluation_only_end_ms']
    _,starts,counts=np.unique(old['game'],return_index=True,return_counts=True)
    samples=Counter();frames=Counter();rolecounts=Counter();complement=Counter();endcounts=defaultdict(Counter)
    previous_scope=-1
    for gid,row in enumerate(entries):
        if gid%100==0:guard()
        wanted=req[row['match_id']];role=row['role'];need(row['game_index']==gid and row['scope_index']>previous_scope,'Game order differs')
        previous_scope=row['scope_index']
        for key in ('role','source_role','patch','game_creation_ms','scope_index','cohort'):need(row[key]==wanted[key],'Metadata differs')
        path=(CACHE/row['file']).resolve(strict=True);need(path.is_relative_to(CACHE) and file_hash(path)==row['sha256'],'Invalid shard binding')
        with np.load(path,allow_pickle=False) as saved:a={k:saved[k] for k in saved.files}
        n=row['rows'];clock=a['timestamp_ms'];y=a['y']
        need(set(a)=={'x_rich','x_combat','y','timestamp_ms','fit_selected_local'},'Unexpected shard fields')
        need(a['x_rich'].shape==(n,61) and a['x_combat'].shape==(n,112) and a['x_rich'].dtype==a['x_combat'].dtype==np.float32,'Feature schema differs')
        need(np.isfinite(a['x_rich']).all() and np.isfinite(a['x_combat']).all(),'Nonfinite features')
        need(y.shape==clock.shape==(n,) and y.dtype==np.int8 and clock.dtype==np.int64 and len(set(y.tolist()))==1
             and set(y.tolist())<={0,1} and np.all(np.diff(clock)>0) and clock.min()>=180000 and clock.max()<=2400000,'Label/time differs')
        buckets=defaultdict(list)
        if role=='fit':
            for i,t in enumerate(clock):buckets[int(t)//300000].append(i)
        chosen=sorted(min(v,key=lambda i:hashlib.sha256(f'20260929:{gid}:{clock[i]}'.encode()).digest()) for v in buckets.values())
        need(a['fit_selected_local'].dtype==np.int64 and a['fit_selected_local'].tolist()==chosen and len(chosen)==row['fit_sampled_frames'],'Sampling differs')
        ending=row['evaluation_only_end_ms'];need(ending==-1 or ending>int(clock.max()),'Invalid evaluation end')
        if gid<6000:
            need(row['match_id']==pilot['games'][gid]['match_id'] and row['cohort']=='pilot','Pilot identity changed')
            take=slice(int(starts[gid]),int(starts[gid]+counts[gid]))
            for actual,expected in ((a['x_rich'],old['x'][take]),(a['x_combat'],combat[take]),(y,old['y'][take]),(clock,old['timestamp_ms'][take])):
                need(np.array_equal(actual,expected),'Exact pilot array parity failed')
            need(set(old['role'][take])=={role} and set(old['patch'][take])=={row['patch']} and np.all(ends[take]==ending),'Pilot metadata/end changed')
            samples['pilot']+=len(chosen)
        else:need(row['cohort']=='complement','Unexpected pilot outside prefix');complement[role]+=1
        samples[role]+=len(chosen);frames[role]+=n;rolecounts[role]+=1;endcounts[role]['known' if ending>=0 else 'unknown']+=1
    need(samples['pilot']==22554 and sum(frames.values())==cache['frames'],'Frame/sample totals differ')
    need(dict(rolecounts)==cache['role_game_counts'] and dict(complement)==cache['complement_role_game_counts'],'Role totals differ')
    need({k:dict(v) for k,v in endcounts.items()}==cache['end_time_coverage'],'End coverage differs')
    # Fixed hash-based replay sample across each newly added role/patch cell.
    cells=defaultdict(list)
    for row in entries[6000:]:cells[row['role'],row['patch']].append(row)
    selected=[r for cell in sorted(cells) for r in sorted(cells[cell],key=lambda r:hashlib.sha256(('scale-audit-v1/'+r['match_id']).encode()).digest())[:2]]
    selection=OUT.with_name('wp_scale_cache_audit_v1_selection.json');write(selection,{'selected_before_raw_replay':True,'games':selected})
    static=read(ROOT/'artifacts/boots-static/20260930-v1/boots_static.json')['patches'];rawrows=0
    ledger=OUT.with_name('wp_scale_cache_audit_v1_access.jsonl')
    with ledger.open('x',encoding='utf-8') as log:
        for row in selected:
            guard();mid=row['match_id'];payload={}
            for endpoint in ('match','timeline'):
                binding=available[mid][endpoint];log.write(json.dumps({'match_id':mid,'endpoint':endpoint,**binding})+'\n');log.flush()
                payload[endpoint]=read_frozen_payload(binding,endpoint,mid,Path(admitted['source_data_root']))
            replay=convert_game(payload['match'],payload['timeline'],req[mid],static[row['patch']]);need(replay is not None,'Replay qualification changed')
            with np.load(CACHE/row['file'],allow_pickle=False) as saved:
                for key,value in replay['arrays'].items():need(np.array_equal(value,saved[key]),'Raw feature/label replay mismatch')
            need(replay['evaluation_only_end_ms']==row['evaluation_only_end_ms'] and replay['game_start_ms']==row['game_start_ms'],'Raw timing mismatch')
            rawrows+=len(replay['arrays']['y']);del payload,replay
    need(file_hash(CACHE/'manifest.json')==csha and not ids&current_reservations(admitted),'Cache/reservations changed')
    result={'schema':'wp-scale-cache-audit-v1','state':'passed','cache_manifest_sha256':csha,'audit_code_sha256':file_hash(__file__),
        'games':len(entries),'frames':sum(frames.values()),'role_games':dict(rolecounts),'role_frames':dict(frames),'fit_samples':samples['fit'],
        'pilot_games_exactly_verified':6000,'pilot_fit_samples':22554,'all_shards_verified':len(entries),'new_payload_reads_accounted':len(access),
        'raw_replay_games':len(selected),'raw_replay_frames':rawrows,'raw_replay_sources':2*len(selected),'reserved_intersections':0,
        'selection_sha256':file_hash(selection),'audit_access_sha256':file_hash(ledger),'elapsed_s':time.monotonic()-started,
        'limits':['Replay uses previously audited frozen extractors; it is not an independent second feature implementation.',
                  'New raw replay checks a deterministic role/patch sample; full shards, pilot parity and scope are checked exhaustively.',
                  'No model or development metric evaluated. All confirmation payloads remain closed.']}
    write(OUT,result);print(json.dumps(result,indent=2))


if __name__=='__main__':run()
