"""Ten deterministic FIT games: current combat-field availability, no outcomes."""
from collections import Counter,defaultdict
import argparse
import ctypes
import hashlib
import json
import math
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

FIELDS=('health','healthMax','armor','magicResist','attackDamage','attackSpeed','abilityPower','abilityHaste',
        'movementSpeed','power','powerMax')
SALT='current-state-schema-v1/'


def need(value,message):
    if not value:raise ValueError(message)


def numeric(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def run(output):
    started=time.monotonic();output=Path(output).resolve();need(not output.exists(),'Refuse schema inspection overwrite')
    def guard():
        need(time.monotonic()-started<120,'Schema inspection120second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            value=Memory();value.length=ctypes.sizeof(value)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)) and value.available>=.75*1024**3,
                 'Schema inspection RAM floor0.75GiB')
    safe.set_below_normal_priority();guard()
    original=ROOT/'data/runs/w1-pilot-20260929-v3'
    source,bindings=safe.verify_inputs(original/'cache/states.npz',original/'cache/source_manifest.json',original/'experiment')
    manifest=json.loads(Path(source['admission_manifest']).read_text());forbidden=current_reservations(manifest)
    selected=[]
    for patch in sorted({row['patch'] for row in source['games'] if row['role']=='fit'}):
        candidates=[row for row in source['games'] if row['role']=='fit' and row['patch']==patch]
        selected.extend(sorted(candidates,key=lambda row:hashlib.sha256((SALT+row['match_id']).encode()).digest())[:5])
    need(len(selected)==10 and all(row['match_id'] not in forbidden for row in selected),'Unexpected/forbidden schema scope')
    paths=[Path(__file__),ROOT/'scripts/rich_experiment.py',ROOT/'scripts/setback_experiment.py',
           ROOT/'winprob/admission.py',ROOT/'winprob/iql_dataset_contract.py']
    bindings.update({str(path.resolve()):safe.digest(path) for path in paths})
    output.mkdir(parents=True,exist_ok=False)
    safe.write_json(output/'selection.json',{'selection_precedes_source_payloads':True,'salt':SALT,'per_FIT_patch':5,
        'games':selected,'bindings_sha256':bindings,'outcomes_used_for_selection':False})
    with np.load(original/'cache/states.npz',allow_pickle=False) as cache:
        game_index=cache['game'];timestamps=cache['timestamp_ms']
    compressed={row['match_id']:row for row in source['compressed_sources']}
    source_root=Path(manifest['source_data_root']).resolve(strict=True)
    values=defaultdict(list);missing=Counter();observations=Counter();schemas=Counter();coverage=[]
    def record(name,value):
        observations[name]+=1
        if numeric(value):values[name].append(float(value))
        else:missing[name]+=1
    with (output/'access_ledger.jsonl').open('x',encoding='utf-8') as ledger:
        for row in selected:
            guard();mid=row['match_id'];payloads={}
            for endpoint in ('match','timeline'):
                binding=compressed[mid][endpoint]
                ledger.write(json.dumps({'match_id':mid,'role':'fit','endpoint':endpoint,**binding})+'\n');ledger.flush()
                payloads[endpoint]=read_frozen_payload(binding,endpoint,mid,source_root)
            info=payloads['match']['info']
            need(info['queueId']==420 and info['gameCreation']==row['game_creation_ms'],'Source metadata differs')
            roster={int(player['participantId']):(int(player['teamId']),int(player['championId'])) for player in info['participants']}
            need(set(roster)==set(range(1,11)) and Counter(team for team,champ in roster.values())=={100:5,200:5},'Roster malformed')
            raw_frames=payloads['timeline']['info']['frames'];frames={int(frame['timestamp']):frame for frame in raw_frames}
            need(len(frames)==len(raw_frames),'Duplicate source timestamp')
            eligible=timestamps[game_index==row['game_index']]
            need(len(eligible)==row['rows'] and all(int(t) in frames for t in eligible),'Cached frame alignment differs')
            for t in eligible:
                frame=frames[int(t)]
                for pid in roster:
                    player=frame['participantFrames'][str(pid)];stats=player.get('championStats',{})
                    schemas.update('championStats.'+key for key in stats)
                    for key in ('currentGold','totalGold','level','xp'):record(key,player.get(key))
                    for key in FIELDS:record('championStats.'+key,stats.get(key))
                    health,maximum=stats.get('health'),stats.get('healthMax')
                    record('derived.health_fraction',health/maximum if numeric(health) and numeric(maximum) and maximum>0 else None)
                    gold,wallet=player.get('totalGold'),player.get('currentGold')
                    record('derived.earned_minus_wallet',gold-wallet if numeric(gold) and numeric(wallet) else None)
            coverage.append({'match_id':mid,'patch':row['patch'],'frames':len(eligible),'player_frames':10*len(eligible)})
            del payloads,info,raw_frames,frames
    for path,digest in bindings.items():need(safe.digest(path)==digest,'Source/code changed during inspection')
    admission.verify_manifest_sources(manifest);need(not {r['match_id'] for r in selected}&current_reservations(manifest),'Reservations changed')
    guard();statistics={}
    for name,count in observations.items():
        raw=np.asarray(values[name])
        statistics[name]={'observations':count,'finite':len(raw),'missing_or_invalid':missing[name],
            'zero':int((raw==0).sum()),'negative':int((raw<0).sum()),
            'quantiles_min_p25_p50_p75_max':np.percentile(raw,[0,25,50,75,100]).tolist() if len(raw) else None}
    result={'schema':'current-combat-field-inspection-v1','state':'complete','games':10,'coverage':coverage,
        'statistics':statistics,'observed_champion_stat_key_counts':dict(schemas),'bindings_sha256':bindings,
        'selection_sha256':safe.digest(output/'selection.json'),'access_ledger_sha256':safe.digest(output/'access_ledger.jsonl'),
        'raw_sources_read':20,'source_json_deserialization_includes_outcomes':True,'outcome_fields_indexed':False,
        'cache_y_array_accessed':False,'development_calibration_or_confirmation_payloads_read':False,
        'models_fitted':False,'elapsed_s':time.monotonic()-started,
        'limits':['Ten FIT games across two training patches; schema reconnaissance, not full coverage or predictive evidence.',
                  'Health-positive is a recorded-value proxy, not proof of every champion active/dead state.',
                  'Earned minus wallet is not certified inventory value; stat units are not inferred from observed ranges.']}
    safe.write_json(output/'report.json',result)
    (output/'report.sha256').write_text(safe.digest(output/'report.json')+' report.json\n',encoding='ascii')
    print(json.dumps({'games':10,'player_frames':sum(r['player_frames'] for r in coverage),'statistics':statistics,'elapsed_s':result['elapsed_s']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
