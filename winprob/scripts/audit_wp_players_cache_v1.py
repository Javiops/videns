"""Independent token algebra, scope ledger and sampled raw-value reconstruction."""
from collections import Counter,defaultdict
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
from winprob.wp_scale_scope_v1 import checked_json,verify_scope
from winprob.iql_dataset_contract import file_hash,current_reservations
from winprob.features_wp_players_v1 import FIELDS
from rich_experiment import read_frozen_payload

CACHE=ROOT/'data/runs/wp-players-cache-20260930-v1/cache'
OUT=ROOT/'data/reports/wp_players_cache_audit_v1.json'
TAGS=('Assassin','Fighter','Mage','Marksman','Support','Tank')
SCALES={'attackDamage':500.,'abilityPower':1000.,'armor':100.,'magicResist':100.,'attackSpeed':100.,'movementSpeed':500.}


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def independent_token(raw,side,tags):
    def number(v):return float(v) if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) else None
    position=raw.get('position');position=position if isinstance(position,dict) else {};px=number(position.get('x'));py=number(position.get('y'))
    known=px is not None and py is not None and 0<px<=14870 and 0<py<=14870
    gold=number(raw.get('totalGold'));wallet=number(raw.get('currentGold'));stats=raw.get('championStats')
    stats=stats if isinstance(stats,dict) else {};need(gold is not None and gold>=0,'Invalid independently reconstructed gold')
    h=number(stats.get('health'));h=h if h is not None and h>=0 else None
    maximum=number(stats.get('healthMax'));maximum=maximum if maximum is not None and maximum>0 else None
    ratio=h/maximum if h is not None and maximum is not None and h<=maximum else None
    level=number(raw.get('level'));level=level if level is not None and 1<=level<=30 else None
    result={'position_x':px/14870 if known else 0.,'position_y':py/14870 if known else 0.,'position_observed':int(known),
        'total_gold_over10000':gold/10000,'wallet_over2000':wallet/2000 if wallet is not None else 0.,'wallet_observed':int(wallet is not None),
        'net_gold_outflow_over10000':(gold-wallet)/10000 if wallet is not None else 0.,'health_over2000':h/2000 if h is not None else 0.,
        'health_observed':int(h is not None),'health_max_over2000':maximum/2000 if maximum is not None else 0.,
        'health_max_observed':int(maximum is not None),'health_ratio':ratio if ratio is not None else 0.,
        'health_ratio_observed':int(ratio is not None),'health_positive':int(h is not None and h>0),
        'level_over18':level/18 if level is not None else 0.,'level_observed':int(level is not None),'side_blue_positive':1. if side==100 else -1.}
    for field,scale in SCALES.items():
        value=number(stats.get(field));result[field+'_scaled']=value/scale if value is not None else 0.;result[field+'_observed']=int(value is not None)
    result.update({'tag_'+tag:int(tag in tags)/len(tags) for tag in TAGS})
    need(set(result)==set(FIELDS),'Independent token definition incomplete');return np.asarray([result[k] for k in FIELDS],dtype=np.float32)


def run():
    started=time.monotonic();need(not OUT.exists(),'Refuse audit overwrite')
    def guard():
        need(time.monotonic()-started<900,'Player cache audit900second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            m=Memory();m.length=ctypes.sizeof(m)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) and m.available>=.75*1024**3,'Audit RAM floor')
    need(read(CACHE.parent/'launcher_status.json')['state']=='completed','Require terminal cache producer')
    cache,csha=checked_json(CACHE/'manifest.json');need(cache['state']=='complete' and cache['training_authorized']
        and cache['schema']=='wp-joint-player-cache-v1' and cache['feature_names']==list(FIELDS),'Incomplete player cache')
    parent,psha=checked_json(cache['parent_manifest_path']);need(psha==cache['parent_manifest_sha256'],'Parent changed')
    scope,ssha=checked_json(cache['scope_path']);need(ssha==cache['scope_sha256'],'Scope changed');admission,pilot,requested=verify_scope(scope)
    for path,digest in cache['bindings_sha256'].items():need(file_hash(path)==digest,'Changed bound input')
    sources,source_sha=checked_json(cache['source_manifest_path']);need(source_sha==cache['source_manifest_sha256'],'Source bindings changed')
    source={r['match_id']:r for r in sources['compressed_sources']};ids={r['match_id'] for r in cache['games']}
    need(len(ids)==len(cache['games'])==17195 and ids=={r['match_id'] for r in requested} and not ids&current_reservations(admission),'Invalid source scope')
    need(file_hash(CACHE/'access_ledger.jsonl')==cache['access_ledger_sha256'],'Source access ledger changed')
    access=set();roles={r['match_id']:r['role'] for r in cache['games']}
    for line in (CACHE/'access_ledger.jsonl').read_text().splitlines():
        row=json.loads(line);key=(row['match_id'],row['endpoint']);need(key not in access,'Duplicate raw access');access.add(key)
        need({k:row[k] for k in ('path','bytes','sha256')}==source[key[0]][key[1]] and row['role']==roles[key[0]],'Wrong raw binding/role')
    need(access=={(mid,endpoint) for mid in ids for endpoint in ('match','timeline')} and len(access)==34390,'Raw scope accounting differs')
    tensor_path=CACHE/cache['tensor_file'];need(file_hash(tensor_path)==cache['tensor_sha256'],'Tensor changed')
    tokens=np.load(tensor_path,mmap_mode='r',allow_pickle=False);need(tokens.shape==(396667,10,35) and tokens.dtype==np.float32,'Tensor schema differs')
    ix={name:i for i,name in enumerate(FIELDS)};quality={r:Counter() for r in ('fit','calibration','development')};offset=0
    masks=[name for name in FIELDS if name.endswith('_observed')]+['health_positive']
    for gid,(row,old) in enumerate(zip(cache['games'],parent['games'],strict=True)):
        if gid%100==0:guard()
        need(row['game_index']==gid and row['offset']==offset,'Offset/order mismatch')
        for key,value in old.items():need(row[{'file':'parent_shard_file','sha256':'parent_shard_sha256'}.get(key,key)]==value,'Parent row identity mismatch')
        a=tokens[offset:offset+row['rows']];need(np.isfinite(a).all(),'Nonfinite tokens');q=quality[row['role']]
        need(np.all(a[:,:5,-1]==1) and np.all(a[:,5:,-1]==-1),'Wrong side order')
        for name in masks:need(np.all((a[:,:,ix[name]]==0)|(a[:,:,ix[name]]==1)),'Nonbinary availability')
        pos=a[:,:,ix['position_observed']].astype(bool);coordinates=a[:,:,:2]
        need(np.all(coordinates[~pos]==0) and np.all((coordinates[pos]>0)&(coordinates[pos]<=1)),'Invalid masked position')
        h=a[:,:,ix['health_over2000']];known=a[:,:,ix['health_observed']].astype(bool)
        need(np.all(h>=0) and np.all(h[~known]==0) and np.array_equal(a[:,:,ix['health_positive']],(h>0).astype(np.float32)),'Invalid masked health')
        ratio=a[:,:,ix['health_ratio']];rk=a[:,:,ix['health_ratio_observed']].astype(bool)
        need(np.all((ratio>=0)&(ratio<=1)) and np.all(ratio[~rk]==0),'Invalid masked health ratio')
        wallet=a[:,:,ix['wallet_over2000']];wk=a[:,:,ix['wallet_observed']].astype(bool);outflow=a[:,:,ix['net_gold_outflow_over10000']]
        need(np.all(wallet[~wk]==0) and np.all(outflow[~wk]==0),'Invalid masked wallet')
        need(np.allclose(outflow[wk],(a[:,:,ix['total_gold_over10000']]-wallet*.2)[wk],rtol=0,atol=1e-6),'Net-outflow algebra differs')
        need(np.allclose(a[:,:,[ix['tag_'+tag] for tag in TAGS]].sum(axis=2),1,rtol=0,atol=1e-6),'Tag mass differs')
        q['player_frames']+=10*row['rows'];q['position_observed']+=int(pos.sum())
        for field,key in (('health','health_observed'),('maximum','health_max_observed'),('ratio','health_ratio_observed'),('wallet','wallet_observed'),('level','level_observed')):
            q[field+'_observed']+=int(a[:,:,ix[key]].sum())
        offset+=row['rows']
    need(offset==396667 and {k:dict(v) for k,v in quality.items()}==cache['quality'],'Full token quality accounting differs')
    cells=defaultdict(list)
    for row in cache['games']:cells[row['role'],row['patch']].append(row)
    chosen=[row for key in sorted(cells) for row in sorted(cells[key],key=lambda r:hashlib.sha256(('wp-player-audit-v1/'+r['match_id']).encode()).digest())[:2]]
    selection=OUT.with_name('wp_players_cache_audit_v1_selection.json')
    with selection.open('x',encoding='utf-8') as handle:json.dump({'chosen_before_raw_reads':True,'games':chosen},handle,indent=2)
    static=read(ROOT/'artifacts/boots-static/20260930-v1/boots_static.json')['patches'];raw_frames=0;maximum_error=0.
    ledger=OUT.with_name('wp_players_cache_audit_v1_access.jsonl')
    with ledger.open('x',encoding='utf-8') as log:
        for row in chosen:
            guard();mid=row['match_id'];payload={}
            for endpoint in ('match','timeline'):
                binding=source[mid][endpoint];log.write(json.dumps({'match_id':mid,'endpoint':endpoint,**binding})+'\n');log.flush()
                payload[endpoint]=read_frozen_payload(binding,endpoint,mid,Path(admission['source_data_root']))
            roster=sorted([(r['participantId'],r['teamId'],r['championId']) for r in payload['match']['info']['participants']],key=lambda r:(r[1],r[0]))
            frames={int(f['timestamp']):f for f in payload['timeline']['info']['frames']}
            shard=Path(cache['parent_manifest_path']).parent/row['parent_shard_file'];need(file_hash(shard)==row['parent_shard_sha256'],'Sample parent shard changed')
            with np.load(shard,allow_pickle=False) as old:clock=old['timestamp_ms']
            for j,t in enumerate(clock):
                frame=frames[int(t)];expected=np.stack([independent_token(frame['participantFrames'][str(pid)],side,static[row['patch']]['champion_tags'][str(champion)]) for pid,side,champion in roster])
                actual=tokens[row['offset']+j];error=float(np.max(np.abs(expected-actual)));maximum_error=max(maximum_error,error)
                need(np.array_equal(expected,actual),'Independent raw token reconstruction differs');raw_frames+=1
    need(file_hash(CACHE/'manifest.json')==csha and not ids&current_reservations(admission),'Manifest/reservations changed')
    result={'schema':'wp-player-cache-independent-audit-v1','state':'passed','cache_manifest_sha256':csha,
        'audit_code_sha256':file_hash(__file__),'all_frames_algebra_checked':396667,'games':17195,'raw_accesses_accounted':34390,
        'raw_sample_games':len(chosen),'raw_sample_frames':raw_frames,'maximum_raw_feature_error':maximum_error,
        'selection_sha256':file_hash(selection),'audit_access_sha256':file_hash(ledger),'reserved_intersections':0,
        'outcome_fields_indexed_in_raw_audit':False,'models_fitted':False,'elapsed_s':time.monotonic()-started,
        'limits':['Independent raw reconstruction covers the fixed role/patch sample; all tensor rows receive scope/algebra/quality checks.',
                  'Parent shard content is reverified in full by model loading; this audit rehashes the raw-replay sample only.']}
    with OUT.open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2,sort_keys=True);handle.write('\n')
    OUT.with_suffix('.sha256').write_text(file_hash(OUT)+' '+OUT.name+'\n',encoding='ascii');print(json.dumps(result,indent=2))


if __name__=='__main__':run()
