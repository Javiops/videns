"""Export already-stored public KDA for exactly the original admitted FIT states."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import closing
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import export_winprob_trajectories_v1 as producer

CONSUMER=ROOT.parent/'winprob'
BASE=CONSUMER/'data/runs/iql-cache-20260930-v1/cache/manifest.json'
HISTORY=CONSUMER/'data/runs/history-cache-20260930-v1/cache/manifest.json'
PROTOCOL=CONSUMER/'docs/scoreboard_information_protocol_v1.md'
BASE_SHA='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea'
HISTORY_SHA='1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b'


def require(ok,message):
    if not ok:raise ValueError(message)


def ordered_scores(event,participant_id,champion_index,expected_champions):
    """Only roster identity and cumulative public KDA enter this projection."""
    board=event.get('board')
    require(isinstance(board,list) and len(board)==10,'complete ten-player board required')
    pids=[p.get('participant_id') for p in board]
    require(type(participant_id) is int and 1<=participant_id<=10,'invalid focal participant')
    require(all(type(pid) is int for pid in pids) and set(pids)==set(range(1,11)) and len(set(pids))==10,'invalid/duplicate participant roster')
    me=next(p for p in board if p['participant_id']==participant_id)
    require(me.get('team_id') in (100,200),'invalid focal side');side=me['team_id']
    require(sum(p.get('is_self') is True for p in board)==1 and me.get('is_self') is True,'focal board identity mismatch')
    allies=[p for p in board if p['participant_id']!=participant_id and p.get('team_id')==side]
    enemies=[p for p in board if p.get('team_id') in (100,200) and p['team_id']!=side]
    require(len(allies)==4 and len(enemies)==5,'four allies/five enemies required')
    players=[me]+sorted(allies,key=lambda p:p['champion_id'])+sorted(enemies,key=lambda p:p['champion_id'])
    require(all(type(p.get('champion_id')) is int and str(p['champion_id']) in champion_index for p in players),'unknown champion')
    indices=[champion_index[str(p['champion_id'])] for p in players]
    require(np.array_equal(indices,expected_champions),'source roster order differs from compiled cache')
    scores=[]
    for p in players:
        values=[p.get(key) for key in ('kills','deaths','assists')]
        require(all(type(v) is int and 0<=v<=np.iinfo(np.int32).max for v in values),'missing/invalid public KDA')
        scores.append(values)
    return np.asarray(scores,dtype=np.int32),np.asarray([p['participant_id'] for p in players],dtype=np.uint8)


def poison_non_features(event):
    changed=deepcopy(event)
    changed.update(win_team=999,winner='poison',label_bought=[{'item_id':999999,'count':99}],bought=[],purchase_actions=[],
                   is_save='poison',save_kind='poison',gold_arrival_true=-999,total_gold=-999,team_position='poison')
    for player in changed.get('board',[]):
        player.update(team_position='poison',gold=-999,items=[],win='poison',future={'kills':999})
    return changed


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
    path.with_suffix('.sha256').write_text(producer.file_hash(path)+' '+path.name+'\n',encoding='ascii')


def run(output):
    started=time.monotonic();require(not output.exists(),'immutable new export directory required')
    base,bs=producer.load_bound(BASE);history,hs=producer.load_bound(HISTORY)
    require(bs==BASE_SHA and hs==HISTORY_SHA,'original cache/selection changed')
    admission_path=Path(base['source_admission_path']);admitted,tests,validation,confirm,bindings=producer.validate_admission(admission_path)
    denied=tests|validation|confirm;games=base['game_metadata'];wanted={g['match_id'] for g in games};admit={g['match_id']:g for g in admitted['games']}
    require(len(wanted)==len(games)==4001 and not wanted&denied,'invalid or reserved FIT population')
    require(all(admit[mid]['role']=='train' and admit[mid]['historical_validation'] is False for mid in wanted),'non-FIT source game')
    bindings.update({str(BASE):bs,str(HISTORY):hs,str(PROTOCOL):producer.file_hash(PROTOCOL)})
    a={}
    for name in ('timestamp_ms','game_index','episode_index','champion_indices','global_public'):
        path=BASE.parent/base['arrays'][name]['file'];sha=base['files'][path.name]
        require(producer.file_hash(path)==sha,'base state metadata/feature array changed');bindings[str(path)]=sha
        a[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    path=HISTORY.parent/history['arrays']['source_event_index']['file'];sha=history['files'][path.name]
    require(producer.file_hash(path)==sha,'original source event index changed');bindings[str(path)]=sha
    event_index=np.load(path,mmap_mode='r',allow_pickle=False)
    metadata={}
    for key in ('episode_metadata_file','feature_metadata_file'):
        path=BASE.parent/base[key];require(producer.file_hash(path)==base['files'][path.name],'cache metadata changed')
        bindings[str(path)]=base['files'][path.name];metadata[key]=producer.load_json(path)
    feature_meta=metadata['feature_metadata_file'];champion_index=feature_meta['champion_index']
    columns=[feature_meta['global_names'].index('own_'+key+'_over_20') for key in ('kills','deaths','assists')]
    episodes=metadata['episode_metadata_file'];n=base['state_count'];require(n==392448 and len(event_index)==n,'state count changed')
    by_game=defaultdict(list)
    for episode_index,ep in enumerate(episodes):
        lo=ep['first_state'];hi=lo+ep['states'];gi=ep['game_index'];require(ep['match_id']==games[gi]['match_id'],'episode/game mismatch')
        require(np.all(a['game_index'][lo:hi]==gi),'episode row range changed')
        require(np.all(a['episode_index'][lo:hi]==episode_index),'episode index changed')
        by_game[ep['match_id']].append(ep)
    require(set(by_game)==wanted,'episode coverage changed')
    sources=[]
    for directory in base['source_directories']:
        path=Path(directory)/'manifest.json';m,sha=producer.load_bound(path);bindings[str(path)]=sha
        require(m['schema']=='winprob-fit-trajectories-v2' and m['state']=='complete' and m['patch'] in ('16.17','16.18'),'wrong prior source release')
        for raw,want in {**m['input_metadata_sha256'],**m['source_code_sha256']}.items():
            require(producer.file_hash(raw)==want,'original source metadata/code changed');bindings[raw]=want
        ids={g['match_id'] for g in games if g['patch']==m['patch']}
        receipts={g['match_id']:g for g in m['game_bindings']}
        require(ids==set(m['eligible_game_ids']) and ids<=receipts.keys(),'source/cache admitted games differ')
        database=Path(m['source_database']).resolve(strict=True)
        require(database.is_relative_to(ROOT/'data/corpora') and database.name=='tracker.db','database escaped frozen source corpus')
        sources.append((m['patch'],database,ids,receipts))
    require(len(sources)==2 and {x[0] for x in sources}=={'16.17','16.18'},'two source patches required')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'selection.json',{'schema':'winprob-scoreboard-selection-v1','game_ids':sorted(wanted),'state_count':n,'base_manifest_sha256':bs,
        'history_manifest_sha256':hs,'protocol_sha256':bindings[str(PROTOCOL)],'before_sql_payload_reads':True})
    counts=np.lib.format.open_memmap(output/'player_kda.npy',mode='w+',dtype=np.int32,shape=(n,10,3));counts[:]=-1
    pids=np.lib.format.open_memmap(output/'participant_ids.npy',mode='w+',dtype=np.uint8,shape=(n,10));pids[:]=0
    seen=np.zeros(n,dtype=bool);ledger=[];plans={};mutation_rows=0
    code={str(Path(__file__).resolve()):producer.file_hash(__file__),str(Path(producer.__file__).resolve()):producer.file_hash(producer.__file__)}
    for patch,database,ids,receipts in sources:
        before=(database.stat().st_size,database.stat().st_mtime_ns)
        with closing(sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=2)) as conn:
            conn.execute('PRAGMA query_only=ON');conn.execute('PRAGMA cache_size=-2048')
            query='SELECT event_index,ts,ts_end,type,payload_json FROM game_events WHERE match_id=? ORDER BY event_index'
            plan=conn.execute('EXPLAIN QUERY PLAN '+query,(sorted(ids)[0],)).fetchall();plans[patch]=[list(x) for x in plan]
            require(any('SEARCH' in str(x).upper() for x in plan) and not any('SCAN GAME_EVENTS' in str(x).upper() for x in plan),'indexed source query required')
            reader=producer.ApprovedGameReader(conn,ids,denied)
            for mid in sorted(ids):
                require(time.monotonic()-started<1200,'fixed20minute export bound exceeded')
                if len(ledger)%100==0:producer.validate_admission(admission_path)
                game,events=reader.game(mid)
                receipt=receipts[mid]
                require(receipt['raw_replay_matches_indexed_rows'] is True and receipt['admitted_mask_reproduced'] is True,'original replay receipt missing')
                require(producer.value_hash({'game':game,'events':events})==receipt['indexed_rows_sha256'],'indexed source changed since certified FIT export')
                event_map={e['event_index']:e for e in events};matched=0
                require(len(event_map)==len(events),'duplicate source event index')
                for ep in by_game[mid]:
                    for row in range(ep['first_state'],ep['first_state']+ep['states']):
                        ev=event_map.get(int(event_index[row]));require(ev is not None and ev['participant_id']==ep['participant_id'] and ev['ts']==int(a['timestamp_ms'][row]),'state/event identity mismatch')
                        value,order=ordered_scores(ev,ep['participant_id'],champion_index,a['champion_indices'][row])
                        expected=(value[0].astype(np.float64)/20.).astype(np.float32)
                        require(np.array_equal(expected,a['global_public'][row,columns]),'focal scoreboard differs from original public features')
                        if matched==0:
                            alternate=ordered_scores(poison_non_features(ev),ep['participant_id'],champion_index,a['champion_indices'][row])
                            require(np.array_equal(alternate[0],value) and np.array_equal(alternate[1],order),'forbidden fields changed KDA projection');mutation_rows+=1
                        require(not seen[row],'duplicate state output');counts[row]=value;pids[row]=order;seen[row]=True;matched+=1
                ledger.append({'match_id':mid,'patch':patch,'matched_states':matched,'indexed_rows_sha256':receipt['indexed_rows_sha256']})
                if len(ledger)%100==0:
                    print(json.dumps({'games':len(ledger),'states':int(seen.sum()),'elapsed_s':time.monotonic()-started}),flush=True)
                del game,events,event_map
        require(before==(database.stat().st_size,database.stat().st_mtime_ns),'frozen source database changed during read')
    require(len(ledger)==4001 and bool(seen.all()) and np.all(counts>=0) and np.all(pids>0),'incomplete feature publication')
    counts.flush();pids.flush();maximum=counts.max(axis=(0,1)).tolist();other_nonzero=int(np.count_nonzero(np.any(counts[:,1:,:]!=0,axis=(1,2))))
    del counts,pids
    for raw,want in {**bindings,**code}.items():require(producer.file_hash(raw)==want,'input/code changed during export')
    producer.validate_admission(admission_path)
    write(output/'game_ledger.json',ledger)
    manifest={'schema':'winprob-scoreboard-v1','state':'complete','states':n,'games':4001,'player_order':'buyer; four allied champions ascending; five enemy champions ascending',
        'score_columns':['kills','deaths','assists'],'score_dtype':'int32','arrays':{'player_kda':[n,10,3],'participant_ids':[n,10]},
        'base_manifest_path':str(BASE),'base_manifest_sha256':bs,'history_manifest_path':str(HISTORY),'history_manifest_sha256':hs,
        'admission_path':str(admission_path),'protocol_path':str(PROTOCOL),'protocol_sha256':producer.file_hash(PROTOCOL),
        'bindings_sha256':bindings,'code_sha256':code,'files':{name:producer.file_hash(output/name) for name in ('selection.json','player_kda.npy','participant_ids.npy','game_ledger.json')},
        'sql_query_plans':plans,'all_state_identities_matched':True,'focal_score_mismatches':0,'roster_mismatches':0,'mutated_example_games':mutation_rows,
        'rows_with_nonzero_other_player_scores':other_nonzero,'maximum_scores':maximum,
        'raw_archive_reads':False,'whole_database_scanned':False,'test_payload_reads':False,'confirmation_payload_reads':False,
        'development_game_payload_reads':False,'calibration_game_payload_reads':False,'training_performed':False,'consumer_runtime_imported':False,
        'purchase_fields_deserialized':True,'outcome_metadata_hashed_for_identity':True,'purchase_labels_used_or_analyzed':False,'outcomes_used_or_analyzed':False,
        'elapsed_s':time.monotonic()-started,'limits':['Counts come from the original stored observation time, including the earlier death snapshot for synthetic no-buy nodes.',
            'Exact death age and event recency are excluded. Buyer KDA is exported for parity only and must be zeroed in an incremental feature experiment.',
            'Availability/coverage do not demonstrate a purchase-accuracy gain.']}
    write(output/'manifest.json',manifest);print(json.dumps({'state':'complete','states':n,'games':4001,'elapsed_s':manifest['elapsed_s']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output.resolve())
