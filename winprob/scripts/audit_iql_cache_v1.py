"""Read-only NumPy audit of the completed IQL cache against all FIT episodes."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from threadpoolctl import threadpool_limits
from winprob.iql_dataset_contract import validate_sources,iter_episodes
from winprob.iql_features import PublicFeatureCompiler
from boots_support import load_static
from setback_experiment import set_below_normal_priority

PATCHES=('16.17','16.18','16.19')
STATE_TAILS={'champion_indices':('int32',(10,)),'inventory_static':('float32',(10,22)),
             'visible_level':('float32',(10,1)),'global_public':('float32',(37,)),
             'game_index':('int32',()),'episode_index':('int32',()),'full_action_id':('int32',()),
             'main_group_id':('int32',()),'patch_index':('uint8',()),'fold':('uint8',()),
             'next_state_index':('int64',()),'reward':('float32',()),'terminated':('bool',()),
             'timestamp_ms':('int64',()),'elapsed_ms':('int64',())}


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def sidecar(path):
    value=sha(path)
    require(path.with_suffix('.sha256').read_text().split()==[value,path.name],'Sidecar mismatch: '+str(path))
    return value
def fold_of(mid):return int.from_bytes(hashlib.sha256(('iql-v1-fold:'+mid).encode()).digest()[:8],'big')%2


def run(cache,output,deadline=150):
    started=time.monotonic();cache=Path(cache).resolve(strict=True);output=Path(output).resolve()
    require(not output.exists(),'Immutable audit output exists')
    output.mkdir(parents=True,exist_ok=False);set_below_normal_priority()
    def check():require(time.monotonic()-started<=deadline,'Audit deadline reached before complete verification')
    def progress(stage,**extra):
        value={'stage':stage,'elapsed_s':time.monotonic()-started,**extra}
        (output/'status.json').write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps(value),flush=True)
    try:
        progress('hashing_completed_artifacts')
        manifest_path=cache/'manifest.json';manifest_hash=sidecar(manifest_path);m=read(manifest_path)
        require(m.get('schema')=='iql-dataset-v1' and m.get('state')=='complete' and m.get('smoke') is False
                and m.get('training_authorized') is True,'Not a complete production IQL cache')
        require(all(m.get(key) is False for key in ('test_payload_reads','confirmation_payload_reads',
                                                   'development_terminal_targets_read','raw_sources_reopened')),'Cache scope mismatch')
        for name,expected in m['files'].items():
            path=(cache/name).resolve(strict=True)
            require(path.is_relative_to(cache) and sha(path)==expected,'Cache file digest mismatch: '+name)
        for path,expected in m['bindings'].items():
            require('raw_match_v5' not in Path(path).parts and Path(path).suffix.lower()!='.db','Audit refused raw/database binding')
            require(sha(path)==expected,'Frozen metadata/code binding mismatch: '+path)
        bundle=validate_sources(m['source_directories'],m['source_admission_path'],m['catalogue_path'])
        require(bundle['catalogue_sha256']==m['catalogue_sha256'],'Catalogue identity differs')
        static,static_bindings=load_static(Path(m['static_export_path']))
        require(sha(m['static_export_path'])==m['static_export_sha256'],'Static export hash mismatch')
        compiler=PublicFeatureCompiler(static)
        feature_meta=read(cache/m['feature_metadata_file'])
        require(json.loads(json.dumps(compiler.metadata()))==feature_meta,'Public feature schema/vocabulary differs')
        require(m['patches']==list(PATCHES),'Native static patch order changed')
        arrays={}
        for name,spec in m['arrays'].items():
            require(spec['file'] in m['files'],'Array lacks a file binding')
            value=np.load(cache/spec['file'],mmap_mode='r',allow_pickle=False)
            require(value.dtype==np.dtype(spec['dtype']) and value.shape==tuple(spec['shape']),'Array shape/dtype differs: '+name)
            arrays[name]=value
        n=m['state_count'];k=13166
        for name,(dtype,tail) in STATE_TAILS.items():
            require(arrays[name].dtype==np.dtype(dtype) and arrays[name].shape==(n,*tail),'State schema mismatch: '+name)
        require(set(arrays)==set(STATE_TAILS)|{'inventory_offsets','inventory_item_indices','inventory_item_counts',
            'action_static','action_static_valid','action_offsets','action_item_indices','action_item_counts','full_to_group','fold_action_counts'},
            'Unexpected cache array inventory')
        for name in ('inventory_static','visible_level','global_public','reward'):
            for start in range(0,n,4096):require(np.isfinite(arrays[name][start:start+4096]).all(),'Nonfinite state feature/reward')
        require(np.array_equal(arrays['full_to_group'],np.asarray(bundle['catalogue']['full_to_group'],dtype=np.int32)),
                'Full-action→main-group map differs')
        ids=arrays['full_action_id'];require(bool(((ids>=0)&(ids<k)).all()),'Invalid cached full action ID')
        require(np.array_equal(arrays['main_group_id'],arrays['full_to_group'][ids]),'Cached main-group IDs are not the true full-action projection')
        require(set(np.unique(arrays['patch_index']))<={0,1},'Non-FIT patch entered state cache')
        require(arrays['action_static_valid'][arrays['patch_index'],ids].all(),'Observed action has no native static encoding')
        offsets=arrays['inventory_offsets'];items=arrays['inventory_item_indices'];counts=arrays['inventory_item_counts']
        require(offsets.dtype==np.int64 and offsets.shape==(n*10+1,) and offsets[0]==0 and offsets[-1]==len(items)==len(counts),
                'Ragged inventory axes/endpoints mismatch')
        require(np.all(offsets[1:]>=offsets[:-1]) and np.all(counts>0) and np.all(items>0)
                and int(items.max())<=len(compiler.item_index),'Invalid inventory ragged bag')
        require(int(np.max(offsets[1:]-offsets[:-1]))==m['inventory_unique_items_max']
                and len(items)==m['inventory_ragged_tokens'],'Inventory token/capacity counts mismatch')
        check();progress('checking_native_static_action_tables')
        action_offsets=arrays['action_offsets'];action_items=arrays['action_item_indices'];action_counts=arrays['action_item_counts']
        require(action_offsets.shape==(k+1,) and action_offsets[0]==0 and action_offsets[-1]==len(action_items)==len(action_counts),
                'Full-action bag axes mismatch')
        native_valid_counts={patch:0 for patch in PATCHES}
        for aid,(basket,kind) in enumerate(zip(bundle['catalogue']['full_actions'],bundle['catalogue']['full_action_kinds'],strict=True)):
            lo,hi=map(int,action_offsets[aid:aid+2])
            expected_bag=[(compiler.item_index[item],count) for item,count in basket]
            require(np.array_equal(action_items[lo:hi],np.asarray([item for item,count in expected_bag],dtype=np.int32))
                    and np.array_equal(action_counts[lo:hi],np.asarray([count for item,count in expected_bag],dtype=np.int32)),
                    'Full-action bag lost item/copy identity')
            for pi,patch in enumerate(PATCHES):
                try:_,feature=compiler.full_action(basket,kind,patch)
                except (KeyError,ValueError):
                    require(not arrays['action_static_valid'][pi,aid] and not arrays['action_static'][pi,aid].any(),
                            'Unsupported native action carries fabricated features')
                else:
                    require(bool(arrays['action_static_valid'][pi,aid]) and np.array_equal(feature,arrays['action_static'][pi,aid]),
                            'Native action attributes differ from bound compiler/static source')
                    native_valid_counts[patch]+=1
        episodes=read(cache/m['episode_metadata_file']);games=m['game_metadata']
        require(len(episodes)==m['episode_count'] and len(games)==m['game_count'],'Metadata counts differ')
        mids=[row['match_id'] for row in games]
        require(len(set(mids))==len(mids) and set(mids)==set(bundle['eligible_fit_ids']),'Cache game IDs differ from allowed eligible FIT IDs')
        require(not m['encoding_exclusions'] and (cache/'encoding_exclusions.jsonl').stat().st_size==0,'This audit requires the declared zero feature exclusions')
        fold_game_counts=Counter();fold_transitions=np.zeros(2,dtype=np.int64);fold_terminal=np.zeros((2,2),dtype=np.int64)
        by_action=np.zeros((2,k),dtype=np.int64);cursor=0;episode_ids=set()
        for game in games:
            require(game['fold']==fold_of(game['match_id']),'Game-hash fold differs')
            fold_game_counts[game['fold']]+=1
        for ei,episode in enumerate(episodes):
            first,length=episode['first_state'],episode['states'];end=first+length;gi=episode['game_index'];fold=episode['fold']
            require(type(first) is int and type(length) is int and length>0 and first==cursor and end<=n,'Episode ranges do not partition cache')
            require(episode['trajectory_id'] not in episode_ids,'Duplicate cached trajectory')
            episode_ids.add(episode['trajectory_id'])
            require(games[gi]['match_id']==episode['match_id'] and games[gi]['patch']==episode['patch']
                    and fold==games[gi]['fold']==fold_of(episode['match_id']),'Episode game/patch/fold changed')
            sl=slice(first,end)
            require(np.all(arrays['episode_index'][sl]==ei) and np.all(arrays['game_index'][sl]==gi)
                    and np.all(arrays['fold'][sl]==fold) and np.all(arrays['patch_index'][sl]==PATCHES.index(episode['patch'])),
                    'Transition graph crosses episode/game/fold/native patch')
            nxt=np.arange(first+1,end+1,dtype=np.int64);nxt[-1]=-1
            require(np.array_equal(arrays['next_state_index'][sl],nxt),'Missing/skipped/cross-episode next-state edge')
            require(not arrays['terminated'][first:end-1].any() and bool(arrays['terminated'][end-1]),'Absorbing terminal placement mismatch')
            require(not arrays['reward'][first:end-1].any() and arrays['reward'][end-1] in (0,1),'Reward paid before the terminal transition')
            require(np.all(arrays['elapsed_ms'][sl]>0) and np.array_equal(np.diff(arrays['timestamp_ms'][sl]),arrays['elapsed_ms'][first:end-1]),
                    'Transition elapsed time mismatches observed state chronology')
            fold_transitions[fold]+=length;fold_terminal[fold,int(arrays['reward'][end-1])]+=1
            by_action[fold]+=np.bincount(ids[sl],minlength=k);cursor=end
        require(cursor==n and int(arrays['terminated'].sum())==len(episodes),'Graph does not cover every transition exactly once')
        require(np.array_equal(by_action,arrays['fold_action_counts']),'Fold action counts mismatch')
        for fold in (0,1):
            g=m['gates']['folds'][str(fold)];supported=int(by_action[fold][by_action[1-fold]>0].sum())
            require(g['games']==fold_game_counts[fold] and g['transitions']==int(fold_transitions[fold])
                    and g['terminal_classes']==fold_terminal[fold].tolist()
                    and g['opposite_fold_supported_transitions']==supported
                    and abs(g['opposite_fold_action_coverage']-supported/fold_transitions[fold])<1e-12,'Recomputed fold support gate differs')
        denominator=sum(bundle['source_complete_player_games'].values())
        require(denominator==m['gates']['otherwise_source_complete_player_games']
                and abs(len(episodes)/denominator-m['gates']['complete_chain_survival'])<1e-12,'Source survival denominator differs')
        check();progress('streaming_all_source_episodes',episodes=len(episodes),transitions=n)
        sampled=Counter();sample_states=Counter();sample_ids=[];ordinal=0;checked_transitions=0;mutation_checks=0
        with threadpool_limits(limits=1):
            for source_index,record,compiled in iter_episodes(bundle):
                episode=episodes[ordinal];first=episode['first_state'];length=len(compiled);sl=slice(first,first+length)
                require(episode['trajectory_id']==record['trajectory_id'] and episode['source_index']==source_index
                        and episode['states']==length and episode['participant_id']==record['participant_id'],
                        'Cache/source episode ordering, identity or length differs')
                expected={'full_action_id':[row['full_action_id'] for row in compiled],
                          'main_group_id':[row['main_group_id'] for row in compiled],
                          'reward':[row['reward'] for row in compiled],
                          'terminated':[row['terminated'] for row in compiled],
                          'elapsed_ms':[row['elapsed_ms'] for row in compiled],
                          'timestamp_ms':[row['timestamp_ms'] for row in record['decisions']],
                          'next_state_index':[-1 if row['terminated'] else first+j+1 for j,row in enumerate(compiled)]}
                for key,value in expected.items():
                    require(np.array_equal(arrays[key][sl],np.asarray(value,dtype=arrays[key].dtype)),
                            'Cache differs from source transition field: '+key)
                patch=record['patch']
                if sampled[patch]<16:
                    sampled[patch]+=1;sample_ids.append(record['trajectory_id'])
                    for j,node in enumerate(record['decisions']):
                        feature=compiler.state(node['prestate'],patch);row=first+j
                        for key in ('champion_indices','inventory_static','visible_level','global_public'):
                            require(np.array_equal(feature[key],arrays[key][row]),'Fixed-sample feature parity failed: '+key)
                        for player,bag in enumerate(feature['inventory_bags']):
                            lo,hi=map(int,offsets[row*10+player:row*10+player+2])
                            require(np.array_equal(items[lo:hi],np.asarray([item for item,count in bag],dtype=np.int32))
                                    and np.array_equal(counts[lo:hi],np.asarray([count for item,count in bag],dtype=np.int32)),
                                    'Fixed-sample inventory multiplicity/order parity failed')
                        if j==0:
                            altered=deepcopy(node['prestate']);altered.update(role='MUTATED',team_position='MUTATED',win=True,
                                source_target_id=99999999,total_gold=10**9,game_duration=10**9,bot_quest_complete=True)
                            for other in altered['others']:other.update(role='MUTATED',team_position='MUTATED',gold=10**9,win=True)
                            changed=compiler.state(altered,patch)
                            require(all(np.array_equal(feature[key],changed[key]) for key in ('champion_indices','inventory_static','visible_level','global_public'))
                                    and feature['inventory_bags']==changed['inventory_bags'],'Ignored role/future/privileged fields influenced compiler')
                            mutation_checks+=1
                        sample_states[patch]+=1
                ordinal+=1;checked_transitions+=length
                if ordinal%1000==0:check();progress('streaming_all_source_episodes',checked_episodes=ordinal,checked_transitions=checked_transitions)
        require(bundle['verified_complete'] is True and ordinal==len(episodes) and checked_transitions==n,'Source stream did not exhaust exactly')
        require('torch' not in sys.modules,'Audit unexpectedly imported Torch')
        check()
        for name,expected in m['files'].items():require(sha(cache/name)==expected,'Cache file changed during audit: '+name)
        for path,expected in m['bindings'].items():require(sha(path)==expected,'Bound input changed during audit')
        require(sidecar(manifest_path)==manifest_hash,'Cache manifest changed during audit')
        report={'schema':'iql-cache-independent-audit-v1','state':'complete','passed':True,
                'audited_utc':datetime.now(timezone.utc).isoformat(),'cache_directory':str(cache),'cache_manifest_sha256':manifest_hash,
                'auditor_sha256':sha(__file__),'cache_files_verified':len(m['files']),'manifest_bindings_verified':len(m['bindings']),
                'state_arrays_verified':len(arrays),'states':n,'episodes':len(episodes),'games':len(games),
                'all_source_episodes_compared':ordinal,'all_source_transitions_compared':checked_transitions,
                'fields_compared_exhaustively':list(expected_key for expected_key in ('full_action_id','main_group_id','reward','terminated','elapsed_ms','timestamp_ms','next_state_index')),
                'source_iterator_exhaustion_verified':True,'graph_cross_episode_game_fold_errors':0,
                'terminal_count':int(arrays['terminated'].sum()),'fold_games':dict(fold_game_counts),
                'fold_transitions':fold_transitions.tolist(),'fold_terminal_classes':fold_terminal.tolist(),
                'fold_opposite_action_coverage':[m['gates']['folds'][str(fold)]['opposite_fold_action_coverage'] for fold in (0,1)],
                'source_complete_player_games':denominator,'catalogue_and_feature_chain_survival':len(episodes)/denominator,
                'native_action_vectors_checked':3*k,'native_action_valid_counts':native_valid_counts,
                'feature_sample_rule':'First16 source episodes per native patch; every decision state in those episodes.',
                'feature_sample_episodes':dict(sampled),'feature_sample_states':dict(sample_states),'feature_sample_trajectory_ids':sample_ids,
                'ignored_field_mutation_checks':mutation_checks,'inventory_ragged_tokens':len(items),
                'inventory_max_unique_items_per_player':m['inventory_unique_items_max'],'feature_encoding_exclusions':0,
                'raw_sources_read':False,'sqlite_opened':False,'heldout_game_payloads_read':False,'torch_imported':False,'training_performed':False,
                'limitations':['Source episode and transition equality is exhaustive; raw Riot truth was not re-read.',
                               'Feature parity uses the already bound compiler on a fixed sample, not independent reimplementation for all states.',
                               'Native static action features were reproduced for every catalogue action and all three public patches.',
                               'This certifies cache integrity and declared population only, not IQL improvement or causal value.'],
                'elapsed_s':time.monotonic()-started}
        path=output/'audit.json';path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        path.with_suffix('.sha256').write_text(sha(path)+'  audit.json\n',encoding='ascii')
        progress('complete',report=str(path),passed=True)
        return report
    except Exception as error:
        progress('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'data/runs/iql-cache-20260930-v1/cache')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/audits/iql-cache-20260930-v1')
    parser.add_argument('--deadline-seconds',type=int,default=150)
    args=parser.parse_args();run(args.cache,args.output,args.deadline_seconds)


if __name__=='__main__':main()
