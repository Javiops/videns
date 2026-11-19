"""Action-only cache validation and bounded stores for the history experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from winprob import admission
from winprob.full_action_catalogue import load_catalogue
from winprob.iql_dataset_contract import current_reservations
from winprob.iql_features import GLOBAL_NAMES,INVENTORY_NAMES
from winprob.iql_training import (Dataset as OldDataset,TensorStore,file_hash,fold_for_game,
    EXPECTED_DTYPES,refuse_quarantine)
from winprob.history_probe_models import HistoryBatch,set_history_validation

SEED=20260930
BATCH_SIZE=256
EPOCHS=20
ARMS=('state','history','shuffled')
BASE_CACHE_SHA256='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea'
BASE_ARRAYS=('champion_indices','inventory_static','visible_level','global_public','inventory_offsets',
    'inventory_item_indices','inventory_item_counts','game_index','episode_index','full_action_id',
    'main_group_id','patch_index','fold','timestamp_ms','action_static','action_static_valid','action_offsets',
    'action_item_indices','action_item_counts','full_to_group')
HISTORY_DTYPES={'prev_state_index':'int64','history_present':'bool','history_age':'float32',
    'donor_history_state_index':'int64','visit_end_ms':'int64','selected_state_indices':'int64','source_event_index':'int64'}


def bound_json(path):
    path=Path(path).resolve(strict=True);refuse_quarantine(path)
    sha=file_hash(path)
    if path.with_suffix('.sha256').read_text().split()!=[sha,path.name]:raise ValueError('manifest sidecar mismatch')
    return json.loads(path.read_text(encoding='utf-8')),sha


def safe_file(directory,name):
    result=(directory/name).resolve(strict=True)
    if not result.is_relative_to(directory):raise ValueError('file escaped artifact directory')
    return result


def validate_history_arrays(a,h,games,episodes):
    """Independent causal/fold checks without opening reward/terminal arrays."""
    n=len(a['game_index']);prev=h['prev_state_index'];present=h['history_present'];age=h['history_age']
    if (prev.shape!=(n,2) or present.shape!=(n,2) or age.shape!=(n,2)
            or h['visit_end_ms'].shape!=(n,) or h['donor_history_state_index'].shape!=(2,n)
            or not np.isfinite(age).all() or np.any(age<0)):
        raise ValueError('history shape/age contract failed')
    if np.any(h['visit_end_ms']<a['timestamp_ms']):raise ValueError('visit closes before it starts')
    end=0
    for ei,episode in enumerate(episodes):
        lo,count=episode['first_state'],episode['states'];hi=lo+count;gi=episode['game_index'];game=games[gi]
        if (lo!=end or count<1 or hi>n or episode['match_id']!=game['match_id']
                or episode['fold']!=game['fold'] or episode['patch']!=game['patch']
                or not np.all(a['episode_index'][lo:hi]==ei) or not np.all(a['game_index'][lo:hi]==gi)
                or not np.all(a['fold'][lo:hi]==game['fold'])
                or np.any(np.diff(a['timestamp_ms'][lo:hi])<=0)):
            raise ValueError('complete episode/fold alignment failed')
        end=hi
        indices=np.arange(lo,hi)
        purchases=indices[a['full_action_id'][lo:hi]!=0]
        preceding=np.searchsorted(purchases,indices,side='left')
        for slot in (0,1):
            position=preceding-slot-1;exists=position>=0
            old=(purchases[np.maximum(position,0)] if len(purchases) else np.full(count,lo,dtype=np.int64))
            safe=np.where(exists,old,lo)
            available=exists&(h['visit_end_ms'][safe]<a['timestamp_ms'][indices])
            expected=np.where(available,old,-1)
            if not np.array_equal(prev[indices,slot],expected) or not np.array_equal(present[indices,slot],available):
                raise ValueError('history must use previous real purchases, excluding SAVE before adjacency and masking closure ties')
            expected_age=np.where(available,(a['timestamp_ms'][indices]-h['visit_end_ms'][safe])/360000.,0.).astype(np.float32)
            if not np.array_equal(age[indices,slot],expected_age):raise ValueError('history age is not time since prior visit END')
    if end!=n:raise ValueError('episodes do not cover all cached states')
    for gi,game in enumerate(games):
        if game['fold']!=fold_for_game(game['match_id']):raise ValueError('whole-game fold changed')
    patterns=present[:,0].astype(np.uint8)+2*present[:,1].astype(np.uint8)
    for fold in (0,1):
        donor=h['donor_history_state_index'][fold]
        if np.any(donor<-1) or np.any(donor>=n):raise ValueError('donor index out of range')
        rows=np.flatnonzero(donor>=0);chosen=donor[rows]
        if (np.any(a['fold'][chosen]==fold) or np.any(a['game_index'][chosen]==a['game_index'][rows])
                or np.any(a['champion_indices'][chosen,0]!=a['champion_indices'][rows,0])
                or np.any(a['patch_index'][chosen]!=a['patch_index'][rows])
                or np.any(a['timestamp_ms'][chosen]//180000!=a['timestamp_ms'][rows]//180000)
                or np.any(patterns[chosen]!=patterns[rows])):
            raise ValueError('donor crosses fold/game or changes matching stratum')
    selected=h['selected_state_indices']
    if (selected.ndim!=1 or np.any(selected<0) or np.any(selected>=n)
            or len(np.unique(a['game_index'][selected]))!=len(selected)):
        raise ValueError('evaluation must contain one selected state per game')
    return {'states':n,'games':len(games),'episodes':len(episodes),'selected_games':len(selected),
        'history_present_slots':int(present.sum()),'folds':{str(f):{
            'train_transitions':int((a['fold']!=f).sum()),'train_games':sum(g['fold']!=f for g in games),
            'selected_games':int((a['fold'][selected]==f).sum()),
            'donor_missing_contexts':int((h['donor_history_state_index'][f]<0).sum())} for f in (0,1)}}


def verify_selection(entries,selected,a,games,episodes):
    by_player={(e['match_id'],e['participant_id']):e for e in episodes}
    by_game={g['match_id']:(i,g) for i,g in enumerate(games)}
    if len(entries)!=len(games) or {e['match_id'] for e in entries}!=set(by_game):raise ValueError('incomplete selection ledger')
    result=[]
    for entry in entries:
        gi,game=by_game[entry['match_id']]
        if 'game_index' in entry and entry['game_index']!=gi:raise ValueError('selection base game index changed')
        mid=game['match_id'];pid=1+int.from_bytes(hashlib.sha256(('history-probe-v1/focal/'+mid).encode()).digest()[:8],'big')%10
        if entry['match_id']!=mid or entry['participant_id']!=pid or entry['fold']!=game['fold']:
            raise ValueError('fixed focal selection identity changed')
        episode=by_player.get((mid,pid))
        if episode is None:
            if entry['status']!='no_eligible_focal_episode' or entry.get('selected_state_index') is not None:
                raise ValueError('missing focal episode was replaced')
            continue
        lo=episode['first_state']
        def rank(ordinal):
            ts=int(a['timestamp_ms'][lo+ordinal])
            return hashlib.sha256(f'history-probe-v1/visit/{mid}|{pid}|{ordinal}|{ts}'.encode()).digest(),ts,ordinal
        ordinal=min(range(episode['states']),key=rank);index=lo+ordinal
        if (entry['status']!='selected' or entry['selected_state_index']!=index
                or entry['source_node_ordinal']!=ordinal or entry['timestamp_ms']!=int(a['timestamp_ms'][index])):
            raise ValueError('preselected visit changed')
        result.append(index)
    if not np.array_equal(selected,np.asarray(result,dtype=np.int64)):raise ValueError('selected array/ledger order differs')


class HistoryDataset:
    """Duck-compatible with frozen TensorStore; old Dataset constructor is never used."""
    def __init__(self,directory,protocol_path):
        self.directory=Path(directory).resolve(strict=True)
        hp=self.directory/'manifest.json';manifest,sha=bound_json(hp)
        if (manifest.get('schema')!='history-probe-dataset-v1' or manifest.get('state')!='complete'
                or manifest.get('training_authorized') is not True or manifest.get('smoke') is not False
                or manifest.get('gates',{}).get('passed') is not True):raise ValueError('completed authorized full history cache required')
        for key in ('raw_sources_reopened','source_code_imported','test_payload_reads','confirmation_payload_reads',
                    'reward_arrays_opened','terminal_outcomes_used_or_analyzed','training_performed'):
            if manifest.get(key) is not False:raise ValueError('history source boundary failed: '+key)
        if (manifest.get('selection_precedes_current_action_access') is not True
                or manifest.get('gates',{}).get('strict_history_closure_verified') is not True
                or manifest.get('gates',{}).get('source_and_cache_action_parity_verified') is not True):
            raise ValueError('history selection/closure/action parity was not certified')
        bp=Path(manifest['cache_manifest_path']).resolve(strict=True);base,bsha=bound_json(bp)
        if bsha!=manifest['cache_manifest_sha256'] or bsha!=BASE_CACHE_SHA256:raise ValueError('frozen base cache changed')
        self.base_directory=bp.parent;self.base_manifest=base
        protocol=Path(protocol_path).resolve(strict=True)
        if Path(manifest['protocol_path']).resolve()!=protocol or file_hash(protocol)!=manifest['protocol_sha256']:
            raise ValueError('history protocol binding changed')
        self.admission_path=Path(manifest['admission_path']).resolve(strict=True)
        if file_hash(self.admission_path)!=manifest['admission_sha256']:raise ValueError('admission binding changed')
        if (self.admission_path!=Path(base['source_admission_path']).resolve()
                or manifest['admission_sha256']!=base['source_admission_sha256']):raise ValueError('history/base admission lineage differs')
        self.admission=json.loads(self.admission_path.read_text());admission.verify_manifest_sources(self.admission)
        forbidden=current_reservations(self.admission);admitted={r['match_id']:r for r in self.admission['games']}
        self.games=base['game_metadata']
        for game in self.games:
            entry=admitted.get(game['match_id'])
            if (entry is None or entry['role']!='train' or entry['historical_validation'] is not False
                    or entry['patch']!=game['patch'] or game['match_id'] in forbidden):raise ValueError('non-FIT/reserved game in history cache')
        self.bindings={str(hp):sha,str(hp.with_suffix('.sha256')):file_hash(hp.with_suffix('.sha256')),
            str(bp):bsha,str(bp.with_suffix('.sha256')):file_hash(bp.with_suffix('.sha256')),
            str(protocol):file_hash(protocol),str(self.admission_path):file_hash(self.admission_path)}
        self.arrays={}
        for name in BASE_ARRAYS:
            desc=base['arrays'][name];path=safe_file(bp.parent,desc['file']);expected=base['files'][desc['file']]
            if file_hash(path)!=expected:raise ValueError('base cache file changed: '+name)
            self.bindings[str(path)]=expected
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            if str(arr.dtype)!=EXPECTED_DTYPES[name] or list(arr.shape)!=desc['shape']:raise ValueError('base array schema mismatch')
            self.arrays[name]=arr
        self.n=int(base['state_count'])
        if manifest.get('state_count')!=self.n:raise ValueError('history/base state count mismatch')
        self.history={}
        if set(manifest['arrays'])!=set(HISTORY_DTYPES):raise ValueError('unexpected history arrays')
        for name,dtype in HISTORY_DTYPES.items():
            desc=manifest['arrays'][name];path=safe_file(self.directory,desc['file'])
            if file_hash(path)!=manifest['files'][desc['file']]:raise ValueError('history array hash mismatch')
            self.bindings[str(path)]=manifest['files'][desc['file']]
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            if str(arr.dtype)!=dtype or list(arr.shape)!=desc['shape']:raise ValueError('history array schema mismatch')
            self.history[name]=arr
        for key in ('feature_metadata_file','episode_metadata_file'):
            path=safe_file(bp.parent,base[key])
            if file_hash(path)!=base['files'][base[key]]:raise ValueError('base metadata hash mismatch')
            self.bindings[str(path)]=base['files'][base[key]]
            value=json.loads(path.read_text())
            if key=='feature_metadata_file':self.feature_metadata=value
            else:self.episode_metadata=value
        if (self.feature_metadata['global_names']!=list(GLOBAL_NAMES) or self.feature_metadata['inventory_names']!=list(INVENTORY_NAMES)
                or self.feature_metadata['padding_index']!=0 or self.feature_metadata['inventory_truncation'] is not False):
            raise ValueError('public feature schema changed')
        self.num_champions=len(self.feature_metadata['champion_index']);self.num_items=len(self.feature_metadata['item_index'])
        a=self.arrays
        for name in ('inventory_static','visible_level','global_public','action_static'):
            for lo in range(0,len(a[name]),16384):
                if not np.isfinite(a[name][lo:lo+16384]).all():raise ValueError('nonfinite cached feature')
        if np.any(a['champion_indices']<1) or np.any(a['champion_indices']>self.num_champions):raise ValueError('invalid champion index')
        for prefix,entries in (('inventory',self.n*10),('action',13166)):
            off=a[prefix+'_offsets'];ids=a[prefix+'_item_indices'];cnt=a[prefix+'_item_counts']
            if (off.shape!=(entries+1,) or off[0]!=0 or off[-1]!=len(ids) or ids.shape!=cnt.shape
                    or np.any(np.diff(off)<0) or np.any(ids<1) or np.any(ids>self.num_items) or np.any(cnt<1)):
                raise ValueError('invalid canonical ragged bags')
        self.catalogue,self.action_lookup=load_catalogue(base['catalogue_path'])
        self.bindings[str(Path(base['catalogue_path']).resolve())]=base['catalogue_sha256']
        if (np.any(a['full_action_id']<0) or np.any(a['full_action_id']>=13166) or np.any(a['patch_index']>2)
                or not np.array_equal(a['full_to_group'],self.catalogue['full_to_group'])
                or not np.array_equal(a['main_group_id'],a['full_to_group'][a['full_action_id']])
                or not a['action_static_valid'][a['patch_index'],a['full_action_id']].all()):raise ValueError('action/native static mapping failed')
        for name,expected in manifest['files'].items():
            path=safe_file(self.directory,name)
            if file_hash(path)!=expected:raise ValueError('history artifact file changed')
            self.bindings[str(path)]=expected
        for path,expected in manifest['bindings'].items():
            path=Path(path).resolve(strict=True);refuse_quarantine(path)
            if path.name in ('reward.npy','terminated.npy') or 'raw_match_v5' in path.parts:
                raise ValueError('history trainer refuses raw/outcome array bindings')
            if file_hash(path)!=expected:raise ValueError('history source binding changed')
            self.bindings[str(path)]=expected
        selection=json.loads(safe_file(self.directory,manifest.get('selection_file','selection.json')).read_text())
        self.selection=selection['entries'];self.selected=np.asarray(self.history['selected_state_indices'])
        self.audit=validate_history_arrays(a,self.history,self.games,self.episode_metadata)
        verify_selection(self.selection,self.selected,a,self.games,self.episode_metadata)
        if len(self.selected)<3000:raise ValueError('fixed minimum selected-game gate failed')
        self.inventory_max=int(np.max(np.diff(a['inventory_offsets'])))
        self.focal_champion_indices=np.array(a['champion_indices'][:,0],dtype=np.int32,copy=True)
        self.host_features_released=False;self.manifest=manifest;self.validated=True

    def release_host_features(self,*,synchronized_cuda_copy):
        return OldDataset.release_host_features(self,synchronized_cuda_copy=synchronized_cuda_copy)


class HistoryTensorStore:
    def __init__(self,dataset,device,guard):
        self.dataset=dataset;self.device=torch.device(device)
        self.base=TensorStore(dataset,device,guard)
        if not self.base.preloaded:raise RuntimeError('fixed history recipe requires validated resident CUDA states')
        a=dataset.arrays;h=dataset.history
        self.full_ids=torch.tensor(np.asarray(a['full_action_id']),dtype=torch.long,device=device)
        self.main_ids=torch.tensor(np.asarray(a['main_group_id']),dtype=torch.long,device=device)
        self.patches=torch.tensor(np.asarray(a['patch_index']),dtype=torch.long,device=device)
        self.previous=torch.tensor(np.asarray(h['prev_state_index']),dtype=torch.long,device=device)
        self.ages=torch.tensor(np.asarray(h['history_age']),dtype=torch.float32,device=device)
        self.presence=torch.tensor(np.asarray(h['history_present']),dtype=torch.bool,device=device)
        self.donors=torch.tensor(np.asarray(h['donor_history_state_index']),dtype=torch.long,device=device)
        torch.cuda.synchronize(device);guard()

    def batch(self,rows,fold,arm):
        if fold not in (0,1) or arm not in ARMS:raise ValueError('fixed fold and arm required')
        r=torch.as_tensor(np.asarray(rows),dtype=torch.long,device=self.device)
        state=self.base.states(rows);present=self.presence[r];ages=self.ages[r]
        previous=self.previous[r]
        if arm=='shuffled':
            donor=self.donors[fold,r];previous=self.previous[donor.clamp_min(0)]
            previous=torch.where((donor>=0)[:,None],previous,torch.full_like(previous,-1))
        content=(previous>=0)&present
        ids=torch.where(content,self.full_ids[previous.clamp_min(0)],torch.zeros_like(previous))
        patches=torch.where(content,self.patches[previous.clamp_min(0)],torch.zeros_like(previous))
        item=torch.where(content[:,:,None],self.base.action_indices[ids],torch.zeros_like(self.base.action_indices[ids]))
        counts=self.base.action_counts[ids]*content[:,:,None]
        static=self.base.action_static[patches,ids]*content[:,:,None]
        if arm=='state':
            item=torch.zeros_like(item);counts=torch.zeros_like(counts);static=torch.zeros_like(static)
            ages=torch.zeros_like(ages);present=torch.zeros_like(present)
        return state,HistoryBatch(item,counts,static,ages,present),self.main_ids[r]


def enable_validated_history_fastpath(actor,dataset):
    if (getattr(dataset,'validated',False) is not True or dataset.manifest.get('training_authorized') is not True
            or dataset.manifest.get('smoke') is not False):raise ValueError('trusted input requires the validated complete history cache')
    set_history_validation(actor,False)


def shared_epoch_order(indices,fold,epoch):
    seed=int.from_bytes(hashlib.sha256(f'history-probe-v1/order/{SEED}|{fold}|{epoch}'.encode()).digest()[:8],'big')
    return np.random.default_rng(seed).permutation(np.asarray(indices,dtype=np.int64))


def total_training_steps(fold):
    return EPOCHS*len(ARMS)*sum(int(np.ceil(np.count_nonzero(fold!=f)/BATCH_SIZE)) for f in (0,1))
