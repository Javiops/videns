"""Matched public-scoreboard arms and target-independent categorical inference."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from winprob.iql_models import PublicStateEncoder
from winprob.iql_training import TensorStore,file_hash
from winprob.scoreboard_features_v1 import ScoreboardArtifact,normalized_scores,permutation_for_state,shuffle_within_team

ARMS=('zero','scoreboard','shuffled')
SEED=20261001
EPOCHS=20
BATCH=256
FEATURE_SHA='c3a0f735523efac4eb040ca138f720196cb8a323194f3a29666119d9173ae786'


def require(ok,message):
    if not ok:raise ValueError(message)


class ScoreCache:
    def __init__(self,path,dataset,guard=lambda:None):
        path=Path(path).resolve(strict=True);require(file_hash(path)==FEATURE_SHA,'fixed prepared scoreboard changed')
        require(path.with_suffix('.sha256').read_text().split()==[FEATURE_SHA,path.name],'scoreboard manifest sidecar changed')
        m=json.loads(path.read_text());require(m['schema']=='scoreboard-feature-cache-v1' and m['state']=='complete' and m['states']==dataset.n==392448 and m['games']==4001,'wrong scoreboard cache')
        for key in ('focal_channel_zeroed','independent_focal_parity_passed','all_episode_participant_ids_verified'):require(m[key] is True,'scoreboard gate failed')
        for key in ('training_performed','purchase_target_arrays_opened','outcome_arrays_opened','raw_sources_opened','source_runtime_imported','test_payload_reads','confirmation_payload_reads'):require(m[key] is False,'scoreboard boundary failed')
        source=ScoreboardArtifact(m['source_manifest_path']);require(source.digest==m['source_manifest_sha256'],'source scoreboard lineage changed')
        require(source.games==dataset.games and source.episodes==dataset.episode_metadata,'scoreboard/base game/episode order differs')
        self.bindings={**source.bindings,str(path):FEATURE_SHA}
        for raw,want in {**m['bindings_sha256'],**m['code_sha256']}.items():
            require(file_hash(raw)==want,'scoreboard source/code changed');self.bindings[raw]=want
        for name,want in m['files'].items():
            member=(path.parent/name).resolve(strict=True);require(member.is_relative_to(path.parent) and file_hash(member)==want,'scoreboard array changed');self.bindings[str(member)]=want
        self.scores=np.load(path.parent/'scores.npy',mmap_mode='r',allow_pickle=False)
        self.permutations=np.load(path.parent/'permutations.npy',mmap_mode='r',allow_pickle=False)
        require(self.scores.dtype==np.float32 and self.scores.shape==(392448,10,3) and self.permutations.dtype==np.uint8 and self.permutations.shape==(392448,10),'scoreboard feature schema changed')
        for lo in range(0,dataset.n,8192):
            guard();hi=min(lo+8192,dataset.n)
            require(np.array_equal(self.scores[lo:hi],normalized_scores(source.scores[lo:hi])),'prepared scores differ from original counters')
            shuffle_within_team(self.scores[lo:hi],self.permutations[lo:hi])
        for epno,episode in enumerate(source.episodes):
            if epno%1000==0:guard()
            for row in range(episode['first_state'],episode['first_state']+episode['states']):
                expected=permutation_for_state(episode['match_id'],int(episode['participant_id']),int(source.events[row]))
                require(np.array_equal(expected,self.permutations[row]),'control permutation changed')
        self.manifest=m;self.validated=True


@dataclass
class ScoreBatch:
    public:object
    real:torch.Tensor
    shuffled:torch.Tensor

    @property
    def batch_size(self):return self.public.batch_size


class ScoreStore:
    def __init__(self,dataset,cache,device,guard):
        require(cache.validated,'validated scoreboard cache required')
        self.base=TensorStore(dataset,device,guard);self.device=torch.device(device)
        self.real=torch.tensor(np.asarray(cache.scores),dtype=torch.float32,device=device)
        self.shuffled=torch.empty_like(self.real)
        for lo in range(0,dataset.n,8192):
            guard();hi=min(lo+8192,dataset.n)
            x=shuffle_within_team(cache.scores[lo:hi],cache.permutations[lo:hi])
            self.shuffled[lo:hi].copy_(torch.from_numpy(x).to(device))
        torch.cuda.synchronize()
        self.preloaded=self.base.preloaded

    def states(self,rows):
        index=torch.as_tensor(np.asarray(rows),dtype=torch.long,device=self.device)
        return ScoreBatch(self.base.states(rows),self.real[index],self.shuffled[index])


class ScoreEncoder(nn.Module):
    def __init__(self,base,arm):
        super().__init__();self.base=base;self.arm=arm
        self.score_projection=nn.Linear(3,64,bias=False);nn.init.zeros_(self.score_projection.weight)
        self.check_scores=True

    def forward(self,batch):
        if batch.real.shape!=(batch.batch_size,10,3) or batch.shuffled.shape!=batch.real.shape:
            raise ValueError('complete aligned scoreboard batch required')
        if self.check_scores:
            for x in (batch.real,batch.shuffled):
                if x.dtype!=torch.float32 or not bool(torch.isfinite(x).all()) or bool((x[:,0]!=0).any()):raise ValueError('finite float32 scoreboard with masked focal slot required')
        scores=torch.zeros_like(batch.real) if self.arm=='zero' else batch.real if self.arm=='scoreboard' else batch.shuffled
        # Keep the frozen own encoder implementation intact. This local hook
        # adds the declared linear score term before its first player ReLU.
        handle=self.base.player[0].register_forward_hook(lambda module,args,value:value+self.score_projection(scores))
        try:return self.base(batch.public)
        finally:handle.remove()


class ScoreActor(nn.Module):
    def __init__(self,encoder,head,arm):
        super().__init__();require(arm in ARMS,'unknown fixed arm')
        self.state=ScoreEncoder(encoder,arm);self.head=head;self.arm=arm

    def logits(self,board):return self.head(self.state(board))
    def loss(self,board,labels):return nn.functional.cross_entropy(self.logits(board),labels)
    def predict(self,board):return self.logits(board)


def initial_models(num_champions,num_items,group_count,fold):
    require(fold in (0,1),'fixed two-fold design')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED+fold)
        encoder=PublicStateEncoder(num_champions,num_items)
        head=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,group_count))
        models={arm:ScoreActor(deepcopy(encoder),deepcopy(head),arm) for arm in ARMS}
    return models


def enable_fastpath(model,dataset,score_cache):
    require(dataset.validated and score_cache.validated,'trusted fast path requires both validated inputs')
    from winprob.iql_models import set_input_content_validation
    set_input_content_validation(model,False);model.state.check_scores=False


def epoch_order(indices,fold,epoch):
    seed=int.from_bytes(hashlib.sha256(f'scoreboard-probe-v1/order/{fold}/{epoch}'.encode()).digest()[:8],'big')
    return np.random.default_rng(seed).permutation(np.asarray(indices,dtype=np.int64))


def evaluate(labels,scores,patches,sizes,kinds):
    y=np.asarray(labels);n=len(y)
    require(set(scores)=={*ARMS,'memory','old_state'},'all fixed arms/references required')
    draws=np.random.default_rng(SEED).integers(0,n,size=(2000,n));values={};arms={}
    for arm,s in scores.items():
        require(np.shape(s['top1'])==(n,) and np.shape(s['top3'])==(n,3) and np.isfinite(s['target_log_probability']).all(),'complete compact predictions required')
        v={'top1':(s['top1']==y).astype(float),'top3':(s['top3']==y[:,None]).any(1).astype(float),'nll':-s['target_log_probability']};values[arm]=v
        arms[arm]={key:{'mean':float(x.mean()),'interval95':np.quantile(x[draws].mean(1),[.025,.975]).tolist()} for key,x in v.items()}
    paired={}
    for control in ('zero','memory','shuffled','old_state'):
        paired[control]={}
        for key in values[control]:
            d=values['scoreboard'][key]-values[control][key]
            paired[control][key]={'difference':float(d.mean()),'interval95':np.quantile(d[draws].mean(1),[.025,.975]).tolist()}
        d=values['scoreboard']['top1']-values[control]['top1'];paired[control].update(corrected=int(np.sum(d>0)),broken=int(np.sum(d<0)))
    mask=y!=0;delta=values['scoreboard']['top1']-values['zero']['top1']
    require(mask.sum()>=30 and np.all(mask[draws].sum(1)>0),'nonempty-purchase guard has insufficient games')
    guard={'games':int(mask.sum()),'difference':float(delta[mask].mean()),
        'interval95':np.quantile((delta*mask)[draws].sum(1)/mask[draws].sum(1),[.025,.975]).tolist()}
    gate=(all(paired[c]['top1']['difference']>=.03 and paired[c]['top1']['interval95'][0]>0 for c in ('zero','memory'))
          and paired['memory']['top3']['difference']>=0 and guard['interval95'][0]>0)
    masks={**{'patch_'+str(p):np.asarray(patches)==p for p in sorted(set(patches))},
           **{'size_'+str(s):np.asarray(sizes)==s for s in sorted(set(map(int,sizes)))},
           **{'kind_'+str(k):np.asarray(kinds)==k for k in sorted(set(kinds))}}
    slices={}
    for name,m in masks.items():
        slices[name]={'games':int(m.sum()),'status':'descriptive' if m.sum()>=30 else 'too_small_for_rates'}
        if m.sum()>=30:slices[name]['arms']={a:{k:float(v[m].mean()) for k,v in types.items()} for a,types in values.items()}
    return {'games':n,'arms':arms,'scoreboard_minus_control':paired,'nonempty_purchase_guard':guard,'metric_gate_passed':bool(gate),
            'descriptive_slices':slices,'bootstrap':{'draws':2000,'seed':SEED,'unit':'whole game; same draws with ratio denominators for guarded slice'}}
