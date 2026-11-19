"""Matched encoder controls, canonical-order loss and exact catalogue prediction."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import numpy as np
import torch
from torch import nn
from winprob.iql_models import PublicStateEncoder
from winprob.multiset_likelihood_v1 import (build_lattice,build_catalogue_lattice,submultisets,
    MultisetTransitionHead,log_bag_probability,catalogue_log_probabilities)

ARMS=('categorical','marginal','canonical')
SEED=20261001
EPOCHS=20
BATCH=256


@dataclass
class PurchaseTables:
    local:object
    catalogue:object
    path_nodes:torch.Tensor
    path_tokens:torch.Tensor
    path_mask:torch.Tensor
    vocabulary:tuple

    def to(self,device):
        return PurchaseTables(self.local.to(device),self.catalogue.to(device),self.path_nodes.to(device),
                              self.path_tokens.to(device),self.path_mask.to(device),self.vocabulary)


def purchase_tables(bags,maximum=6):
    vocabulary=tuple(sorted({item for bag in bags for item,count in bag}));token={item:i+1 for i,item in enumerate(vocabulary)}
    local=build_lattice(bags,vocabulary,maximum);shared=build_catalogue_lattice(bags,vocabulary,maximum)
    nodes=np.zeros((len(bags),maximum),dtype=np.int64);tokens=np.zeros_like(nodes);mask=np.zeros_like(nodes,dtype=bool)
    for row,entries in enumerate(bags):
        bag=tuple(sorted(map(tuple,entries)));lookup={value:i for i,value in enumerate(submultisets(bag))}
        prefix={};column=0
        for item,count in bag:
            for copy in range(count):
                nodes[row,column]=lookup[tuple(sorted(prefix.items()))];tokens[row,column]=token[item];mask[row,column]=True
                prefix[item]=prefix.get(item,0)+1;column+=1
    return PurchaseTables(local,shared,*(torch.from_numpy(x) for x in (nodes,tokens,mask)),vocabulary)


def canonical_log_probability(logp,local,path_nodes,path_tokens,path_mask):
    row=torch.arange(len(logp),device=logp.device)
    terms=logp[row[:,None],path_nodes,path_tokens]
    total=torch.where(path_mask,terms,torch.zeros_like(terms)).sum(1)
    stop=logp[row,local.targets,0]
    return total+torch.where(local.sizes[row,local.targets]==local.maximum,torch.zeros_like(stop),stop)


class PurchaseActor(nn.Module):
    def __init__(self,encoder,head,arm):
        super().__init__();self.state=encoder;self.head=head;self.arm=arm
        if arm not in ARMS:raise ValueError('unknown fixed arm')

    def loss(self,board,labels,tables):
        context=self.state(board)
        if self.arm=='categorical':return nn.functional.cross_entropy(self.head(context),labels)
        local=tables.local.select(labels);logp=self.head(context,local)
        if self.arm=='marginal':return -log_bag_probability(logp,local).mean()
        return -canonical_log_probability(logp,local,tables.path_nodes[labels],tables.path_tokens[labels],tables.path_mask[labels]).mean()

    def predict(self,board,tables):
        """No true label parameter. Returns raw probability mass of ALL known bags."""
        context=self.state(board)
        if self.arm=='categorical':return torch.log_softmax(self.head(context),dim=-1)
        return catalogue_log_probabilities(self.head,context,tables.catalogue,chunk_nodes=1024)


def initial_models(num_champions,num_items,vocabulary_size,group_count,fold):
    if fold not in (0,1):raise ValueError('fixed two-fold design')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED+fold)
        encoder=PublicStateEncoder(num_champions,num_items)
        categorical=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,group_count))
        generative=MultisetTransitionHead(vocabulary_size)
    return {'categorical':PurchaseActor(deepcopy(encoder),categorical,'categorical'),
            'marginal':PurchaseActor(deepcopy(encoder),deepcopy(generative),'marginal'),
            'canonical':PurchaseActor(deepcopy(encoder),deepcopy(generative),'canonical')}


def epoch_order(indices,fold,epoch):
    seed=int.from_bytes(hashlib.sha256(f'multiset-probe-v1/order/{fold}/{epoch}'.encode()).digest()[:8],'big')
    return np.random.default_rng(seed).permutation(np.asarray(indices,dtype=np.int64))


def compact_raw(log_masses,labels):
    values=np.asarray(log_masses,dtype=np.float64);labels=np.asarray(labels)
    if values.ndim!=2 or labels.shape!=(len(values),) or not np.isfinite(values).all():raise ValueError('finite aligned catalogue masses required')
    maximum=values.max(1);logmass=maximum+np.log(np.exp(values-maximum[:,None]).sum(1))
    if np.any(logmass>np.log1p(1e-5)):raise ValueError('catalogue probability exceeds1')
    top3=np.argsort(-values,axis=1,kind='stable')[:,:3]
    raw=values[np.arange(len(values)),labels]
    return {'top1':top3[:,0],'top3':top3,'target_log_probability':raw-logmass,
            'raw_target_log_probability':raw,'catalogue_log_mass':logmass}


def summarize(labels,scores,sizes,repeats):
    labels=np.asarray(labels);n=len(labels)
    if set(scores)!={*ARMS,'memory','old_state'}:raise ValueError('all fixed comparisons required')
    draws=np.random.default_rng(SEED).integers(0,n,size=(2000,n));values={};arms={}
    for arm,s in scores.items():
        if np.shape(s['top1'])!=(n,) or np.shape(s['top3'])!=(n,3) or not np.isfinite(s['target_log_probability']).all():raise ValueError('incomplete predictions')
        v={'top1':(s['top1']==labels).astype(float),'top3':(s['top3']==labels[:,None]).any(1).astype(float),
           'conditional_nll':-np.asarray(s['target_log_probability'])};values[arm]=v
        arms[arm]={key:{'mean':float(x.mean()),'interval95':np.quantile(x[draws].mean(1),[.025,.975]).tolist()} for key,x in v.items()}
        if 'raw_target_log_probability' in s:
            arms[arm]['raw_nll']=float(-np.mean(s['raw_target_log_probability']))
            mass=np.exp(s['catalogue_log_mass']);arms[arm]['catalogue_mass']={'mean':float(mass.mean()),'minimum':float(mass.min()),'p05':float(np.quantile(mass,.05))}
    paired={}
    for other in ('categorical','memory','canonical','old_state'):
        paired[other]={}
        for key in values[other]:
            d=values['marginal'][key]-values[other][key]
            paired[other][key]={'difference':float(d.mean()),'interval95':np.quantile(d[draws].mean(1),[.025,.975]).tolist()}
        d=values['marginal']['top1']-values[other]['top1'];paired[other].update(corrected=int(np.sum(d>0)),broken=int(np.sum(d<0)))
    gate=all(paired[a]['top1']['difference']>=.03 and paired[a]['top1']['interval95'][0]>0 for a in ('categorical','memory')) and paired['memory']['top3']['difference']>=0
    masks={**{'size_'+str(s):np.asarray(sizes)==s for s in sorted(set(map(int,sizes)))},'repeated_copies':np.asarray(repeats,dtype=bool)}
    slices={name:{'games':int(mask.sum()),'arms':{arm:{key:float(v[mask].mean()) for key,v in value.items()} for arm,value in values.items()}} for name,mask in masks.items() if mask.any()}
    return {'games':n,'arms':arms,'marginal_minus_control':paired,'metric_gate_passed':bool(gate),'descriptive_slices':slices,'bootstrap':{'draws':2000,'seed':SEED,'unit':'game'}}
