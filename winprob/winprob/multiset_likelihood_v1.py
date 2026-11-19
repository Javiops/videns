"""Exact differentiable likelihood of unordered, count-preserving action bags."""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
import numpy as np
import torch
from torch import nn


def canonical_bag(entries, vocabulary, maximum):
    pairs=list(entries)
    if any(len(p)!=2 or type(p[0]) is not int or type(p[1]) is not int or p[0] not in vocabulary or p[1]<=0 for p in pairs):
        raise ValueError('known item IDs and positive integer counts required')
    if len({p[0] for p in pairs})!=len(pairs) or sum(p[1] for p in pairs)>maximum:
        raise ValueError('duplicate item entries or oversized basket')
    return tuple(sorted((p[0],p[1]) for p in pairs))


def submultisets(bag):
    return sorted((tuple((bag[i][0],n) for i,n in enumerate(counts) if n)
                   for counts in itertools.product(*(range(n+1) for _,n in bag))),
                  key=lambda x:(sum(n for _,n in x),x))


@dataclass
class Lattice:
    items: torch.Tensor       # [...,nodes,support], token IDs1..V; padding0
    counts: torch.Tensor      # [...,nodes,support], exact multiplicities
    sizes: torch.Tensor       # [...,nodes], -1 means padding
    predecessors: torch.Tensor # [...,nodes,support], local predecessor node IDs
    tokens: torch.Tensor      # [...,nodes,support], item added on each edge
    edge_valid: torch.Tensor  # [...,nodes,support]
    targets: torch.Tensor     # [...], target endpoint node IDs
    maximum: int

    def to(self,device):
        return Lattice(**{name:(value.to(device) if isinstance(value,torch.Tensor) else value)
                           for name,value in vars(self).items()})

    def select(self,rows):
        if self.sizes.ndim!=2:raise ValueError('select only a batched target lattice')
        return Lattice(**{name:(value[rows] if isinstance(value,torch.Tensor) else value)
                           for name,value in vars(self).items()})


def build_lattice(bags,vocabulary,maximum=6):
    """One local, closed DAG for each target. Metadata only; no state/label I/O."""
    vocabulary=tuple(vocabulary)
    if len(set(vocabulary))!=len(vocabulary) or not vocabulary or maximum<1:
        raise ValueError('nonempty unique vocabulary and positive maximum required')
    token={item:i+1 for i,item in enumerate(vocabulary)}
    canonical=[canonical_bag(bag,token,maximum) for bag in bags]
    if not canonical:raise ValueError('nonempty target list required')
    lattices=[submultisets(bag) for bag in canonical]
    nodes=max(map(len,lattices));support=max(1,max(map(len,canonical)));b=len(canonical)
    items=np.zeros((b,nodes,support),dtype=np.int64);counts=np.zeros_like(items)
    sizes=np.full((b,nodes),-1,dtype=np.int64);pred=np.zeros_like(items);tokens=np.zeros_like(items);valid=np.zeros_like(items,dtype=bool)
    targets=np.zeros(b,dtype=np.int64)
    for row,(bag,lattice) in enumerate(zip(canonical,lattices)):
        lookup={node:i for i,node in enumerate(lattice)};targets[row]=lookup[bag]
        for i,node in enumerate(lattice):
            sizes[row,i]=sum(n for _,n in node)
            for j,(item,n) in enumerate(node):
                items[row,i,j]=token[item];counts[row,i,j]=n
                previous=tuple((other,c-int(other==item)) for other,c in node if c-int(other==item)>0)
                pred[row,i,j]=lookup[previous];tokens[row,i,j]=token[item];valid[row,i,j]=True
    return Lattice(*(torch.from_numpy(x) for x in (items,counts,sizes,pred,tokens,valid,targets)),maximum)


def log_prefix_probability(log_probabilities,lattice):
    """Forward probability of every prefix, before its STOP decision."""
    p=log_probabilities
    if (p.ndim!=3 or lattice.sizes.shape!=p.shape[:2] or lattice.predecessors.shape[:2]!=p.shape[:2]
            or p.shape[2]<2):
        raise ValueError('aligned batch/node/vocabulary dimensions required')
    b,n,v=p.shape
    if torch.any(lattice.tokens>=v):raise ValueError('edge token outside transition vocabulary')
    batch=torch.arange(b,device=p.device)[:,None,None]
    transitions=p[batch,lattice.predecessors,lattice.tokens]
    alpha=torch.full((b,n),-torch.inf,dtype=p.dtype,device=p.device)
    alpha=torch.where(lattice.sizes==0,torch.zeros_like(alpha),alpha)
    for depth in range(1,lattice.maximum+1):
        active=lattice.sizes==depth
        parents=alpha[batch,lattice.predecessors]
        terms=(parents+transitions).masked_fill(~lattice.edge_valid,-torch.inf)
        # Future/padded nodes would have only -inf parents. Replace them before
        # logsumexp, avoiding NaN derivatives from inactive all-impossible sums.
        safe=torch.where(active[:,:,None],terms,torch.zeros_like(terms))
        alpha=torch.where(active,torch.logsumexp(safe,dim=-1),alpha)
    return alpha


def log_bag_probability(log_probabilities,lattice):
    """Sum distinct orderings once. Accept arbitrary normalized transition tables."""
    p=log_probabilities;alpha=log_prefix_probability(p,lattice)
    if lattice.targets.shape!=(len(p),):raise ValueError('one target endpoint per training row required')
    row=torch.arange(len(p),device=p.device);end=lattice.targets
    stop=p[row,end,0]
    stop=torch.where(lattice.sizes[row,end]==lattice.maximum,torch.zeros_like(stop),stop)
    return alpha[row,end]+stop


class MultisetTransitionHead(nn.Module):
    """Prefix-conditioned distribution; STOP0, item tokens1..V, padding is masked."""
    def __init__(self,vocabulary_size,context_dim=128,item_dim=64,hidden=128,maximum=6):
        super().__init__();self.maximum=maximum;self.vocabulary_size=vocabulary_size;self.context_dim=context_dim
        self.items=nn.Embedding(vocabulary_size+1,item_dim,padding_idx=0)
        self.size=nn.Embedding(maximum+1,8)
        self.network=nn.Sequential(nn.Linear(context_dim+item_dim+8,hidden),nn.ReLU(),nn.Linear(hidden,vocabulary_size+1))

    def prefix_transitions(self,context,items,counts,sizes):
        if context.ndim!=2 or context.shape[1]!=self.context_dim or items.shape[0]!=len(context):
            raise ValueError('aligned context and local prefix batch required')
        prefix=(self.items(items)*counts[...,None].to(context.dtype)).sum(-2)
        size=self.size(sizes.clamp_min(0))
        state=context[:,None,:].expand(-1,prefix.shape[1],-1)
        logits=self.network(torch.cat((state,prefix,size),dim=-1))
        return torch.log_softmax(logits,dim=-1)

    def forward(self,context,lattice):
        return self.prefix_transitions(context,lattice.items,lattice.counts,lattice.sizes)

    def log_probability(self,context,lattice):
        return log_bag_probability(self(context,lattice),lattice)


@dataclass
class CatalogueLattice:
    graph:Lattice
    endpoints:torch.Tensor

    def to(self,device):return CatalogueLattice(self.graph.to(device),self.endpoints.to(device))


def build_catalogue_lattice(bags,vocabulary,maximum=6):
    """Share each prefix once across ALL catalogue targets; maintain group order."""
    token={item:i+1 for i,item in enumerate(vocabulary)}
    if not token or len(token)!=len(vocabulary):raise ValueError('unique vocabulary required')
    groups=[canonical_bag(bag,token,maximum) for bag in bags]
    if not groups or len(set(groups))!=len(groups):raise ValueError('unique catalogue bags required')
    nodes=sorted({node for bag in groups for node in submultisets(bag)},key=lambda x:(sum(n for _,n in x),x))
    lookup={node:i for i,node in enumerate(nodes)};support=max(1,max(map(len,nodes)))
    items=np.zeros((1,len(nodes),support),dtype=np.int64);counts=np.zeros_like(items);pred=np.zeros_like(items);tokens=np.zeros_like(items)
    valid=np.zeros_like(items,dtype=bool);sizes=np.zeros((1,len(nodes)),dtype=np.int64)
    for i,node in enumerate(nodes):
        sizes[0,i]=sum(n for _,n in node)
        for j,(item,n) in enumerate(node):
            items[0,i,j]=token[item];counts[0,i,j]=n;tokens[0,i,j]=token[item];valid[0,i,j]=True
            previous=tuple((other,c-int(other==item)) for other,c in node if c-int(other==item)>0)
            pred[0,i,j]=lookup[previous]
    lattice=Lattice(*(torch.from_numpy(x) for x in (items,counts,sizes,pred,tokens,valid,np.zeros(1,dtype=np.int64))),maximum)
    return CatalogueLattice(lattice,torch.tensor([lookup[bag] for bag in groups],dtype=torch.long))


def catalogue_log_probabilities(head,context,catalogue,chunk_nodes=1024):
    """Exact mass for each known bag; do not silently renormalize outside mass."""
    if type(chunk_nodes) is not int or chunk_nodes<1:raise ValueError('positive chunk bound required')
    graph=catalogue.graph;b=len(context)
    expanded=Lattice(**{name:(value.expand(b,*value.shape[1:]) if isinstance(value,torch.Tensor) else value)
                        for name,value in vars(graph).items()})
    parts=[]
    for lo in range(0,graph.sizes.shape[1],chunk_nodes):
        hi=lo+chunk_nodes
        parts.append(head.prefix_transitions(context,expanded.items[:,lo:hi],expanded.counts[:,lo:hi],expanded.sizes[:,lo:hi]))
    logp=torch.cat(parts,dim=1);alpha=log_prefix_probability(logp,expanded)
    end=catalogue.endpoints;stop=logp[:,end,0]
    stop=torch.where(expanded.sizes[:,end]==graph.maximum,torch.zeros_like(stop),stop)
    return alpha[:,end]+stop


def catalogue_facts(bags):
    nodes=[submultisets(tuple(map(tuple,bag))) for bag in bags]
    return {'groups':len(bags),'unique_prefixes':len({n for lattice in nodes for n in lattice}),
            'maximum_local_nodes':max(map(len,nodes)),'local_nodes_total':sum(map(len,nodes)),
            'local_edges_total':sum(len(node) for lattice in nodes for node in lattice),
            'maximum_distinct_orders':max(math.factorial(sum(n for _,n in bag))//math.prod(math.factorial(n) for _,n in bag) for bag in bags)}
