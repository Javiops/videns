"""Compositional current-action models with optional training-only goal supervision."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from winprob.iql_models import PublicStateEncoder,set_input_content_validation


@dataclass(frozen=True)
class BasketTables:
    main_item_indices:torch.Tensor    # [A,L], fixed public-vocabulary indices
    main_item_counts:torch.Tensor
    main_static:torch.Tensor          # [3,A,23], native-patch public attributes
    main_static_valid:torch.Tensor    # [3,A], invalid compositions contribute zero
    goal_item_indices:torch.Tensor    # [G,Lg], exact counted future-goal identities
    goal_item_counts:torch.Tensor
    goal_static:torch.Tensor          # [3,G,23]; EMPTY has all-zero inputs
    goal_static_valid:torch.Tensor


class FutureGoalActor(nn.Module):
    """A free class head plus shared composition; labels never enter forward.

    Missing native action metadata zeros only its composition, never its free
    class score or softmax mass. Observed action labels must be valid upstream.
    """
    def __init__(self,num_champions,num_items,tables):
        super().__init__()
        self.state=PublicStateEncoder(num_champions,num_items)
        self.basket=nn.Sequential(nn.Linear(55,128),nn.ReLU(),nn.Linear(128,128))
        self.class_count=int(tables.main_item_indices.shape[0])
        self.goal_count=int(tables.goal_item_indices.shape[0])
        if self.class_count<3 or self.goal_count<2:raise ValueError('fixed action and goal catalogues required')
        self.main_free=nn.Linear(128,self.class_count)
        self.goal_query=nn.Linear(128,128,bias=False)
        self.goal_bias=nn.Parameter(torch.zeros(self.goal_count))
        self.check_input_contents=True
        for name in tables.__dataclass_fields__:
            value=getattr(tables,name)
            if not isinstance(value,torch.Tensor):raise ValueError('explicit tensor basket tables required')
            self.register_buffer(name,value.detach().clone())
        for prefix,n in (('main',self.class_count),('goal',self.goal_count)):
            ids=getattr(self,prefix+'_item_indices');counts=getattr(self,prefix+'_item_counts')
            attrs=getattr(self,prefix+'_static');valid=getattr(self,prefix+'_static_valid')
            if (ids.ndim!=2 or ids.shape[0]!=n or ids.dtype!=torch.long or counts.shape!=ids.shape
                    or counts.dtype!=torch.float32 or attrs.shape!=(3,n,23) or attrs.dtype!=torch.float32
                    or valid.shape!=(3,n) or valid.dtype!=torch.bool or not bool(torch.isfinite(attrs).all())
                    or bool((ids<0).any()) or bool((ids>num_items).any())
                    or not bool(torch.isfinite(counts).all()) or not torch.equal(counts,counts.round())
                    or bool((counts<0).any()) or bool(((ids==0)&(counts!=0)).any())
                    or bool(((ids!=0)&(counts<=0)).any())):
                raise ValueError('invalid count-preserving native basket table')
        if (bool((self.main_item_indices[0]!=0).any()) or bool((self.main_item_counts[0]!=0).any())
                or bool((self.main_static[:,0,:22]!=0).any()) or not bool((self.main_static[:,0,22]==1).all())):
            raise ValueError('main group zero must retain its explicit SAVE coordinate')
        if (bool((self.goal_item_indices[0]!=0).any()) or bool((self.goal_item_counts[0]!=0).any())
                or bool((self.goal_static[:,0]!=0).any())):
            raise ValueError('future EMPTY must have zero items and all23 static inputs zero')
        if not bool(self.goal_static_valid.all()):raise ValueError('every fixed goal identity needs native metadata')

    def basket_vectors(self,kind,patch):
        if kind not in ('main','goal') or patch not in (0,1,2):raise ValueError('fixed basket world required')
        items=self.state.inventory(getattr(self,kind+'_item_indices'),getattr(self,kind+'_item_counts'))
        value=self.basket(torch.cat((items,getattr(self,kind+'_static')[patch]),dim=1))
        # This does NOT remove a class from the categorical main likelihood.
        return value.masked_fill(~getattr(self,kind+'_static_valid')[patch,:,None],0.)

    def forward(self,state,patch_index,*,compute_goal=False):
        if (patch_index.dtype!=torch.long or patch_index.shape!=(state.batch_size,)
                or type(compute_goal) is not bool):raise ValueError('aligned native patch indices required')
        if self.check_input_contents and bool(((patch_index<0)|(patch_index>2)).any()):
            raise ValueError('unknown native public-static world')
        h=self.state(state);main=self.main_free(h)
        goal=h.new_empty((len(h),self.goal_count)) if compute_goal else None
        for patch in (0,1,2):
            rows=torch.nonzero(patch_index==patch,as_tuple=False).flatten()
            if rows.numel()==0:continue
            current=h.index_select(0,rows)
            main[rows]=main[rows]+current@self.basket_vectors('main',patch).T
            if compute_goal:
                goal[rows]=self.goal_query(current)@self.basket_vectors('goal',patch).T+self.goal_bias
        return {'main_logits':main,'goal_logits':goal}


def set_future_validation(model,enabled=True):
    if type(enabled) is not bool:raise ValueError('explicit boolean validation setting required')
    set_input_content_validation(model,enabled);model.check_input_contents=enabled


def future_supervision_loss(outputs,main_labels,goal_labels=None,*,use_goal=False):
    logits=outputs['main_logits']
    if (logits.ndim!=2 or not len(logits) or main_labels.shape!=(len(logits),)
            or main_labels.dtype!=torch.long or not bool(torch.isfinite(logits).all())):
        raise ValueError('finite main logits and aligned labels required')
    main=nn.functional.cross_entropy(logits,main_labels)
    if not use_goal:
        if goal_labels is not None:raise ValueError('main-only loss must not receive future labels')
        if outputs['goal_logits'] is not None:raise ValueError('main-only arm must leave the goal head inactive')
        total=main;auxiliary=None
    else:
        goal=outputs['goal_logits']
        if (goal is None or goal_labels is None or goal_labels.shape!=(len(logits),)
                or goal_labels.dtype!=torch.long or not bool(torch.isfinite(goal).all())):
            raise ValueError('finite supervised goal logits and aligned targets required')
        auxiliary=nn.functional.cross_entropy(goal,goal_labels);total=main+auxiliary
    if not bool(torch.isfinite(total)):raise ValueError('nonfinite future-supervision loss')
    return total,main,auxiliary
