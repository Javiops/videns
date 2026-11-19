"""Prototype Gaussian fields of observed health/resources; no fitted parameters.

These are proximity associations, not reachability, combat power or causal
action values. Input is the raw35-field player tensor, before standardization.
No real-data experiment is authorized merely by importing this prototype.
"""
import numpy as np
from .features_wp_players_v1 import FIELDS as PLAYER_FIELDS

RADIUS=2000./14870.
CHANNELS=('positive_health','health_fraction','health_weighted_gold','health_weighted_nonnegative_outflow')
STATISTICS=('balance_mean','balance_min','balance_q25')
FIELDS=tuple('kernel_'+channel+'_'+stat+'_'+op for channel in CHANNELS for stat in STATISTICS for op in ('diff','sum'))
FIELDS+=tuple('kernel_'+channel+'_'+name for channel in CHANNELS for name in ('cross_mean_kernel','squared_mmd','both_masses_positive'))
FIELDS+=tuple('kernel_'+name+'_'+op for name in ('anchor_fraction','ratio_position_fraction','wallet_ratio_position_fraction') for op in ('diff','sum'))
assert len(FIELDS)==42
IX={name:i for i,name in enumerate(PLAYER_FIELDS)}


def need(value,message):
    if not value:raise ValueError(message)


def team_stats(values,mask):
    count=mask.sum(axis=1);total=np.where(mask,values,0).sum(axis=1)
    mean=np.divide(total,count,out=np.zeros(len(count)),where=count>0)
    ordered=np.sort(np.where(mask,values,np.inf),axis=1);ordered[count==0]=0.
    minimum=ordered[:,0];where=np.maximum(count-1,0)*.25;low=np.floor(where).astype(int);high=np.ceil(where).astype(int)
    a=ordered[np.arange(len(count)),low];b=ordered[np.arange(len(count)),high]
    return {'balance_mean':mean,'balance_min':minimum,'balance_q25':a+(b-a)*(where-low)}


def kernel_features(players):
    raw=np.asarray(players)
    need(raw.ndim==3 and raw.shape[1:]==(10,35) and np.isfinite(raw).all(),'Finite raw ten-player35-field tensor required')
    p=raw.astype(np.float64);n=len(p);side=p[:,:,-1]
    blue=side==1;red=side==-1
    need(np.all(blue.sum(axis=1)==5) and np.all(red.sum(axis=1)==5),'Exactly five players on each side required')
    mask_fields=('position_observed','health_positive','health_ratio_observed','wallet_observed')
    for name in mask_fields:need(np.all((p[:,:,IX[name]]==0)|(p[:,:,IX[name]]==1)),'Raw binary availability required')
    positioned=p[:,:,IX['position_observed']].astype(bool);positive=p[:,:,IX['health_positive']]
    ratio=p[:,:,IX['health_ratio']];ratio_known=p[:,:,IX['health_ratio_observed']].astype(bool)
    wallet_known=p[:,:,IX['wallet_observed']].astype(bool);gold=p[:,:,IX['total_gold_over10000']]
    need(np.all((ratio>=0)&(ratio<=1)) and np.all(gold>=0),'Raw nonnegative wealth/health ratio required')
    coordinates=p[:,:,:2]
    need(np.all(coordinates[~positioned]==0) and np.all((coordinates[positioned]>0)&(coordinates[positioned]<=1)),
         'Positions must use the raw map-extent scale and explicit unknown mask')
    distance2=((coordinates[:,:,None,:]-coordinates[:,None,:,:])**2).sum(axis=-1)
    kernel=np.exp(-distance2/(2*RADIUS**2))*positioned[:,:,None]*positioned[:,None,:]
    anchors=positioned&(positive>0);hp=ratio*ratio_known
    weights={'positive_health':positive,'health_fraction':hp,'health_weighted_gold':hp*gold,
        'health_weighted_nonnegative_outflow':hp*np.maximum(p[:,:,IX['net_gold_outflow_over10000']],0)*wallet_known}
    result={}
    for channel,weight in weights.items():
        wb=weight*positioned*blue;wr=weight*positioned*red
        blue_field=np.einsum('bij,bj->bi',kernel,wb);red_field=np.einsum('bij,bj->bi',kernel,wr)
        denominator=blue_field+red_field
        balance=np.divide(blue_field-red_field,denominator,out=np.zeros((n,10)),where=denominator>0)
        bs=team_stats(balance,anchors&blue);rs=team_stats(-balance,anchors&red)
        for stat in STATISTICS:
            result['kernel_'+channel+'_'+stat+'_diff']=bs[stat]-rs[stat]
            result['kernel_'+channel+'_'+stat+'_sum']=bs[stat]+rs[stat]
        mb=wb.sum(axis=1);mr=wr.sum(axis=1);valid=(mb>0)&(mr>0)
        nb=np.divide(wb,mb[:,None],out=np.zeros_like(wb),where=mb[:,None]>0)
        nr=np.divide(wr,mr[:,None],out=np.zeros_like(wr),where=mr[:,None]>0)
        cross=np.einsum('bi,bij,bj->b',nb,kernel,nr)
        mmd=np.einsum('bi,bij,bj->b',nb,kernel,nb)+np.einsum('bi,bij,bj->b',nr,kernel,nr)-2*cross
        need(np.all(mmd[valid]>=-1e-12),'Gaussian squared MMD must be nonnegative apart from rounding')
        result['kernel_'+channel+'_cross_mean_kernel']=np.where(valid,cross,0.)
        result['kernel_'+channel+'_squared_mmd']=np.where(valid,np.maximum(mmd,0),0.)
        result['kernel_'+channel+'_both_masses_positive']=valid.astype(float)
    availability={'anchor_fraction':anchors,'ratio_position_fraction':positioned&ratio_known,
                  'wallet_ratio_position_fraction':positioned&ratio_known&wallet_known}
    for name,mask in availability.items():
        b=(mask&blue).sum(axis=1)/5;r=(mask&red).sum(axis=1)/5
        result['kernel_'+name+'_diff']=b-r;result['kernel_'+name+'_sum']=b+r
    answer=np.column_stack([result[name] for name in FIELDS]).astype(np.float32)
    need(answer.shape==(n,42) and np.isfinite(answer).all(),'Invalid Gaussian field features')
    return answer
