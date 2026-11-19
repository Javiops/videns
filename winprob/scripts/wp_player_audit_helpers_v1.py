"""Independent arithmetic and functional neural replay for the player-WP audit."""
import hashlib
import math
import numpy as np
import torch
from torch.nn import functional as F


def independent_normalization(x,p,g):
    ids,starts,counts=np.unique(g,return_index=True,return_counts=True)
    if not np.array_equal(g,np.repeat(ids,counts)):raise ValueError('Canonical game grouping required')
    mx=np.zeros(173);mp=np.zeros(35)
    for start,count in zip(starts,counts):
        mx+=x[start:start+count].astype(np.float64).mean(axis=0)/len(ids)
        mp+=p[start:start+count].astype(np.float64).mean(axis=(0,1))/len(ids)
    vx=np.zeros(173);vp=np.zeros(35)
    for start,count in zip(starts,counts):
        dx=x[start:start+count].astype(np.float64)-mx;dp=p[start:start+count].astype(np.float64)-mp
        vx+=(dx*dx).mean(axis=0)/len(ids);vp+=(dp*dp).mean(axis=(0,1))/len(ids)
    sx=np.sqrt(vx);sp=np.sqrt(vp);sx[sx<1e-12]=1;sp[sp<1e-12]=1
    return {'global_mean':mx,'global_scale':sx,'token_mean':mp,'token_scale':sp}


def apply_normalization(x,p,state):
    gx=np.empty_like(x);pt=np.empty_like(p)
    for lo in range(0,len(x),4096):
        hi=lo+4096;gx[lo:hi]=((x[lo:hi].astype(np.float64)-state['global_mean'])/state['global_scale']).astype(np.float32)
        pt[lo:hi]=((p[lo:hi].astype(np.float64)-state['token_mean'])/state['token_scale']).astype(np.float32)
    return gx,pt


def independent_position_permutations(games,times):
    result=np.empty((len(games),10),dtype=np.uint8)
    for row in range(len(games)):
        for side,positions in ((100,range(5)),(200,range(5,10))):
            scored=[(hashlib.sha256(('20260929:'+str(int(games[row]))+':'+str(int(times[row]))+':'+str(side)+':'+str(slot)).encode()).digest(),slot)
                    for slot in positions]
            ordered=[slot for digest,slot in sorted(scored)];result[row,list(positions)]=ordered
    return result


def forward(weights,arm,global_features,players):
    def linear(x,name):return F.linear(x,weights[name+'.weight'],weights[name+'.bias'])
    def block(query,context,name):
        b,nq,_=query.shape;nk=context.shape[1]
        q=linear(query,name+'.query').reshape(b,nq,4,16).transpose(1,2)
        k=linear(context,name+'.key').reshape(b,nk,4,16).transpose(1,2)
        v=linear(context,name+'.value').reshape(b,nk,4,16).transpose(1,2)
        attention=torch.softmax(torch.matmul(q,k.transpose(-1,-2))/math.sqrt(16),dim=-1)
        value=torch.matmul(attention,v).transpose(1,2).reshape(b,nq,64)
        h=F.layer_norm(query+linear(value,name+'.output'),(64,),weights[name+'.first_norm.weight'],weights[name+'.first_norm.bias'],1e-5)
        ff=linear(F.gelu(linear(h,name+'.feedforward.0')),name+'.feedforward.2')
        return F.layer_norm(h+ff,(64,),weights[name+'.second_norm.weight'],weights[name+'.second_norm.bias'],1e-5)
    if arm=='global':representation=global_features.new_zeros((len(global_features),64))
    elif arm=='deepset':
        tokens=F.gelu(linear(F.gelu(linear(players,'token_encoder.0')),'token_encoder.2'));representation=tokens.mean(dim=1)
    elif arm in ('attention','attention_position_shuffled'):
        tokens=linear(players,'token_encoder')
        for i in (0,1):tokens=block(tokens,tokens,'blocks.'+str(i))
        representation=block(weights['pool_query'].expand(len(tokens),-1,-1),tokens,'pool')[:,0]
    else:raise ValueError('Unknown independent replay arm')
    head=torch.cat((global_features,representation),dim=1)
    return linear(F.gelu(linear(F.gelu(linear(head,'head.0')),'head.2')),'head.4').squeeze(-1)


def predict(weights,arm,x,p,device,positions=None,guard=lambda:None):
    if (arm=='attention_position_shuffled')!=(positions is not None):raise ValueError('Independent control alignment differs')
    # Preserve native nn.Parameter flags. Synthetic attention replay with
    # detached constants differed by about3e-8; matching these flags was exact.
    w={k:v.detach().to(device).requires_grad_(True) for k,v in weights.items()};out=np.empty(len(x),dtype=np.float32)
    with torch.inference_mode():
        for lo in range(0,len(x),512):
            if lo%(8*512)==0:guard()
            hi=min(lo+512,len(x));players=p[lo:hi]
            if positions is not None:
                players=players.copy();players[:,:,:3]=p[lo:hi,:,:3][np.arange(hi-lo)[:,None],positions[lo:hi]]
            logits=forward(w,arm,torch.as_tensor(x[lo:hi],device=device),torch.as_tensor(players,device=device))
            out[lo:hi]=torch.sigmoid(logits).cpu().numpy()
    return out


def grouped_mean(values,games):
    order=np.argsort(games,kind='stable');sorted_games=games[order];starts=np.r_[0,np.flatnonzero(np.diff(sorted_games))+1]
    counts=np.diff(np.r_[starts,len(order)]);return np.add.reduceat(np.asarray(values)[order],starts)/counts
