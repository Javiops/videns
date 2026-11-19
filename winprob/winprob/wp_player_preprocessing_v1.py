"""FIT-only weighted normalization and a deterministic spatial-association control."""
import hashlib
import numpy as np


def need(value,message):
    if not value:raise ValueError(message)


def position_permutations(games,timestamps):
    games=np.asarray(games);timestamps=np.asarray(timestamps)
    need(games.shape==timestamps.shape and games.ndim==1 and np.issubdtype(games.dtype,np.integer)
         and np.issubdtype(timestamps.dtype,np.integer),'Aligned integer game/time identities required')
    result=np.empty((len(games),10),dtype=np.uint8)
    for i,(game,clock) in enumerate(zip(games,timestamps)):
        for side,start in ((100,0),(200,5)):
            slots=sorted(range(start,start+5),key=lambda slot:hashlib.sha256(f'20260929:{int(game)}:{int(clock)}:{side}:{slot}'.encode()).digest())
            result[i,start:start+5]=slots
    return result


def shuffle_positions(players,permutations):
    players=np.asarray(players);permutations=np.asarray(permutations)
    need(players.ndim==3 and players.shape[1:]==(10,35) and permutations.shape==(len(players),10),'Invalid position-shuffle alignment')
    need(np.array_equal(np.sort(permutations[:,:5],axis=1),np.broadcast_to(np.arange(5),(len(players),5)))
         and np.array_equal(np.sort(permutations[:,5:],axis=1),np.broadcast_to(np.arange(5,10),(len(players),5))),
         'Position permutation must stay within the same five-player side')
    result=players.copy();positions=players[:,:,:3]
    result[:,:,:3]=positions[np.arange(len(players))[:,None],permutations]
    return result


def fit_normalization(global_features,players,games):
    x=np.asarray(global_features);p=np.asarray(players);g=np.asarray(games)
    need(x.shape==(len(g),173) and p.shape==(len(g),10,35) and len(g)>0 and np.isfinite(x).all() and np.isfinite(p).all(),
         'Finite aligned sampled FIT arrays required')
    ids,inverse,counts=np.unique(g,return_inverse=True,return_counts=True);w=1./counts[inverse]/len(ids)
    sx=np.zeros(173);sp=np.zeros(35)
    for start in range(0,len(g),2048):
        end=start+2048;xb=x[start:end].astype(np.float64);pb=p[start:end].astype(np.float64);weights=w[start:end]
        sx+=(weights[:,None]*xb).sum(axis=0)
        sp+=(weights[:,None,None]*pb/10).sum(axis=(0,1))
    vx=np.zeros(173);vp=np.zeros(35)
    for start in range(0,len(g),2048):
        end=start+2048;dx=x[start:end].astype(np.float64)-sx;dp=p[start:end].astype(np.float64)-sp;weights=w[start:end]
        vx+=(weights[:,None]*dx*dx).sum(axis=0);vp+=(weights[:,None,None]*dp*dp/10).sum(axis=(0,1))
    def scale(variance):
        result=np.sqrt(np.maximum(0,variance));result[result<1e-12]=1.;return result
    return {'global_mean':sx,'global_scale':scale(vx),'token_mean':sp,'token_scale':scale(vp),
            'fit_games':np.asarray(len(ids),dtype=np.int64),'fit_rows':np.asarray(len(g),dtype=np.int64)}


def normalize(global_features,players,state):
    x=((global_features-state['global_mean'])/state['global_scale']).astype(np.float32)
    p=((players-state['token_mean'])/state['token_scale']).astype(np.float32)
    need(np.isfinite(x).all() and np.isfinite(p).all(),'Invalid normalized inputs')
    return x,p
