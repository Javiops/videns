"""Deterministic fixed-epoch training and fixed-batch inference for player WP."""
import hashlib
import json
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from .wp_player_models_v1 import PlayerWinModel,parameter_count
from .wp_player_preprocessing_v1 import shuffle_positions
from .metrics import game_weights

EPOCHS=30
BATCH=512


def need(value,message):
    if not value:raise ValueError(message)


def configure_torch(device):
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if str(device).startswith('cuda'):
        need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Fixed deterministic CUDA workspace required')
        need(torch.cuda.is_available(),'CUDA required for the declared real-data experiment')


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        header=json.dumps([name,list(array.shape),array.dtype.str],separators=(',',':')).encode()
        digest.update(len(header).to_bytes(8,'little'));digest.update(header);digest.update(array.tobytes())
    return digest.hexdigest()


def normalized_game_weights(games):
    result=game_weights(games);return (result*len(result)/result.sum()).astype(np.float32)


def train_one(arm,seed,x,players,y,weight,guard=lambda:None,status=lambda **kw:None,*,epochs=EPOCHS,batch_size=BATCH):
    need(x.dtype==y.dtype==weight.dtype==torch.float32 and x.ndim==2 and x.shape[1]==173 and y.shape==weight.shape==(len(x),),
         'Aligned float32 FIT tensors required')
    need(len(x)>0 and epochs>0 and batch_size>0 and bool(torch.isfinite(x).all()) and bool(torch.isfinite(weight).all())
         and bool((weight>0).all()) and bool(((y==0)|(y==1)).all()),'Invalid FIT values')
    if arm!='global':need(players is not None and players.shape==(len(x),10,35) and players.device==x.device,'Aligned player FIT tensor required')
    torch.manual_seed(seed)
    if x.device.type=='cuda':torch.cuda.manual_seed_all(seed)
    model=PlayerWinModel(arm);initial=state_digest(model);model.to(x.device);model.check_input_contents=False;model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    rng=np.random.default_rng(seed);permutation_hashes=[];losses=[];updates=0
    for epoch in range(epochs):
        guard();order=rng.permutation(len(x)).astype(np.int64);permutation_hashes.append(hashlib.sha256(order.tobytes()).hexdigest())
        total=0.
        for start in range(0,len(order),batch_size):
            if updates%20==0:guard()
            idx=torch.as_tensor(order[start:start+batch_size],device=x.device);optimizer.zero_grad(set_to_none=True)
            logits=model(x[idx],None if arm=='global' else players[idx])
            each=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[idx],reduction='none');loss=(each*weight[idx]).mean()
            need(bool(torch.isfinite(loss)),'Nonfinite neural FIT loss');loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(idx);updates+=1
        losses.append(total/len(x));status(arm=arm,seed=seed,completed_epochs=epoch+1,total_epochs=epochs,updates=updates,fit_loss=losses[-1])
    model.eval();model.check_input_contents=True
    receipt={'arm':arm,'seed':seed,'epochs':epochs,'batch_size':batch_size,'updates':updates,'fit_frames':len(x),
        'row_visits':epochs*len(x),'parameters':parameter_count(model),'initial_state_sha256':initial,
        'final_state_sha256':state_digest(model),'epoch_permutation_sha256':permutation_hashes,'fit_loss_by_epoch':losses,
        'optimizer':{'name':'AdamW','learning_rate':.0003,'weight_decay':.0001,'gradient_clip':1.},
        'early_stopping':False,'checkpoint_selection':False,'mixed_precision':False}
    del optimizer
    return model,receipt


def predict_raw(model,x,players,device,permutations=None,guard=lambda:None,batch_size=BATCH):
    need(len(x)>0 and x.shape[1:]==(173,) and players.shape==(len(x),10,35),'Aligned inference arrays required')
    if model.arm=='attention_position_shuffled':need(permutations is not None and permutations.shape==(len(x),10),'Shuffled arm needs fixed position permutations')
    else:need(permutations is None,'Only the matched control may shuffle positions')
    out=np.empty(len(x),dtype=np.float32);model.eval()
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            if start%(8*batch_size)==0:guard()
            stop=min(start+batch_size,len(x));p=players[start:stop]
            if permutations is not None:p=shuffle_positions(p,permutations[start:stop])
            gx=torch.as_tensor(x[start:stop],device=device);px=None if model.arm=='global' else torch.as_tensor(p,device=device)
            probability=torch.sigmoid(model(gx,px));out[start:stop]=probability.cpu().numpy()
    need(np.isfinite(out).all() and np.all((out>=0)&(out<=1)),'Invalid neural probability')
    return out


def mean_seed_probability(values):
    values=np.asarray(values);need(values.ndim==2 and values.shape[0]==3 and np.isfinite(values).all()
        and np.all((values>=0)&(values<=1)),'Exactly three finite seed probabilities required')
    return np.mean(values,axis=0,dtype=np.float64)


def logit(probability):
    p=np.clip(probability,1e-7,1-1e-7);return np.log(p/(1-p)).reshape(-1,1)


def fit_calibrator(probability,y,games):
    weight=game_weights(games);weight*=len(weight)/weight.sum()
    calibrator=LogisticRegression(C=1e6,max_iter=500,random_state=20260929)
    calibrator.fit(logit(probability),y,sample_weight=weight);return calibrator


def calibrated(probability,calibrator):return calibrator.predict_proba(logit(probability))[:,1]
