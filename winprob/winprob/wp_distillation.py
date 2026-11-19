"""Current-only win forecasts with FIT-only cross-fitted teacher targets."""
from __future__ import annotations
import hashlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from .metrics import game_weights,losses,per_game_mean,paired_comparison

SEED=20260929
HORIZON_BINS=3
STUDENTS=('hard','current_teacher','future_teacher')
PARAMETERS=dict(max_iter=150,max_leaf_nodes=15,learning_rate=.06,l2_regularization=10,
                min_samples_leaf=100,early_stopping=False,random_state=SEED)


def require(value,message):
    if not value:raise ValueError(message)


def fold_for_match(match_id):
    return int.from_bytes(hashlib.sha256(('wp-distill-v1:'+match_id).encode()).digest()[:8],'big')%4


def privileged_fit_matrix(x,games,timestamps,roles):
    x=np.asarray(x);games=np.asarray(games);timestamps=np.asarray(timestamps);roles=np.asarray(roles)
    require(x.ndim==2 and len(x)>0 and all(v.shape==(len(x),) for v in (games,timestamps,roles)),'Unaligned frame arrays')
    require(set(roles)<= {'fit','calibration','development'},'Forbidden frame role')
    role_by_game={};seen=set();bins={};fit=np.flatnonzero(roles=='fit')
    for i,(game,role) in enumerate(zip(games,roles,strict=True)):
        require(int(game) not in role_by_game or role_by_game[int(game)]==str(role),'A game crosses roles')
        role_by_game[int(game)]=str(role)
        if role!='fit':continue
        key=(int(game),int(timestamps[i]));require(key not in seen,'Duplicate FIT game/timestamp');seen.add(key)
        group=(int(game),int(timestamps[i])//60000)
        if group not in bins or timestamps[i]<timestamps[bins[group]]:bins[group]=i
    require(len(fit)>0 and np.isfinite(x[fit]).all(),'No finite FIT features')
    future=np.array([bins.get((int(games[i]),int(timestamps[i])//60000+HORIZON_BINS),-1) for i in fit],dtype=np.int64)
    present=future>=0;later=np.zeros_like(x[fit]);later[present]=x[future[present]]
    require(np.all(roles[future[present]]=='fit') and np.array_equal(games[future[present]],games[fit[present]])
            and np.all(timestamps[future[present]]>timestamps[fit[present]]),'Privileged state crossed a game/role/time boundary')
    joint=np.column_stack((x[fit],later,present.astype(x.dtype)))
    return fit,joint,future


def teacher_plan(fit_games,source_by_index,selected_local):
    require(all(source_by_index[int(game)]['role']=='fit' for game in np.unique(fit_games)),
            'Teacher planning received an external calibration/development game')
    folds=np.array([fold_for_match(source_by_index[int(game)]['match_id']) for game in fit_games],dtype=np.int8)
    plans=[];all_rows=np.arange(len(fit_games));selected_local=np.asarray(selected_local,dtype=np.int64)
    require(len(np.unique(selected_local))==len(selected_local) and np.all((selected_local>=0)&(selected_local<len(folds))),
            'Invalid sampled FIT row indices')
    for heldout in range(4):
        calibration=(heldout+1)%4
        train=selected_local[(folds[selected_local]!=heldout)&(folds[selected_local]!=calibration)]
        cal=all_rows[folds==calibration];target=all_rows[folds==heldout]
        sets=[set(map(int,fit_games[idx])) for idx in (train,cal,target)]
        require(all(sets) and not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'Teacher game separation failed')
        require(set.union(*sets)==set(map(int,fit_games)),'Teacher plan drops FIT games')
        plans.append({'heldout_fold':heldout,'calibration_fold':calibration,'train':train,'cal':cal,'target':target})
    require(np.array_equal(np.sort(np.concatenate([p['target'] for p in plans])),all_rows),'OOF target coverage differs')
    return folds,plans


class CurrentWinStudent:
    """Serialized inference has a current-state regressor and calibrator only."""
    def __init__(self,feature_names):
        self.feature_names=tuple(feature_names)
        require(len(self.feature_names)>0 and len(set(self.feature_names))==len(self.feature_names),'Distinct current feature names required')
        self.model=None;self.calibrator=None

    def _x(self,x):
        x=np.asarray(x)
        require(x.ndim==2 and x.shape[1]==len(self.feature_names) and np.isfinite(x).all(),
                'Student requires exactly the declared CURRENT feature matrix')
        return x

    @staticmethod
    def _weights(games):
        weight=game_weights(games);return weight*len(weight)/weight.sum()

    @staticmethod
    def logit(p):
        p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7);return np.log(p/(1-p)).reshape(-1,1)

    def fit(self,x,target,games):
        x=self._x(x);target=np.asarray(target,dtype=float);games=np.asarray(games)
        require(target.shape==games.shape==(len(x),) and np.isfinite(target).all() and np.all((target>=0)&(target<=1)),
                'Aligned bounded teacher/hard targets required')
        self.model=HistGradientBoostingRegressor(loss='squared_error',**PARAMETERS)
        self.model.fit(x,target,sample_weight=self._weights(games));return self

    def unbounded_raw(self,x):
        require(self.model is not None,'Student is not fitted')
        result=self.model.predict(self._x(x));require(np.isfinite(result).all(),'Nonfinite student forecast');return result

    def raw(self,x):return np.clip(self.unbounded_raw(x),1e-7,1-1e-7)

    def calibrate(self,x,y,games):
        x=self._x(x);y=np.asarray(y);games=np.asarray(games)
        require(y.shape==games.shape==(len(x),) and set(np.unique(y))=={0,1},'Binary external calibration labels required')
        self.calibrator=LogisticRegression(C=1e6,max_iter=500,random_state=SEED)
        self.calibrator.fit(self.logit(self.raw(x)),y,sample_weight=self._weights(games));return self

    def predict(self,x):
        require(self.calibrator is not None,'Student is not calibrated')
        result=self.calibrator.predict_proba(self.logit(self.raw(x)))[:,1]
        require(np.isfinite(result).all() and np.all((result>=0)&(result<=1)),'Invalid calibrated forecast');return result


def comparison(y,reference,candidate,games):
    result=paired_comparison(y,reference,candidate,games,2000)
    ids,old=per_game_mean(losses(y,reference)['log_loss'],games)
    other,new=per_game_mean(losses(y,candidate)['log_loss'],games);require(np.array_equal(ids,other),'Paired game axis differs')
    rng=np.random.default_rng(SEED);ratios=[]
    for _ in range(2000):
        index=rng.integers(0,len(ids),len(ids));ratios.append(float((old[index]-new[index]).mean()/old[index].mean()))
    result['relative_log_loss_reduction_ci95']=np.quantile(ratios,[.025,.975]).tolist()
    return result


def useful_gate(comparisons):
    gates={}
    for control in ('hard','frozen_rich_classifier'):
        item=comparisons['future_teacher_vs_'+control]
        gates['two_percent_log_loss_vs_'+control]=item['relative_log_loss_reduction']>=.02
        gates['positive_paired_log_loss_vs_'+control]=item['log_loss']['ci95'][0]>0
        gates['positive_paired_brier_vs_'+control]=item['brier']['ci95'][0]>0
    gates['positive_paired_log_loss_vs_current_teacher']=comparisons['future_teacher_vs_current_teacher']['log_loss']['ci95'][0]>0
    return gates
