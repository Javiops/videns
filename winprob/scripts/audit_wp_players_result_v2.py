"""Independent GPU replay; v2 makes proper-score and ECE residual dtypes explicit."""
from collections import Counter
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from winprob.wp_player_data_v1 import load_player_data
from winprob.iql_dataset_contract import current_reservations,file_hash
import wp_player_audit_helpers_v1 as independent

HELPER=ROOT/'data/reports/wp_distillation_audit_v1.py'
spec=importlib.util.spec_from_file_location('independent_metrics',HELPER);metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
metric.per_game=independent.grouped_mean


def float64_loss_vectors(y,p):
    # The producer defines proper scores in float64, including clipping/logs.
    y=np.asarray(y,dtype=np.float64);p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    return {'log_loss':-(y*np.log(p)+(1-y)*np.log1p(-p)),'brier':(y-p)**2,'accuracy':((p>=.5)==y).astype(float)}


metric.loss_vectors=float64_loss_vectors
original_summary_validator=metric.validate_summary


def summary_validator(y,p,games,record,label):
    # Keep p's native dtype for the producer's ECE bin assignment, while the
    # weighted residual p-y is float64, as in its weighted mean calculation.
    return original_summary_validator(np.asarray(y,dtype=np.float64),p,games,record,label)


metric.validate_summary=summary_validator
ARMS=('global','deepset','attention','attention_position_shuffled');SEEDS=(20260929,20260930,20261001)


def need(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def state_hash(weights):
    h=hashlib.sha256()
    for name in sorted(weights):
        a=weights[name].detach().cpu().contiguous().numpy();header=json.dumps([name,list(a.shape),a.dtype.str],separators=(',',':')).encode()
        h.update(len(header).to_bytes(8,'little'));h.update(header);h.update(a.tobytes())
    return h.hexdigest()


def verify_decision(report):
    main=report['cohorts']['full_development'];coverage=report['coverage'];gates={};eligible=[]
    def extra(pair):return pair['relative_log_loss_reduction']>=.005 and pair['log_loss']['ci95'][0]>0 and pair['brier']['estimate']>=0
    for arm in ('deepset','attention'):
        ref=main['comparisons'][arm+'_vs_combat_full'];lead=main['lead_time']['comparisons'][arm+'_vs_combat_full'];seeds=report['raw_seed_log_loss_reductions'][arm]
        checks={'complete_declared_scope':coverage['fit_games']==11468 and coverage['calibration_games']==2866 and coverage['development_games']==2861 and coverage['fit_sampled_frames']==64422,
            'all12_models_and30epochs':coverage['completed_models']==12 and coverage['minimum_completed_epochs']==30 and coverage['maximum_completed_epochs']==30,
            'three_seed_ensemble_and_inference_checks':coverage['three_seed_ensembles'] is True and coverage['inference_checks'] is True,
            'two_percent_loss_vs_strong_reference':ref['relative_log_loss_reduction']>=.02,'positive_loss_vs_strong_reference':ref['log_loss']['ci95'][0]>0,
            'positive_brier_vs_strong_reference':ref['brier']['ci95'][0]>0,'half_percent_resolved_gain_vs_global_neural':extra(main['comparisons'][arm+'_vs_global']),
            'minimum1700_lead_games':coverage['lead_games']>=1700,'half_percent_resolved_lead_gain':extra(lead),
            'at_least_two_positive_raw_seeds':len(seeds)==3 and all(np.isfinite(v) for v in seeds) and sum(v>0 for v in seeds)>=2}
        if arm=='attention':checks['half_percent_resolved_gain_vs_position_shuffle']=extra(main['comparisons']['attention_vs_attention_position_shuffled'])
        gates[arm]=checks
        if all(checks.values()):eligible.append(arm)
    chosen=eligible[0] if len(eligible)==1 else None
    if len(eligible)==2:chosen='attention' if extra(main['comparisons']['attention_vs_deepset']) else 'deepset'
    expected={'gates':gates,'eligible':eligible,'selected_candidate':chosen,'useful_development_signal':chosen is not None,
        'confirmation_completed':False,'practical_benefit_established':False,'cohort_status':'previously exposed development; exploratory selection'}
    need(expected==report['decision'],'Independent gate/selection differs');return chosen


def run(experiment,output,device='cuda:0'):
    started=time.monotonic();experiment=Path(experiment).resolve(strict=True);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    def guard():
        need(time.monotonic()-started<1800,'Player result audit1800second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            m=Memory();m.length=ctypes.sizeof(m)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) and m.available>=.75*1024**3,'Audit RAM floor')
    def status(stage,**extra):
        (output/'status.json').write_text(json.dumps({'stage':stage,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n',encoding='utf-8')
    try:
        need(read(experiment.parent/'launcher_status.json')['state']=='completed','Producer must be complete before result audit')
        report=read(experiment/'report.json');report_sha=file_hash(experiment/'report.json')
        need(report['schema']=='wp-joint-player-comparison-v1' and report['state']=='complete','Incomplete player result')
        need((experiment/'report.sha256').read_text().split()==[report_sha,'report.json'],'Report digest differs')
        for path,digest in report['bindings_sha256'].items():need(file_hash(path)==digest,'Changed bound input/code')
        for name,digest in report['files'].items():need(file_hash(experiment/name)==digest,'Changed experiment file')
        for name,digest in report['calibrators'].items():need(file_hash(experiment/name)==digest,'Changed calibrator')
        lock=read(experiment/'training_lock.json');before=read(experiment/'experiment_lock.json')
        need(lock['all_models_complete'] and not lock['new_development_predictions_inspected'] and not lock['comparison_metrics_inspected'],'Invalid complete training lock')
        need(lock['models']==report['models'] and lock['receipts']==report['receipts'] and lock['calibrators']==report['calibrators'],'Lock/result differ')
        need(lock['bindings_sha256']==before['bindings_sha256']==report['bindings_sha256'],'Input bindings differ across locks')
        expected={arm+'__'+str(seed) for arm in ARMS for seed in SEEDS}
        need(set(report['models'])==set(report['receipts'])==expected,'All four arms/three seeds required')
        for seed in SEEDS:
            rng=np.random.default_rng(seed);hashes=[hashlib.sha256(rng.permutation(64422).astype(np.int64).tobytes()).hexdigest() for _ in range(30)]
            for arm in ARMS:
                r=report['receipts'][arm+'__'+str(seed)]
                need(r['arm']==arm and r['seed']==seed and r['epochs']==30 and r['batch_size']==512 and r['updates']==3780
                     and r['row_visits']==1932660 and r['fit_frames']==64422 and r['epoch_permutation_sha256']==hashes,'Training schedule/identity differs')
                need(r['optimizer']=={'name':'AdamW','learning_rate':.0003,'weight_decay':.0001,'gradient_clip':1.}
                     and not r['early_stopping'] and not r['checkpoint_selection'] and not r['mixed_precision'],'Training recipe differs')
            need(report['receipts']['attention__'+str(seed)]['initial_state_sha256']==report['receipts']['attention_position_shuffled__'+str(seed)]['initial_state_sha256'],
                 'Matched control initial parameters differ')
        status('loading_verified_rows');data,cache,parent,bounds,admission=load_player_data(report['cache_manifest_path'],guard)
        fit,cal,dev=(data[k] for k in ('fit','calibration','development'))
        with np.load(experiment/'normalization.npz',allow_pickle=False) as saved:norm={k:saved[k] for k in saved.files}
        need(int(norm['fit_games'])==11468 and int(norm['fit_rows'])==64422,'Normalization FIT population differs')
        status('independent_fit_normalization');recomputed=independent.independent_normalization(fit['x'],fit['players'],fit['game'])
        for key,value in recomputed.items():need(np.allclose(value,norm[key],rtol=1e-12,atol=1e-10),'Independent normalization differs: '+key)
        with np.load(experiment/'position_permutations.npz',allow_pickle=False) as saved:permutations={k:saved[k] for k in saved.files}
        need(set(permutations)==set(data),'Permutation role population differs')
        for role,part in data.items():
            guard();need(np.array_equal(permutations[role],independent.independent_position_permutations(part['game'],part['timestamp_ms'])),'Independent positional control differs')
        with np.load(experiment/'development_predictions.npz',allow_pickle=False) as saved:p={k:saved[k] for k in saved.files}
        with np.load(experiment/'calibration_seed_predictions.npz',allow_pickle=False) as saved:cp={k:saved[k] for k in saved.files}
        for key in ('y','game','timestamp_ms','patch','pilot'):need(np.array_equal(p[key],dev[key]),'Development identity differs')
        for key in ('y','game','timestamp_ms'):need(np.array_equal(cp[key],cal[key]),'Calibration identity differs')
        lead=(dev['end_ms']>=0)&(dev['end_ms']-dev['timestamp_ms']>=180000)
        need(np.array_equal(p['evaluation_only_end_ms'],dev['end_ms']) and np.array_equal(p['lead_time_mask'],lead),'Lead-time identity differs')
        reference_dir=ROOT/'data/runs/wp-scale-fit-20260930-v2/experiment'
        with np.load(reference_dir/'development_predictions.npz',allow_pickle=False) as old:
            need(np.array_equal(p['combat_full'],old['combat_full']) and np.array_equal(p['combat_full_raw'],old['combat_full_raw']),'Strong reference was replaced')
        with threadpool_limits(limits=1):
            reference=joblib.load(reference_dir/'combat_full.joblib');raw=reference.model.predict_proba(dev['x'])[:,1]
            need(np.array_equal(raw,p['combat_full_raw']),'Reference raw replay differs');metric.close(metric.calibrated(raw,reference.calibrator),p['combat_full'],'Reference calibration')
        del reference,raw
        status('preparing_independent_neural_replay')
        for role in ('calibration','development'):
            part=data[role];part['x'],part['players']=independent.apply_normalization(part['x'],part['players'],norm)
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        need(device=='cuda:0' and torch.cuda.is_available(),'Declared GPU replay required')
        need(torch.__version__==report['runtime']['torch'] and torch.version.cuda==report['runtime']['cuda']
             and torch.cuda.get_device_name(device)==report['runtime']['gpu'],'Exact GPU replay runtime differs')
        result=replay_and_score(experiment,output,report,data,p,cp,permutations,lead,guard,status,device)
        for path,digest in report['bindings_sha256'].items():need(file_hash(path)==digest,'Bound input changed during audit')
        need(file_hash(experiment/'report.json')==report_sha and not {r['match_id'] for r in cache['games']}&current_reservations(admission),'Final reservation/report check failed')
        result.update(schema='wp-player-result-independent-audit-v2',state='passed',report_sha256=report_sha,
            audit_code_sha256=file_hash(__file__),functional_helper_sha256=file_hash(independent.__file__),metric_helper_sha256=file_hash(HELPER),
            verified_bindings=len(report['bindings_sha256']),new_raw_payload_reads=False,neural_models_refitted=False,
            calibrators_recomputed_for_verification=4,normalization_independently_recomputed=True,positional_controls_independently_recomputed=True,
            reserved_intersections=0,elapsed_s=time.monotonic()-started,
            numeric_semantics='float64 proper scores; native probability dtype for ECE bins and float64 weighted ECE residuals; unchanged1e-12 metric tolerance',
            limits=['Independent functional GPU replay covers all calibration and development seed predictions at the original batch512.',
                    'Primary/full, old-pilot and complement proper-score/interval/selection checks are independent; time/patch slices are not separately audited.',
                    'Player-cache row loading is shared, following its separately passed full-algebra and sampled independent raw reconstruction audit.',
                    'This is repeated-development evidence, not confirmation, causal action value or practical benefit.'])
        path=output/'report.json';path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
        path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii');status('complete');return result
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


def replay_and_score(experiment,output,report,data,p,cp,permutations,lead,guard,status,device):
    cal=data['calibration'];dev=data['development'];checks=Counter();errors={};calibration_errors={}
    seed_names={arm+'__'+str(seed) for arm in ARMS for seed in SEEDS}
    need(set(cp)=={'y','game','timestamp_ms'}|seed_names,'Calibration archive fields differ')
    expected={'y','game','timestamp_ms','patch','pilot','evaluation_only_end_ms','lead_time_mask','combat_full','combat_full_raw'}
    expected|=set(ARMS)|{arm+'_raw' for arm in ARMS}|{name+'_seed_raw' for name in seed_names}
    need(set(p)==expected,'Development archive fields differ')
    parameter_counts={'global':38785,'deepset':45249,'attention':141569,'attention_position_shuffled':141569}
    for arm in ARMS:
        for seed in SEEDS:
            guard();name=arm+'__'+str(seed);status('independent_functional_gpu_replay',model=name)
            artifact=report['models'][name];path=(experiment/artifact['file']).resolve(strict=True)
            need(path.is_relative_to(experiment) and path.name==name+'.pt' and file_hash(path)==artifact['sha256'],'Model path/digest differs')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            need(saved['schema']=='wp-player-weights-v1' and saved['arm']==artifact['arm']==arm and saved['seed']==artifact['seed']==seed,'Checkpoint identity differs')
            weights=saved['model_state'];need(all(v.dtype==torch.float32 and bool(torch.isfinite(v).all()) for v in weights.values()),'Invalid model tensors')
            need(sum(v.numel() for v in weights.values())==artifact['parameters']==parameter_counts[arm],'Fixed architecture parameter count differs')
            need(state_hash(weights)==report['receipts'][name]['final_state_sha256'],'Final parameter receipt differs')
            replay_cal=independent.predict(weights,arm,cal['x'],cal['players'],device,
                permutations['calibration'] if arm=='attention_position_shuffled' else None,guard)
            replay_dev=independent.predict(weights,arm,dev['x'],dev['players'],device,
                permutations['development'] if arm=='attention_position_shuffled' else None,guard)
            cal_error=float(np.max(np.abs(replay_cal-cp[name])));dev_error=float(np.max(np.abs(replay_dev-p[name+'_seed_raw'])))
            errors[name]={'calibration':cal_error,'development':dev_error}
            need(np.array_equal(replay_cal,cp[name]) and np.array_equal(replay_dev,p[name+'_seed_raw']),'Functional GPU replay must match every stored probability exactly')
            checks['calibration_predictions']+=len(replay_cal);checks['development_predictions']+=len(replay_dev)
            del weights,saved,replay_cal,replay_dev
        # Average every fixed seed in float64; no seed selection or weighting.
        mean_cal=sum(cp[arm+'__'+str(seed)].astype(np.float64) for seed in SEEDS)/3
        mean_dev=sum(p[arm+'__'+str(seed)+'_seed_raw'].astype(np.float64) for seed in SEEDS)/3
        metric.close(mean_dev,p[arm+'_raw'],arm+'/raw ensemble')
        calibrator=joblib.load(experiment/(arm+'_calibrator.joblib'))
        metric.close(metric.calibrated(mean_dev,calibrator),p[arm],arm+'/calibrated ensemble')
        ids,starts,counts=np.unique(cal['game'],return_index=True,return_counts=True);w=np.empty(len(cal['game']),dtype=np.float64)
        need(np.array_equal(cal['game'],np.repeat(ids,counts)),'Calibration groups are not canonical')
        for first,count in zip(starts,counts):w[first:first+count]=1./count
        w*=len(w)/w.sum();clipped=np.clip(mean_cal,1e-7,1-1e-7);logits=np.log(clipped/(1-clipped)).reshape(-1,1)
        with threadpool_limits(limits=1):
            recomputed=LogisticRegression(C=1e6,max_iter=500,random_state=20260929).fit(logits,cal['y'],sample_weight=w)
        need(np.allclose(recomputed.coef_,calibrator.coef_,rtol=0,atol=1e-9)
             and np.allclose(recomputed.intercept_,calibrator.intercept_,rtol=0,atol=1e-9),'Independent calibration fit differs')
        calibration_errors[arm]={'coefficient':float(np.max(np.abs(recomputed.coef_-calibrator.coef_))),
            'intercept':float(np.max(np.abs(recomputed.intercept_-calibrator.intercept_)))}
    # The sums below give equal weight to complete games, not individual frames.
    for name,take in (('full_development',np.ones(len(dev['y']),dtype=bool)),('original_pilot',dev['pilot']),('old_pilot_complement',~dev['pilot'])):
        guard();status('independent_metrics',cohort=name);cohort=report['cohorts'][name]
        need(cohort['games']==len(np.unique(dev['game'][take])) and cohort['frames']==int(take.sum()),'Cohort counts differ')
        for suffix,mask,part in (('',take,cohort),('/lead',take&lead,cohort['lead_time'])):
            guard();need(part['games']==len(np.unique(dev['game'][mask])) and part['frames']==int(mask.sum()),'Stratum counts differ')
            for model,summary in part['methods'].items():
                metric.validate_summary(dev['y'][mask],p[model][mask],dev['game'][mask],summary,name+suffix+'/'+model);checks['summaries']+=1
            for pair,record in part['comparisons'].items():
                a,b=pair.split('_vs_');metric.validate_pair(dev['y'][mask],p[b][mask],p[a][mask],dev['game'][mask],record,name+suffix+'/'+pair);checks['comparisons']+=1
    for name,summary in report['raw_primary_methods'].items():
        guard();metric.validate_summary(dev['y'],p[name+'_raw'],dev['game'],summary,name+'/raw');checks['summaries']+=1
    for name,summary in report['raw_primary_seed_methods'].items():
        guard();metric.validate_summary(dev['y'],p[name+'_seed_raw'],dev['game'],summary,name+'/seed');checks['summaries']+=1
    ref_losses=metric.loss_vectors(dev['y'],p['combat_full_raw'])['log_loss'];base=independent.grouped_mean(ref_losses,dev['game']).mean()
    for arm in ('deepset','attention'):
        gains=[]
        for seed in SEEDS:
            loss=metric.loss_vectors(dev['y'],p[arm+'__'+str(seed)+'_seed_raw'])['log_loss']
            gains.append(float(base-independent.grouped_mean(loss,dev['game']).mean()))
        metric.close(gains,report['raw_seed_log_loss_reductions'][arm],arm+'/raw seed gains')
    coverage={'fit_games':len(np.unique(data['fit']['game'])),'calibration_games':len(np.unique(cal['game'])),
        'development_games':len(np.unique(dev['game'])),'fit_sampled_frames':len(data['fit']['y']),'completed_models':12,
        'minimum_completed_epochs':30,'maximum_completed_epochs':30,'three_seed_ensembles':True,
        'inference_checks':len(report['permutation_checks'])==12 and max(report['permutation_checks'].values())<=1e-5,
        'lead_games':len(np.unique(dev['game'][lead]))}
    need(coverage==report['coverage'],'Verified coverage differs');chosen=verify_decision(report)
    return {'coverage':coverage,'models_replayed':12,'calibration_predictions_replayed':checks['calibration_predictions'],
        'development_predictions_replayed':checks['development_predictions'],'maximum_probability_errors':errors,
        'independent_calibration_parameter_errors':calibration_errors,'verified_summaries':checks['summaries'],
        'verified_paired_comparisons':checks['comparisons'],'decision_reproduced':True,'selected_candidate':chosen}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    print(json.dumps(run(args.experiment,args.output,args.device),indent=2))
