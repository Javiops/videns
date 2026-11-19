"""Fixed four-arm, three-seed joint-player WP experiment on exposed development."""
from __future__ import annotations
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import joblib
import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits
import setback_experiment as safe
from winprob.iql_dataset_contract import current_reservations
from winprob.wp_player_data_v1 import load_player_data
from winprob.wp_player_models_v1 import ARMS,SEEDS,PlayerWinModel
from winprob.wp_player_preprocessing_v1 import fit_normalization,normalize,position_permutations,shuffle_positions
from winprob.wp_player_training_v1 import configure_torch,train_one,predict_raw,normalized_game_weights,mean_seed_probability,fit_calibrator,calibrated,state_digest
from winprob.wp_player_selection_v1 import select_player_candidate
from winprob.metrics import summarize
from winprob.wp_distillation import comparison

PROTOCOL=ROOT/'docs/wp_player_relations_protocol_v1.md'
REFERENCE=ROOT/'data/runs/wp-scale-fit-20260930-v2/experiment'


def need(value,message):
    if not value:raise ValueError(message)


def write(path,value):
    safe.write_json(path,value);Path(path).with_suffix('.sha256').write_text(safe.digest(path)+' '+Path(path).name+'\n',encoding='ascii')


def model_name(arm,seed):return arm+'__'+str(seed)


def run(cache_manifest,output,device='cuda:0'):
    started=time.monotonic();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    safe.set_below_normal_priority();bindings={};models={};receipts={};permutation_checks={}
    def guard():
        need(time.monotonic()-started<5400,'Player experiment5400second cap')
        if os.name=='nt':
            class Memory(ctypes.Structure):
                _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(k,ctypes.c_ulonglong) for k in
                    ('total','available','page_total','page_available','virtual_total','virtual_available','extended')]
            m=Memory();m.length=ctypes.sizeof(m)
            need(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)) and m.available>=.75*1024**3,'Player fit RAM floor0.75GiB')
    def bind(path,want=None):
        path=Path(path).resolve(strict=True);safe.reject_quarantined(path);digest=safe.digest(path)
        need(want is None or digest==want,'Changed bound experiment input');bindings[str(path)]=digest
    def status(stage,**values):write(output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-started,**values})
    try:
        code=list(sorted((ROOT/'winprob').glob('*.py')))+[Path(__file__),PROTOCOL,ROOT/'scripts/setback_experiment.py']
        for path in code:bind(path)
        need(safe.digest(PROTOCOL)=='4095f3f2b908e174ed80b438c316bf303973e03e6307e3f4df0c81905e762763','Changed player protocol')
        audit_path=ROOT/'data/reports/wp_players_cache_audit_v1.json';audit=json.loads(audit_path.read_text());bind(audit_path)
        need(audit['state']=='passed' and audit['cache_manifest_sha256']==safe.digest(cache_manifest),'Passed independent player-cache audit required')
        guard();status('loading_verified_player_rows')
        data,cache,parent,old_bounds,admission=load_player_data(cache_manifest,guard)
        for path,digest in old_bounds.items():bind(path,digest)
        fit,cal,dev=(data[role] for role in ('fit','calibration','development'))
        need([len(np.unique(part['game'])) for part in (fit,cal,dev)]==[11468,2866,2861],'Wrong chronological role populations')
        for a,b in ((fit,cal),(fit,dev),(cal,dev)):need(not set(a['game'])&set(b['game']),'Game split overlap')
        dates={role:[r['game_creation_ms'] for r in cache['games'] if r['role']==role] for role in data}
        need(max(dates['fit'])<min(dates['calibration']) and max(dates['calibration'])<min(dates['development']),'Chronological split differs')
        reference_report=json.loads((REFERENCE/'report.json').read_text())
        bind(REFERENCE/'report.json','6e10781c05ce85d706ef87f723c36bdabebbc75613624193bd2144a8fa2fe014')
        bind(REFERENCE/'combat_full.joblib',reference_report['artifacts']['combat_full'])
        bind(REFERENCE/'development_predictions.npz',reference_report['files']['development_predictions.npz'])
        with np.load(REFERENCE/'development_predictions.npz',allow_pickle=False) as saved:
            for key in ('y','game','timestamp_ms','patch','pilot'):need(np.array_equal(dev[key],saved[key]),'Reference population differs')
            reference_probability=saved['combat_full'].copy();reference_raw=saved['combat_full_raw'].copy()
            need(np.array_equal(dev['end_ms'],saved['evaluation_only_end_ms']),'Reference end timing differs')
        with threadpool_limits(limits=1):
            reference=joblib.load(REFERENCE/'combat_full.joblib')
            need(np.array_equal(reference.predict(dev['x']),reference_probability) and np.array_equal(reference.raw(dev['x']),reference_raw),
                 'Strong reference predictions do not reproduce exactly')
        del reference
        need(device=='cuda:0','Declared GPU device required');configure_torch(device)
        free_gpu,total_gpu=torch.cuda.mem_get_info(device);need(free_gpu>=4*1024**3,'GPU startup4GiB available required')
        torch.cuda.reset_peak_memory_stats(device)
        status('fitting_fit_only_normalization');state=fit_normalization(fit['x'],fit['players'],fit['game'])
        np.savez_compressed(output/'normalization.npz',**state)
        permutations={}
        for role,part in data.items():
            guard();part['x'],part['players']=normalize(part['x'],part['players'],state)
            permutations[role]=position_permutations(part['game'],part['timestamp_ms'])
        np.savez_compressed(output/'position_permutations.npz',**permutations)
        write(output/'experiment_lock.json',{'published_before_neural_fit':True,'arms':list(ARMS),'seeds':list(SEEDS),'epochs':30,'batch_size':512,
            'bindings_sha256':bindings,'cache_manifest_path':str(Path(cache_manifest).resolve()),'cache_manifest_sha256':safe.digest(cache_manifest),
            'normalization_sha256':safe.digest(output/'normalization.npz'),'position_permutations_sha256':safe.digest(output/'position_permutations.npz'),
            'fit_games':11468,'calibration_games':2866,'development_games':2861,'fit_sampled_frames':64422,
            'primary_cohort':'full previously exposed development','reference_replay_exact':True,'new_development_predictions_inspected':False})
        gx=torch.as_tensor(fit['x'],device=device);px=torch.as_tensor(fit['players'],device=device)
        shuffled_fit=shuffle_positions(fit['players'],permutations['fit']);sx=torch.as_tensor(shuffled_fit,device=device);del shuffled_fit
        gy=torch.as_tensor(fit['y'].astype(np.float32),device=device);gw=torch.as_tensor(normalized_game_weights(fit['game']),device=device)
        seed_calibration={};held_permutation=np.array([4,2,0,3,1,9,5,8,6,7])
        for arm in ARMS:
            for seed in SEEDS:
                guard();name=model_name(arm,seed);status('training_fixed_model',arm=arm,seed=seed,completed_epochs=0,total_epochs=30)
                model,receipt=train_one(arm,seed,gx,None if arm=='global' else sx if arm=='attention_position_shuffled' else px,gy,gw,
                    guard=guard,status=lambda **kw:status('training_fixed_model',**kw))
                need(receipt['epochs']==30 and receipt['updates']==3780 and receipt['row_visits']==1932660,'Incomplete fixed training schedule')
                # Check record-order invariance on calibration inputs, never on new development.
                probe=cal['players'][:512]
                if arm=='attention_position_shuffled':probe=shuffle_positions(probe,permutations['calibration'][:512])
                with torch.inference_mode():
                    probe_x=torch.as_tensor(cal['x'][:512],device=device);probe_p=torch.as_tensor(probe,device=device)
                    first=torch.sigmoid(model(probe_x,probe_p));second=torch.sigmoid(model(probe_x,probe_p[:,held_permutation]))
                    difference=float((first-second).abs().max())
                need(difference<=1e-5,'Player-order invariance check failed');permutation_checks[name]=difference
                path=output/(name+'.pt');weights={k:v.detach().cpu() for k,v in model.state_dict().items()}
                torch.save({'schema':'wp-player-weights-v1','arm':arm,'seed':seed,'model_state':weights},path)
                models[name]={'file':path.name,'sha256':safe.digest(path),'arm':arm,'seed':seed,'parameters':receipt['parameters']};receipts[name]=receipt
                path.with_suffix('.sha256').write_text(models[name]['sha256']+' '+path.name+'\n',encoding='ascii')
                seed_calibration[name]=predict_raw(model,cal['x'],cal['players'],device,
                    permutations['calibration'] if arm=='attention_position_shuffled' else None,guard)
                del model,weights,probe_x,probe_p,first,second
        del gx,px,sx,gy,gw
        need(len(models)==12,'All12 models required')
        for seed in SEEDS:
            expected=receipts[model_name('global',seed)]['epoch_permutation_sha256']
            need(all(receipts[model_name(arm,seed)]['epoch_permutation_sha256']==expected for arm in ARMS),'Arm batch order differs')
            need(receipts[model_name('attention',seed)]['initial_state_sha256']==receipts[model_name('attention_position_shuffled',seed)]['initial_state_sha256'],
                 'Attention/control initialization differs')
        np.savez_compressed(output/'calibration_seed_predictions.npz',y=cal['y'],game=cal['game'],timestamp_ms=cal['timestamp_ms'],**seed_calibration)
        calibrators={};calibration_hashes={}
        with threadpool_limits(limits=1):
            for arm in ARMS:
                raw=mean_seed_probability([seed_calibration[model_name(arm,seed)] for seed in SEEDS])
                calibrators[arm]=fit_calibrator(raw,cal['y'],cal['game']);path=output/(arm+'_calibrator.joblib')
                joblib.dump(calibrators[arm],path,compress=3);calibration_hashes[path.name]=safe.digest(path)
        write(output/'training_lock.json',{'all_models_complete':True,'new_development_predictions_inspected':False,'comparison_metrics_inspected':False,
            'models':models,'receipts':receipts,'calibrators':calibration_hashes,'permutation_checks':permutation_checks,'bindings_sha256':bindings})
        # Evaluation is intentionally after the complete immutable training lock.
        result=evaluate(output,data,models,receipts,calibrators,permutations,reference_probability,reference_raw,permutation_checks,guard,status,device)
        status('verifying_final_inputs')
        for path,digest in bindings.items():need(safe.digest(path)==digest,'Bound input/code changed during experiment')
        parent_root=Path(cache['parent_manifest_path']).parent
        for j,row in enumerate(parent['games']):
            if j%250==0:guard()
            need(safe.digest(parent_root/row['file'])==row['sha256'],'Parent shard changed during experiment')
        need(not {r['match_id'] for r in cache['games']}&current_reservations(admission),'Reservations changed during experiment')
        snapshot=output/'code_snapshot';snapshot.mkdir();copied={}
        for path in code:
            relative=Path(path).resolve().relative_to(ROOT);dest=snapshot/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest);copied[relative.as_posix()]=safe.digest(dest)
        write(snapshot/'manifest.json',{'files':copied})
        result.update(schema='wp-joint-player-comparison-v1',state='complete',bindings_sha256=bindings,models=models,receipts=receipts,
            calibrators=calibration_hashes,cache_manifest_path=str(Path(cache_manifest).resolve()),permutation_checks=permutation_checks,
            files={name:safe.digest(output/name) for name in ('normalization.npz','position_permutations.npz','experiment_lock.json','training_lock.json',
                'calibration_seed_predictions.npz','development_predictions.npz')},
            runtime={'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'cuda':torch.version.cuda,
                'gpu':torch.cuda.get_device_name(device),'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(device),
                'peak_reserved_gpu_bytes':torch.cuda.max_memory_reserved(device)},elapsed_s=time.monotonic()-started,
            original_reference_predictions_exact=True,new_raw_payload_reads=False,sealed_or_confirmation_payload_reads=False,
            practical_or_confirmatory_benefit_established=False,live_safe=False,causal_claim=False,
            limits=['All2861 primary development games were previously exposed; this is exploratory representation selection.',
                'Intervals condition on the fitted three-seed ensembles; seed variation is not training-set uncertainty.',
                'The spatial shuffle is a matched information control, not a realistic intervention or causal action comparison.',
                'Positive forecast performance alone is not confirmed practical review or decision benefit.'])
        guard();write(output/'report.json',result);status('complete',report=str(output/'report.json'));return result
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


def evaluate(output,data,models,receipts,calibrators,permutations,reference,reference_raw,permutation_checks,guard,status,device):
    dev=data['development'];raw={'combat_full':reference_raw};probability={'combat_full':reference};seeds={}
    for arm in ARMS:
        for seed in SEEDS:
            guard();name=model_name(arm,seed);status('predicting_locked_model',arm=arm,seed=seed)
            artifact=models[name];path=output/artifact['file'];need(safe.digest(path)==artifact['sha256'],'Locked model changed')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            need(saved['schema']=='wp-player-weights-v1' and saved['arm']==arm and saved['seed']==seed,'Wrong model checkpoint identity')
            model=PlayerWinModel(arm);model.load_state_dict(saved['model_state'],strict=True)
            need(state_digest(model)==receipts[name]['final_state_sha256'],'Saved model state differs from training receipt')
            model.to(device);seeds[name]=predict_raw(model,dev['x'],dev['players'],device,
                permutations['development'] if arm=='attention_position_shuffled' else None,guard)
            del model,saved
        raw[arm]=mean_seed_probability([seeds[model_name(arm,seed)] for seed in SEEDS])
        probability[arm]=calibrated(raw[arm],calibrators[arm])
        need(np.isfinite(probability[arm]).all() and np.all((probability[arm]>=0)&(probability[arm]<=1)),'Invalid calibrated ensemble')
    lead=(dev['end_ms']>=0)&(dev['end_ms']-dev['timestamp_ms']>=180000)
    np.savez_compressed(output/'development_predictions.npz',y=dev['y'],game=dev['game'],timestamp_ms=dev['timestamp_ms'],
        patch=dev['patch'],pilot=dev['pilot'],evaluation_only_end_ms=dev['end_ms'],lead_time_mask=lead,
        **probability,**{name+'_raw':p for name,p in raw.items()},**{name+'_seed_raw':p for name,p in seeds.items()})
    pairs=[(arm,'combat_full') for arm in ARMS]+[(arm,'global') for arm in ('deepset','attention')]
    pairs += [('attention','attention_position_shuffled'),('attention','deepset')]
    cohorts={}
    for cohort,take in (('full_development',np.ones(len(dev['y']),dtype=bool)),('original_pilot',dev['pilot']),('old_pilot_complement',~dev['pilot'])):
        guard();status('evaluating_fixed_cohort',cohort=cohort)
        methods={name:summarize(dev['y'][take],p[take],dev['game'][take],2000) for name,p in probability.items()}
        contrasts={a+'_vs_'+b:comparison(dev['y'][take],probability[b][take],probability[a][take],dev['game'][take]) for a,b in pairs}
        ahead=take&lead
        lead_methods={name:summarize(dev['y'][ahead],p[ahead],dev['game'][ahead],2000) for name,p in probability.items()}
        lead_comparisons={a+'_vs_combat_full':comparison(dev['y'][ahead],reference[ahead],probability[a][ahead],dev['game'][ahead])
                          for a in ('deepset','attention')}
        cohorts[cohort]={'games':len(np.unique(dev['game'][take])),'frames':int(take.sum()),'methods':methods,'comparisons':contrasts,
            'lead_time':{'games':len(np.unique(dev['game'][ahead])),'frames':int(ahead.sum()),'methods':lead_methods,'comparisons':lead_comparisons}}
    guard();status('evaluating_raw_seed_stability')
    raw_methods={name:summarize(dev['y'],p,dev['game'],2000) for name,p in raw.items()}
    seed_methods={name:summarize(dev['y'],p,dev['game'],2000) for name,p in seeds.items()}
    baseline_loss=raw_methods['combat_full']['log_loss']['estimate']
    raw_seed_gains={arm:[baseline_loss-seed_methods[model_name(arm,seed)]['log_loss']['estimate'] for seed in SEEDS]
                    for arm in ('deepset','attention')}
    coverage={'fit_games':len(np.unique(data['fit']['game'])),'calibration_games':len(np.unique(data['calibration']['game'])),
        'development_games':len(np.unique(dev['game'])),'fit_sampled_frames':len(data['fit']['y']),'completed_models':len(models),
        'minimum_completed_epochs':min(r['epochs'] for r in receipts.values()),'maximum_completed_epochs':max(r['epochs'] for r in receipts.values()),
        'three_seed_ensembles':all(len([r for r in receipts.values() if r['arm']==arm])==3 for arm in ARMS),
        'inference_checks':len(permutation_checks)==12 and max(permutation_checks.values())<=1e-5,'lead_games':int(len(np.unique(dev['game'][lead])))}
    main=cohorts['full_development'];decision=select_player_candidate(main['comparisons'],main['lead_time']['comparisons'],coverage,raw_seed_gains)
    slices={};groups=[(f'{lo}-{hi}min',(dev['timestamp_ms']>=lo*60000)&(dev['timestamp_ms']<(hi if hi<40 else 41)*60000))
        for lo,hi in ((3,10),(10,20),(20,30),(30,40))]
    groups += [('patch'+str(p),dev['patch']==p) for p in np.unique(dev['patch'])]
    for name,take in groups:
        guard();status('evaluating_declared_slice',slice=name);ng=len(np.unique(dev['game'][take]))
        slices[name]={'games':ng,'status':'not_measured_below30_games'} if ng<30 else {'games':ng,
            'methods':{key:summarize(dev['y'][take],p[take],dev['game'][take],2000) for key,p in probability.items()}}
    return {'decision':decision,'coverage':coverage,'primary_cohort':'full_development','cohorts':cohorts,
        'raw_primary_methods':raw_methods,'raw_primary_seed_methods':seed_methods,'raw_seed_log_loss_reductions':raw_seed_gains,
        'secondary_slices':slices,'inference_batch_size':512,'ensemble_rule':'mean of three raw probabilities, then common weighted Platt calibration'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    result=run(args.cache_manifest,args.output,args.device)
    print(json.dumps({'decision':result['decision'],'coverage':result['coverage'],'elapsed_s':result['elapsed_s']},indent=2))
