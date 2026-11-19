"""Unchanged history recipe with the explicit45-minute runtime addendum v2."""
from __future__ import annotations
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from winprob import admission
from winprob.iql_models import PublicBoardBatch
from winprob.iql_training import RuntimeGuard,adam,file_hash,tree_hash,write_json as _write_json,set_below_normal_priority
from winprob.history_probe_models import HistoryProbeActor,HistoryBatch,main_action_loss,set_history_validation
from winprob.history_probe_training import (HistoryDataset,HistoryTensorStore,enable_validated_history_fastpath,
    shared_epoch_order,total_training_steps,SEED,BATCH_SIZE,EPOCHS,ARMS)
from winprob.history_probe_evaluation import (summarize_logits,evaluate_history_probe,canonical_inventory_signature,
    fit_frequency,score_frequency,serialize_frequency,validate_summary)

PROTOCOL=ROOT/'docs/history_probe_protocol_v1.md'
RUNTIME_ADDENDUM=ROOT/'docs/history_probe_runtime_v2.md'
PRIOR_PREFLIGHT=ROOT/'data/runs/history-probe-20260930-v1/experiment/preflight.json'


def sidecar(path):
    path=Path(path);path.with_suffix('.sha256').write_text(f'{file_hash(path)} {path.name}\n',encoding='ascii')


def write_json(path,payload):
    _write_json(path,payload);sidecar(path)


def checkpoint(path,payload):
    tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(path);sidecar(path);return file_hash(path)


def finish_integrity(dataset,bindings,guard):
    for path,expected in bindings.items():
        if file_hash(path)!=expected:raise ValueError('bound input/code changed: '+path)
    admission.verify_manifest_sources(dataset.admission);guard()


def runtime_projection(step_means,gather_means,inference_means,fold,selected_fold):
    updates_per_arm=EPOCHS*sum(int(np.ceil(np.count_nonzero(fold!=f)/BATCH_SIZE)) for f in (0,1))
    validation_batches=sum(int(np.ceil(np.count_nonzero(selected_fold==f)/BATCH_SIZE)) for f in (0,1))
    train=sum((step_means[arm]+gather_means[arm])*updates_per_arm for arm in ARMS)
    validate=2*sum((inference_means[arm]+gather_means[arm])*validation_batches for arm in ARMS)
    return {'training_updates':updates_per_arm*3,'updates_per_arm':updates_per_arm,
        'selected_validation_batches_per_arm':validation_batches,'validation_forward_passes_per_batch':2,
        'training_s':train,'validation_s':validate,
        'measured_work_with_1_25_factor_s':1.25*(train+validate),'fixed_overhead_s':180.,
        'projected_remaining_s':1.25*(train+validate)+180.,
        'overhead_covers':['fold frequency fitting/scoring','checkpoint and prediction serialization','code snapshot',
                           'paired bootstrap','final full-hash/admission revalidation'],
        'preparation_accounting':'already elapsed load/preload/preflight time is subtracted from the stage deadline'}


def synthetic_inputs(dataset,store,device,arm):
    n=BATCH_SIZE;width=max(1,dataset.inventory_max);aw=store.base.action_indices.shape[1]
    champ=torch.arange(1,11,device=device).expand(n,-1)
    item=torch.arange(1,width+1,device=device).view(1,1,-1).expand(n,10,-1)
    counts=torch.ones(n,10,width,device=device)
    ally=torch.tensor([False]+[True]*4+[False]*5,device=device).expand(n,-1)
    enemy=torch.tensor([False]*5+[True]*5,device=device).expand(n,-1)
    state=PublicBoardBatch(champ,item,counts,torch.zeros(n,10,22,device=device),torch.full((n,10,1),.5,device=device),
                           ally,enemy,torch.zeros(n,37,device=device))
    hi=torch.arange(1,aw+1,device=device).view(1,1,-1).expand(n,2,-1).clone()
    hc=torch.ones(n,2,aw,device=device);hs=torch.zeros(n,2,23,device=device);hs[:,:,0]=1.
    age=torch.tensor([.1,.3],device=device).expand(n,-1).clone();present=torch.ones(n,2,dtype=torch.bool,device=device)
    if arm=='state':hi.zero_();hc.zero_();hs.zero_();age.zero_();present.zero_()
    elif arm=='shuffled':hi=hi.flip(2).contiguous()
    return state,HistoryBatch(hi,hc,hs,age,present),torch.arange(n,device=device)%9333


def preflight(dataset,store,device,guard):
    steps={};inference={};gather={}
    for armno,arm in enumerate(ARMS):
        torch.manual_seed(SEED+9000+armno);torch.cuda.manual_seed_all(SEED+9000+armno)
        model=HistoryProbeActor(dataset.num_champions,dataset.num_items).to(device);optimizer=adam(model.parameters())
        state,history,labels=synthetic_inputs(dataset,store,device,arm)
        with torch.no_grad():model(state,history)
        enable_validated_history_fastpath(model,dataset)
        times=[]
        for i in range(30):
            guard();torch.cuda.synchronize(device);start=time.monotonic()
            optimizer.zero_grad(set_to_none=True);loss=main_action_loss(model(state,history),labels)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize(device)
            if i>=10:times.append(time.monotonic()-start)
        steps[arm]=float(np.mean(times));times=[];model.eval()
        with torch.inference_mode():
            for i in range(8):
                torch.cuda.synchronize(device);start=time.monotonic()
                logits=model(state,history);summarize_logits(logits.cpu().numpy(),labels.cpu().numpy())
                torch.cuda.synchronize(device)
                if i>=3:times.append(time.monotonic()-start)
        inference[arm]=float(np.mean(times))
        del model,optimizer,state,history,labels,logits,loss;gc.collect();torch.cuda.empty_cache()
        rng=np.random.default_rng(SEED+9100+armno);times=[]
        for i in range(25):
            rows=rng.choice(dataset.n,min(BATCH_SIZE,dataset.n),replace=False)
            torch.cuda.synchronize(device);start=time.monotonic();batch=store.batch(rows,i%2,arm);torch.cuda.synchronize(device)
            if i>=5:times.append(time.monotonic()-start)
            del batch
        gather[arm]=float(np.mean(times));guard()
    result=runtime_projection(steps,gather,inference,dataset.arrays['fold'],dataset.arrays['fold'][dataset.selected])
    result.update(synthetic_optimizer_steps=90,optimizer_warmup_per_arm=10,timed_optimizer_steps_per_arm=20,
        step_mean_s=steps,gather_mean_s=gather,inference_summary_mean_s=inference,real_neural_training_performed=False,
        batch_size=BATCH_SIZE,inventory_width=dataset.inventory_max,history_basket_width=int(store.base.action_indices.shape[1]),
        state_cache_preloaded=store.base.preloaded,host_release=store.base.host_release_report)
    return result


def changed_selected_future_labels(store,dataset,fold):
    a=dataset.arrays;selected=dataset.selected[a['fold'][dataset.selected]==fold]
    cut=np.full(len(dataset.games),np.iinfo(np.int64).max,dtype=np.int64)
    cut[a['game_index'][selected]]=a['timestamp_ms'][selected]
    mask=(a['fold']==fold)&(a['timestamp_ms']>=cut[a['game_index']])
    if np.any(a['fold'][mask]!=fold):raise ValueError('mutation touches a donor/training game')
    altered=store.full_ids.clone();gpu_mask=torch.as_tensor(mask,dtype=torch.bool,device=store.device)
    altered[gpu_mask]=(altered[gpu_mask]+1)%13166
    return altered,int(mask.sum())


def score_model(model,store,rows,fold,arm,guard,altered_full_ids):
    values={key:[] for key in ('target_log_probability','top1','top3')};model.eval()
    checked=0
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH_SIZE):
            guard();batch_rows=rows[start:start+BATCH_SIZE];state,history,labels=store.batch(batch_rows,fold,arm)
            logits=model(state,history);original=store.full_ids
            try:
                store.full_ids=altered_full_ids
                changed_state,changed_history,_=store.batch(batch_rows,fold,arm)
            finally:store.full_ids=original
            for name in history.__dataclass_fields__:
                if not torch.equal(getattr(history,name),getattr(changed_history,name)):
                    raise ValueError('current/future target mutation changed history features')
            if not torch.equal(logits,model(changed_state,changed_history)):
                raise ValueError('current/future target mutation changed full9333 logits')
            checked+=len(batch_rows)
            summary=summarize_logits(logits.cpu().numpy(),labels.cpu().numpy())
            for key,value in summary.items():values[key].append(value)
    return {key:np.concatenate(value) for key,value in values.items()},checked


def prediction_receipts(game_ids,selected,fold,labels,kinds,scores):
    for value in scores.values():validate_summary(value,len(game_ids),9333)
    rows=[]
    for i,mid in enumerate(game_ids):
        row={'game_id':mid,'selected_state_index':int(selected[i]),'prediction_fold':int(fold[i]),
             'main_label':int(labels[i]),'target_kind':kinds[i],
             'arms':{arm:{'target_log_probability':float(value['target_log_probability'][i]),
                          'top1':int(value['top1'][i]),'top3':value['top3'][i].tolist()} for arm,value in scores.items()}}
        row['row_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        rows.append(row)
    return rows


def frequency_fold(dataset,train,selected,allowed,guard):
    """Reopen only three immutable buyer-inventory maps after GPU feature release.

    No closed ndarray is reused; the local maps close after owned canonical
    tuples have been consumed. No outcome, terminal, or source JSON is opened.
    """
    specs=dataset.base_manifest['arrays'];maps={}
    try:
        for name in ('inventory_offsets','inventory_item_indices','inventory_item_counts'):
            maps[name]=np.load(dataset.base_directory/specs[name]['file'],mmap_mode='r',allow_pickle=False)
        def signatures(indices):
            for count,index in enumerate(indices):
                if count%4096==0:guard()
                bag=int(index)*10;lo=int(maps['inventory_offsets'][bag]);hi=int(maps['inventory_offsets'][bag+1])
                yield canonical_inventory_signature(maps['inventory_item_indices'][lo:hi],maps['inventory_item_counts'][lo:hi])
        a=dataset.arrays
        model=fit_frequency(dataset.focal_champion_indices[train],signatures(train),a['main_group_id'][train],
            game_ids=(dataset.games[int(i)]['match_id'] for i in a['game_index'][train]),allowed_training_game_ids=allowed)
        scores=score_frequency(model,dataset.focal_champion_indices[selected],signatures(selected),a['main_group_id'][selected])
        payload=serialize_frequency(model);del model
        return scores,payload
    finally:
        for array in maps.values():array._mmap.close()


def run(args):
    if not 1<=args.deadline_seconds<=2700:raise ValueError('model-stage ceiling is2700seconds under runtime addendum v2')
    args.output.mkdir(parents=True,exist_ok=False)
    guard=RuntimeGuard(args.deadline_seconds,.75);set_below_normal_priority();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    def status(stage,**kw):write_json(args.output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-guard.started,**kw})
    def log(value):
        with (args.output/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        status('validating_action_only_history_cache');guard()
        if not torch.cuda.is_available() or torch.device(args.device).type!='cuda':raise RuntimeError('own idle CUDA runtime required; no CPU training fallback')
        device=torch.device(args.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        paths=[Path(__file__),PROTOCOL,RUNTIME_ADDENDUM]+[ROOT/'winprob'/name for name in ('history_probe_models.py','history_probe_training.py',
            'history_probe_evaluation.py','history_probe_data.py','iql_training.py','iql_models.py','iql_features.py',
            'ipo_residual.py','full_action_catalogue.py','admission.py','iql_dataset_contract.py','trajectory_contract.py')]
        initial_code={str(path.resolve()):file_hash(path) for path in paths}
        prior=json.loads(PRIOR_PREFLIGHT.read_text(encoding='utf-8'))
        if (prior.get('projection_gate_passed') is not False or prior.get('real_neural_training_performed') is not False
                or PRIOR_PREFLIGHT.with_suffix('.sha256').read_text().split()!=[file_hash(PRIOR_PREFLIGHT),PRIOR_PREFLIGHT.name]):
            raise ValueError('Runtime addendum requires the preserved pre-fit failed resource receipt')
        initial_code[str(PRIOR_PREFLIGHT.resolve())]=file_hash(PRIOR_PREFLIGHT)
        guard.bind(initial_code);guard()
        dataset=HistoryDataset(args.history_manifest.parent,PROTOCOL)
        if args.history_manifest.resolve()!=(dataset.directory/'manifest.json').resolve():raise ValueError('expected the bound history manifest')
        for path,expected in initial_code.items():
            if path in dataset.bindings and dataset.bindings[path]!=expected:raise ValueError('entry code/source binding conflict')
        bindings={**dataset.bindings,**initial_code}
        guard.bind(bindings);finish_integrity(dataset,bindings,guard)
        write_json(args.output/'input_audit.json',{'dataset':dataset.audit,'bindings_sha256':bindings,
            'reward_arrays_opened':False,'history_unit':'previous2realpurchasevisits; syntheticSAVEremovedbeforeadjacency'})
        status('loading_validated_cuda_inputs');store=HistoryTensorStore(dataset,device,guard)
        status('synthetic_preflight');timing=preflight(dataset,store,device,guard)
        timing['remaining_time_before_fit_s']=args.deadline_seconds-(time.monotonic()-guard.started)
        timing['projection_gate_passed']=timing['projected_remaining_s']<=.85*timing['remaining_time_before_fit_s']
        write_json(args.output/'preflight.json',timing)
        if not timing['projection_gate_passed']:raise RuntimeError('complete20epoch/3arm/2foldrecipe fails fixed preflight budget')
        if not args.execute:
            finish_integrity(dataset,bindings,guard);status('preflight_complete_no_real_fit');return
        a=dataset.arrays;selected=dataset.selected;size=len(selected)
        scores={arm:{'target_log_probability':np.full(size,np.nan),'top1':np.full(size,-1,dtype=np.int64),
                     'top3':np.full((size,3),-1,dtype=np.int64)} for arm in (*ARMS,'frequency')}
        checkpoints={};initial={};orders={};train_ids={};step_counts={};initial_receipts={};optimizer_initial={}
        mutation_counts=dict.fromkeys(ARMS,0);mutation_changed_rows={}
        for fold in (0,1):
            train=np.flatnonzero(a['fold']!=fold);positions=np.flatnonzero(a['fold'][selected]==fold);test=selected[positions]
            allowed=sorted(g['match_id'] for g in dataset.games if g['fold']!=fold);train_ids[str(fold)]=allowed
            seed=SEED+100+fold;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            template=HistoryProbeActor(dataset.num_champions,dataset.num_items);initial_state=template.state_dict()
            initial[str(fold)]=tree_hash(initial_state)
            altered_full_ids,changed_count=changed_selected_future_labels(store,dataset,fold)
            mutation_changed_rows[str(fold)]=changed_count
            for arm in ARMS:
                guard();model=HistoryProbeActor(dataset.num_champions,dataset.num_items)
                model.load_state_dict(initial_state,strict=True)
                if tree_hash(model.state_dict())!=initial[str(fold)]:raise ValueError('arm initialization differs')
                initial_receipts[f'{fold}:{arm}']=tree_hash(model.state_dict())
                model=model.to(device);enable_validated_history_fastpath(model,dataset);optimizer=adam(model.parameters());steps=0
                optimizer_initial[f'{fold}:{arm}']=tree_hash(optimizer.state_dict())
                if any(optimizer_initial[f'{fold}:{other}']!=optimizer_initial[f'{fold}:{arm}'] for other in ARMS if f'{fold}:{other}' in optimizer_initial):
                    raise ValueError('fresh Adam initial states differ across arms')
                for epoch in range(EPOCHS):
                    status('training',fold=fold,arm=arm,epoch=epoch+1,required_epochs=EPOCHS)
                    order=shared_epoch_order(train,fold,epoch);key=f'{fold}:{epoch}';sha=hashlib.sha256(order.tobytes()).hexdigest()
                    if key in orders and orders[key]!=sha:raise ValueError('arm batch order changed')
                    orders[key]=sha;total=0.;maxnorm=0.;start=time.monotonic();model.train()
                    for offset in range(0,len(order),BATCH_SIZE):
                        if offset%(BATCH_SIZE*20)==0:guard()
                        batch=order[offset:offset+BATCH_SIZE];state,history,y=store.batch(batch,fold,arm)
                        optimizer.zero_grad(set_to_none=True);loss=main_action_loss(model(state,history),y);loss.backward()
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
                        steps+=1;total+=float(loss.detach())*len(batch);maxnorm=max(maxnorm,float(norm))
                    log({'fold':fold,'arm':arm,'epoch':epoch+1,'transitions':len(train),'updates':steps,
                         'mean_training_nll':total/len(train),'maximum_gradient_before_clip':maxnorm,'elapsed_s':time.monotonic()-start})
                expected=EPOCHS*int(np.ceil(len(train)/BATCH_SIZE))
                if steps!=expected:raise ValueError('incomplete model epochs')
                if {int(state['step']) for state in optimizer.state.values()}!={expected}:raise ValueError('Adam step accounting differs')
                name=f'actor_{arm}_fold{fold}.pt';checkpoints[name]=checkpoint(args.output/name,{
                    'actor':model.state_dict(),'optimizer':optimizer.state_dict(),'epochs':EPOCHS,'steps':steps,
                    'seed':seed,'arm':arm,'scored_fold':fold,'training_game_ids':allowed,'initial_state_sha256':initial[str(fold)]})
                step_counts[f'{fold}:{arm}']=steps
                # Checkpoint frozen before any selected target score is computed.
                compact,checked=score_model(model,store,test,fold,arm,guard,altered_full_ids)
                mutation_counts[arm]+=checked
                for key,value in compact.items():scores[arm][key][positions]=value
                del model,optimizer,state,history,y,loss,compact;gc.collect();torch.cuda.empty_cache()
            del template,initial_state,altered_full_ids
        if sum(step_counts.values())!=total_training_steps(a['fold']):raise ValueError('matched recipe update count changed')
        del store;gc.collect();torch.cuda.empty_cache();guard()
        for fold in (0,1):
            status('fitting_fold_frequency_control',fold=fold)
            train=np.flatnonzero(a['fold']!=fold);positions=np.flatnonzero(a['fold'][selected]==fold)
            compact,payload=frequency_fold(dataset,train,selected[positions],train_ids[str(fold)],guard)
            for key,value in compact.items():scores['frequency'][key][positions]=value
            name=f'frequency_fold{fold}.json.gz'
            with gzip.open(args.output/name,'wt',encoding='utf-8') as stream:json.dump(payload,stream,separators=(',',':'))
            checkpoints[name]=file_hash(args.output/name);del payload,compact;gc.collect();guard()
        finish_integrity(dataset,bindings,guard)
        if mutation_counts!=dict.fromkeys(ARMS,size):raise ValueError('selected mutation coverage incomplete')
        lock={'all_neural_epochs_complete':True,'epochs_per_arm_fold':20,'initial_state_hashes':initial,
            'initial_parameter_receipts_by_arm':initial_receipts,'initial_adam_receipts_by_arm':optimizer_initial,
            'shared_epoch_order_hashes':orders,'updates_per_arm_fold':step_counts,'training_game_ids_by_scored_fold':train_ids,
            'checkpoints':checkpoints,'bindings_sha256':bindings,'initial_code_hashes_before_input_reads':initial_code,
            'selected_full_logit_mutation_checked_by_arm':mutation_counts,'mutation_failures':0,
            'mutation_changed_label_rows_by_scored_fold':mutation_changed_rows,'comparison_metrics_inspected':False}
        write_json(args.output/'training_lock.json',lock)
        game_ids=[dataset.games[int(i)]['match_id'] for i in a['game_index'][selected]]
        labels=np.array(a['main_group_id'][selected]);kinds=[]
        for action_id in a['full_action_id'][selected]:
            action_id=int(action_id);basket=dataset.catalogue['full_actions'][action_id]
            if action_id==0:kind='observed_no_buy_death'
            elif not any(item not in (2055,772043) for item,count in basket):kind='ward_only_purchase'
            elif any(item in (2055,772043) for item,count in basket):kind='mixed_purchase_with_wards'
            else:kind='other_purchase'
            kinds.append(kind)
        prediction={'game_ids':np.asarray(game_ids),'labels':labels,'selected_state_indices':np.array(selected),
            'prediction_fold':np.array(a['fold'][selected]),'target_kind':np.asarray(kinds)}
        for arm,value in scores.items():
            for key,array in value.items():prediction[f'{arm}_{key}']=array
        receipts=prediction_receipts(game_ids,selected,a['fold'][selected],labels,kinds,scores)
        prediction['row_sha256']=np.asarray([row['row_sha256'] for row in receipts])
        with (args.output/'prediction_rows.jsonl').open('w',encoding='utf-8') as stream:
            for row in receipts:stream.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        sidecar(args.output/'prediction_rows.jsonl')
        np.savez_compressed(args.output/'predictions.npz',**prediction)
        sidecar(args.output/'predictions.npz')
        status('evaluating_frozen_oof_predictions')
        result=evaluate_history_probe(game_ids,labels,scores,selected_state_indices=selected,prediction_fold=a['fold'][selected],
            training_game_ids_by_fold=train_ids,full_selection_game_ids=[e['match_id'] for e in dataset.selection],
            selection_statuses=[e['status'] for e in dataset.selection],target_kinds=kinds)
        snapshot=args.output/'code_snapshot';snapshot.mkdir();saved={}
        for path in paths:
            relative=path.resolve().relative_to(ROOT);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination);saved[relative.as_posix()]=file_hash(destination)
        write_json(snapshot/'manifest.json',{'files':saved,'source_code_modified':False,'raw_data_copied':False})
        finish_integrity(dataset,bindings,guard)
        proof={'upstream_dataset_gates':dataset.manifest['gates']['passed'],'complete_recipe':sum(step_counts.values())==total_training_steps(a['fold']),
            'same_initial_parameters':all(initial_receipts[f'{f}:{arm}']==initial[str(f)] for f in (0,1) for arm in ARMS),
            'same_fresh_adam':all(optimizer_initial[f'{f}:{arm}']==optimizer_initial[f'{f}:state'] for f in (0,1) for arm in ARMS),
            'matched_epoch_orders':len(orders)==40,'full_selected_mutation_coverage':mutation_counts==dict.fromkeys(ARMS,size),
            'zero_mutation_failures':True,'entry_and_final_code_bindings_valid':True,'whole_game_oof_membership_valid':True,
            'prediction_receipts_complete':len(receipts)==size,'no_outcome_features_or_reward_arrays':True}
        result.update(schema='history-probe-experiment-v1',state='complete',preflight=timing,dataset_audit=dataset.audit,
            model_stage_limit_s=2700,runtime_addendum_path=str(RUNTIME_ADDENDUM),runtime_addendum_sha256=file_hash(RUNTIME_ADDENDUM),
            prior_preflight_path=str(PRIOR_PREFLIGHT),prior_preflight_sha256=file_hash(PRIOR_PREFLIGHT),
            checkpoints=checkpoints,bindings_sha256=bindings,predictions_sha256=file_hash(args.output/'predictions.npz'),
            prediction_rows_sha256=file_hash(args.output/'prediction_rows.jsonl'),proof_gates=proof,
            useful_internal_signal=bool(result['metric_gate_passed'] and all(proof.values())),
            training_lock_sha256=file_hash(args.output/'training_lock.json'),training_performed=True,
            reward_arrays_opened=False,raw_payload_reads=False,test_payload_reads=False,confirmation_payload_reads=False,
            outcomes_analyzed=False,legality_masking=False,
            policy_benefit_established=False,causal_claim=False,elapsed_s=time.monotonic()-guard.started,
            runtime={'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device)},
            host_memory=guard.memory_report(),peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)))
        write_json(args.output/'report.json',result);status('complete',report=str((args.output/'report.json').resolve()))
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-manifest',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--deadline-seconds',type=int,default=2700)
    parser.add_argument('--execute',action='store_true');run(parser.parse_args())


if __name__=='__main__':main()
