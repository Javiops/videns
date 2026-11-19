"""Fixed future-goal auxiliary supervision; no oracle goal inputs or reward learning."""
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
from winprob.iql_models import PublicBoardBatch
from winprob.iql_training import RuntimeGuard,adam,file_hash,tree_hash,set_below_normal_priority
from winprob.future_goal_models import FutureGoalActor,future_supervision_loss
from winprob.future_goal_training import (FutureGoalDataset,FutureGoalStore,enable_validated_future_fastpath,
    verify_optimizer_steps,shared_epoch_order,ARMS,EPOCHS,BATCH_SIZE,SEED)
from winprob.future_goal_evaluation import evaluate_future_goal_probe,summarize_logits
from train_history_probe import write_json,sidecar,checkpoint,finish_integrity,frequency_fold,prediction_receipts

PROTOCOL=ROOT/'docs/future_goal_protocol_v1.md'


def synthetic_inputs(dataset,device):
    n=BATCH_SIZE;width=max(1,dataset.inventory_max)
    champ=torch.arange(1,11,device=device).expand(n,-1)
    item=torch.arange(1,width+1,device=device).view(1,1,-1).expand(n,10,-1)
    count=torch.ones(n,10,width,device=device)
    ally=torch.tensor([False]+[True]*4+[False]*5,device=device).expand(n,-1)
    enemy=torch.tensor([False]*5+[True]*5,device=device).expand(n,-1)
    state=PublicBoardBatch(champ,item,count,torch.zeros(n,10,22,device=device),torch.full((n,10,1),.5,device=device),
                          ally,enemy,torch.zeros(n,37,device=device))
    share=float(np.mean(dataset.arrays['patch_index']==1));second=max(1,min(n-1,round(n*share)))
    patch=torch.tensor([0]*(n-second)+[1]*second,dtype=torch.long,device=device)
    return state,patch,torch.arange(n,device=device)%9333,torch.arange(n,device=device)%258


def runtime_projection(step,gather,inference,fold,selected_fold):
    per_arm=EPOCHS*sum(int(np.ceil(np.count_nonzero(fold!=f)/BATCH_SIZE)) for f in (0,1))
    evaluation_batches=sum(int(np.ceil(np.count_nonzero(selected_fold==f)/BATCH_SIZE)) for f in (0,1))
    training=sum((step[arm]+gather[arm])*per_arm for arm in ARMS)
    evaluation=2*sum((inference[arm]+gather[arm])*evaluation_batches for arm in ARMS)
    return {'training_updates':3*per_arm,'updates_per_arm':per_arm,'validation_batches_per_arm':evaluation_batches,
        'validation_forwards_per_batch':2,'projected_training_s':training,'projected_validation_s':evaluation,
        'work_margin':1.25,'fixed_overhead_s':180.,'projected_remaining_s':1.25*(training+evaluation)+180.,
        'overhead_covers':['frequency fitting/scoring','checkpoints','goal diagnostic inference','bootstrap','hash checks','code snapshot'],
        'preparation_accounting':'already elapsed cache validation/preload/preflight is removed from remaining3600s'}


def auxiliary_gradient_proof(model,state,patch,goal_labels):
    model.zero_grad(set_to_none=True)
    result=model(state,patch,compute_goal=True)
    loss=torch.nn.functional.cross_entropy(result['goal_logits'],goal_labels);loss.backward()
    def nonzero(parameters):
        return any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad!=0).any()) for p in parameters)
    proof={'served_state_encoder':nonzero(model.state.parameters()),
           'shared_item_embedding':nonzero([model.state.inventory.embedding.weight]),
           'served_basket_encoder':nonzero(model.basket.parameters()),
           'main_free_head_not_directly_updated':all(p.grad is None for p in model.main_free.parameters())}
    if not all(proof.values()):raise ValueError('auxiliary gradient does not reach the declared served representation')
    model.zero_grad(set_to_none=True)
    return proof


def preflight(dataset,store,tables,device,guard):
    step={};gather={};inference={};proof=None
    for index,arm in enumerate(ARMS):
        torch.manual_seed(SEED+9000+index);torch.cuda.manual_seed_all(SEED+9000+index)
        model=FutureGoalActor(dataset.num_champions,dataset.num_items,tables).to(device)
        state,patch,labels,goals=synthetic_inputs(dataset,device)
        with torch.no_grad():model(state,patch,compute_goal=arm!='mainonly')
        if arm=='future':proof=auxiliary_gradient_proof(model,state,patch,goals)
        enable_validated_future_fastpath(model,dataset);optimizer=adam(model.parameters());times=[]
        for iteration in range(30):
            guard();torch.cuda.synchronize(device);start=time.monotonic();optimizer.zero_grad(set_to_none=True)
            outputs=model(state,patch,compute_goal=arm!='mainonly')
            loss,main,aux=future_supervision_loss(outputs,labels,None if arm=='mainonly' else goals,use_goal=arm!='mainonly')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
            float(loss.detach());float(norm)
            torch.cuda.synchronize(device)
            if iteration>=10:times.append(time.monotonic()-start)
        step[arm]=float(np.mean(times));times=[];model.eval()
        with torch.inference_mode():
            for iteration in range(8):
                torch.cuda.synchronize(device);start=time.monotonic()
                output=model(state,patch)['main_logits'];summarize_logits(output.cpu().numpy(),labels.cpu().numpy())
                torch.cuda.synchronize(device)
                if iteration>=3:times.append(time.monotonic()-start)
        inference[arm]=float(np.mean(times))
        patch_counts=torch.bincount(patch,minlength=3).cpu().tolist()
        del model,optimizer,state,patch,labels,goals,outputs,output,loss,main,aux,norm;gc.collect();torch.cuda.empty_cache()
        rng=np.random.default_rng(SEED+9200+index);training=np.flatnonzero(dataset.arrays['fold']!=0);times=[]
        for iteration in range(25):
            rows=rng.choice(training,min(BATCH_SIZE,len(training)),replace=False)
            torch.cuda.synchronize(device);start=time.monotonic();state,patch=store.inputs(rows);labels=store.main_labels(rows)
            goals=None if arm=='mainonly' else store.training_goal_labels(rows,0,arm)
            torch.cuda.synchronize(device)
            if iteration>=5:times.append(time.monotonic()-start)
            del state,patch,labels,goals
        gather[arm]=float(np.mean(times));guard()
    result=runtime_projection(step,gather,inference,dataset.arrays['fold'],dataset.arrays['fold'][dataset.selected])
    result.update(synthetic_only=True,actual_neural_training_performed=False,synthetic_optimizer_steps=90,
        warmup_steps_per_arm=10,timed_steps_per_arm=20,batch_size=BATCH_SIZE,mixed_native_patch_counts=patch_counts,
        optimizer_step_mean_s=step,gather_mean_s=gather,inference_summary_mean_s=inference,
        auxiliary_gradient_proof=proof,host_release=store.base.host_release_report)
    return result


def changed_heldout_targets(store,dataset,heldout):
    a=dataset.arrays;selected=dataset.selected[a['fold'][dataset.selected]==heldout]
    thresholds=np.full(len(dataset.games),np.iinfo(np.int64).max,dtype=np.int64)
    thresholds[a['game_index'][selected]]=a['timestamp_ms'][selected]
    changed=(a['fold']==heldout)&(a['timestamp_ms']>=thresholds[a['game_index']])
    main=store.main_ids.clone();mask=torch.as_tensor(changed,dtype=torch.bool,device=store.device)
    main[mask]=(main[mask]+1)%9333
    real=np.array(store.real_goals,copy=True);real[changed]=(real[changed]+1)%258
    shuffled=np.array(store.shuffled_goals,copy=True);shuffled[:,changed]=0
    return main,real,shuffled,int(changed.sum())


def score_model(model,store,rows,guard,altered):
    values={key:[] for key in ('target_log_probability','top1','top3')};checked=0;model.eval()
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH_SIZE):
            guard();batch=rows[start:start+BATCH_SIZE];state,patch=store.inputs(batch);labels=store.main_labels(batch)
            logits=model(state,patch)['main_logits'];old=(store.main_ids,store.real_goals,store.shuffled_goals)
            try:
                store.main_ids,store.real_goals,store.shuffled_goals=altered[:3]
                changed_state,changed_patch=store.inputs(batch)
                changed_logits=model(changed_state,changed_patch)['main_logits']
            finally:store.main_ids,store.real_goals,store.shuffled_goals=old
            if not torch.equal(logits,changed_logits):raise ValueError('held-out current/future/goal labels changed main logits')
            checked+=len(batch);summary=summarize_logits(logits.cpu().numpy(),labels.cpu().numpy())
            for key,value in summary.items():values[key].append(value)
    return {key:np.concatenate(value) for key,value in values.items()},checked


def goal_diagnostics(model,store,rows,guard):
    values={key:[] for key in ('target_log_probability','top1','top3')}
    with torch.inference_mode():
        for start in range(0,len(rows),BATCH_SIZE):
            guard();batch=rows[start:start+BATCH_SIZE];state,patch=store.inputs(batch)
            output=model(state,patch,compute_goal=True)['goal_logits']
            # Observed goals are used ONLY to score this frozen auxiliary head.
            labels=np.asarray(store.real_goals[batch]);summary=summarize_logits(output.cpu().numpy(),labels)
            for key,value in summary.items():values[key].append(value)
    return {key:np.concatenate(value) for key,value in values.items()}


def run(args):
    if not 1<=args.deadline_seconds<=3600:raise ValueError('future model stage hard cap is3600seconds')
    args.output.mkdir(parents=True,exist_ok=False);guard=RuntimeGuard(args.deadline_seconds,.75)
    set_below_normal_priority();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    def status(stage,**kw):write_json(args.output/'status.json',{'stage':stage,'elapsed_s':time.monotonic()-guard.started,**kw})
    def log(value):
        with (args.output/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')
    try:
        paths=[Path(__file__),PROTOCOL,ROOT/'scripts/train_history_probe.py']+[ROOT/'winprob'/name for name in (
            'future_goal_models.py','future_goal_training.py','future_goal_data.py','future_goal_evaluation.py',
            'history_probe_training.py','history_probe_models.py','history_probe_evaluation.py','history_probe_data.py',
            'iql_training.py','iql_models.py','iql_features.py','ipo_residual.py','admission.py','full_action_catalogue.py','iql_dataset_contract.py','trajectory_contract.py')]
        entry_code={str(path.resolve()):file_hash(path) for path in paths};guard.bind(entry_code);guard()
        status('validating_frozen_future_supervision_cache');dataset=FutureGoalDataset(args.future_manifest.parent,PROTOCOL)
        if args.future_manifest.resolve()!=(dataset.future_directory/'manifest.json').resolve():raise ValueError('expected bound future manifest')
        for path,expected in entry_code.items():
            if path in dataset.bindings and dataset.bindings[path]!=expected:raise ValueError('entry/source code digest conflict')
        bindings={**dataset.bindings,**entry_code};guard.bind(bindings);finish_integrity(dataset,bindings,guard)
        write_json(args.output/'input_audit.json',{'dataset':dataset.audit,'goal_control':dataset.goal_control_audit,
            'independent_cache_audit':dataset.independent_cache_audit,
            'native_composition_fallbacks':dataset.native_fallbacks,'bindings_sha256':bindings,'reward_arrays_opened':False,
            'terminal_timing_arrays_loaded':False,'timing_ancestor_bytes_hashed':dataset.timing_ancestor_bytes_hashed})
        if not torch.cuda.is_available() or torch.device(args.device).type!='cuda':raise RuntimeError('idle CUDA required; no CPU training fallback')
        device=torch.device(args.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tables=dataset.cpu_tables();status('loading_validated_cuda_states');store=FutureGoalStore(dataset,device,guard)
        status('mandatory_synthetic_preflight');timing=preflight(dataset,store,tables,device,guard)
        timing['remaining_time_before_fit_s']=args.deadline_seconds-(time.monotonic()-guard.started)
        timing['projection_gate_passed']=timing['projected_remaining_s']<=.85*timing['remaining_time_before_fit_s']
        write_json(args.output/'preflight.json',timing)
        if not timing['projection_gate_passed']:raise RuntimeError('complete fixed future-supervision recipe fails resource projection')
        if not args.execute:finish_integrity(dataset,bindings,guard);status('preflight_complete_no_real_fit');return
        store.goal_label_gathers=dict.fromkeys(ARMS,0)
        a=dataset.arrays;selected=dataset.selected;n=len(selected)
        scores={arm:{'target_log_probability':np.full(n,np.nan),'top1':np.full(n,-1,dtype=np.int64),'top3':np.full((n,3),-1,dtype=np.int64)} for arm in (*ARMS,'frequency')}
        auxiliary={arm:{'target_log_probability':np.full(n,np.nan),'top1':np.full(n,-1,dtype=np.int64),'top3':np.full((n,3),-1,dtype=np.int64)} for arm in ('future','shuffled')}
        initial={};init_receipts={};adam_initial={};orders={};checkpoints={};steps_by_arm={};parameter_steps={};training_ids={}
        mutation=dict.fromkeys(ARMS,0);changed_rows={}
        for heldout in (0,1):
            train=np.flatnonzero(a['fold']!=heldout);positions=np.flatnonzero(a['fold'][selected]==heldout);test=selected[positions]
            allowed=sorted(row['match_id'] for row in dataset.games if row['fold']!=heldout);training_ids[str(heldout)]=allowed
            seed=SEED+100+heldout;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            template=FutureGoalActor(dataset.num_champions,dataset.num_items,tables);initial_state=template.state_dict();initial[str(heldout)]=tree_hash(initial_state)
            altered=changed_heldout_targets(store,dataset,heldout);changed_rows[str(heldout)]=altered[3]
            for arm in ARMS:
                model=FutureGoalActor(dataset.num_champions,dataset.num_items,tables);model.load_state_dict(initial_state,strict=True)
                key=f'{heldout}:{arm}';init_receipts[key]=tree_hash(model.state_dict())
                if init_receipts[key]!=initial[str(heldout)]:raise ValueError('unmatched model initialization')
                model=model.to(device);enable_validated_future_fastpath(model,dataset);optimizer=adam(model.parameters())
                adam_initial[key]=tree_hash(optimizer.state_dict())
                if any(adam_initial[f'{heldout}:{other}']!=adam_initial[key] for other in ARMS if f'{heldout}:{other}' in adam_initial):
                    raise ValueError('fresh Adam initialization differs')
                steps=0
                for epoch in range(EPOCHS):
                    status('training',fold=heldout,arm=arm,epoch=epoch+1,required_epochs=EPOCHS)
                    order=shared_epoch_order(train,heldout,epoch);order_key=f'{heldout}:{epoch}';sha=hashlib.sha256(order.tobytes()).hexdigest()
                    if order_key in orders and orders[order_key]!=sha:raise ValueError('arm epoch permutation differs')
                    orders[order_key]=sha;main_total=0.;goal_total=0.;maximum_gradient=0.;start=time.monotonic();model.train()
                    for offset in range(0,len(order),BATCH_SIZE):
                        if offset%(20*BATCH_SIZE)==0:guard()
                        rows=order[offset:offset+BATCH_SIZE];state,patch=store.inputs(rows);labels=store.main_labels(rows)
                        goals=None if arm=='mainonly' else store.training_goal_labels(rows,heldout,arm)
                        optimizer.zero_grad(set_to_none=True);outputs=model(state,patch,compute_goal=arm!='mainonly')
                        loss,main_loss,goal_loss=future_supervision_loss(outputs,labels,goals,use_goal=arm!='mainonly')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
                        steps+=1;main_total+=float(main_loss.detach())*len(rows)
                        if goal_loss is not None:goal_total+=float(goal_loss.detach())*len(rows)
                        maximum_gradient=max(maximum_gradient,float(norm))
                    log({'fold':heldout,'arm':arm,'epoch':epoch+1,'transitions':len(train),'updates':steps,
                        'main_training_nll':main_total/len(train),'goal_training_nll':None if arm=='mainonly' else goal_total/len(train),
                        'lambda':0 if arm=='mainonly' else 1,'maximum_gradient_before_clip':maximum_gradient,'elapsed_s':time.monotonic()-start})
                expected=EPOCHS*int(np.ceil(len(train)/BATCH_SIZE))
                if steps!=expected:raise ValueError('incomplete fixed epochs')
                parameter_steps[key]=verify_optimizer_steps(model,optimizer,expected,arm);steps_by_arm[key]=steps
                name=f'actor_{arm}_fold{heldout}.pt';checkpoints[name]=checkpoint(args.output/name,{'actor':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'epochs':EPOCHS,'steps':steps,'seed':seed,'arm':arm,'scored_fold':heldout,
                    'training_game_ids':allowed,'initial_state_sha256':initial[str(heldout)],'goal_input_at_inference':False})
                compact,checked=score_model(model,store,test,guard,altered);mutation[arm]+=checked
                for field,value in compact.items():scores[arm][field][positions]=value
                if arm!='mainonly':
                    diag=goal_diagnostics(model,store,test,guard)
                    for field,value in diag.items():auxiliary[arm][field][positions]=value
                    del diag
                del model,optimizer,state,patch,labels,goals,outputs,loss,main_loss,goal_loss,norm,compact;gc.collect();torch.cuda.empty_cache()
            del template,initial_state,altered
        expected_updates=EPOCHS*3*sum(int(np.ceil(np.count_nonzero(a['fold']!=f)/BATCH_SIZE)) for f in (0,1))
        if sum(steps_by_arm.values())!=expected_updates or mutation!=dict.fromkeys(ARMS,n):raise ValueError('recipe/mutation coverage incomplete')
        label_gathers=dict(store.goal_label_gathers)
        if label_gathers!={'mainonly':0,'future':EPOCHS*dataset.n,'shuffled':EPOCHS*dataset.n}:
            raise ValueError('training goal target access does not match the complete fixed recipe')
        del store;gc.collect();torch.cuda.empty_cache();guard()
        for heldout in (0,1):
            status('frequency_control',fold=heldout);train=np.flatnonzero(a['fold']!=heldout);positions=np.flatnonzero(a['fold'][selected]==heldout)
            compact,payload=frequency_fold(dataset,train,selected[positions],training_ids[str(heldout)],guard)
            for field,value in compact.items():scores['frequency'][field][positions]=value
            name=f'frequency_fold{heldout}.json.gz'
            with gzip.open(args.output/name,'wt',encoding='utf-8') as stream:json.dump(payload,stream,separators=(',',':'))
            checkpoints[name]=file_hash(args.output/name);sidecar(args.output/name);del compact,payload;gc.collect();guard()
        finish_integrity(dataset,bindings,guard)
        lock={'epochs_per_arm_fold':EPOCHS,'initialization':'fresh per-fold PyTorch default Linears; state item/champion embeddings zero; goal bias zero; exact template copied across arms',
            'initial_state_hashes':initial,'initial_parameter_receipts_by_arm':init_receipts,'initial_adam_receipts_by_arm':adam_initial,
            'shared_epoch_order_hashes':orders,'updates_per_arm_fold':steps_by_arm,'parameter_step_receipts':parameter_steps,
            'training_game_ids_by_scored_fold':training_ids,'checkpoints':checkpoints,'bindings_sha256':bindings,
            'entry_code_hashes_before_input_reads':entry_code,'selected_main_mutation_checked_by_arm':mutation,'mutation_failures':0,
            'changed_heldout_target_rows':changed_rows,'goal_label_gathers_by_arm':label_gathers,'comparison_metrics_inspected':False}
        write_json(args.output/'training_lock.json',lock)
        game_ids=[dataset.games[int(i)]['match_id'] for i in a['game_index'][selected]];labels=np.array(a['main_group_id'][selected]);kinds=[]
        for full_id in a['full_action_id'][selected]:
            full_id=int(full_id);basket=dataset.catalogue['full_actions'][full_id]
            if full_id==0:kind='observed_no_buy_death'
            elif not any(item not in (2055,772043) for item,count in basket):kind='ward_only_purchase'
            elif any(item in (2055,772043) for item,count in basket):kind='mixed_purchase_with_wards'
            else:kind='other_purchase'
            kinds.append(kind)
        receipts=prediction_receipts(game_ids,selected,a['fold'][selected],labels,kinds,scores)
        prediction={'game_ids':np.asarray(game_ids),'labels':labels,'selected_state_indices':np.array(selected),
            'prediction_fold':np.array(a['fold'][selected]),'target_kind':np.asarray(kinds),
            'row_sha256':np.asarray([row['row_sha256'] for row in receipts])}
        for arm,value in scores.items():
            for field,array in value.items():prediction[f'{arm}_{field}']=array
        with (args.output/'prediction_rows.jsonl').open('w',encoding='utf-8') as stream:
            for row in receipts:stream.write(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        sidecar(args.output/'prediction_rows.jsonl');np.savez_compressed(args.output/'predictions.npz',**prediction);sidecar(args.output/'predictions.npz')
        auxiliary_file={'real_goal_id':np.array(dataset.future_arrays['real_goal_id'][selected])}
        for arm,value in auxiliary.items():
            for field,array in value.items():auxiliary_file[f'{arm}_{field}']=array
        np.savez_compressed(args.output/'auxiliary_diagnostics.npz',**auxiliary_file);sidecar(args.output/'auxiliary_diagnostics.npz')
        status('evaluating_frozen_oof_main_predictions')
        result=evaluate_future_goal_probe(game_ids,labels,scores,selected_state_indices=selected,prediction_fold=a['fold'][selected],
            training_game_ids_by_fold=training_ids,full_selection_game_ids=[e['match_id'] for e in dataset.selection],
            selection_statuses=[e['status'] for e in dataset.selection],target_kinds=kinds)
        snapshot=args.output/'code_snapshot';snapshot.mkdir();saved={}
        for path in paths:
            relative=path.resolve().relative_to(ROOT);destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,destination);saved[relative.as_posix()]=file_hash(destination)
        write_json(snapshot/'manifest.json',{'files':saved,'raw_data_copied':False,'old_source_code_modified':False})
        finish_integrity(dataset,bindings,guard)
        proof={'upstream_gates':dataset.manifest['gates']['passed'],'all_epochs_updates':sum(steps_by_arm.values())==expected_updates,
            'same_initial_parameters':all(init_receipts[f'{f}:{arm}']==initial[str(f)] for f in (0,1) for arm in ARMS),
            'same_initial_adam':all(adam_initial[f'{f}:{arm}']==adam_initial[f'{f}:mainonly'] for f in (0,1) for arm in ARMS),
            'same_epoch_orders':len(orders)==40,'active_and_inactive_parameter_steps_verified':len(parameter_steps)==6,
            'all_selected_main_mutations':mutation==dict.fromkeys(ARMS,n),'no_baseline_goal_label_access':label_gathers['mainonly']==0,
            'auxiliary_gradients_reach_served_representation':all(timing['auxiliary_gradient_proof'].values()),
            'all9333_classes_retained':True,'all258_goals_valid':True,'entry_final_bindings_and_oof_membership_valid':True}
        result.update(schema='future-goal-experiment-v1',state='complete',preflight=timing,proof_gates=proof,
            useful_internal_signal=bool(result['metric_gate_passed'] and all(proof.values())),checkpoints=checkpoints,bindings_sha256=bindings,
            dataset_audit=dataset.audit,goal_control_audit=dataset.goal_control_audit,native_composition_fallbacks=dataset.native_fallbacks,
            independent_cache_audit=dataset.independent_cache_audit,
            auxiliary_diagnostics={arm:{'nll':float(-value['target_log_probability'].mean()),
                'top1_accuracy':float(np.mean(value['top1']==auxiliary_file['real_goal_id'])),
                'scope':'frozen auxiliary head only; cannot rescue primary'} for arm,value in auxiliary.items()},
            files={name:file_hash(args.output/name) for name in ('training_lock.json','predictions.npz','prediction_rows.jsonl','auxiliary_diagnostics.npz')},
            training_performed=True,reward_arrays_opened=False,terminal_timing_arrays_loaded=False,raw_payload_reads=False,
            test_payload_reads=False,confirmation_payload_reads=False,outcomes_analyzed=False,goal_labels_used_at_main_inference=False,
            policy_benefit_established=False,causal_claim=False,elapsed_s=time.monotonic()-guard.started,
            host_memory=guard.memory_report(),peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            runtime={'torch':str(torch.__version__),'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device)})
        write_json(args.output/'report.json',result);status('complete',report=str((args.output/'report.json').resolve()))
    except Exception as error:status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--future-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--deadline-seconds',type=int,default=3600);parser.add_argument('--execute',action='store_true');run(parser.parse_args())


if __name__=='__main__':main()
