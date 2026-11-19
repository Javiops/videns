"""Fixed20-epoch,3-arm,2-fold purchase likelihood experiment with resource preflight."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[variable]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from winprob import admission
from winprob.history_probe_training import HistoryDataset
from winprob.iql_dataset_contract import current_reservations
from winprob.iql_models import PublicBoardBatch,set_input_content_validation
from winprob.iql_training import RuntimeGuard,TensorStore,adam,file_hash,tree_hash,set_below_normal_priority
from winprob.multiset_probe_v1 import ARMS,SEED,EPOCHS,BATCH,purchase_tables,initial_models,epoch_order,compact_raw,summarize

PROTOCOL=ROOT/'docs/multiset_probe_protocol_v1.md'
CACHE=ROOT/'data/runs/history-cache-20260930-v1/cache'
MEMORY=ROOT/'data/runs/purchase-memory-20261001-v1/experiment/predictions.npz'
MEMORY_SHA='204e23aa1843e237f6517cc5cdf40262a69402f64ae5bd782d9639b6fa3d3fc2'
OLD=ROOT/'data/runs/history-probe-20260930-v3/experiment/predictions.npz'
OLD_SHA='27eb094eb1dcfc1551b375446684b0b423d86c03b79077b2ccb5772c8c0968f5'


def sidecar(path):path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
    sidecar(path)


def save_checkpoint(path,value):
    tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);tmp.replace(path);sidecar(path);return file_hash(path)


def synthetic_board(dataset,device):
    n=BATCH;width=max(1,dataset.inventory_max)
    return PublicBoardBatch(torch.arange(1,11,device=device).expand(n,-1),
        torch.arange(1,width+1,device=device).view(1,1,-1).expand(n,10,-1),
        torch.ones(n,10,width,device=device),torch.zeros(n,10,22,device=device),torch.full((n,10,1),.5,device=device),
        torch.tensor([False]+[True]*4+[False]*5,device=device).expand(n,-1),
        torch.tensor([False]*5+[True]*5,device=device).expand(n,-1),torch.zeros(n,37,device=device))


def preflight(dataset,store,tables,device,guard):
    models=initial_models(dataset.num_champions,dataset.num_items,len(tables.vocabulary),9333,0)
    step_time={};infer_time={};board=synthetic_board(dataset,device)
    synthetic_labels=torch.tensor(np.random.default_rng(SEED).integers(0,9333,size=BATCH),device=device)
    for arm in ARMS:
        model=models.pop(arm).to(device);optimizer=adam(model.parameters())
        # Validate synthetic public inputs once before enabling the trusted path.
        model.loss(board,synthetic_labels,tables);set_input_content_validation(model,False)
        times=[]
        for step in range(30):
            guard();torch.cuda.synchronize();started=time.monotonic()
            optimizer.zero_grad(set_to_none=True);loss=model.loss(board,synthetic_labels,tables)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
            torch.cuda.synchronize()
            if step>=10:times.append(time.monotonic()-started)
            if not bool(torch.isfinite(loss)):raise ValueError('nonfinite synthetic preflight loss')
        step_time[arm]=float(np.mean(times));times=[];model.eval()
        with torch.inference_mode():
            for repeat in range(6):
                guard();torch.cuda.synchronize();started=time.monotonic()
                raw=model.predict(store.states(dataset.selected[:8]),tables)
                compact_raw(raw.cpu().numpy(),np.zeros(8,dtype=np.int64))
                torch.cuda.synchronize()
                if repeat>=2:times.append(time.monotonic()-started)
        infer_time[arm]=float(np.mean(times))
        del model,optimizer,loss,raw;gc.collect();torch.cuda.empty_cache()
    gather=[];rng=np.random.default_rng(SEED+99)
    for repeat in range(15):
        rows=rng.choice(dataset.n,BATCH,replace=False);torch.cuda.synchronize();started=time.monotonic()
        temporary=store.states(rows);torch.cuda.synchronize()
        if repeat>=5:gather.append(time.monotonic()-started)
        del temporary
    fold=dataset.arrays['fold'];selected_fold=fold[dataset.selected]
    updates=EPOCHS*sum(int(np.ceil(np.sum(fold!=f)/BATCH)) for f in (0,1))
    batches=sum(int(np.ceil(np.sum(selected_fold==f)/8)) for f in (0,1))
    train=sum(step_time.values())*updates+len(ARMS)*updates*np.mean(gather)
    inference=2*batches*sum(infer_time.values())
    projected=1.25*(train+inference)+180.
    return {'synthetic_optimizer_steps':90,'real_optimizer_steps':0,'updates_per_arm':updates,'all_updates':len(ARMS)*updates,
        'step_mean_s':step_time,'state_gather_mean_s':float(np.mean(gather)),'inference_batch8_mean_s':infer_time,
        'inference_batches_per_arm':batches,'inference_passes_per_batch':2,'projected_training_s':float(train),
        'projected_inference_s':float(inference),'safety_factor':1.25,'fixed_overhead_s':180.,'projected_remaining_s':float(projected),
        'remaining_stage_s':guard.deadline-(time.monotonic()-guard.started),'preloaded_states':store.preloaded,
        'projection_gate_passed':bool(projected<=.8*(guard.deadline-(time.monotonic()-guard.started)))}


def load_reference_scores(selected,labels,fold,game_index):
    if file_hash(MEMORY)!=MEMORY_SHA or file_hash(OLD)!=OLD_SHA:raise ValueError('frozen stronger/reference predictions changed')
    with np.load(MEMORY,allow_pickle=False) as z:
        if (not np.array_equal(z['selected_state_indices'],selected) or not np.array_equal(z['labels'],labels)
                or not np.array_equal(z['fold'],fold) or not np.array_equal(z['game_index'],game_index)):
            raise ValueError('frozen memory evaluation population differs')
        memory={key:np.array(z['memory_'+key]) for key in ('top1','top3','target_log_probability')}
    with np.load(OLD,allow_pickle=False) as z:
        if not np.array_equal(z['selected_state_indices'],selected) or not np.array_equal(z['labels'],labels):raise ValueError('old state population differs')
        old={key:np.array(z['state_'+key]) for key in ('top1','top3','target_log_probability')}
    if int(np.sum(memory['top1']==labels))!=1584 or int(np.sum(old['top1']==labels))!=1471:raise ValueError('frozen reference score identity differs')
    return memory,old


def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    if args.deadline_seconds!=3600:raise ValueError('fixed60minute worker bound required')
    guard=RuntimeGuard(args.deadline_seconds);set_below_normal_priority()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    real_started=False
    def status(stage,**extra):
        value={'stage':stage,'elapsed_s':time.monotonic()-guard.started,'real_training_started':real_started,**extra}
        (args.output/'status.json').write_text(json.dumps(value,allow_nan=False));print(json.dumps(value),flush=True)
    try:
        status('validating')
        paths=[Path(__file__),PROTOCOL,ROOT/'docs/multiset_likelihood_design_v1.md']+sorted((ROOT/'winprob').glob('*.py'))
        bindings={str(p.resolve()):file_hash(p) for p in paths}
        if file_hash(CACHE/'manifest.json')!='1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b':raise ValueError('history cache changed')
        dataset=HistoryDataset(CACHE,ROOT/'docs/history_probe_protocol_v1.md');bindings.update(dataset.bindings)
        bindings[str(MEMORY)]=MEMORY_SHA;bindings[str(OLD)]=OLD_SHA
        for raw,want in bindings.items():
            if file_hash(raw)!=want:raise ValueError('initial binding changed: '+raw)
        guard.bind(bindings);guard()
        a=dataset.arrays;selected=np.array(dataset.selected);n=len(selected)
        if n!=3350 or len(np.unique(a['game_index'][selected]))!=3350:raise ValueError('fixed selected game population changed')
        selected_labels=np.array(a['main_group_id'][selected]);selected_fold=np.array(a['fold'][selected]);game_index=np.array(a['game_index'][selected])
        memory,old=load_reference_scores(selected,selected_labels,selected_fold,game_index)
        write(args.output/'input_audit.json',{'dataset':dataset.audit,'bindings_sha256':bindings,'selection':dataset.selection,
            'training_only':True,'reference_memory_top1_hits':1584,'reference_old_state_top1_hits':1471,'reward_arrays_opened':False})
        device=torch.device('cuda:0')
        if not torch.cuda.is_available():raise RuntimeError('idle CUDA runtime required')
        tables=purchase_tables(dataset.catalogue['group_main']).to(device)
        if len(tables.vocabulary)!=196:raise ValueError('frozen output vocabulary changed')
        store=TensorStore(dataset,device,guard)
        labels_gpu=torch.tensor(np.asarray(a['main_group_id']),dtype=torch.long,device=device)
        status('full_synthetic_preflight')
        timing=preflight(dataset,store,tables,device,guard);write(args.output/'preflight.json',timing)
        if not timing['projection_gate_passed']:raise RuntimeError('full recipe does not fit fixed80percent resource projection gate')
        if not args.execute:status('preflight_complete_no_real_fit');return
        scores={arm:{'top1':np.full(n,-1,dtype=np.int64),'top3':np.full((n,3),-1,dtype=np.int64),
            **{key:np.full(n,np.nan) for key in ('target_log_probability','raw_target_log_probability','catalogue_log_mass')}} for arm in ARMS}
        scores.update(memory=memory,old_state=old)
        checkpoints={};orders={};initial={};steps_by_model={};parameters={};training_ids={};mutation={arm:0 for arm in ARMS}
        real_started=True
        for fold in (0,1):
            train=np.flatnonzero(a['fold']!=fold);positions=np.flatnonzero(selected_fold==fold);queries=selected[positions]
            allowed=sorted(g['match_id'] for g in dataset.games if g['fold']!=fold);training_ids[str(fold)]=allowed
            if set(a['game_index'][train])&set(game_index[positions]):raise ValueError('training/query whole-game overlap')
            models=initial_models(dataset.num_champions,dataset.num_items,196,9333,fold)
            encoder_hashes={arm:tree_hash(model.state.state_dict()) for arm,model in models.items()}
            head_hashes={arm:tree_hash(model.head.state_dict()) for arm,model in models.items()}
            if len(set(encoder_hashes.values()))!=1 or head_hashes['marginal']!=head_hashes['canonical']:raise ValueError('initial parameter matching failed')
            initial[str(fold)]={'encoder':encoder_hashes,'head':head_hashes}
            original_main,original_full=a['main_group_id'],a['full_action_id']
            altered_main=np.array(original_main);altered_full=np.array(original_full)
            heldout=a['fold']==fold;altered_main[heldout]=(altered_main[heldout]+1)%9333;altered_full[heldout]=(altered_full[heldout]+1)%13166
            for arm in ARMS:
                model=models.pop(arm).to(device);set_input_content_validation(model,False);optimizer=adam(model.parameters());steps=0
                parameters[arm]={'encoder':sum(p.numel() for p in model.state.parameters()),'head':sum(p.numel() for p in model.head.parameters())}
                for epoch in range(EPOCHS):
                    status('training',fold=fold,arm=arm,epoch=epoch+1)
                    order=epoch_order(train,fold,epoch);order_sha=hashlib.sha256(order.tobytes()).hexdigest();key=f'{fold}:{epoch}'
                    if key in orders and orders[key]!=order_sha:raise ValueError('epoch order differs across arms')
                    orders[key]=order_sha;total=0.;maxnorm=0.;started=time.monotonic();model.train()
                    for offset in range(0,len(order),BATCH):
                        if offset%(20*BATCH)==0:guard()
                        rows=order[offset:offset+BATCH];board=store.states(rows);y=labels_gpu[torch.as_tensor(rows,device=device)]
                        optimizer.zero_grad(set_to_none=True);loss=model.loss(board,y,tables)
                        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite real loss')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
                        steps+=1;total+=float(loss.detach())*len(rows);maxnorm=max(maxnorm,float(norm))
                    epoch_record={'fold':fold,'arm':arm,'epoch':epoch+1,'transitions':len(train),'updates':steps,
                                  'mean_training_loss':total/len(train),'max_gradient_before_clip':maxnorm,'elapsed_s':time.monotonic()-started}
                    with (args.output/'epochs.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(epoch_record)+'\n')
                expected=EPOCHS*int(np.ceil(len(train)/BATCH))
                if steps!=expected or {int(s['step']) for s in optimizer.state.values()}!={expected}:raise ValueError('incomplete optimizer accounting')
                steps_by_model[f'{fold}:{arm}']=steps
                name=f'actor_{arm}_fold{fold}.pt'
                checkpoints[name]=save_checkpoint(args.output/name,{'actor':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'arm':arm,'scored_fold':fold,'seed':SEED+fold,'epochs':EPOCHS,'steps':steps,'training_game_ids':allowed,
                    'vocabulary':tables.vocabulary,'initial':initial[str(fold)],'protocol_sha256':file_hash(PROTOCOL)})
                status('predicting_frozen_checkpoint',fold=fold,arm=arm);model.eval()
                with torch.inference_mode():
                    for offset in range(0,len(queries),8):
                        if offset%128==0:guard()
                        rows=queries[offset:offset+8];raw=model.predict(store.states(rows),tables)
                        try:
                            a['main_group_id'],a['full_action_id']=altered_main,altered_full
                            changed=model.predict(store.states(rows),tables)
                        finally:a['main_group_id'],a['full_action_id']=original_main,original_full
                        if not torch.equal(raw,changed):raise ValueError('heldout target mutation changed full prediction')
                        pos=positions[offset:offset+len(rows)];summary=compact_raw(raw.cpu().numpy(),selected_labels[pos])
                        for key,value in summary.items():scores[arm][key][pos]=value
                        mutation[arm]+=len(rows)
                del model,optimizer,loss,board,y,raw,changed;gc.collect();torch.cuda.empty_cache();guard()
            del models,altered_main,altered_full
        if sum(steps_by_model.values())!=timing['all_updates'] or mutation!=dict.fromkeys(ARMS,n):raise ValueError('complete training/mutation coverage failed')
        write(args.output/'training_lock.json',{'epochs_complete':True,'updates':steps_by_model,'shared_epoch_order_sha256':orders,
            'initial_parameter_sha256':initial,'training_game_ids_by_fold':training_ids,'checkpoints':checkpoints,
            'heldout_target_mutation_rows':mutation,'comparison_metrics_inspected':False})
        arrays={'selected_state_indices':selected,'labels':selected_labels,'fold':selected_fold,'game_index':game_index}
        for arm,score in scores.items():
            for key,value in score.items():arrays[arm+'_'+key]=value
        np.savez_compressed(args.output/'predictions.npz',**arrays);sidecar(args.output/'predictions.npz')
        status('evaluating_complete_recipe')
        bags=dataset.catalogue['group_main'];sizes=np.array([sum(c for _,c in bags[g]) for g in selected_labels]);repeated=np.array([any(c>1 for _,c in bags[g]) for g in selected_labels])
        result=summarize(selected_labels,scores,sizes,repeated)
        snapshot=args.output/'code_snapshot';snapshot.mkdir()
        for path in paths:
            dest=snapshot/path.resolve().relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        for path,want in bindings.items():
            guard()
            if file_hash(path)!=want:raise ValueError('bound input/code changed before publication: '+path)
        admission.verify_manifest_sources(dataset.admission)
        if {g['match_id'] for g in dataset.games}&current_reservations(dataset.admission):raise ValueError('FIT game became reserved')
        result.update(schema='multiset-probe-experiment-v1',state='complete',scope='already exposed internal whole-game OOF; no native legality decoder',
            ledger_games=4001,unavailable_focal_games=651,parameters=parameters,preflight=timing,checkpoints=checkpoints,
            training_lock_sha256=file_hash(args.output/'training_lock.json'),predictions_sha256=file_hash(args.output/'predictions.npz'),
            bindings_sha256=bindings,epochs_per_model=20,updates=steps_by_model,heldout_target_mutation_rows=mutation,
            training_performed=True,raw_payload_reads=False,reward_arrays_opened=False,outcomes_analyzed=False,
            test_payload_reads=False,confirmation_payload_reads=False,source_code_imported=False,
            policy_benefit_established=False,breakthrough_established=False,promotion_authorized=False,
            elapsed_s=time.monotonic()-guard.started,host_memory=guard.memory_report(),peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            runtime={'torch':str(torch.__version__),'numpy':np.__version__,'gpu':torch.cuda.get_device_name(device)})
        write(args.output/'report.json',result);status('complete',metric_gate_passed=result['metric_gate_passed'])
    except Exception as error:status('failed',error=str(error),error_type=type(error).__name__);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--deadline-seconds',type=int,default=3600);parser.add_argument('--execute',action='store_true')
    run(parser.parse_args())
