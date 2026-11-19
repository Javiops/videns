"""Replay all scoreboard models and independently recompute the fixed decision gates."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from winprob import admission
from winprob.history_probe_training import HistoryDataset
from winprob.history_probe_evaluation import summarize_logits
from winprob.iql_dataset_contract import current_reservations
from winprob.iql_training import RuntimeGuard,file_hash,tree_hash,set_below_normal_priority
from winprob.scoreboard_probe_v1 import ScoreCache,ScoreStore,initial_models,enable_fastpath,epoch_order,ARMS


def check(ok,message):
    if not ok:raise ValueError(message)


def run(directory,output):
    output.mkdir(parents=True,exist_ok=False);guard=RuntimeGuard(900);set_below_normal_priority()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rp=directory/'report.json';report=json.loads(rp.read_text())
    check(report['schema']=='scoreboard-probe-experiment-v1' and report['state']=='complete','complete scoreboard result required')
    check(rp.with_suffix('.sha256').read_text().split()==[file_hash(rp),rp.name],'report digest changed')
    for raw,want in report['bindings_sha256'].items():check(file_hash(raw)==want,'bound artifact/code changed: '+raw)
    check(file_hash(directory/'training_lock.json')==report['training_lock_sha256'],'training lock changed')
    check(file_hash(directory/'predictions.npz')==report['predictions_sha256'],'predictions changed')
    lock=json.loads((directory/'training_lock.json').read_text());epochs=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
    check(len(epochs)==120 and lock['epochs_complete'] is True,'all120 model epochs required')
    dataset=HistoryDataset(ROOT/'data/runs/history-cache-20260930-v1/cache',ROOT/'docs/history_probe_protocol_v1.md')
    feature_path=ROOT/'data/runs/scoreboard-features-20261001-v1/cache/manifest.json';cache=ScoreCache(feature_path,dataset,guard)
    a=dataset.arrays;selected=np.array(dataset.selected);labels=np.array(a['main_group_id'][selected]);folds=np.array(a['fold'][selected]);gi=np.array(a['game_index'][selected])
    p=np.load(directory/'predictions.npz',allow_pickle=False)
    check(len(selected)==3350 and len(np.unique(gi))==3350 and np.array_equal(selected,p['selected_state_indices'])
          and np.array_equal(labels,p['labels']) and np.array_equal(folds,p['fold']) and np.array_equal(gi,p['game_index']),'query identity changed')
    for arm,source,prefix in (
        ('memory',ROOT/'data/runs/purchase-memory-20261001-v1/experiment/predictions.npz','memory'),
        ('old_state',ROOT/'data/runs/history-probe-20260930-v3/experiment/predictions.npz','state')):
        check(file_hash(source)==report['bindings_sha256'][str(source)],'frozen reference changed')
        with np.load(source,allow_pickle=False) as prior:
            check(np.array_equal(prior['selected_state_indices'],selected) and np.array_equal(prior['labels'],labels),'reference population differs')
            for key in ('top1','top3','target_log_probability'):check(np.array_equal(prior[prefix+'_'+key],p[arm+'_'+key]),'reference prediction changed')
    device=torch.device('cuda:0');store=ScoreStore(dataset,cache,device,guard)
    check(store.preloaded==report['preflight']['preloaded_states'],'same state padding/store path required for exact replay')
    batch_size=report['preflight']['inference_batch_size'];check(batch_size==256,'inference batching changed')
    replayed=0;mismatches=0;max_error=0.;checkpoint_count=0
    for fold in (0,1):
        train=np.flatnonzero(a['fold']!=fold);positions=np.flatnonzero(folds==fold);queries=selected[positions]
        allowed=sorted(g['match_id'] for g in dataset.games if g['fold']!=fold);updates=20*int(np.ceil(len(train)/256))
        check(allowed==lock['training_game_ids_by_fold'][str(fold)] and not set(a['game_index'][train])&set(a['game_index'][queries]),'whole-game membership differs')
        models=initial_models(dataset.num_champions,dataset.num_items,9333,fold)
        initial={arm:tree_hash(model.state_dict()) for arm,model in models.items()}
        check(initial==lock['initial_model_sha256'][str(fold)] and len(set(initial.values()))==1,'initial tensors do not reproduce')
        for epoch in range(20):
            order=epoch_order(train,fold,epoch)
            check(np.array_equal(np.sort(order),train) and hashlib.sha256(order.tobytes()).hexdigest()==lock['shared_epoch_order_sha256'][f'{fold}:{epoch}'],'epoch order incomplete or changed')
        for arm in ARMS:
            guard();name=f'actor_{arm}_fold{fold}.pt';path=directory/name
            check(file_hash(path)==report['checkpoints'][name]==lock['checkpoints'][name],'checkpoint changed')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            check(saved['arm']==arm and saved['scored_fold']==fold and saved['epochs']==20 and saved['steps']==updates
                  and saved['training_game_ids']==allowed and saved['feature_manifest_sha256']==file_hash(feature_path),'checkpoint identity/recipe differs')
            check({int(s['step']) for s in saved['optimizer']['state'].values()}=={updates},'Adam updates incomplete')
            for group in saved['optimizer']['param_groups']:check(group['lr']==3e-4 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0,'optimizer settings differ')
            records=[r for r in epochs if r['fold']==fold and r['arm']==arm]
            check([r['epoch'] for r in records]==list(range(1,21)),'epoch ledger incomplete')
            for r in records:check(r['transitions']==len(train) and r['updates']==r['epoch']*int(np.ceil(len(train)/256)) and np.isfinite(r['mean_training_loss']) and np.isfinite(r['max_gradient_before_clip']),'epoch accounting differs')
            model=models.pop(arm).to(device);model.load_state_dict(saved['actor'],strict=True);del saved
            check(report['parameters'][arm]=={'encoder':sum(x.numel() for x in model.state.parameters()),'head':sum(x.numel() for x in model.head.parameters()),'score_projection':192},'parameter count differs')
            if arm=='zero':check(not bool(model.state.score_projection.weight.any()),'zero-input projection changed')
            model.eval();enable_fastpath(model,dataset,cache)
            with torch.inference_mode():
                for lo in range(0,len(queries),batch_size):
                    guard();rows=queries[lo:lo+batch_size];pos=positions[lo:lo+len(rows)]
                    raw=model.predict(store.states(rows));score=summarize_logits(raw.cpu().numpy(),labels[pos])
                    mismatches+=int(np.sum(score['top3']!=p[arm+'_top3'][pos]));max_error=max(max_error,float(np.max(np.abs(score['target_log_probability']-p[arm+'_target_log_probability'][pos]))));replayed+=len(rows)
            checkpoint_count+=1;del model,raw;gc.collect();torch.cuda.empty_cache()
    check(checkpoint_count==6 and replayed==10050 and mismatches==0 and max_error==0.,'full functional replay differs')
    check(report['heldout_target_mutation_rows']==dict.fromkeys(ARMS,3350) and sum(report['updates'].values())==92040,'mutation/update counts differ')
    check(report['elapsed_s']<3600 and report['preflight']['projection_gate_passed'] is True and report['preflight']['projected_remaining_s']<=.8*report['preflight']['remaining_stage_s'],'resource gate failed')
    check(report['preflight']['initial_base_parity_verified'] is True,'initial base-prediction parity was not established')
    draws=np.random.default_rng(20261001).integers(0,3350,size=(2000,3350));values={}
    for arm in (*ARMS,'memory','old_state'):
        check(np.array_equal(p[arm+'_top1'],p[arm+'_top3'][:,0]),'top1/top3 inconsistency')
        v={'top1':(p[arm+'_top1']==labels).astype(float),'top3':(p[arm+'_top3']==labels[:,None]).any(1).astype(float),'nll':-p[arm+'_target_log_probability']};values[arm]=v
        for key,x in v.items():
            expected=report['arms'][arm][key]
            check(abs(float(x.mean())-expected['mean'])<1e-12 and np.allclose(np.quantile(x[draws].mean(1),[.025,.975]),expected['interval95'],atol=1e-12,rtol=0),'aggregate/interval differs')
    for control in ('zero','memory','shuffled','old_state'):
        for key in values[control]:
            d=values['scoreboard'][key]-values[control][key];expected=report['scoreboard_minus_control'][control][key]
            check(abs(float(d.mean())-expected['difference'])<1e-12 and np.allclose(np.quantile(d[draws].mean(1),[.025,.975]),expected['interval95'],atol=1e-12,rtol=0),'paired contrast differs')
        d=values['scoreboard']['top1']-values[control]['top1'];expected=report['scoreboard_minus_control'][control]
        check(int(np.sum(d>0))==expected['corrected'] and int(np.sum(d<0))==expected['broken'],'movement differs')
    nonempty=labels!=0;d=values['scoreboard']['top1']-values['zero']['top1'];slice_ci=np.quantile((d*nonempty)[draws].sum(1)/nonempty[draws].sum(1),[.025,.975])
    expected=report['nonempty_purchase_guard'];check(int(nonempty.sum())==expected['games'] and abs(float(d[nonempty].mean())-expected['difference'])<1e-12 and np.allclose(slice_ci,expected['interval95'],rtol=0,atol=1e-12),'nonempty-purchase guard differs')
    groups=dataset.catalogue['group_main'];sizes=np.array([sum(c for i,c in groups[int(g)]) for g in labels]);patches=np.array([dataset.games[int(i)]['patch'] for i in gi]);kinds=[]
    for row in selected:
        fid=int(a['full_action_id'][row]);bag=dataset.catalogue['full_actions'][fid]
        kinds.append('no_buy' if fid==0 else 'ward_only' if not any(i not in (2055,772043) for i,c in bag) else 'mixed' if any(i in (2055,772043) for i,c in bag) else 'purchase')
    kinds=np.array(kinds);masks={**{'patch_'+str(x):patches==x for x in sorted(set(patches))},**{'size_'+str(x):sizes==x for x in sorted(set(sizes))},**{'kind_'+str(x):kinds==x for x in sorted(set(kinds))}}
    check(set(masks)==set(report['descriptive_slices']),'slice set changed')
    for name,mask in masks.items():
        s=report['descriptive_slices'][name];check(int(mask.sum())==s['games'],'slice count differs')
        if mask.sum()<30:check(s['status']=='too_small_for_rates' and 'arms' not in s,'small slice received rates')
        else:
            for arm,v in values.items():
                for key,array in v.items():check(abs(float(array[mask].mean())-s['arms'][arm][key])<1e-12,'slice metric differs')
    paired=report['scoreboard_minus_control'];gate=all(paired[c]['top1']['difference']>=.03 and paired[c]['top1']['interval95'][0]>0 for c in ('zero','memory')) and paired['memory']['top3']['difference']>=0 and slice_ci[0]>0
    check(bool(gate)==report['metric_gate_passed'],'gate conclusion differs')
    for key in ('raw_payload_reads','reward_arrays_opened','outcomes_analyzed','test_payload_reads','confirmation_payload_reads','source_runtime_imported'):check(report[key] is False,'experiment boundary differs')
    for raw,want in report['bindings_sha256'].items():check(file_hash(raw)==want,'binding changed during audit')
    admission.verify_manifest_sources(dataset.admission);check(not {g['match_id'] for g in dataset.games}&current_reservations(dataset.admission),'FIT game became reserved')
    result={'schema':'scoreboard-probe-audit-v1','state':'complete','passed':True,'report_sha256':file_hash(rp),'auditor_sha256':file_hash(__file__),
        'models_replayed':checkpoint_count,'query_predictions_replayed':replayed,'top3_cell_mismatches':mismatches,'max_target_logp_error':max_error,
        'epochs_verified':120,'updates_verified':92040,'metric_gate_passed':bool(gate),'arms':report['arms'],'paired':paired,'nonempty_purchase_guard':expected,
        'elapsed_s':time.monotonic()-guard.started,'test_payload_reads':False,'confirmation_payload_reads':False,
        'limits':['Exact replay validates the exposed internal comparison; it is not a direct M2 deployment result or causal claim.']}
    rp=output/'report.json';rp.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False));rp.with_suffix('.sha256').write_text(file_hash(rp)+' '+rp.name+'\n',encoding='ascii')
    print(json.dumps({k:result[k] for k in ('state','passed','query_predictions_replayed','max_target_logp_error','metric_gate_passed','elapsed_s')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.directory.resolve(),args.output.resolve())
