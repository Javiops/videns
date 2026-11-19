"""Replay all fixed OOF neural predictions and independently check training/metrics."""
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
from winprob.iql_dataset_contract import current_reservations
from winprob.iql_models import set_input_content_validation
from winprob.iql_training import RuntimeGuard,TensorStore,file_hash,tree_hash,set_below_normal_priority
from winprob.multiset_probe_v1 import initial_models,purchase_tables,epoch_order,compact_raw,ARMS


def check(ok,message):
    if not ok:raise ValueError(message)


def run(directory,output):
    output.mkdir(parents=True,exist_ok=False);guard=RuntimeGuard(900);set_below_normal_priority()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report_path=directory/'report.json';report=json.loads(report_path.read_text())
    check(report['state']=='complete' and report['schema']=='multiset-probe-experiment-v1','incomplete/wrong model result')
    check(report_path.with_suffix('.sha256').read_text().split()==[file_hash(report_path),report_path.name],'report sidecar mismatch')
    for path,want in report['bindings_sha256'].items():check(file_hash(path)==want,'changed bound input/code: '+path)
    check(file_hash(directory/'training_lock.json')==report['training_lock_sha256'],'training lock changed')
    check(file_hash(directory/'predictions.npz')==report['predictions_sha256'],'prediction artifact changed')
    lock=json.loads((directory/'training_lock.json').read_text());epochs=[json.loads(line) for line in (directory/'epochs.jsonl').read_text().splitlines()]
    check(len(epochs)==120 and lock['epochs_complete'] is True,'all120 model epochs required')
    dataset=HistoryDataset(ROOT/'data/runs/history-cache-20260930-v1/cache',ROOT/'docs/history_probe_protocol_v1.md')
    a=dataset.arrays;selected=np.array(dataset.selected);labels=np.array(a['main_group_id'][selected]);folds=np.array(a['fold'][selected])
    check(len(selected)==3350 and len(np.unique(a['game_index'][selected]))==3350,'fixed focal sample changed')
    prediction=np.load(directory/'predictions.npz',allow_pickle=False)
    check(np.array_equal(selected,prediction['selected_state_indices']) and np.array_equal(labels,prediction['labels'])
          and np.array_equal(folds,prediction['fold']) and np.array_equal(a['game_index'][selected],prediction['game_index']),'prediction identity differs')
    for arm,source_path,prefix in (
        ('memory',ROOT/'data/runs/purchase-memory-20261001-v1/experiment/predictions.npz','memory'),
        ('old_state',ROOT/'data/runs/history-probe-20260930-v3/experiment/predictions.npz','state')):
        check(file_hash(source_path)==report['bindings_sha256'][str(source_path)],'frozen reference changed')
        with np.load(source_path,allow_pickle=False) as reference:
            check(np.array_equal(reference['selected_state_indices'],selected) and np.array_equal(reference['labels'],labels),'reference identity differs')
            for key in ('top1','top3','target_log_probability'):
                check(np.array_equal(reference[prefix+'_'+key],prediction[arm+'_'+key]),'reference predictions changed in comparison')
    device=torch.device('cuda:0');tables=purchase_tables(dataset.catalogue['group_main']).to(device);store=TensorStore(dataset,device,guard)
    check(store.preloaded==report['preflight']['preloaded_states'],'functional replay requires same state padding/store path')
    checkpoint_count=0;replay_count=0;max_error=0.;maximum_raw_error=0.;maximum_mass_error=0.;top_mismatches=0
    for fold in (0,1):
        train=np.flatnonzero(a['fold']!=fold);positions=np.flatnonzero(folds==fold);queries=selected[positions]
        allowed=sorted(g['match_id'] for g in dataset.games if g['fold']!=fold);expected_steps=20*int(np.ceil(len(train)/256))
        check(allowed==lock['training_game_ids_by_fold'][str(fold)],'training IDs differ')
        check(not set(a['game_index'][train])&set(a['game_index'][queries]),'train/query game overlap')
        templates=initial_models(dataset.num_champions,dataset.num_items,196,9333,fold)
        expected_encoder={arm:tree_hash(model.state.state_dict()) for arm,model in templates.items()}
        expected_head={arm:tree_hash(model.head.state_dict()) for arm,model in templates.items()}
        check(lock['initial_parameter_sha256'][str(fold)]=={'encoder':expected_encoder,'head':expected_head},'initial tensors do not reproduce')
        check(len(set(expected_encoder.values()))==1 and expected_head['marginal']==expected_head['canonical'],'initial matching failed')
        for epoch in range(20):
            order=epoch_order(train,fold,epoch)
            check(np.array_equal(np.sort(order),train) and hashlib.sha256(order.tobytes()).hexdigest()==lock['shared_epoch_order_sha256'][f'{fold}:{epoch}'],'epoch row order incomplete or changed')
        for arm in ARMS:
            guard();name=f'actor_{arm}_fold{fold}.pt';path=directory/name
            check(file_hash(path)==report['checkpoints'][name]==lock['checkpoints'][name],'checkpoint changed')
            saved=torch.load(path,map_location='cpu',weights_only=True)
            check(saved['arm']==arm and saved['scored_fold']==fold and saved['epochs']==20 and saved['steps']==expected_steps
                  and saved['training_game_ids']==allowed and saved['vocabulary']==tables.vocabulary,'checkpoint identity/recipe differs')
            check({int(s['step']) for s in saved['optimizer']['state'].values()}=={expected_steps},'Adam updates incomplete')
            for group in saved['optimizer']['param_groups']:
                check(group['lr']==3e-4 and group['weight_decay']==0 and tuple(group['betas'])==(.9,.999) and group['eps']==1e-8,'optimizer recipe changed')
            records=[e for e in epochs if e['arm']==arm and e['fold']==fold]
            check([e['epoch'] for e in records]==list(range(1,21)),'epoch ledger incomplete')
            for e in records:
                check(e['transitions']==len(train) and e['updates']==e['epoch']*int(np.ceil(len(train)/256))
                      and np.isfinite(e['mean_training_loss']) and np.isfinite(e['max_gradient_before_clip']),'epoch accounting/finite-gradient receipt failed')
            model=templates.pop(arm).to(device);model.load_state_dict(saved['actor'],strict=True);del saved
            check(report['parameters'][arm]=={'encoder':sum(p.numel() for p in model.state.parameters()),
                'head':sum(p.numel() for p in model.head.parameters())},'parameter accounting differs')
            model.eval();set_input_content_validation(model,False)
            with torch.inference_mode():
                for lo in range(0,len(queries),8):
                    if lo%128==0:guard()
                    rows=queries[lo:lo+8];pos=positions[lo:lo+len(rows)];raw=model.predict(store.states(rows),tables)
                    replay=compact_raw(raw.cpu().numpy(),labels[pos])
                    top_mismatches+=int(np.sum(replay['top3']!=prediction[arm+'_top3'][pos]))
                    max_error=max(max_error,float(np.max(np.abs(replay['target_log_probability']-prediction[arm+'_target_log_probability'][pos]))))
                    maximum_raw_error=max(maximum_raw_error,float(np.max(np.abs(replay['raw_target_log_probability']-prediction[arm+'_raw_target_log_probability'][pos]))))
                    maximum_mass_error=max(maximum_mass_error,float(np.max(np.abs(replay['catalogue_log_mass']-prediction[arm+'_catalogue_log_mass'][pos]))))
                    replay_count+=len(rows)
            checkpoint_count+=1;del model,raw;gc.collect();torch.cuda.empty_cache()
    check(checkpoint_count==6 and replay_count==10050 and top_mismatches==0 and max_error==maximum_raw_error==maximum_mass_error==0.,'full neural prediction replay differs')
    check(report['heldout_target_mutation_rows']==dict.fromkeys(ARMS,3350) and sum(report['updates'].values())==92040,'mutation/update totals differ')
    check(report['elapsed_s']<3600 and report['preflight']['projection_gate_passed'] is True
          and report['preflight']['projected_remaining_s']<=.8*report['preflight']['remaining_stage_s'],
          'declared resource bound/preflight did not hold')
    for field in ('raw_payload_reads','reward_arrays_opened','outcomes_analyzed','test_payload_reads','confirmation_payload_reads','source_code_imported'):
        check(report[field] is False,'experiment boundary failed: '+field)
    all_arms=(*ARMS,'memory','old_state');draws=np.random.default_rng(20261001).integers(0,3350,size=(2000,3350));values={}
    for arm in all_arms:
        check(np.array_equal(prediction[arm+'_top1'],prediction[arm+'_top3'][:,0]),'first prediction differs from top3')
        values[arm]={'top1':(prediction[arm+'_top1']==labels).astype(float),
                     'top3':(prediction[arm+'_top3']==labels[:,None]).any(1).astype(float),
                     'conditional_nll':-prediction[arm+'_target_log_probability']}
        for key,v in values[arm].items():
            expected=report['arms'][arm][key];check(abs(float(v.mean())-expected['mean'])<1e-12,'metric mean differs')
            check(np.allclose(np.quantile(v[draws].mean(1),[.025,.975]),expected['interval95'],atol=1e-12,rtol=0),'metric interval differs')
        if arm in ARMS:
            check(abs(float(-prediction[arm+'_raw_target_log_probability'].mean())-report['arms'][arm]['raw_nll'])<1e-12,'raw NLL differs')
            mass=np.exp(prediction[arm+'_catalogue_log_mass']);expect=report['arms'][arm]['catalogue_mass']
            check(abs(float(mass.mean())-expect['mean'])<1e-12 and abs(float(mass.min())-expect['minimum'])<1e-12
                  and abs(float(np.quantile(mass,.05))-expect['p05'])<1e-12,'catalogue mass differs')
    for control in ('categorical','memory','canonical','old_state'):
        for key in values[control]:
            d=values['marginal'][key]-values[control][key];expected=report['marginal_minus_control'][control][key]
            check(abs(float(d.mean())-expected['difference'])<1e-12 and np.allclose(np.quantile(d[draws].mean(1),[.025,.975]),expected['interval95'],atol=1e-12,rtol=0),'paired contrast differs')
        d=values['marginal']['top1']-values[control]['top1'];expected=report['marginal_minus_control'][control]
        check(int(np.sum(d>0))==expected['corrected'] and int(np.sum(d<0))==expected['broken'],'movement differs')
    bags=dataset.catalogue['group_main']
    sizes=np.array([sum(count for item,count in bags[int(g)]) for g in labels])
    repeated=np.array([any(count>1 for item,count in bags[int(g)]) for g in labels])
    masks={**{'size_'+str(size):sizes==size for size in sorted(set(map(int,sizes)))},'repeated_copies':repeated}
    masks={name:mask for name,mask in masks.items() if mask.any()}
    check(set(masks)==set(report['descriptive_slices']),'slice definitions changed')
    for name,mask in masks.items():
        expected=report['descriptive_slices'][name];check(int(mask.sum())==expected['games'],'slice denominator differs')
        for arm,v in values.items():
            for key,array in v.items():check(abs(float(array[mask].mean())-expected['arms'][arm][key])<1e-12,'slice metric differs')
    paired=report['marginal_minus_control'];gate=all(paired[c]['top1']['difference']>=.03 and paired[c]['top1']['interval95'][0]>0 for c in ('categorical','memory')) and paired['memory']['top3']['difference']>=0
    check(bool(gate)==report['metric_gate_passed'],'gate differs')
    for raw,want in report['bindings_sha256'].items():check(file_hash(raw)==want,'input changed during audit')
    admission.verify_manifest_sources(dataset.admission);check(not {g['match_id'] for g in dataset.games}&current_reservations(dataset.admission),'FIT game became reserved')
    result={'schema':'multiset-probe-audit-v1','state':'complete','passed':True,'report_sha256':file_hash(report_path),'auditor_sha256':file_hash(__file__),
        'checkpoints_replayed':checkpoint_count,'query_predictions_replayed':replay_count,'top3_cell_mismatches':top_mismatches,
        'max_conditional_logp_error':max_error,'max_raw_logp_error':maximum_raw_error,'max_catalogue_logmass_error':maximum_mass_error,
        'epochs_verified':120,'updates_verified':92040,'metric_gate_passed':bool(gate),'arms':report['arms'],'paired':paired,
        'elapsed_s':time.monotonic()-guard.started,'test_payload_reads':False,'confirmation_payload_reads':False,
        'limits':['Exact replay and recomputation validate this exposed internal experiment; they do not establish M2 benefit or causality.']}
    path=output/'report.json';path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')
    print(json.dumps({k:result[k] for k in ('state','passed','query_predictions_replayed','max_conditional_logp_error','metric_gate_passed','elapsed_s')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.directory.resolve(),args.output.resolve())
