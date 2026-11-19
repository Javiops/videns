"""Independent receipt/metric audit; no fitting or source payload access."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ['OMP_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import current_reservations


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def check(condition,message):
    if not condition: raise ValueError(message)


def run(directory,output):
    report_path=directory/'report.json';report=json.loads(report_path.read_text())
    check(report['state']=='complete' and report['schema']=='purchase-memory-probe-v1','incomplete/wrong experiment')
    check(report_path.with_suffix('.sha256').read_text().split()==[digest(report_path),'report.json'],'report digest mismatch')
    check(digest(directory/'predictions.npz')==report['predictions_sha256'],'prediction digest mismatch')
    for raw,expected in report['bindings_sha256'].items(): check(digest(raw)==expected,'bound artifact changed: '+raw)
    base_path=ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json'
    check(digest(base_path)=='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea','base cache mismatch')
    base=json.loads(base_path.read_text());admission_path=Path(base['source_admission_path'])
    check(digest(admission_path)==base['source_admission_sha256'],'admission mismatch')
    admitted=json.loads(admission_path.read_text());forbidden=current_reservations(admitted)
    game_ids=[g['match_id'] for g in base['game_metadata']]
    check(not set(game_ids)&forbidden,'reserved game in FIT')
    arrays={}
    for name in ('main_group_id','game_index','fold','champion_indices','patch_index','timestamp_ms'):
        path=base_path.parent/base['arrays'][name]['file']
        check(digest(path)==base['files'][path.name],'cache array changed')
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    p=np.load(directory/'predictions.npz',allow_pickle=False)
    selected=p['selected_state_indices'];labels=p['labels'];gi=arrays['game_index'];folds=arrays['fold']
    check(len(selected)==3350 and len(np.unique(gi[selected]))==3350,'fixed unique game count changed')
    check(np.array_equal(labels,arrays['main_group_id'][selected]),'observed targets mismatch')
    check(np.array_equal(p['game_index'],gi[selected]) and np.array_equal(p['fold'],folds[selected]),'query identity mismatch')
    old_path=ROOT/'data/runs/history-probe-20260930-v3/experiment/predictions.npz'
    check(digest(old_path)=='27eb094eb1dcfc1551b375446684b0b423d86c03b79077b2ccb5772c8c0968f5','old prediction changed')
    with np.load(old_path,allow_pickle=False) as old:
        check(np.array_equal(selected,old['selected_state_indices']),'focal selection changed')
        check(np.array_equal(p['state_top1'],old['state_top1']) and np.array_equal(p['state_top3'],old['state_top3']),'state ranks changed')
        check(np.max(np.abs(p['state_target_log_probability']-old['state_target_log_probability']))<=1e-5,'state logp changed')
    labels_all=arrays['main_group_id'];champ=arrays['champion_indices'][:,0]
    shuffled={}
    for fold in (0,1):
        groups=defaultdict(list);train=np.flatnonzero(folds!=fold)
        for r in train: groups[(int(champ[r]),int(arrays['patch_index'][r]),int(arrays['timestamp_ms'][r]//180000))].append(int(r))
        result=np.full(len(folds),-1,dtype=labels_all.dtype);rng=np.random.default_rng(20261001+fold)
        for key in sorted(groups):
            members=np.asarray(sorted(groups[key]));result[members]=labels_all[rng.permutation(members)]
        shuffled[fold]=result
    logp_errors=[];receipt_count=0
    for j,row in enumerate(selected):
        ids=p['neighbor_rows'][j];weights=p['neighbor_weights'][j];dist=p['neighbor_distances'][j]
        valid=ids>=0;ids=ids[valid];w=weights[valid];d=dist[valid];receipt_count+=len(ids)
        check(np.all(weights[~valid]==0) and np.all(np.isnan(dist[~valid])),'neighbor padding changed')
        check(len(ids)<=32 and len(np.unique(gi[ids]))==len(ids),'duplicate neighbor game')
        check(np.all(folds[ids]!=folds[row]) and np.all(gi[ids]!=gi[row]),'neighbor game leakage')
        check(np.all(champ[ids]==champ[row]),'wrong neighbor champion')
        check(np.all(np.isfinite(d)) and np.all(d>=0) and np.all(np.diff(d)>=0),'invalid ordered distances')
        if len(ids):
            temp=max(float(d[-1])/4.,1e-6);expected=np.exp(-(d-d[0])/temp);expected/=expected.sum()
            check(np.allclose(w,expected,atol=1e-14,rtol=0),'kernel weight mismatch')
        for arm,source in (('memory',labels_all),('shuffled',shuffled[int(folds[row])])):
            base_probability=np.exp(p['state_target_log_probability'][j])
            expected=(.5*base_probability+.5*float(w[source[ids]==labels[j]].sum())) if len(ids) else base_probability
            error=abs(np.log(expected)-p[arm+'_target_log_probability'][j]);logp_errors.append(float(error))
            check(error<1e-10,'fixed mixture target probability mismatch')
    rng=np.random.default_rng(20261001);draws=rng.integers(0,len(labels),size=(2000,len(labels)))
    values={};recomputed={}
    for arm in ('state','memory','shuffled'):
        v={'top1':(p[arm+'_top1']==labels).astype(float),
           'top3':(p[arm+'_top3']==labels[:,None]).any(axis=1).astype(float),
           'nll':-p[arm+'_target_log_probability']};values[arm]=v
        recomputed[arm]={k:float(a.mean()) for k,a in v.items()}
        for key,a in v.items():
            check(abs(float(a.mean())-report['arms'][arm][key]['value'])<1e-12,'metric mean mismatch')
            check(np.allclose(np.quantile(a[draws].mean(axis=1),[.025,.975]),report['arms'][arm][key]['interval95'],atol=1e-12,rtol=0),'metric interval mismatch')
    comparisons={}
    for arm in ('state','shuffled'):
        comparisons[arm]={}
        for key in ('top1','top3','nll'):
            delta=values['memory'][key]-values[arm][key]
            ci=np.quantile(delta[draws].mean(axis=1),[.025,.975]);r=report['memory_minus_control'][arm][key]
            check(abs(float(delta.mean())-r['difference'])<1e-12 and np.allclose(ci,r['interval95'],atol=1e-12,rtol=0),'paired contrast mismatch')
            comparisons[arm][key]={'difference':float(delta.mean()),'interval95':ci.tolist()}
    gate=(comparisons['state']['top1']['difference']>=.03 and comparisons['state']['top1']['interval95'][0]>0
          and comparisons['shuffled']['top1']['interval95'][0]>0 and comparisons['state']['nll']['difference']<=0)
    check(bool(gate)==report['metric_gate_passed'],'wrong conclusion')
    check(report['heldout_target_mutation_rows']==3350 and not report['training_performed'] and not report['test_payload_reads']
          and not report['confirmation_payload_reads'] and not report['outcomes_analyzed'],'scope/mutation receipt failed')
    result={'schema':'purchase-memory-audit-v1','state':'complete','passed':True,'report_sha256':digest(report_path),
        'auditor_sha256':digest(__file__),'queries':len(selected),'neighbor_receipts':receipt_count,
        'maximum_independent_mixture_target_logp_error':max(logp_errors),'arms':recomputed,'paired':comparisons,
        'metric_gate_passed':bool(gate),'raw_payload_reads':False,'source_code_imported':False,
        'limitations':['Nearest-neighbor search and full ranking were not independently replayed; receipt checks, mixture target probabilities and all reported metrics were recomputed.','This is an internal unmasked catalogue screen, not an M2 displayed-policy comparison.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
    output.with_suffix('.sha256').write_text(digest(output)+' '+output.name+'\n',encoding='ascii')
    print(json.dumps({k:v for k,v in result.items() if k not in ('limitations','paired')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.directory.resolve(),args.output.resolve())
