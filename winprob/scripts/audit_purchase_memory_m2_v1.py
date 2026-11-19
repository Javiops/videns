"""Recompute M2 memory reranking, observed-purchase credits and paired intervals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import current_reservations


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check(ok,message):
    if not ok:raise ValueError(message)


def run(directory,output):
    report_path=directory/'report.json';report=json.loads(report_path.read_text())
    check(report['state']=='complete' and report['schema']=='purchase-memory-m2-v1','incomplete/wrong report')
    for raw,want in report['bindings_sha256'].items():check(digest(raw)==want,'changed binding: '+raw)
    for name,want in report['files'].items():check(digest(directory/name)==want,'changed output: '+name)
    orders=json.loads((directory/'prediction_orders.json').read_text())
    check(digest(directory/'prediction_orders.json')==report['prediction_orders_sha256_before_targets'],'prediction identity changed')
    base_path=ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json'
    check(digest(base_path)=='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea','base cache changed')
    base=json.loads(base_path.read_text());admitted=json.loads(Path(base['source_admission_path']).read_text())
    forbidden=current_reservations(admitted)
    check(not {g['match_id'] for g in base['game_metadata']}&forbidden,'reserved FIT game')
    arrays={}
    for name in ('main_group_id','game_index','fold','champion_indices','patch_index','timestamp_ms'):
        path=base_path.parent/base['arrays'][name]['file'];check(digest(path)==base['files'][path.name],'changed array')
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    root=ROOT.parent/'build_tracker/data/exports/winprob-ordinary'
    source_path=root/'ordinary-contexts-20260930-v1/manifest.json';source=json.loads(source_path.read_text())
    targets_path=source_path.parent/'targets.jsonl';check(digest(targets_path)==source['files']['targets.jsonl'],'changed observed targets')
    targets=[json.loads(line) for line in targets_path.read_text().splitlines()]
    ref_path=root/'ordinary-reference-20260930-v3/manifest.json';ref=json.loads(ref_path.read_text())
    refs_path=ref_path.parent/'rows.jsonl';check(digest(refs_path)==ref['files']['rows.jsonl'],'changed M2 options')
    refs=[json.loads(line) for line in refs_path.read_text().splitlines()]
    check(len(orders)==len(targets)==len(refs)==859,'population changed')
    check(not {t['match_id'] for t in targets}&forbidden,'reserved query')
    check(not {t['match_id'] for t in targets}&{g['match_id'] for g in base['game_metadata']},'query/FIT overlap')
    y=arrays['main_group_id'];folds=arrays['fold'];gi=arrays['game_index'];champs=arrays['champion_indices'][:,0]
    shuffled={}
    for fold in (0,1):
        buckets=defaultdict(list)
        for row in np.flatnonzero(folds!=fold):buckets[(int(champs[row]),int(arrays['patch_index'][row]),int(arrays['timestamp_ms'][row]//180000))].append(int(row))
        result=np.full(len(y),-1,dtype=y.dtype);rng=np.random.default_rng(20261001+fold)
        for key in sorted(buckets):
            members=np.asarray(sorted(buckets[key]));result[members]=y[rng.permutation(members)]
        shuffled[fold]=result
    memory=np.load(directory/'neighbors.npz',allow_pickle=False)
    fallbacks={r['row'] for r in report['feature_fallbacks']};credits={arm:{k:[] for k in ('main','typed_main','complete')} for arm in ('m2','memory','shuffled')}
    receipt_count=0
    for i,(order,target,reference) in enumerate(zip(orders,targets,refs,strict=True)):
        check(order['sample_id']==target['sample_id']==reference['sample_id'],'identity mismatch')
        original=reference['m2_options'];groups=np.array([o['group'] for o in original]);p=np.array(order['m2_candidate_probabilities'])
        check(order['ranked']['m2']==original,'changed M2 order')
        for arm,source_labels in (('memory',y),('shuffled',None)):
            scores=p.copy() if i in fallbacks else .5*p
            if i not in fallbacks:
                for fold in (0,1):
                    ids=memory['rows'][fold,i];valid=ids>=0;ids=ids[valid];w=memory['weights'][fold,i][valid];d=memory['distances'][fold,i][valid]
                    if arm=='memory':receipt_count+=len(ids)
                    check(np.all(folds[ids]!=fold) and len(np.unique(gi[ids]))==len(ids),'wrong memory fold/game')
                    if len(ids):
                        expected=np.exp(-(d-d[0])/max(float(d[-1])/4.,1e-6));expected/=expected.sum()
                        check(np.allclose(w,expected,atol=1e-14,rtol=0),'kernel weight changed')
                        labels=source_labels[ids] if arm=='memory' else shuffled[fold][ids]
                        scores+=.25*np.array([sum(w[labels==g]) for g in groups])
                    else:scores+=.25*p
            check(np.allclose(scores,order['candidate_scores'][arm],atol=1e-14,rtol=0),'mixture candidate score mismatch')
            ranked=[original[j] for j in sorted(range(len(original)),key=lambda j:(-scores[j],j))]
            check(ranked==order['ranked'][arm],'reranking/candidate membership changed')
        counts=target['target']['label_counts'];full=tuple(sorted((int(k),v) for k,v in counts.items()))
        if full==((999999,1),):full=()
        main=tuple(pair for pair in full if pair[0] not in (2055,772043));wardonly=bool(full) and not main
        for arm,ranked in order['ranked'].items():
            predicted=tuple(sorted((int(k),v) for k,v in ranked[0]['items'].items()))
            credits[arm]['main'].append(float(predicted==main))
            credits[arm]['typed_main'].append(float(predicted==main and not wardonly))
            credits[arm]['complete'].append(float(predicted==full))
    draws=np.random.default_rng(20261001).integers(0,859,size=(2000,859))
    values={a:{k:np.array(v) for k,v in types.items()} for a,types in credits.items()}
    for arm,types in values.items():
        for key,v in types.items():
            expected=report['arms'][arm][key]
            check(int(v.sum())==expected['hits'] and abs(float(v.mean())-expected['accuracy'])<1e-12,'observed-purchase metric mismatch')
            check(np.allclose(np.quantile(v[draws].mean(1),[.025,.975]),expected['interval95'],atol=1e-12,rtol=0),'arm interval mismatch')
    for control in ('m2','shuffled'):
        for key in ('main','typed_main','complete'):
            d=values['memory'][key]-values[control][key];expected=report['memory_minus_control'][control][key]
            check(abs(float(d.mean())-expected['gain'])<1e-12 and np.allclose(np.quantile(d[draws].mean(1),[.025,.975]),expected['interval95'],atol=1e-12,rtol=0),'paired interval mismatch')
            check(int(np.sum(d>0))==expected['corrected'] and int(np.sum(d<0))==expected['broken'],'movement mismatch')
    primary=report['memory_minus_control']['m2']['main'];negative=report['memory_minus_control']['shuffled']['main']
    gate=primary['gain']>=.03 and primary['interval95'][0]>0 and negative['interval95'][0]>0
    check(gate==report['useful_development_gate_passed'],'wrong gate conclusion')
    result={'schema':'purchase-memory-m2-audit-v1','state':'complete','passed':True,'report_sha256':digest(report_path),'auditor_sha256':digest(__file__),
        'games':859,'neighbor_receipts':receipt_count,'candidate_rankings_recomputed':1718,'observed_multisets_scored':859,
        'arms':report['arms'],'paired':report['memory_minus_control'],'useful_development_gate_passed':gate,
        'limitations':['Encoder/nearest search not rerun; neighbor weight and label aggregation, full candidate rankings, actual targets and all metrics were independently recomputed.','Prior native legality receipts are bound; original candidate objects are unchanged.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
    output.with_suffix('.sha256').write_text(digest(output)+' '+output.name+'\n',encoding='ascii')
    print(json.dumps({'passed':True,'games':859,'primary':primary,'gate':gate,'report':str(output)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.directory.resolve(),args.output.resolve())
