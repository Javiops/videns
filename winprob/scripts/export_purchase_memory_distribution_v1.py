"""Reconstruct fixed full-catalogue distributions from already frozen memory receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from winprob import admission
from winprob.iql_dataset_contract import current_reservations
from winprob.ordinary_evaluation_io import validate_source,load_manifest,read_rows,digest
from winprob.purchase_memory_v1 import distribution,shuffled_labels

PROTOCOL=ROOT/'docs/purchase_memory_candidates_protocol_v1.md'
PRIOR=ROOT/'data/runs/purchase-memory-m2-20261001-v1/experiment'
SOURCE=ROOT.parent/'build_tracker/data/exports/winprob-ordinary/ordinary-contexts-20260930-v1/manifest.json'
REFERENCE=SOURCE.parent.parent/'ordinary-reference-20260930-v3/manifest.json'
BASE=ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json'


def check(ok,message):
    if not ok:raise ValueError(message)


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
    path.with_suffix('.sha256').write_text(digest(path)+' '+path.name+'\n',encoding='ascii')


def run(output):
    started=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    bindings={str(p.resolve()):digest(p) for p in (Path(__file__),PROTOCOL,ROOT/'winprob/purchase_memory_v1.py',ROOT/'winprob/ordinary_evaluation_io.py')}
    source,selection,contexts,admitted=validate_source(SOURCE,bindings)
    reference=load_manifest(REFERENCE,'winprob-ordinary-reference-v1',bindings)
    check(digest(REFERENCE)=='b1fdc8741109000d6732c133b35c0d15c54c6a59bda6eb915174401f561ddf90','frozen M2 reference changed')
    check(digest(PRIOR/'report.json')=='521c10ff70a38daa18686c8dd02d740eb1019ed25efafac0245ec940b1a32dfb','prior memory recipe changed')
    audit=ROOT/'data/reports/purchase_memory_m2_audit_v1.json'
    check(digest(audit)=='9bb81ec748afaac1e1143dea858322e3793e0f66f5e54b35261c859ae95eed0e','prior memory audit changed')
    check(json.loads(audit.read_text())['passed'] is True,'prior audit did not pass')
    prior=json.loads((PRIOR/'report.json').read_text())
    bindings[str(PRIOR/'report.json')]=digest(PRIOR/'report.json');bindings[str(audit)]=digest(audit)
    for name in ('neighbors.npz','prediction_orders.json'):
        check(digest(PRIOR/name)==prior['files'][name],'prior prediction receipt changed');bindings[str(PRIOR/name)]=prior['files'][name]
    check(digest(BASE)=='1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea','FIT cache changed')
    bindings[str(BASE)]=digest(BASE);base=json.loads(BASE.read_text())
    fit_admission=Path(base['source_admission_path']);check(digest(fit_admission)==base['source_admission_sha256'],'FIT admission changed')
    bindings[str(fit_admission)]=digest(fit_admission);fit=json.loads(fit_admission.read_text());admission.verify_manifest_sources(fit)
    forbidden=current_reservations(fit)|current_reservations(admitted);fit_ids={g['match_id'] for g in base['game_metadata']}
    query_ids={c['match_id'] for c in contexts};admit={g['match_id']:g for g in fit['games']}
    check(not (fit_ids|query_ids)&forbidden and not fit_ids&query_ids,'FIT/query/reservation overlap')
    check(all(admit[g]['role']=='train' and admit[g]['historical_validation'] is False for g in fit_ids),'memory contains non-FIT games')
    a={}
    for name in ('main_group_id','fold','game_index','champion_indices','patch_index','timestamp_ms'):
        path=BASE.parent/base['arrays'][name]['file'];check(digest(path)==base['files'][path.name],'FIT array changed')
        bindings[str(path)]=digest(path);a[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    refs=read_rows(REFERENCE.parent/'rows.jsonl');groups=json.loads((REFERENCE.parent/'group_main.json').read_text())
    orders=json.loads((PRIOR/'prediction_orders.json').read_text())
    with np.load(REFERENCE.parent/'probabilities.npz',allow_pickle=False) as archive:m2=archive['probabilities']
    check(m2.dtype==np.float32 and m2.shape==(859,9333),'M2 matrix shape/type changed')
    check(len(refs)==len(orders)==len(contexts)==859 and all(c['status']=='available' for c in contexts),'complete859 rows required')
    for i,(context,ref,order) in enumerate(zip(contexts,refs,orders)):
        check(context['sample_id']==ref['sample_id']==order['sample_id'] and ref['m2_tensor_row']==i,'prediction row order changed')
    ledger=[{'sample_id':c['sample_id'],'match_id':c['match_id'],'participant_id':c['participant_id'],'ordinal':i} for i,c in enumerate(contexts)]
    write(output/'selection.json',ledger)
    matrices={arm:.5*m2.astype(np.float64) for arm in ('memory','shuffled')}
    fallbacks={entry['row'] for entry in prior['feature_fallbacks']}
    with np.load(PRIOR/'neighbors.npz',allow_pickle=False) as archive:
        nr=archive['rows'];nw=archive['weights'];nd=archive['distances']
    for fold in (0,1):
        train=np.flatnonzero(a['fold']!=fold)
        shuffled,_=shuffled_labels(a['main_group_id'],train,a['champion_indices'][:,0],a['patch_index'],a['timestamp_ms'],fold)
        for i in range(859):
            if i in fallbacks:
                for matrix in matrices.values():matrix[i]=m2[i]
                continue
            valid=nr[fold,i]>=0;ids=nr[fold,i][valid];w=nw[fold,i][valid];d=nd[fold,i][valid]
            check(np.all(a['fold'][ids]!=fold) and len(np.unique(a['game_index'][ids]))==len(ids),'neighbor exclusion failed')
            if len(ids):
                expected=np.exp(-(d-d[0])/max(float(d[-1])/4.,1e-6));expected/=expected.sum()
                check(np.allclose(w,expected,atol=1e-14,rtol=0),'neighbor weights changed')
            for arm,labels in (('memory',a['main_group_id']),('shuffled',shuffled)):
                matrices[arm][i]+=.25*(distribution(labels[ids],w,9333) if len(ids) else m2[i])
    for arm,matrix in matrices.items():
        check(np.isfinite(matrix).all() and np.all(matrix>=0) and np.all(np.abs(matrix.sum(1)-1)<2e-6),'invalid distribution')
        for i,order in enumerate(orders):check(np.allclose(matrix[i,order['candidate_groups']],order['candidate_scores'][arm],atol=1e-14,rtol=0),'previous restricted scores do not reproduce')
    np.savez_compressed(output/'distributions.npz',m2=m2,**matrices)
    write(output/'group_main.json',groups)
    for raw,want in bindings.items():check(digest(raw)==want,'binding changed during export')
    check(not (fit_ids|query_ids)&(current_reservations(fit)|current_reservations(admitted)),'reservation changed during export')
    files={name:digest(output/name) for name in ('selection.json','distributions.npz','group_main.json')}
    manifest={'schema':'winprob-purchase-memory-distribution-v1','state':'complete','files':files,'selection_count':859,'group_count':9333,
        'source_manifest_path':str(SOURCE),'source_manifest_sha256':digest(SOURCE),'reference_manifest_path':str(REFERENCE),'reference_manifest_sha256':digest(REFERENCE),
        'protocol_path':str(PROTOCOL),'protocol_sha256':digest(PROTOCOL),'bindings_sha256':bindings,'prior_scores_reproduced':True,
        'row_order_sha256':hashlib.sha256(json.dumps([r['sample_id'] for r in ledger],separators=(',',':')).encode()).hexdigest(),
        'target_file_opened':False,'test_payload_reads':False,'confirmation_payload_reads':False,'outcomes_analyzed':False,'training_performed':False,
        'source_runtime_imported':False,'weights':{'m2':.5,'each_memory':.25},'elapsed_s':time.monotonic()-started}
    write(output/'manifest.json',manifest);print(json.dumps({'state':'complete','manifest':str(output/'manifest.json'),'elapsed_s':manifest['elapsed_s']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    run(parser.parse_args().output.resolve())
