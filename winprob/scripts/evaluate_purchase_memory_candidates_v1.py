"""Compare complete native-decoded purchase memory with unchanged M2."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from winprob import admission
from winprob.iql_dataset_contract import current_reservations
from winprob.ordinary_evaluation_io import (validate_source,validate_targets,validate_audit_entries,read_rows,
    digest,load_catalogue,canonical)
from winprob.ordinary_policy_evaluation import target_description,recommendation_credit,option_multiset

PROTOCOL=ROOT/'docs/purchase_memory_candidates_protocol_v1.md'
ARMS=('m2','memory','shuffled','restricted_memory')
SOURCE=ROOT.parent/'build_tracker/data/exports/winprob-ordinary/ordinary-contexts-20260930-v1/manifest.json'
PRIOR=ROOT/'data/runs/purchase-memory-m2-20261001-v1/experiment'


def check(ok,message):
    if not ok:raise ValueError(message)


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
    path.with_suffix('.sha256').write_text(digest(path)+' '+path.name+'\n',encoding='ascii')


def bound(path,schema,bindings):
    check(path.with_suffix('.sha256').read_text().split()==[digest(path),path.name],'manifest sidecar mismatch')
    data=json.loads(path.read_text());check(data['schema']==schema and data['state']=='complete','wrong/incomplete artifact')
    bindings[str(path)]=digest(path)
    for name,want in data['files'].items():
        check(digest(path.parent/name)==want,'changed artifact member');bindings[str(path.parent/name)]=want
    for group in ('bindings_sha256','code_sha256'):
        for raw,want in data.get(group,{}).items():
            check(Path(raw).name!='targets.jsonl','prediction artifact unexpectedly follows targets')
            check(digest(raw)==want,'changed lineage');bindings[raw]=want
    return data


def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    bindings={str(p.resolve()):digest(p) for p in (Path(__file__),PROTOCOL)}
    source,selection,contexts,admitted=validate_source(SOURCE,bindings)
    dist=bound(args.distribution_manifest,'winprob-purchase-memory-distribution-v1',bindings)
    decoded=bound(args.decoded_manifest,'source-purchase-memory-decoder-v1',bindings)
    native=bound(args.native_manifest,'source-purchase-memory-native19-v1',bindings)
    for data in (dist,decoded,native):
        check(data['protocol_sha256']==digest(PROTOCOL) and data['selection_count']==859,'protocol/population mismatch')
        for flag in ('target_file_opened','test_payload_reads','confirmation_payload_reads','outcomes_analyzed','training_performed'):
            check(data[flag] is False,'source boundary failed: '+flag)
    check(decoded['distribution_manifest_sha256']==native['distribution_manifest_sha256']==digest(args.distribution_manifest),'distribution lineage mismatch')
    check(decoded['world']=='16.18' and native['world']=='16.19' and decoded['m2_replay_mismatches']==0
          and decoded['mutation_checked']==2577,'native replay/world/mutation mismatch')
    options=read_rows(args.decoded_manifest.parent/'rows.jsonl');audits=read_rows(args.native_manifest.parent/'rows.jsonl')
    check(len(options)==len(audits)==len(contexts)==859,'query population lost')
    catalogue,lookup=load_catalogue(source['catalogue_path'])
    check(json.loads((args.distribution_manifest.parent/'group_main.json').read_text())==catalogue['group_main'],'catalogue mapping changed')
    check(digest(PRIOR/'report.json')=='521c10ff70a38daa18686c8dd02d740eb1019ed25efafac0245ec940b1a32dfb','restricted recipe changed')
    prior_report=json.loads((PRIOR/'report.json').read_text());check(digest(PRIOR/'prediction_orders.json')==prior_report['files']['prediction_orders.json'],'restricted orders changed')
    prior=json.loads((PRIOR/'prediction_orders.json').read_text());bindings[str(PRIOR/'prediction_orders.json')]=digest(PRIOR/'prediction_orders.json')
    invalid=Counter();candidate_changes=0
    for i,(context,row,audit,old) in enumerate(zip(contexts,options,audits,prior)):
        check(row['sample_id']==audit['sample_id']==old['sample_id']==context['sample_id'] and row['ordinal']==audit['ordinal']==i,'query identity mismatch')
        for arm in ('m2','memory','shuffled'):
            for world,entries in (('native18',row['audits'][arm]),('native19',audit['audits'][arm])):
                receipt=validate_audit_entries(entries,row['options'][arm],context,catalogue['group_main'])
                invalid[world+'_'+arm]+=sum(not x['feasible'] for x in receipt.values())
        check(row['options']['m2']==old['ranked']['m2'],'M2 reproduction mismatch')
        candidate_changes+=int({o['group'] for o in row['options']['memory']}!={o['group'] for o in row['options']['m2']})
        row['options']['restricted_memory']=old['ranked']['memory']
    prediction_sha=hashlib.sha256(canonical(options)).hexdigest()
    write(args.output/'validated_prediction_orders.json',options)
    targets=validate_targets(SOURCE.parent/'targets.jsonl',source,contexts,catalogue,lookup,bindings)
    scores=[];credits={arm:{k:[] for k in ('main','typed_main','complete','top3')} for arm in ARMS}
    for context,row,target in zip(contexts,options,targets):
        truth=target_description(target['target']['label_counts']);entry={'sample_id':context['sample_id'],'match_id':context['match_id'],'target_kind':truth['kind'],'credits':{}}
        for arm in ARMS:
            opts=row['options'][arm];v=recommendation_credit(truth,opts[0]);v['top3']=float(any(option_multiset(o)==truth['main'] for o in opts))
            entry['credits'][arm]=v
            for key,value in v.items():credits[arm][key].append(value)
        scores.append(entry)
    values={a:{k:np.array(v) for k,v in metrics.items()} for a,metrics in credits.items()}
    check(values['m2']['main'].sum()==454 and values['m2']['complete'].sum()==354 and values['m2']['top3'].sum()==591,'original M2 metrics do not reproduce')
    check(values['restricted_memory']['main'].sum()==464,'restricted primary does not reproduce')
    draws=np.random.default_rng(20261001).integers(0,859,size=(2000,859));metrics={};paired={}
    for arm,v in values.items():metrics[arm]={k:{'hits':int(x.sum()),'accuracy':float(x.mean()),'interval95':np.quantile(x[draws].mean(1),[.025,.975]).tolist()} for k,x in v.items()}
    for control in ('m2','shuffled','restricted_memory'):
        paired[control]={}
        for key in ('main','typed_main','complete','top3'):
            delta=values['memory'][key]-values[control][key]
            paired[control][key]={'gain':float(delta.mean()),'interval95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                'corrected':int(np.sum(delta>0)),'broken':int(np.sum(delta<0))}
    primary=paired['m2']['main'];stat=primary['gain']>=.03 and primary['interval95'][0]>0 and paired['shuffled']['main']['interval95'][0]>0 and paired['m2']['top3']['gain']>=0
    check(hashlib.sha256(canonical(options)).hexdigest()==prediction_sha,'orders changed after target access')
    for raw,want in bindings.items():check(digest(raw)==want,'binding changed before publication')
    admission.verify_manifest_sources(admitted);check(not {c['match_id'] for c in contexts}&current_reservations(admitted),'query became reserved')
    write(args.output/'scored_rows.json',scores)
    result={'schema':'purchase-memory-candidates-result-v1','state':'complete','games':859,'scope':'exposed development; full native catalogue decode',
        'arms':metrics,'memory_minus_control':paired,'candidate_sets_changed_games':candidate_changes,'infeasible_counts':dict(invalid),
        'statistics_gate_passed':bool(stat),'useful_development_gate_passed':bool(stat and not any(invalid.values())),
        'm2_replayed':True,'restricted_primary_reproduced':True,'prediction_orders_sha256_before_targets':prediction_sha,
        'bindings_sha256':bindings,'files':{name:digest(args.output/name) for name in ('scored_rows.json','validated_prediction_orders.json')},
        'source_manifests':{k:str(v) for k,v in vars(args).items() if k.endswith('_manifest')},
        'training_performed':False,'test_payload_reads':False,'confirmation_payload_reads':False,'outcomes_analyzed':False,'promotion_authorized':False,'breakthrough_established':False,
        'limits':['Previously exposed development, not confirmation.','Legal recommendations are conditional on estimated wallet.','Ward quantities are outside the main action space.']}
    write(args.output/'report.json',result)
    print(json.dumps({'state':'complete','primary':primary,'top3':paired['m2']['top3'],'gate':result['useful_development_gate_passed'],'infeasible_counts':dict(invalid)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('distribution-manifest','decoded-manifest','native-manifest','output'):parser.add_argument('--'+name,required=True,type=Path)
    run(parser.parse_args())
