"""Independent observed-multiset and whole-game interval audit of native outputs."""
from __future__ import annotations
import argparse
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


def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines()]
def basket(counts):return tuple(sorted((int(k),int(v)) for k,v in counts.items()))


def run(directory,output):
    path=directory/'report.json';report=json.loads(path.read_text())
    check(report['state']=='complete' and report['schema']=='purchase-memory-candidates-result-v1','wrong/incomplete report')
    check(path.with_suffix('.sha256').read_text().split()==[digest(path),path.name],'report sidecar changed')
    for raw,want in report['bindings_sha256'].items():check(digest(raw)==want,'changed binding: '+raw)
    for name,want in report['files'].items():check(digest(directory/name)==want,'changed report member')
    prediction=json.loads((directory/'validated_prediction_orders.json').read_text())
    prediction_bytes=json.dumps(prediction,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    check(hashlib.sha256(prediction_bytes).hexdigest()==report['prediction_orders_sha256_before_targets'],'prediction order changed')
    source_path=ROOT.parent/'build_tracker/data/exports/winprob-ordinary/ordinary-contexts-20260930-v1/manifest.json'
    source=json.loads(source_path.read_text());target_path=source_path.parent/'targets.jsonl'
    check(digest(target_path)==source['files']['targets.jsonl'],'actual target file changed')
    targets=read_rows(target_path);admitted=json.loads(Path(source['admission_path']).read_text())
    check(not {t['match_id'] for t in targets}&current_reservations(admitted),'query became reserved')
    decoded_path=Path(report['source_manifests']['decoded_manifest']);decoded=json.loads(decoded_path.read_text())
    native_path=Path(report['source_manifests']['native_manifest']);native=json.loads(native_path.read_text())
    decoded_rows=read_rows(decoded_path.parent/'rows.jsonl');native_rows=read_rows(native_path.parent/'rows.jsonl')
    dist_path=Path(report['source_manifests']['distribution_manifest']);dist=json.loads(dist_path.read_text())
    check(decoded['distribution_manifest_sha256']==native['distribution_manifest_sha256']==digest(dist_path),'distribution lineage changed')
    check(decoded['m2_replay_mismatches']==0 and decoded['mutation_checked']==2577,'missing decoder parity/mutation evidence')
    arms=('m2','memory','shuffled','restricted_memory')
    values={arm:{k:[] for k in ('main','typed_main','complete','top3')} for arm in arms};invalid={};costs={arm:[] for arm in arms[:3]};changed=0
    check(len(prediction)==len(targets)==len(decoded_rows)==len(native_rows)==859,'incomplete query ledger')
    for row,target,decoded_row,native_row in zip(prediction,targets,decoded_rows,native_rows,strict=True):
        check(row['sample_id']==target['sample_id']==decoded_row['sample_id']==native_row['sample_id'],'query identity mismatch')
        full=basket(target['target']['label_counts'])
        if full==((999999,1),):full=()
        main=tuple(p for p in full if p[0] not in (2055,772043));ward_only=bool(full) and not main
        for arm in arms:
            options=row['options'][arm];first=basket(options[0]['items'])
            check(1<=len(options)<=3 and len({o['group'] for o in options})==len(options),'invalid card set')
            values[arm]['main'].append(float(first==main));values[arm]['typed_main'].append(float(first==main and not ward_only))
            values[arm]['complete'].append(float(first==full));values[arm]['top3'].append(float(any(basket(o['items'])==main for o in options)))
            if arm=='restricted_memory':continue
            check(options==decoded_row['options'][arm],'post-decoder options changed')
            for world,audits in (('native18',decoded_row['audits'][arm]),('native19',native_row['audits'][arm])):
                check(len(audits)==len(options),'missing native audit')
                for option,receipt in zip(options,audits,strict=True):
                    raw=json.dumps(option,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
                    check(receipt['option_sha256']==hashlib.sha256(raw).hexdigest() and receipt['group']==option['group'],'audit candidate mismatch')
                invalid.setdefault(world+'_'+arm,0);invalid[world+'_'+arm]+=sum(not r['feasible'] for r in audits)
            costs[arm].append(native_row['audits'][arm][0]['incremental_cost'])
        changed+=int({o['group'] for o in row['options']['memory']}!={o['group'] for o in row['options']['m2']})
    check(invalid==report['infeasible_counts'] and changed==report['candidate_sets_changed_games'],'candidate/legality counts differ')
    values={a:{k:np.asarray(v) for k,v in types.items()} for a,types in values.items()}
    draws=np.random.default_rng(20261001).integers(0,859,size=(2000,859))
    for arm,types in values.items():
        for key,v in types.items():
            m=report['arms'][arm][key];check(int(v.sum())==m['hits'] and abs(v.mean()-m['accuracy'])<1e-12,'metric differs')
            check(np.allclose(np.quantile(v[draws].mean(1),[.025,.975]),m['interval95'],rtol=0,atol=1e-12),'arm CI differs')
    for control in ('m2','shuffled','restricted_memory'):
        for key in ('main','typed_main','complete','top3'):
            delta=values['memory'][key]-values[control][key];m=report['memory_minus_control'][control][key]
            check(abs(delta.mean()-m['gain'])<1e-12 and int(np.sum(delta>0))==m['corrected'] and int(np.sum(delta<0))==m['broken'],'paired mean/movement differs')
            check(np.allclose(np.quantile(delta[draws].mean(1),[.025,.975]),m['interval95'],rtol=0,atol=1e-12),'paired CI differs')
    p=report['memory_minus_control']['m2']['main'];s=report['memory_minus_control']['shuffled']['main']
    stats=p['gain']>=.03 and p['interval95'][0]>0 and s['interval95'][0]>0 and report['memory_minus_control']['m2']['top3']['gain']>=0
    check(bool(stats)==report['statistics_gate_passed'] and bool(stats and not any(invalid.values()))==report['useful_development_gate_passed'],'gate conclusion differs')
    result={'schema':'purchase-memory-candidates-audit-v1','state':'complete','passed':True,'games':859,'report_sha256':digest(path),'auditor_sha256':digest(__file__),
        'candidate_sets_changed_games':changed,'infeasible_counts':invalid,'arms':report['arms'],'paired':report['memory_minus_control'],
        'useful_development_gate_passed':report['useful_development_gate_passed'],
        'first_option_native19_costs':{arm:{'known':sum(x is not None for x in vals),'mean':float(np.mean([x for x in vals if x is not None]))} for arm,vals in costs.items()},
        'limits':['Native decoder and economic engine were not independently replayed; their immutable source receipts are checked.','Every observed target multiset, metric, movement count, bootstrap interval and gate was independently recomputed.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False)
    output.with_suffix('.sha256').write_text(digest(output)+' '+output.name+'\n',encoding='ascii')
    print(json.dumps({'passed':True,'primary':p,'top3':report['memory_minus_control']['m2']['top3'],'gate':report['useful_development_gate_passed'],'costs':result['first_option_native19_costs']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();run(args.directory.resolve(),args.output.resolve())
