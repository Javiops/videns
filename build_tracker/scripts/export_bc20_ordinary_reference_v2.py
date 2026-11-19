"""Capture unchanged native M2 options/P and decode frozen BC on the same rows.

This source-owned stage never opens targets, SQLite, raw games or a mixed
training export. It must run in its own clean process, pinned to M2's16.18
runtime. The subsequent16.19 audit is a separate process.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import export_winprob_trajectories_v1 as producer
import export_bc20_ordinary_contexts_v1 as source

SCHEMA='winprob-ordinary-reference-v1'
PACKAGE_DIGEST='ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9'
CHECKPOINT_DIGEST='57fca2f787aaff6dce43bfd729e34e90464de2f536abdf4ba534d5d01e5b1f0d'
REVIEW=ROOT/'data/acceptance/16.18_reconstructible_20260923/inputs/static_review.json'
IDENTITY_KEYS=('sample_id','match_id','participant_id','role','patch','status')


def require(condition,message):
    if not condition: raise ValueError(message)


def read_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def write_json(path,value):
    path=Path(path)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    source.sidecar(path)


def revalidate(bindings):
    for path,expected in bindings.items():
        require(producer.file_hash(path)==expected,'Bound input/code changed: '+str(path))


def metadata_bindings(values):
    """A target digest may be copied, but never followed even for hashing."""
    require(isinstance(values,dict) and bool(values),'Metadata/code bindings required')
    result={}
    for raw,expected in values.items():
        path=Path(raw).resolve(strict=True)
        require(path.name!='targets.jsonl' and 'raw_match_v5' not in path.parts
                and path.suffix.lower() not in ('.gz','.db','.sqlite'),
                'Forbidden raw/target/database provenance path')
        require(isinstance(expected,str) and len(expected)==64,'Malformed input digest')
        result[str(path)]=expected
    revalidate(result)
    return result


def validate_context_rows(manifest,selection,rows):
    entries=selection['entries']
    require(len(rows)==len(entries)==859,'Every selected game must remain in the ledger')
    next_tensor=0
    for ordinal,(entry,row) in enumerate(zip(entries,rows,strict=True)):
        mid=entry['match_id'];choice=entry['selected_decision']
        require(not any(key in row for key in ('target','label_counts','win','outcome','full_action_id','main_group_id')),
                'Observed target/outcome reached the context ledger')
        require(row.get('ordinal')==ordinal and row.get('sample_id')==entry['sample_id']
                and row.get('match_id')==mid and row.get('participant_id')==entry['pid']
                and row.get('role')=='development' and row.get('patch')=='16.19'
                and row.get('selected_decision')==choice,'Context selection identity changed')
        require(row.get('status') in ('available','unavailable','no_admitted_decision'),'Unknown context status')
        if choice is None:
            require(row['status']=='no_admitted_decision' and row.get('tensor_row') is None,'Missing decision replaced')
        else:
            require(row['status']!='no_admitted_decision','Selected decision disappeared')
        if row['status']=='available':
            projection=row.get('source_projection')
            require(type(row.get('tensor_row')) is int and row['tensor_row']==next_tensor
                    and isinstance(projection,dict) and producer.value_hash(projection)==row.get('projection_sha256'),
                    'Invalid available projection/index')
            require(type(row.get('context_timestamp_ms')) is int and row['context_timestamp_ms']==choice['ts']
                    and projection.get('ts')==choice['ts'] and row.get('prestate',{}).get('ts')==choice['ts'],
                    'Projection is not the selected predecision time')
            require(all(row.get('checks',{}).get(key) is True for key in
                        ('selected_identity_matches','prefix_wallet_reproduced','target_future_projection_mutation_equal')),
                    'Context reconstruction certificate incomplete')
            require(not any(key in projection for key in ('target','label_counts','win','outcome','source_target_id')),
                    'Projection includes forbidden observed target/outcome fields')
            next_tensor+=1
        else:
            require(row.get('tensor_row') is None,'Unavailable context has tensor index')
    require(manifest.get('available_contexts')==next_tensor,'Source availability accounting changed')
    return next_tensor


def load_contexts(path):
    path=Path(path).resolve(strict=True);manifest,digest=producer.load_bound(path)
    require(manifest.get('schema')==source.CONTEXT_SCHEMA and manifest.get('state')=='complete'
            and manifest.get('smoke') is False and manifest.get('full_cohort_complete') is True
            and manifest.get('selection_count')==859,'Completed full859 context artifact required')
    for flag in ('test_payload_reads','confirmation_payload_reads','outcomes_exported','outcomes_analyzed','training_performed'):
        require(manifest.get(flag) is False,'Context boundary flag failed: '+flag)
    bindings=metadata_bindings(manifest['bindings'])
    bindings[str(path)]=digest
    for name in ('selection.json','contexts.jsonl'):
        value=producer.file_hash(path.parent/name)
        require(value==manifest['files'][name],'Context artifact file changed: '+name)
        bindings[str(path.parent/name)]=value
    # Do not open/hash targets.jsonl; preserve only its opaque manifest identity.
    target=manifest['files'].get('targets.jsonl')
    require(isinstance(target,str) and len(target)==64,'Separate opaque target digest missing')
    selection,native_source,selected_bindings,denied=source.load_selection(path.parent/'selection.json')
    require(set(selection['admitted_ids']).isdisjoint(denied),'Reserved identity in ordinary population')
    bindings.update(selected_bindings)
    for key in ('admission','protocol','static_export','catalogue'):
        item=Path(manifest[key+'_path']).resolve(strict=True)
        require(producer.file_hash(item)==manifest[key+'_sha256'],'Source metadata changed: '+key)
        bindings[str(item)]=manifest[key+'_sha256']
    require(manifest['admission_sha256']==selection['admission_sha256']
            and manifest['protocol_sha256']==selection['protocol_sha256'],'Context/selection provenance disagreement')
    rows=read_lines(path.parent/'contexts.jsonl')
    validate_context_rows(manifest,selection,rows)
    return manifest,rows,selection,native_source,bindings


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def validate_probabilities(array,rows,groups=9333):
    require(array.dtype==np.float32 and array.shape==(rows,groups),'Wrong probability axes/dtype')
    require(np.isfinite(array).all() and (array>=0).all()
            and np.all(np.abs(array.sum(1,dtype=np.float64)-1)<=2e-6),
            'Probability mass violates the unchanged2e-6 guard')


def validate_actor_rows(rows,actor_rows,probabilities):
    require(len(rows)==len(actor_rows)==859,'Actor ledger dropped a selected game')
    next_tensor=0
    for context,actor in zip(rows,actor_rows,strict=True):
        require(all(actor.get(key)==context.get(key) for key in IDENTITY_KEYS)
                and actor.get('source_tensor_row')==context.get('tensor_row'),'Actor/context row identity differs')
        state=actor.get('bc_status')
        if context['status']=='available':
            require(state in ('scored','bc_feature_fallback'),'Available actor context has invalid status')
        else:
            require(state==('no_admitted_decision' if context['status']=='no_admitted_decision' else 'source_unavailable'),
                    'Unavailable actor row was silently replaced')
        if state=='scored':
            require(type(actor.get('bc_tensor_row')) is int and actor['bc_tensor_row']==next_tensor
                    and actor.get('ignored_field_mutation_equal') is True,'Actor index/mutation coverage failed')
            require(next_tensor<len(probabilities) and actor.get('row_sha256',{}).get('group_probabilities')==array_hash(probabilities[next_tensor]),
                    'Actor probability row digest differs')
            next_tensor+=1
        else:
            require(actor.get('bc_tensor_row') is None,'Fallback actor row has tensor data')
    validate_probabilities(probabilities,next_tensor)
    return next_tensor


def load_actor(path,source_path,source_manifest,rows):
    path=Path(path).resolve(strict=True);manifest,digest=producer.load_bound(path)
    require(manifest.get('schema')=='winprob-ordinary-bc20-export-v1' and manifest.get('state')=='complete'
            and manifest.get('selection_count')==859 and manifest.get('group_count')==9333
            and manifest.get('checkpoint_sha256')==CHECKPOINT_DIGEST,'Only frozen BC20 ordinary export is supported')
    for flag in ('target_file_opened','raw_payload_reads','test_payload_reads','confirmation_payload_reads',
                 'targets_analyzed','outcomes_analyzed','training_performed','mixture_or_accuracy_computed'):
        require(manifest.get(flag) is False,'Actor boundary flag failed: '+flag)
    require(Path(manifest['source_manifest_path']).resolve()==Path(source_path).resolve()
            and manifest['source_manifest_sha256']==producer.file_hash(source_path)
            and manifest['source_contexts_sha256']==source_manifest['files']['contexts.jsonl']
            and manifest['source_selection_sha256']==source_manifest['files']['selection.json']
            and manifest['opaque_targets_sha256_not_opened']==source_manifest['files']['targets.jsonl'],
            'Actor source lineage differs from reference contexts')
    bindings=metadata_bindings(manifest['bindings_sha256']);bindings[str(path)]=digest
    for name in ('distribution.npz','rows.jsonl','group_main.json','full_to_group.json'):
        value=producer.file_hash(path.parent/name)
        require(value==manifest['files'][name],'Actor file digest changed: '+name)
        bindings[str(path.parent/name)]=value
    actor_rows=read_lines(path.parent/'rows.jsonl')
    with np.load(path.parent/'distribution.npz',allow_pickle=False) as archive:
        probabilities=archive['group_probabilities']
    groups=producer.load_json(path.parent/'group_main.json')
    require(len(groups)==9333 and groups[0]==[],'Actor canonical class map invalid')
    count=validate_actor_rows(rows,actor_rows,probabilities)
    require(count==manifest.get('scored_rows')==manifest.get('mutation_checked_rows')
            and manifest.get('mutation_failures')==0,'Actor mutation/scoring accounting incomplete')
    return manifest,actor_rows,probabilities,groups,bindings


def model_row(context):
    """Labels exist only to satisfy ShopDataset; they are constant placeholders."""
    return {**deepcopy(context['source_projection']),'match_id':context['match_id'],
            'label_counts':{'999999':1},'target_encoding':'item-counts-v1',
            'label_id':999999,'save_kind':'no_buy_death'}


def mutate_ignored(row):
    changed=deepcopy(row)
    changed.update(label_counts={'2055':1},label_id=2055,save_kind='build_spend',
                   source_target_id=999999999,gold_arrival_true=10**9,win='mutation-only',
                   outcome='mutation-only',decision='mutation-only',gold_after_complete=-1,
                   future_state={'win':False,'gold':10**9},target={'label_counts':{'1001':1}})
    return changed


def options_override(policy,rows,probabilities):
    original=policy.probabilities
    def supplied(requested):
        require(requested is rows,'Native options changed its supplied row batch')
        return probabilities
    try:
        policy.probabilities=supplied
        return policy.options(rows)
    finally:
        policy.probabilities=original


def capture_native(policy,rows):
    """One unmodified neural call inside unchanged native options; no reranking."""
    original=policy.probabilities;captured=[]
    def capture(requested):
        require(requested is rows,'Native options changed its original row batch')
        values=original(requested);captured.append(values);return values
    try:
        policy.probabilities=capture
        options=policy.options(rows)
    finally:
        policy.probabilities=original
    require(len(captured)==1,'Native options did not invoke exactly one original probability call')
    return captured[0],options


def validate_options(options,groups):
    require(isinstance(options,list) and 1<=len(options)<=3,'Native candidate count outside1..3')
    seen=set()
    for option in options:
        gid=option.get('group')
        require(type(gid) is int and 0<=gid<len(groups) and gid not in seen,'Bad or repeated candidate group')
        require(option.get('legal') is True and option.get('items')=={str(item):count for item,count in groups[gid]}
                and option.get('kind')==('save' if gid==0 else 'buy'),'Candidate object does not represent its canonical group')
        seen.add(gid)


def audit_options(options,row,groups,purchase_sequence,dragon):
    validate_options(options,groups);result=[]
    for option in options:
        gid=option['group'];action=tuple(item for item,count in groups[gid] for _ in range(count))
        sequence=[] if gid==0 else purchase_sequence(action,row,float(row['gold_est']),dragon)
        cost=None
        if sequence is not None:
            require(isinstance(sequence,list),'Unexpected native purchase_sequence schema')
            costs=[step['cost'] for step in sequence]
            require(all(isinstance(cost,(int,float)) and not isinstance(cost,bool) and math.isfinite(cost) and cost>=0 for cost in costs),
                    'Invalid incremental native cost')
            require(Counter(step['item_id'] for step in sequence)==Counter(action),'Native purchase sequence changed the candidate multiset')
            cost=float(sum(costs))
            require(cost<=float(row['gold_est'])+1e-6,'Native feasible sequence exceeds supplied wallet')
        result.append({'group':gid,'option_sha256':producer.value_hash(option),'feasible':sequence is not None,
                       'incremental_cost':cost,'sequence':sequence})
    return result


def runtime_code_bindings():
    values={}
    for module in list(sys.modules.values()):
        raw=getattr(module,'__file__',None)
        if not raw: continue
        path=Path(raw).resolve()
        if path.is_file() and path.is_relative_to(ROOT) and path.suffix=='.py' and '.venv' not in path.parts and 'data' not in path.relative_to(ROOT).parts:
            values[str(path)]=producer.file_hash(path)
    values[str(Path(__file__).resolve())]=producer.file_hash(__file__)
    return values


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Refuse to overwrite an ordinary reference artifact')
    require(1<=args.batch_size<=16 and 1<=args.deadline_seconds<=3000,'Bounded CPU batch/deadline required')
    manifest,contexts,selection,native_source,bindings=load_contexts(args.source_manifest)
    actor_manifest,actor_rows,bc,groups,actor_bindings=load_actor(args.actor_manifest,args.source_manifest,manifest,contexts)
    bindings.update(actor_bindings)
    import app.mixture_policy as mp
    active=mp.activate_runtime(producer.PACKAGE,REVIEW)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from app.conditional_policy import purchase_sequence
    policy=mp.load_package('mixture-v4-m2-v1',device='cpu')
    actual_groups=[[[int(item),int(count)] for item,count in group] for group in policy.group_main]
    require(policy.digest==PACKAGE_DIGEST and policy.n_act==9333 and actual_groups==groups
            and policy.shortlist==128 and policy.n_options==3,'Frozen native policy identity/decoder changed')
    for path in (producer.PACKAGE/'manifest.json',producer.PACKAGE/'package_id.json',REVIEW):
        bindings[str(path.resolve())]=producer.file_hash(path)
    for entry in policy.manifest['weights'].values():
        path=producer.PACKAGE/entry['file'];bindings[str(path.resolve())]=entry['sha256']
    code=runtime_code_bindings();revalidate({**bindings,**code})
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n',encoding='utf-8')
    def guard():
        require(time.monotonic()-started<=args.deadline_seconds,'Ordinary reference deadline exceeded')
    available=[i for i,row in enumerate(contexts) if row['status']=='available']
    matrix=np.empty((len(available),9333),dtype=np.float32)
    ledger=[{**{key:row[key] for key in IDENTITY_KEYS},'ordinal':i,'source_tensor_row':row.get('tensor_row'),
             'context_timestamp_ms':row.get('context_timestamp_ms'),'projection_sha256':row.get('projection_sha256'),
             'm2_tensor_row':None,'bc_tensor_row':actor_rows[i].get('bc_tensor_row'),'bc_status':actor_rows[i]['bc_status'],
             'reference_status':'source_unavailable' if row['status']!='no_admitted_decision' else 'no_admitted_decision',
             'm2_options':[],'direct_bc_options':[],'native18_audit':{'m2':[],'direct_bc':[]},
             'zero_residual_native_options_equal':None,'whole_distribution_label_mutation_equal':None,
             'direct_bc_fallback':None} for i,row in enumerate(contexts)]
    try:
        for start in range(0,len(available),args.batch_size):
            guard();indexes=available[start:start+args.batch_size];rows=[model_row(contexts[i]) for i in indexes]
            changed_rows=[mutate_ignored(row) for row in rows]
            with torch.inference_mode():
                p,native=capture_native(policy,rows)
                changed=policy.probabilities(changed_rows)
                require(torch.equal(p,changed),'M2 distribution depends on forbidden targets/future fields')
                values=p.detach().cpu().numpy();validate_probabilities(values,len(rows))
                replay=options_override(policy,rows,p)
                require(producer.canonical(native)==producer.canonical(replay),'Exact-P native decoder parity failed')
                changed_options=options_override(policy,changed_rows,p)
                require(producer.canonical(native)==producer.canonical(changed_options),'Native options depend on forbidden targets/future fields')
                direct_values=np.array(values,copy=True)
                for local,i in enumerate(indexes):
                    if actor_rows[i]['bc_status']=='scored':direct_values[local]=bc[actor_rows[i]['bc_tensor_row']]
                validate_probabilities(direct_values,len(rows))
                direct_p=torch.from_numpy(direct_values).to(policy.device)
                direct=options_override(policy,rows,direct_p)
                changed_direct=options_override(policy,changed_rows,direct_p)
                require(producer.canonical(direct)==producer.canonical(changed_direct),'Direct BC native options depend on forbidden targets/future fields')
            matrix[start:start+len(rows)]=values
            for local,i in enumerate(indexes):
                for opts in (native[local],direct[local]):validate_options(opts,groups)
                native18={'m2':audit_options(native[local],rows[local],groups,purchase_sequence,policy.dragon),
                          'direct_bc':audit_options(direct[local],rows[local],groups,purchase_sequence,policy.dragon)}
                require(all(entry['feasible'] for arm in native18.values() for entry in arm),'Native16.18 returned an infeasible candidate')
                fallback=actor_rows[i]['bc_status']!='scored'
                if fallback:require(native[local]==direct[local],'BC feature fallback altered original M2 options')
                ledger[i].update(reference_status='scored',m2_tensor_row=start+local,
                                 m2_options=native[local],direct_bc_options=direct[local],native18_audit=native18,
                                 zero_residual_native_options_equal=True,whole_distribution_label_mutation_equal=True,
                                 native_options_label_mutation_equal=True,
                                 target_future_native_options_equal=True,target_future_direct_bc_options_equal=True,
                                 probability_row_sha256=array_hash(values[local]),probability_row_sum=float(values[local].sum(dtype=np.float64)),
                                 direct_bc_fallback='original_m2' if fallback else None)
            status('scoring',processed_contexts=min(start+args.batch_size,len(available)),available_contexts=len(available))
        guard();revalidate({**bindings,**code});producer.validate_admission(manifest['admission_path'])
        require(runtime_code_bindings()==code,'Inference loaded an unbound source module')
        np.savez_compressed(output/'probabilities.npz',probabilities=matrix)
        write_json(output/'group_main.json',groups)
        with (output/'rows.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for row in ledger:stream.write(producer.canonical(row).decode('utf-8')+'\n')
        files={name:source.sidecar(output/name) for name in ('rows.jsonl','probabilities.npz','group_main.json')}
        result={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'files':files,'selection_count':859,'scored_rows':len(available),'group_count':9333,
                'row_order_sha256':producer.value_hash([row['sample_id'] for row in ledger]),
                'package_identity_sha256':policy.digest,'reference_world_patch':'16.18','native_context_patch':'16.19',
                'source_manifest_path':str(Path(args.source_manifest).resolve()),'source_manifest_sha256':producer.file_hash(args.source_manifest),
                'source_selection_sha256':manifest['files']['selection.json'],'source_contexts_sha256':manifest['files']['contexts.jsonl'],
                'actor_manifest_path':str(Path(args.actor_manifest).resolve()),'actor_manifest_sha256':producer.file_hash(args.actor_manifest),
                'opaque_targets_sha256_not_opened':manifest['files']['targets.jsonl'],'protocol_path':manifest['protocol_path'],
                'protocol_sha256':manifest['protocol_sha256'],'bindings_sha256':bindings,'code_sha256':code,'reference_runtime':active,
                'parity_checked_rows':len(available),'mutation_checked_rows':len(available),'zero_residual_native_options_mismatches':0,
                'whole_distribution_label_mutation_equal':True,'direct_bc_decoded_rows':sum(row['bc_status']=='scored' for row in ledger),
                'native_options_label_mutation_equal':True,
                'target_future_native_options_equal':True,'target_future_direct_bc_options_equal':True,
                'native_options_mutation_checked_rows':len(available),'direct_bc_options_mutation_checked_rows':len(available),
                'direct_bc_fallback_rows':sum(row['direct_bc_fallback']=='original_m2' for row in ledger),
                'status_counts':dict(Counter(row['reference_status'] for row in ledger)),
                'original_probability_calls':2*((len(available)+args.batch_size-1)//args.batch_size),
                'capture_contract':'One original forward inside unchanged M2.options, one forbidden-target/future mutation forward; exact-P override ordered full-object parity.',
                'decoder_contract':{'shortlist':128,'options':3,'fallback':'native SAVE','direct_bc':'same native options with exported BC group masses'},
                'probability_array':{'file':'probabilities.npz','key':'probabilities','dtype':'float32','shape':list(matrix.shape),'row_sum_atol':2e-6},
                'target_file_opened':False,'targets_analyzed':False,'outcomes_analyzed':False,'raw_payload_reads':False,
                'test_payload_reads':False,'confirmation_payload_reads':False,'training_performed':False,'accuracy_computed':False,
                'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'python':sys.version,'device':'cpu','threads':1,'batch_size':args.batch_size},
                'elapsed_s':time.monotonic()-started,
                'limits':['Development cohort is exposed, not confirmatory.','Supplied causal estimated wallet is not an observed live exact wallet.',
                          'Native16.19 legality is audited separately without replacing these candidates.','Main groups omit control wards; group0 is not a full SAVE-action claim.']}
        write_json(output/'manifest.json',result);status('complete',scored_rows=len(available));return result
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True,type=Path)
    parser.add_argument('--actor-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--deadline-seconds',type=int,default=2400)
    result=run(parser.parse_args())
    print(json.dumps({key:result[key] for key in ('scored_rows','parity_checked_rows','mutation_checked_rows','elapsed_s')}))


if __name__=='__main__':main()
