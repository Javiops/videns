"""Explicit observed-target feasibility diagnostic; no models or raw games.

DO NOT RUN until the owner authorizes reading the named observed target file.
This is a post-result explanation of the fixed108 catalogue-top3 misses,
not a new policy, score, tuned decoder, or promotion. Each patch requires its
own clean process. Budget relaxation is diagnostic only and never exported
as a recommendation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import export_bc20_ordinary_reference_v2 as reference
import audit_bc20_ordinary_native19_v2 as native_audit
producer=reference.producer
source=reference.source
require=reference.require
SCHEMA='winprob-ordinary-target-feasibility-diagnostic-v1'
GENEROUS_WALLET=1000000
EXPECTED_CASES=108


def check_sequence(sequence,action,wallet):
    if sequence is None:return None
    require(isinstance(sequence,list) and Counter(step['item_id'] for step in sequence)==Counter(action),
            'Native sequence changed the complete candidate multiset')
    costs=[step['cost'] for step in sequence]
    require(all(isinstance(cost,(int,float)) and not isinstance(cost,bool) and math.isfinite(cost) and cost>=0 for cost in costs),
            'Native sequence has an invalid cost')
    cost=float(sum(costs));require(cost<=wallet+1e-6,'Feasible sequence exceeds supplied diagnostic wallet')
    return {'sequence':sequence,'incremental_cost':cost,'wallet':wallet}


def diagnose_action(action,row,purchase_sequence,dragon,guard=lambda:None):
    """Original context is immutable; only the explicit wallet input changes."""
    require(bool(action) and all(type(item) is int and item>0 and item!=999999 for item in action),
            'Lost catalogue main actions must be actual nonempty purchases')
    original=float(row['gold_est'])
    require(math.isfinite(original) and 0<=original<GENEROUS_WALLET,'Invalid original diagnostic wallet')
    before=producer.canonical(row);calls=0
    def probe(wallet):
        nonlocal calls
        guard();calls+=1
        # Synchronize row wallet and explicit wallet without changing any
        # inventory, role, timing, slot or footwear field of the frozen row.
        supplied=deepcopy(row);supplied['gold_est']=wallet
        sequence=purchase_sequence(tuple(action),supplied,float(wallet),dragon)
        require(producer.canonical(row)==before,'Native economics mutated the frozen original row')
        return check_sequence(sequence,action,float(wallet))
    observed=probe(original);generous=probe(GENEROUS_WALLET)
    result={'original_wallet':original,'generous_wallet':GENEROUS_WALLET,
            'original_receipt':observed,'generous_receipt':generous,
            'minimum_feasible_integer_wallet':None,'minimum_integer_wallet_shortfall':None,
            'minimum_wallet_receipt':None,'lower_integer_wallet_receipt':None}
    if observed is not None:
        require(generous is not None,'Native affordability is nonmonotone under increased wallet')
        result['classification']='feasible_at_original_wallet'
    elif generous is None:
        result['classification']='structural_or_unresolved_at_generous_wallet'
    else:
        low=0;high=GENEROUS_WALLET
        while low<high:
            mid=(low+high)//2
            if probe(mid) is None:low=mid+1
            else:high=mid
        minimum=probe(low);lower=probe(low-1) if low else None
        require(minimum is not None and (low==0 or lower is None),'Integer feasibility boundary failed')
        result.update(classification='wallet_sensitive',minimum_feasible_integer_wallet=low,
                      minimum_integer_wallet_shortfall=float(low-original),minimum_wallet_receipt=minimum,
                      lower_integer_wallet_receipt=lower)
    result['native_economic_calls']=calls
    return result


def verify_report(path,input_paths):
    path=Path(path).resolve(strict=True);report,digest=producer.load_bound(path)
    require(report.get('schema')=='winprob-ordinary-evaluation-v1' and report.get('state')=='complete'
            and report.get('outcomes_analyzed') is False and report.get('test_payload_reads') is False
            and report.get('training_performed') is False,'Completed outcome-free ordinary evaluation required')
    bindings={str(path):digest}
    for key,supplied in input_paths.items():
        original=report['input_manifests'][key];supplied=Path(supplied).resolve(strict=True)
        require(Path(original['path']).resolve()==supplied and producer.file_hash(supplied)==original['sha256'],
                'Diagnostic input is not the completed evaluation artifact: '+key)
        bindings[str(supplied)]=original['sha256']
    outputs={}
    for name in ('predictions.json','scored_rows.json'):
        value,observed=producer.load_bound(path.parent/name)
        require(observed==report['files'][name],'Completed evaluation row file changed: '+name)
        bindings[str(path.parent/name)]=observed;outputs[name]=value
    return report,outputs,bindings


def target_main(target,groups):
    require(target.get('encoding')=='item-counts-v1','Unknown observed target encoding')
    raw=target['main_label_counts']
    require(isinstance(raw,dict) and all(str(key).isdecimal() and str(int(key))==str(key)
            and int(key)>0 and int(key) not in (2055,772043,999999) and type(value) is int and value>0
            for key,value in raw.items()),'Malformed main target multiset')
    key=[[int(item),count] for item,count in sorted(raw.items(),key=lambda pair:int(pair[0]))]
    gid=target['main_group_id']
    if gid is None:
        require(key not in groups,'Declared unsupported main target is in the catalogue')
    else:
        require(type(gid) is int and 0<=gid<len(groups) and groups[gid]==key,'Observed main target/catalogue mismatch')
    return gid,key


def select_cases(contexts,targets,reference_rows,predictions,scored_rows,probabilities,groups,expected=EXPECTED_CASES):
    n=len(contexts)
    require(n==859 and all(len(rows)==n for rows in (targets,reference_rows,predictions,scored_rows)),
            'Diagnostic must account for all859 originally evaluated rows')
    selected=[];counts=Counter()
    for context,target_row,ref,prediction,scored in zip(contexts,targets,reference_rows,predictions,scored_rows,strict=True):
        require(len({row['sample_id'] for row in (context,target_row,ref,prediction,scored)})==1
                and len({row['match_id'] for row in (context,target_row,ref,prediction,scored)})==1,
                'Observed target join changed sample identity')
        require(context['status']=='available' and target_row['status']=='available'
                and ref['reference_status']=='scored','This fixed diagnosis requires the completed859 available rows')
        gid,main=target_main(target_row['target'],groups)
        direct=[entry['group'] for entry in ref['direct_bc_options']]
        require(prediction['direct_options']==ref['direct_bc_options'],'Frozen direct candidates differ')
        cat=gid in prediction['catalogue_top_groups'];displayed=gid in direct
        require(cat==bool(scored['catalogue_diagnostic_top3_main_credit'])
                and displayed==bool(scored['secondary_direct_main_ceiling']),'Recorded recall flags changed')
        counts['both' if cat and displayed else 'catalogue_only' if cat else 'direct_only' if displayed else 'neither']+=1
        if not(cat and not displayed):continue
        require(type(ref.get('bc_tensor_row')) is int and gid is not None and bool(main),'Lost target lacks frozen actor mass')
        values=probabilities[ref['bc_tensor_row']];greater=int(np.sum(values>values[gid]));equal=int(np.sum(values==values[gid]))
        require(greater<3 and equal==1 and values[gid]>0,'Loss might be probability ties/underflow/rank mismatch')
        require(gid not in [entry['group'] for entry in ref['m2_options']],'Lost direct target is feasible in original M2 options')
        selected.append({'sample_id':context['sample_id'],'match_id':context['match_id'],
                        'participant_id':context['participant_id'],'role':context['role'],'patch':context['patch'],
                        'context_timestamp_ms':context['context_timestamp_ms'],'source_visit_end_ms':context['source_visit_end_ms'],
                        'projection_sha256':context['projection_sha256'],'observed_main_group':gid,'observed_main_counts':main,
                        'observed_target_kind':target_row['target']['kind'],'catalogue_probability_rank':greater+1,
                        'probability_equal_count':equal,'observed_target_probability':float(values[gid]),
                        'source_projection':context['source_projection']})
    require(len(selected)==expected,'Fixed post-result108-case diagnostic scope changed')
    return selected,dict(counts)


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Refuse to overwrite diagnostic artifact')
    require(args.patch in ('16.18','16.19') and 1<=args.deadline_seconds<=120,'Clean single-patch bounded diagnostic required')
    require(args.read_observed_targets is True,'This diagnostic explicitly opens observed purchase targets; acknowledge --read-observed-targets only after owner authorization')
    def guard():require(time.monotonic()-started<=args.deadline_seconds,'Diagnostic deadline exceeded')
    report,outputs,bindings=verify_report(args.evaluation_report,{'source':args.source_manifest,'actor':args.actor_manifest,
                                         'reference':args.reference_manifest,'native19':args.native19_manifest})
    source_manifest,contexts,selection,native_source,source_bindings=reference.load_contexts(args.source_manifest)
    actor,actor_rows,probabilities,groups,actor_bindings=reference.load_actor(args.actor_manifest,args.source_manifest,source_manifest,contexts)
    ref,ref_rows,ref_groups,ref_bindings=native_audit.load_reference(args.reference_manifest,args.source_manifest,source_manifest,contexts)
    require(groups==ref_groups,'Diagnostic actor/reference catalogue mismatch')
    native_manifest,native_hash=producer.load_bound(args.native19_manifest)
    require(native_manifest.get('schema')==native_audit.SCHEMA and native_manifest.get('state')=='complete'
            and native_manifest['reference_manifest_sha256']==producer.file_hash(args.reference_manifest)
            and native_manifest['source_manifest_sha256']==producer.file_hash(args.source_manifest)
            and native_manifest['native_world_patch']=='16.19','Completed native19 candidate audit changed')
    for extra in (source_bindings,actor_bindings,ref_bindings,
                  reference.metadata_bindings({**native_manifest['bindings_sha256'],**native_manifest['code_sha256']})):
        bindings.update(extra)
    bindings[str(Path(args.native19_manifest).resolve())]=native_hash
    # This is the sole observed-target access boundary, after current
    # admission/reservation/859-cohort and artifact identities are verified.
    targets_path=Path(args.targets).resolve(strict=True)
    require(targets_path==(Path(args.source_manifest).resolve().parent/'targets.jsonl')
            and producer.file_hash(targets_path)==source_manifest['files']['targets.jsonl'],
            'Explicit observed target path/digest differs from the completed evaluation')
    bindings[str(targets_path)]=source_manifest['files']['targets.jsonl']
    targets=reference.read_lines(targets_path)
    cases,overlap=select_cases(contexts,targets,ref_rows,outputs['predictions.json'],outputs['scored_rows.json'],probabilities,groups)
    guard()
    if args.patch=='16.18':
        import benchmark_1618_runtime as runtime
        active=runtime.activate(reference.REVIEW)
        from app.ddragon import dragon_for_patch
        dragon=dragon_for_patch('16.18')
        require(active==ref['reference_runtime'],'Diagnostic native18 runtime differs from the frozen M2 decoder')
    else:
        active,_,_,dragon,_=source.native.activate_native(native_source)
        require(active==native_manifest['native_runtime'],'Diagnostic native19 runtime differs from the completed audit')
    from app.conditional_policy import purchase_sequence
    require(dragon.version==args.patch+'.1','Diagnostic static patch changed')
    if args.patch=='16.19':require(dragon.signature==native_manifest['static_data_sha256'],'Diagnostic native19 static data changed')
    else:
        package=producer.load_json(producer.PACKAGE/'manifest.json')
        require(producer.value_hash(package)==reference.PACKAGE_DIGEST and dragon.signature==package['static_data_sha256'],
                'Diagnostic native18 package/static identity changed')
    code=reference.runtime_code_bindings();code[str(Path(__file__).resolve())]=producer.file_hash(__file__)
    reference.revalidate({**bindings,**code})
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n',encoding='utf-8')
    counts=Counter();calls=0;shortfalls=[]
    try:
        with (output/'rows.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for index,case in enumerate(cases):
                action=tuple(item for item,count in case['observed_main_counts'] for _ in range(count))
                outcome=diagnose_action(action,case['source_projection'],purchase_sequence,dragon,guard)
                record={key:value for key,value in case.items() if key!='source_projection'}
                record.update(schema=SCHEMA,diagnostic_patch=args.patch,**outcome)
                record['native18_rejection_reproduced']=(outcome['original_receipt'] is None) if args.patch=='16.18' else None
                counts[outcome['classification']]+=1;calls+=outcome['native_economic_calls']
                if outcome['minimum_integer_wallet_shortfall'] is not None:shortfalls.append(outcome['minimum_integer_wallet_shortfall'])
                stream.write(producer.canonical(record).decode('utf-8')+'\n')
                if index%10==0:stream.flush();status('diagnosing',cases=index+1,native_economic_calls=calls)
        guard();reference.revalidate({**bindings,**code});producer.validate_admission(source_manifest['admission_path'])
        require(reference.runtime_code_bindings()==code,'Diagnostic loaded unbound source code')
        files={'rows.jsonl':source.sidecar(output/'rows.jsonl')}
        summary={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                 'patch':args.patch,'files':files,'observed_target_file_opened':str(targets_path),'observed_target_rows_read':len(targets),
                 'selected_cases':len(cases),'selection_rule':'The fixed completed-evaluation catalogue-top3 hits missing from native direct-BC options; unique positive exported-P rank1..3.',
                 'selected_case_ids_sha256':producer.value_hash([row['sample_id'] for row in cases]),
                 'original_overlap_counts':overlap,'classification_counts':dict(counts),'native_economic_calls':calls,
                 'generous_wallet':GENEROUS_WALLET,'minimum_wallet_semantics':'Minimum nonnegative integer supplied wallet with native feasibility; immediately lower integer tested infeasible.',
                 'wallet_shortfall_quantiles':dict(zip(['min','p25','median','p75','p90','max'],np.quantile(shortfalls,[0,.25,.5,.75,.9,1]).tolist())) if shortfalls else None,
                 'native18_all_rejections_reproduced':(counts['feasible_at_original_wallet']==0) if args.patch=='16.18' else None,
                 'native_runtime':active,'ddragon_version':dragon.version,'static_data_sha256':dragon.signature,
                 'bindings_sha256':bindings,'code_sha256':code,'input_manifests':report['input_manifests'],
                 'evaluation_report_path':str(Path(args.evaluation_report).resolve()),'evaluation_report_sha256':producer.file_hash(args.evaluation_report),
                 'raw_payload_reads':False,'test_payload_reads':False,'confirmation_payload_reads':False,'terminal_outcomes_read':False,
                 'training_performed':False,'policy_model_loaded':False,'new_policy_accuracy_computed':False,
                 'candidate_recommendations_changed':False,'promotion_authorized':False,'elapsed_s':time.monotonic()-started,
                 'limits':['Post-result, target-dependent diagnostic on exposed development cases; no new policy evidence.',
                           'Wallet sensitivity identifies incompatibility with the supplied estimate, not the actual historical wallet or its error.',
                           'Failure at1,000,000 gold remains structural or unresolved; it is not automatically attributed to slots.',
                           'No new raw events are read, so intra-visit income/sales/timing explanations remain unverified.',
                           'Integer minimum is an economic feasibility threshold, never a suggested wallet override.']}
        reference.write_json(output/'manifest.json',summary);status('complete',cases=len(cases));return summary
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-report','source-manifest','targets','actor-manifest','reference-manifest','native19-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--patch',choices=('16.18','16.19'),required=True)
    parser.add_argument('--read-observed-targets',action='store_true',help='Explicit observed purchase-target access; requires prior owner authorization.')
    parser.add_argument('--deadline-seconds',type=int,default=120)
    result=run(parser.parse_args())
    print(json.dumps({key:result[key] for key in ('patch','selected_cases','classification_counts','native_economic_calls','elapsed_s')}))


if __name__=='__main__':main()
