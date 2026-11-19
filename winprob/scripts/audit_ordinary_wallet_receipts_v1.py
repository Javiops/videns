"""Independent artifact/receipt audit; never imports or reruns source economics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def check(condition,message):
    if not condition:raise ValueError(message)


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_rows(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def bind(path,want,bindings):
    path=Path(path).resolve(strict=True)
    check('raw_match_v5' not in path.parts and path.suffix.lower() not in ('.gz','.db','.sqlite'),
          'Audit must not read raw archives/databases')
    name=str(path)
    check(name not in bindings or bindings[name]==want,'Conflicting artifact identity')
    if name not in bindings:check(digest(path)==want,'Artifact/code digest changed: '+name)
    bindings[name]=want


def bound_json(path,bindings):
    path=Path(path).resolve(strict=True);observed=digest(path)
    check(path.with_suffix('.sha256').read_text(encoding='ascii').split()==[observed,path.name],'Sidecar mismatch: '+str(path))
    bind(path,observed,bindings)
    return read_json(path)


def quantiles(values):
    return dict(zip(['min','p25','median','p75','p90','max'],np.quantile(values,[0,.25,.5,.75,.9,1]).tolist()))


def receipt_check(receipt,counts,wallet):
    check(isinstance(receipt,dict) and receipt['wallet']==wallet,'Receipt wallet binding changed')
    sequence=receipt['sequence']
    check(isinstance(sequence,list) and Counter(step['item_id'] for step in sequence)==Counter(dict(counts)),
          'Receipt does not preserve target item multiplicities')
    costs=[step['cost'] for step in sequence]
    check(all(type(cost) in (int,float) and math.isfinite(cost) and cost>=0 for cost in costs),'Invalid sequence cost')
    total=float(sum(costs))
    check(total==receipt['incremental_cost'] and total<=wallet,'Incremental cost arithmetic mismatch')
    return total


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    check(not output.exists() and not output.with_suffix('.sha256').exists(),'Refuse to overwrite audit')
    bindings={};manifests={};all_rows={};paths={'16.18':args.native18,'16.19':args.native19}
    flags=('raw_payload_reads','test_payload_reads','confirmation_payload_reads','terminal_outcomes_read',
           'training_performed','policy_model_loaded','new_policy_accuracy_computed','candidate_recommendations_changed','promotion_authorized')
    for patch,path in paths.items():
        path=Path(path).resolve(strict=True);manifest=bound_json(path,bindings)
        check(manifest['schema']=='winprob-ordinary-target-feasibility-diagnostic-v1' and manifest['state']=='complete'
              and manifest['patch']==patch and manifest['ddragon_version']==patch+'.1','Wrong completed diagnostic/world')
        check(manifest['selected_cases']==108 and manifest['observed_target_rows_read']==859
              and manifest['generous_wallet']==1000000 and all(manifest.get(flag) is False for flag in flags),'Diagnostic boundary/scope changed')
        for file,want in manifest['files'].items():
            check(Path(file).name==file,'Nonlocal diagnostic artifact')
            bind(path.parent/file,want,bindings)
        for key in ('bindings_sha256','code_sha256'):
            for name,want in manifest[key].items():bind(name,want,bindings)
        rows=read_rows(path.parent/'rows.jsonl')
        check(len(rows)==108 and len({row['sample_id'] for row in rows})==108,'Diagnostic case count/uniqueness changed')
        check(hashlib.sha256(canonical([row['sample_id'] for row in rows])).hexdigest()==manifest['selected_case_ids_sha256'],
              'Diagnostic ordered case digest changed')
        manifests[patch]=manifest;all_rows[patch]=rows
    first=manifests['16.18'];second=manifests['16.19']
    check(first['evaluation_report_path']==second['evaluation_report_path']
          and first['evaluation_report_sha256']==second['evaluation_report_sha256']
          and first['input_manifests']==second['input_manifests'],'Patch diagnostics use different completed evaluations')
    check(first['selected_case_ids_sha256']==second['selected_case_ids_sha256'],'Patch case selection changed')
    report_path=Path(first['evaluation_report_path']);report=bound_json(report_path,bindings)
    check(digest(report_path)==first['evaluation_report_sha256'],'Evaluation identity changed')
    for name,want in report['files'].items():bind(report_path.parent/name,want,bindings)
    predictions=read_json(report_path.parent/'predictions.json');scored=read_json(report_path.parent/'scored_rows.json')
    source_path=Path(report['input_manifests']['source']['path']);source=bound_json(source_path,bindings)
    check(digest(source_path)==report['input_manifests']['source']['sha256'],'Original source identity changed')
    check(source['selection_count']==source['available_contexts']==859 and source['outcomes_exported'] is False,'Source scope changed')
    for name in ('contexts.jsonl','targets.jsonl'):bind(source_path.parent/name,source['files'][name],bindings)
    contexts=read_rows(source_path.parent/'contexts.jsonl');targets=read_rows(source_path.parent/'targets.jsonl')
    check(len(contexts)==len(targets)==len(predictions)==len(scored)==859,'Original evaluated population differs')
    selected={};overlap=Counter()
    for context,target,prediction,score in zip(contexts,targets,predictions,scored,strict=True):
        check(len({row['sample_id'] for row in (context,target,prediction,score)})==1,'Source/evaluation sample join differs')
        gid=target['target']['main_group_id'];catalogue=gid in prediction['catalogue_top_groups'];direct=gid in [x['group'] for x in prediction['direct_options']]
        check(catalogue==bool(score['catalogue_diagnostic_top3_main_credit']) and direct==bool(score['secondary_direct_main_ceiling']),
              'Evaluator target membership credit differs')
        overlap['both' if catalogue and direct else 'catalogue_only' if catalogue else 'direct_only' if direct else 'neither']+=1
        if catalogue and not direct:selected[context['sample_id']]=(context,target['target'])
    check(dict(overlap)==first['original_overlap_counts']==second['original_overlap_counts']
          and set(selected)=={row['sample_id'] for row in all_rows['16.18']},'Independent108-case selection differs')
    observed={};receipts_checked=0
    for patch,rows in all_rows.items():
        manifest=manifests[patch];classification=Counter();shortfalls=[];economic_calls=0
        for row in rows:
            context,target=selected[row['sample_id']]
            check(row['schema']==manifest['schema'] and row['diagnostic_patch']==patch
                  and row['patch']=='16.19' and row['role']=='development','Row schema or economic/context world changed')
            for key in ('match_id','participant_id','projection_sha256','context_timestamp_ms','source_visit_end_ms'):
                check(row[key]==context[key],'Diagnostic/source identity mismatch: '+key)
            check(row['projection_sha256']==hashlib.sha256(canonical(context['source_projection'])).hexdigest(),'Source projection digest mismatch')
            wanted=[[int(item),count] for item,count in sorted(target['main_label_counts'].items(),key=lambda x:int(x[0]))]
            check(row['observed_main_counts']==wanted and row['observed_main_group']==target['main_group_id']
                  and row['observed_target_kind']==target['kind'] and bool(wanted),'Observed main target changed')
            check(row['original_wallet']==float(context['source_projection']['gold_est'])
                  and row['original_receipt'] is None and row['classification']=='wallet_sensitive','Original rejection or classification differs')
            check(row['probability_equal_count']==1 and row['catalogue_probability_rank'] in (1,2,3)
                  and row['observed_target_probability']>0,'Case lost through rank/zero-mass ambiguity')
            minimum=row['minimum_feasible_integer_wallet'];shortfall=row['minimum_integer_wallet_shortfall']
            check(type(minimum) is int and 0<minimum<=1000000 and shortfall==minimum-row['original_wallet'] and shortfall>0,
                  'Minimum integer wallet/shortfall arithmetic failed')
            check(row['lower_integer_wallet_receipt'] is None,'Immediately lower integer was recorded feasible')
            low_cost=receipt_check(row['minimum_wallet_receipt'],wanted,minimum)
            generous_cost=receipt_check(row['generous_receipt'],wanted,1000000)
            check(low_cost>minimum-1,'Returned minimum-wallet sequence itself works at a lower integer')
            check(row['native18_rejection_reproduced'] is (True if patch=='16.18' else None),'Native18 rejection marker differs')
            check(type(row['native_economic_calls']) is int and 4<=row['native_economic_calls']<=24,'Unexpected bounded bisection call count')
            classification[row['classification']]+=1;shortfalls.append(shortfall);economic_calls+=row['native_economic_calls'];receipts_checked+=2
        check(dict(classification)==manifest['classification_counts'] and economic_calls==manifest['native_economic_calls'],
              'Diagnostic aggregate accounting mismatch')
        recomputed=quantiles(shortfalls)
        check(recomputed==manifest['wallet_shortfall_quantiles'],'Diagnostic quantiles differ from receipts')
        observed[patch]={'cases':len(rows),'classification_counts':dict(classification),'native_economic_calls_recorded':economic_calls,
                         'shortfall_quantiles':recomputed,'all_original_receipts_infeasible':True,
                         'all_minimum_receipts_feasible_and_costs_above_next_lower_integer':True}
    check(first['native18_all_rejections_reproduced'] is True and second['native18_all_rejections_reproduced'] is None,
          'Patch summary reproduction flags differ')
    differences=[]
    excluded={'diagnostic_patch','native18_rejection_reproduced'}
    for left,right in zip(all_rows['16.18'],all_rows['16.19'],strict=True):
        unequal=[key for key in left if key not in excluded and left[key]!=right[key]]
        if unequal:differences.append({'sample_id':left['sample_id'],'fields':unequal})
    check(not differences,'Patch economic receipts differ; inspect instead of claiming equality')
    durations=[c['source_visit_end_ms']-c['context_timestamp_ms'] for c,t in selected.values()]
    size_counts=Counter(sum(t['main_label_counts'].values()) for c,t in selected.values())
    code_path=Path(__file__).resolve();bindings[str(code_path)]=digest(code_path)
    for name,want in bindings.items():check(digest(name)==want,'Input changed during audit: '+name)
    result={'schema':'winprob-ordinary-wallet-diagnostic-audit-v1','state':'complete','passed':True,
            'created_utc':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,
            'diagnostics':{patch:{'manifest_path':str(Path(paths[patch]).resolve()),'manifest_sha256':digest(paths[patch]),**observed[patch]} for patch in paths},
            'bindings_sha256':bindings,'verified_binding_count':len(bindings),
            'independent_evaluation_overlap_counts':dict(overlap),'original_population':859,'fixed_selected_cases':108,
            'independent_sequence_receipts_checked':receipts_checked,'source_context_and_target_joins_checked':216,
            'world_comparison':{'identical_case_order':True,'different_static_world_digests':first['static_data_sha256']!=second['static_data_sha256'],
                                'identical_all_case_fields_except_world_markers':True,'identical_minimum_wallets':108,'identical_sequence_receipts':216},
            'selected_case_descriptions':{'main_action_size_counts':dict(size_counts),'visit_duration_ms_quantiles':quantiles(durations),
                                           'zero_duration_visits':sum(value==0 for value in durations)},
            'authorization':'User explicitly authorized exported purchases from the859 development games for audit and legality diagnosis.',
            'observed_target_file_read':str(source_path.parent/'targets.jsonl'),'observed_target_rows_read':859,
            'raw_payload_reads':False,'terminal_outcomes_read':False,'economic_search_rerun':False,'source_code_imported':False,
            'policy_model_loaded':False,'training_performed':False,'promotion_authorized':False,
            'finding':'All108 previously identified catalogue-top3 losses are wallet-sensitive under both economic worlds; receipts and integer wallet thresholds agree case-by-case.',
            'interpretation_limits':[
                'This independently validates artifacts, joins and receipt arithmetic. Negative feasibility at the immediately lower integer is the producer recorded result, not an independently repeated search.',
                'Increasing only supplied wallet restores feasibility in these reconstructed contexts; this does not measure actual historical wallet or isolate estimator error from cutoff/inventory/target semantics.',
                'The108 post-result cases are a conditional subset, not an estimate of wallet error across every shop decision.',
                'Agreement between native18 and native19 applies only to these108 observed actions/contexts, not all patch effects.',
                'The failed859-game primary policy gate remains failed; no decoder change, policy accuracy gain or causal win benefit is established.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);stream.write('\n')
    output.with_suffix('.sha256').write_text(digest(output)+' '+output.name+'\n',encoding='ascii')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native18',required=True,type=Path);parser.add_argument('--native19',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    result=run(parser.parse_args())
    print(json.dumps({key:result[key] for key in ('passed','verified_binding_count','independent_sequence_receipts_checked','fixed_selected_cases','world_comparison','elapsed_s')},indent=2))


if __name__=='__main__':main()
