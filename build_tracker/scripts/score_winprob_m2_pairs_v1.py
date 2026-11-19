"""Export exact frozen M2 group masses for the ten certified pre-A smoke pairs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import export_winprob_trajectories_v1 as producer

DEFAULT_CONTEXTS=ROOT/'data/exports/winprob-revision-contexts/alignment-smoke-20260930-v1'
REVIEW=ROOT/'data/acceptance/16.18_reconstructible_20260923/inputs/static_review.json'
SCHEMA='winprob-m2-pair-reference-v1'
REQUIRED_CHECKS=('three_exact_operations','no_opening_or_extra_operations','source_observes_retention',
                 'qualified_ts_equals_pre_a','qualified_ts_end_equals_b','opening_sales_absent',
                 'physical_preinventory_coherent','singleton_b_retained','source_mask_admitted',
                 'wallet_strict_prefix_reproduced','label_future_mutation_invariant')


def require(condition,message):
    if not condition:raise ValueError(message)


def context_row(context,allowed,denied,label=None):
    mid=context['match_id']
    require(mid in allowed and mid not in denied,'Pair context identity is not permitted')
    require(context.get('status')=='aligned' and context.get('role')=='fit' and context.get('patch')=='16.18',
            'This smoke scores only aligned16.18fit contexts')
    require(all(context.get('checks',{}).get(key) is True for key in REQUIRED_CHECKS),'Context lacks an exact pre-A certificate')
    projection=context['source_projection']
    require(producer.value_hash(projection)==context['projection_sha256'],'Context projection digest mismatch')
    require(projection['ts']==context['context_timestamp_ms']<context['certificate_b_timestamp_ms'],
            'Context is not earlier than the B certificate')
    item=context['action_b'] if label is None else label
    return {**projection,'match_id':mid,'label_counts':{str(item):1},'target_encoding':'item-counts-v1',
            'label_id':item,'save_kind':'build_spend'}


def singleton_group_map(groups):
    mapping={}
    for group,entries in enumerate(groups):
        if len(entries)==1 and int(entries[0][1])==1:
            item=int(entries[0][0])
            require(item not in mapping,'Duplicate singleton group')
            mapping[item]=group
    return mapping


def candidate_statistics(probability_a,probability_b):
    require(all(math.isfinite(p) and 0<=p<=1 for p in (probability_a,probability_b)),'Invalid exact M2 group probability')
    total=probability_a+probability_b
    return {'probability_a':probability_a,'probability_b':probability_b,
            'pair_conditional_b':probability_b/total if total>0 else None,
            'log_probability_b_minus_a':math.log(probability_b)-math.log(probability_a) if min(probability_a,probability_b)>0 else None,
            'zero_numerical_mass':min(probability_a,probability_b)==0,
            'preference':'B' if probability_b>probability_a else 'A' if probability_a>probability_b else 'tie'}


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Immutable M2 pair output already exists')
    manifest,tests,validation,confirm,bindings=producer.validate_admission(args.admission)
    allowed={row['match_id'] for row in manifest['games'] if row['role']=='train' and row['historical_validation'] is False}
    denied=tests|validation|confirm
    context_dir=Path(args.contexts).resolve(strict=True)
    report,report_hash=producer.load_bound(context_dir/'report.json')
    require(report.get('schema')=='winprob-revision-source-alignment-v1' and report.get('state')=='complete'
            and report.get('test_payload_reads') is False and report.get('confirmation_payload_reads') is False,
            'Expected a complete clean alignment artifact')
    context_path=context_dir/'contexts.jsonl'
    require(producer.file_hash(context_path)==report['contexts_sha256'],'Alignment rows hash mismatch')
    for path,digest in {**report['input_metadata_sha256'],**report['code_sha256']}.items():
        require(producer.file_hash(path)==digest,'Alignment source/code changed: '+path)
    contexts=[json.loads(line) for line in context_path.read_text(encoding='utf-8').splitlines() if line]
    require(1<=len(contexts)<=10 and len(contexts)==report['pairs_examined'],'Alignment smoke size changed')
    rows=[context_row(context,allowed,denied) for context in contexts]
    # Fresh process activation BEFORE any trainer or policy consumer import.
    import app.mixture_policy as mp
    active=mp.activate_runtime(producer.PACKAGE,REVIEW)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy=mp.load_package('mixture-v4-m2-v1',device='cpu')
    require(policy.manifest['featurization'].get('history') is False,'Unexpected M2 history requirement')
    code_paths=(Path(__file__),Path(producer.__file__),ROOT/'app/mixture_policy.py',ROOT/'scripts/train_prefix.py')
    code={str(path.resolve()):producer.file_hash(path) for path in code_paths}
    bindings.update({str(context_dir/'report.json'):report_hash,str(context_path):report['contexts_sha256']})
    mutated=[]
    for context,row in zip(contexts,rows,strict=True):
        other=context_row(context,allowed,denied,label=context['action_a'])
        other.update(source_target_id=999999999,gold_arrival_true=10**9,win='mutation-only',decision='changed',save_kind='no_buy_death')
        mutated.append(other)
    with torch.no_grad():
        probabilities=policy.probabilities(rows).cpu()
        changed=policy.probabilities(mutated).cpu()
    require(probabilities.shape==(len(rows),policy.n_act),'Unexpected M2 probability shape')
    require(bool(torch.isfinite(probabilities).all()) and bool((probabilities>=0).all()),'Nonfinite M2 masses')
    require(bool(torch.allclose(probabilities.sum(dim=1),torch.ones(len(rows)),atol=2e-6,rtol=0)),'M2 masses not normalized')
    require(torch.equal(probabilities,changed),'Frozen M2 scores depend on labels or future metadata')
    groups=singleton_group_map(policy.group_main)
    from app.conditional_policy import purchase_sequence
    displayed=policy.options(rows)
    records=[]
    for index,(context,row) in enumerate(zip(contexts,rows,strict=True)):
        a,b=context['action_a'],context['action_b']
        record={'schema':SCHEMA,'match_id':context['match_id'],'participant_id':context['participant_id'],
                'source_pair_line':context['source_pair_line'],'game_candidate_index':context['game_candidate_index'],
                'context_timestamp_ms':context['context_timestamp_ms'],'certificate_b_timestamp_ms':context['certificate_b_timestamp_ms'],
                'action_a':a,'action_b':b,'role':'fit','patch':'16.18','projection_sha256':context['projection_sha256'],
                'package_identity_sha256':policy.digest,'source_model_exposure':'M2 original training population',
                'label_future_mutation_invariant':True,'probability_row_sum':float(probabilities[index].sum()),
                'displayed_groups':[option['group'] for option in displayed[index]],
                'estimated_budget':float(row['gold_est']),
                'legal_a_at_estimated_pre_a_wallet':purchase_sequence((a,),row,float(row['gold_est']),policy.dragon) is not None,
                'legal_b_at_estimated_pre_a_wallet':purchase_sequence((b,),row,float(row['gold_est']),policy.dragon) is not None,
                'legality_scope':'Frozen model estimate at pre-A, distinct from transactional feasibility certificate at B.'}
        if a not in groups or b not in groups:
            record.update(status='candidate_not_in_frozen_group_catalogue',missing_items=[item for item in (a,b) if item not in groups])
        else:
            record.update(status='scored',group_a=groups[a],group_b=groups[b],
                          **candidate_statistics(float(probabilities[index,groups[a]]),float(probabilities[index,groups[b]])))
        records.append(record)
    producer.validate_admission(args.admission)
    require(all(producer.file_hash(path)==digest for path,digest in {**bindings,**code}.items()),'Frozen input/code changed during M2 scoring')
    require(time.monotonic()-started<=150,'M2 smoke exceeded150second deadline')
    output.mkdir(parents=True,exist_ok=False)
    path=output/'m2_pair_probabilities.jsonl'
    with path.open('x',encoding='utf-8',newline='\n') as handle:
        for record in records:handle.write(producer.canonical(record).decode()+'\n')
    rows_hash=producer.file_hash(path);path.with_suffix('.sha256').write_text(rows_hash+'  '+path.name+'\n',encoding='ascii')
    result={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
            'rows_file':path.name,'rows_sha256':rows_hash,'pairs':len(records),
            'status_counts':dict(Counter(row['status'] for row in records)),
            'fit_smoke_preference_counts':dict(Counter(row['preference'] for row in records if row['status']=='scored')),
            'package_identity_sha256':policy.digest,'runtime':active,'group_count':policy.n_act,
            'probability_semantics':'Exact normalized full-catalogue grouped M2 masses before legal display filtering, no truncation/renormalization.',
            'pair_conditional_semantics':'Optional pB/(pA+pB) diagnostic; not a calibrated human preference probability.',
            'whole_distribution_label_mutation_equal':True,'input_sha256':bindings,'code_sha256':code,
            'raw_sources_reopened':False,'training_performed':False,'outcomes_analyzed':False,'elapsed_s':time.monotonic()-started,
            'limits':['Ten FIT contexts establish adapter validity only; no held-out model comparison.',
                      'These16.18games were in M2 training; preferences here cannot establish generalization.',
                      'Reference uses legacy inferred roles by exact frozen-model contract.',
                      'Transaction certificate atB does not assert both candidates legal at model-estimated preA wallet.']}
    report_path=output/'report.json';report_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    report_path.with_suffix('.sha256').write_text(producer.file_hash(report_path)+'  report.json\n',encoding='ascii')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts',type=Path,default=DEFAULT_CONTEXTS)
    parser.add_argument('--admission',type=Path,default=producer.ADMISSION)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args)
    print(json.dumps({key:report[key] for key in ('pairs','status_counts','fit_smoke_preference_counts','group_count','whole_distribution_label_mutation_equal','rows_sha256','elapsed_s')},indent=2))


if __name__=='__main__':main()
