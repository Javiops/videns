"""Bounded CPU export of every frozen M2 group mass on certified pre-A contexts.

Native-patch context production and this16.18-pinned scorer run in separate
processes. The dense float32 matrix is disk backed; only small batches of
source tensors are in memory. Full production execution requires --full.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import export_winprob_trajectories_v1 as producer
import score_winprob_m2_pairs_v1 as smoke

SCHEMA='winprob-m2-preference-v1'
ROLE_MAP={'fit':'train','calibration':'calibration','development':'development'}
PATCHES={'16.17','16.18','16.19'}


def require(condition,message):
    if not condition:raise ValueError(message)


def source_row(context,admitted,denied,label=None):
    mid=context['match_id'];entry=admitted.get(mid)
    require(mid not in denied and entry is not None,'Context identity is not currently admitted')
    role=context.get('role');patch=context.get('patch')
    require(role in ROLE_MAP and patch in PATCHES and entry['role']==ROLE_MAP[role]
            and entry['patch']==patch and entry.get('historical_validation') is False,'Context role/patch changed')
    require(role!='development' or patch=='16.19','Primary preference development is16.19 only')
    require(context.get('schema') in ('winprob-revision-source-alignment-v2','winprob-revision-source-alignment-v3') and context.get('status')=='aligned',
            'Only certified native-v2/v3 contexts may be scored')
    require(all(context.get('checks',{}).get(key) is True for key in smoke.REQUIRED_CHECKS),'Incomplete pre-A certificate')
    projection=context['source_projection']
    require(producer.value_hash(projection)==context['projection_sha256'],'Source projection hash changed')
    ta=context['context_timestamp_ms'];tu=context['undo_timestamp_ms'];tb=context['decision_timestamp_ms']
    require(projection['ts']==ta<tu<tb and tb==context['certificate_b_timestamp_ms'],'Pre-A/undo/B identity mismatch')
    expected='|'.join(str(value) for value in (mid,context['participant_id'],ta,tu,tb,context['action_a'],context['action_b']))
    require(context['pair_id']==expected,'Stable pair identity changed')
    weight=context.get('random_focal_pair_weight')
    require(isinstance(weight,(int,float)) and not isinstance(weight,bool) and math.isfinite(weight) and 0<weight<=.1,
            'Missing original random-focal preference weight')
    item=context['action_b'] if label is None else label
    return {**projection,'match_id':mid,'label_counts':{str(item):1},'target_encoding':'item-counts-v1',
            'label_id':item,'save_kind':'build_spend'}


def options_with_exported_probabilities(policy,rows,probabilities):
    """Native decoder with exact unchanged P: this is the W=0 reference path."""
    original=policy.probabilities
    def cached(requested_rows):
        require(requested_rows is rows,'Native decoder changed its row batch')
        return probabilities
    try:
        policy.probabilities=cached
        return policy.options(rows)
    finally:
        policy.probabilities=original


def load_contexts(directories,admitted,denied):
    contexts=[];excluded=[];bindings={};runtimes=[];seen=set()
    for directory in directories:
        directory=Path(directory).resolve(strict=True)
        require(not any((parent/'QUARANTINED.md').exists() for parent in (directory,*directory.parents)),'Quarantined context input')
        report,report_hash=producer.load_bound(directory/'report.json')
        require(report.get('schema') in ('winprob-revision-source-alignment-v2','winprob-revision-source-alignment-v3') and report.get('state')=='complete'
                and report.get('test_payload_reads') is False and report.get('confirmation_payload_reads') is False
                and report.get('training_performed') is False,'Expected complete native context export')
        rows_path=directory/'contexts.jsonl';selection_path=directory/'selection.json'
        require(producer.file_hash(rows_path)==report['contexts_sha256'],'Native context file digest changed')
        require(producer.file_hash(selection_path)==report['selection_sha256'],'Native context selection changed')
        for path,expected in {**report['input_metadata_sha256'],**report['code_sha256']}.items():
            require(producer.file_hash(path)==expected,'Native context source/code changed: '+path)
            bindings[path]=expected
        bindings.update({str(directory/'report.json'):report_hash,str(rows_path):report['contexts_sha256'],
                         str(selection_path):report['selection_sha256']})
        runtimes.append({'context_directory':str(directory),'patch':report['patch'],'runtime':report['runtime'],
                         'report_sha256':report_hash,'contexts_sha256':report['contexts_sha256']})
        read=0
        with rows_path.open('r',encoding='utf-8') as handle:
            for line in handle:
                context=json.loads(line);read+=1
                identity=context['pair_id']
                require(identity not in seen,'Duplicate pair across context artifacts')
                seen.add(identity)
                mid=context['match_id'];entry=admitted.get(mid)
                require(mid not in denied and entry is not None and entry['role']==ROLE_MAP.get(context['role'])
                        and entry['patch']==context['patch'],'Source context or exclusion changed its admitted role')
                if context.get('status')!='aligned':
                    excluded.append({**context,'context_artifact':str(rows_path)})
                    continue
                source_row(context,admitted,denied)
                contexts.append({**context,'context_artifact':str(rows_path),'context_artifact_sha256':report['contexts_sha256']})
        require(read==report['pairs_examined'],'Native context row count mismatch')
    require(bool(contexts),'No aligned contexts to score')
    return contexts,excluded,bindings,runtimes


def write_matrix_archive(temporary,destination):
    """Stream an on-disk NPY matrix into a standard compressed NPZ archive."""
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        archive.write(temporary,arcname='probabilities.npy')
    with zipfile.ZipFile(destination) as archive:
        require(archive.namelist()==['probabilities.npy'] and archive.testzip() is None,'Dense probability NPZ publication failed')


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Immutable dense M2 export already exists')
    require(1<=args.batch_size<=16,'Dense reference export uses CPU batches of at most16')
    manifest,tests,validation,confirm,bindings=producer.validate_admission(args.admission)
    admitted={row['match_id']:row for row in manifest['games']};denied=tests|validation|confirm
    contexts,excluded,context_bindings,native_runtimes=load_contexts(args.contexts,admitted,denied)
    bindings.update(context_bindings)
    require(args.full or len(contexts)<=20,'More than20 contexts requires the reviewed --full run')
    require(len(contexts)<=10000,'Dense export exceeds its fixed10k-pair cap')
    # Only this stage activates M2's own original16.18 reference world.
    import app.mixture_policy as mp
    active=mp.activate_runtime(producer.PACKAGE,smoke.REVIEW)
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    policy=mp.load_package('mixture-v4-m2-v1',device='cpu')
    require(policy.n_act==9333 and policy.manifest['featurization'].get('history') is False,'Unexpected frozen M2 action/feature schema')
    groups=smoke.singleton_group_map(policy.group_main)
    group_main=[[[int(item),int(count)] for item,count in entries] for entries in policy.group_main]
    require(group_main[0]==[] and len({producer.canonical(row) for row in group_main})==len(group_main),'Invalid canonical group mapping')
    source_split=producer.assignment(producer.GENERATION/'split_assignment.json')
    bindings[str(producer.GENERATION/'split_assignment.json')]=producer.file_hash(producer.GENERATION/'split_assignment.json')
    for name in ('manifest.json','package_id.json'):
        bindings[str(producer.PACKAGE/name)]=producer.file_hash(producer.PACKAGE/name)
    code_paths=[Path(__file__),Path(producer.__file__),Path(smoke.__file__),ROOT/'app/mixture_policy.py',
                ROOT/'scripts/train_prefix.py',ROOT/'scripts/basket_econ_cache.py',ROOT/'app/conditional_policy.py']
    code={str(path.resolve()):producer.file_hash(path) for path in code_paths}
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n')
    stage_path=output/'probabilities.staging.npy'
    matrix=np.lib.format.open_memmap(stage_path,mode='w+',dtype=np.float32,shape=(len(contexts),policy.n_act))
    parity_checked=mutation_checked=parity_mismatches=0;status_counts=Counter();exposures=Counter();role_counts=Counter()
    pairs_order=[];deadline=1800 if args.full else 150
    try:
        with (output/'contexts.jsonl').open('x',encoding='utf-8',newline='\n') as handle:
            for start in range(0,len(contexts),args.batch_size):
                require(time.monotonic()-started<=deadline,'Dense M2 export deadline exceeded')
                if start%160==0:producer.validate_admission(args.admission)
                selected=contexts[start:start+args.batch_size]
                rows=[source_row(context,admitted,denied) for context in selected]
                changed_rows=[]
                for context in selected:
                    changed=source_row(context,admitted,denied,label=context['action_a'])
                    changed.update(source_target_id=999999999,gold_arrival_true=10**9,win='mutation-only',
                                   decision='changed',save_kind='no_buy_death')
                    changed_rows.append(changed)
                with torch.no_grad():
                    p=policy.probabilities(rows).cpu();mutated=policy.probabilities(changed_rows).cpu()
                require(p.dtype==torch.float32 and p.shape==(len(rows),9333),'M2 probability dtype/shape mismatch')
                require(bool(torch.isfinite(p).all()) and bool((p>=0).all())
                        and bool(torch.allclose(p.sum(dim=1),torch.ones(len(rows)),atol=2e-6,rtol=0)),
                        'Full M2 probability rows are invalid or unnormalized')
                require(torch.equal(p,mutated),'Full M2 distribution depends on label/future metadata')
                mutation_checked+=len(rows)
                native=policy.options(rows)
                replay=options_with_exported_probabilities(policy,rows,p)
                failures=sum(producer.canonical(a)!=producer.canonical(b) for a,b in zip(native,replay,strict=True))
                parity_checked+=len(rows);parity_mismatches+=failures
                require(failures==0,'W=0 cached-P native decoder differs from original options')
                values=p.numpy();matrix[start:start+len(rows)]=values
                require(np.array_equal(matrix[start:start+len(rows)],values),'Disk-backed probability matrix differs from source output')
                for offset,(context,row) in enumerate(zip(selected,rows,strict=True)):
                    a,b=context['action_a'],context['action_b'];mid=context['match_id']
                    exposure=('M2_training_exposed' if source_split.get(mid)=='train' else
                              'M2_cohort_exposed' if source_split.get(mid)=='val' else 'not_in_declared_M2_training_or_cohort')
                    exposures[exposure]+=1;role_counts[context['role']]+=1
                    record={**context,'schema':SCHEMA,'tensor_row':start+offset,
                            'champion_id':int(row['champion_id']),'team_id':int(row['team_id']),
                            'package_identity_sha256':policy.digest,'reference_exposure':exposure,
                            'reference_world_patch':'16.18','native_context_patch':context['patch'],
                            'group_a':groups.get(a),'group_b':groups.get(b),'native_options':native[offset],
                            'native_zero_residual_options':replay[offset],'zero_residual_native_options_equal':True,
                            'whole_distribution_label_mutation_equal':True,
                            'probability_row_sha256':hashlib.sha256(values[offset].tobytes(order='C')).hexdigest(),
                            'probability_row_sum':float(p[offset].sum()),
                            'immutable_roster':{'focal_champion_id':int(row['champion_id']),
                                'other_champion_team_ids':[[int(member['champion_id']),int(member['team_id'])] for member in row['others']]}}
                    if a not in groups or b not in groups:
                        record.update(status='candidate_not_in_frozen_group_catalogue',missing_items=[item for item in (a,b) if item not in groups])
                    else:
                        stats=smoke.candidate_statistics(float(p[offset,groups[a]]),float(p[offset,groups[b]]))
                        record.update(status='scored' if not stats['zero_numerical_mass'] else 'zero_numerical_candidate_mass',**stats)
                    status_counts[record['status']]+=1;pairs_order.append(context['pair_id'])
                    handle.write(producer.canonical(record).decode('utf-8')+'\n')
                handle.flush();matrix.flush()
                status('scoring',processed_pairs=min(start+args.batch_size,len(contexts)),pairs=len(contexts),
                       parity_checked_pairs=parity_checked,mutation_checked_pairs=mutation_checked)
        matrix.flush();del matrix
        write_matrix_archive(stage_path,output/'probabilities.npz')
        (output/'group_main.json').write_text(json.dumps(group_main,separators=(',',':'))+'\n',encoding='utf-8')
        with (output/'exclusions.jsonl').open('x',encoding='utf-8',newline='\n') as handle:
            for row in excluded:handle.write(producer.canonical(row).decode('utf-8')+'\n')
        producer.validate_admission(args.admission)
        require(all(producer.file_hash(path)==expected for path,expected in {**bindings,**code}.items()),'Dense M2 input/code changed during export')
        require(parity_checked==mutation_checked==len(contexts) and parity_mismatches==0,'Parity/mutation checks did not cover every output row')
        files={name:producer.file_hash(output/name) for name in ('probabilities.npz','group_main.json','contexts.jsonl','exclusions.jsonl')}
        for name,digest in files.items():
            path=output/name;path.with_suffix('.sha256').write_text(digest+'  '+name+'\n',encoding='ascii')
        result={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'files':files,'package_identity_sha256':policy.digest,'group_count':9333,'pairs':len(contexts),
                'probability_array':{'file':'probabilities.npz','key':'probabilities','dtype':'float32','shape':[len(contexts),9333],
                                     'axes':['contexts.jsonl row order','group_main.json index'],'row_sum_atol':2e-6},
                'pair_order_sha256':hashlib.sha256(('\n'.join(pairs_order)+'\n').encode()).hexdigest(),
                'status_counts':dict(status_counts),'role_counts':dict(role_counts),'reference_exposure_counts':dict(exposures),
                'source_unaligned_pairs':len(excluded),'input_pairs_including_unaligned':len(contexts)+len(excluded),
                'zero_residual_native_options_mismatches':parity_mismatches,'parity_checked_pairs':parity_checked,
                'whole_distribution_label_mutation_equal':True,'mutation_checked_pairs':mutation_checked,
                'zero_residual_reference_path':'Original native policy.options versus same native decoder with exact exportedP override; ordered complete options compared.',
                'input_source_sha256':bindings,'producer_code_sha256':code,'native_context_runtimes':native_runtimes,
                'reference_runtime':active,'test_payload_reads':False,'confirmation_payload_reads':False,
                'training_performed':False,'raw_sources_reopened':False,'outcomes_analyzed':False,'elapsed_s':time.monotonic()-started,
                'limits':['Full group masses are frozen M2 reference outputs, not calibrated human logging propensities.',
                          'M2 reference compatibility includes legacy inferred roles; new policy inputs must respect their separate feature contract.',
                          'Pre-A model-wallet legality is distinct from receipt-certified feasibility at B.',
                          'Native decoder parity covers canonical exported decisions only, not broader live/display regressions.']}
        meta=output/'manifest.json';meta.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        meta.with_suffix('.sha256').write_text(producer.file_hash(meta)+'  manifest.json\n',encoding='ascii')
        # This file was created solely for bounded-memory publication here.
        stage_path.unlink()
        status('complete',pairs=len(contexts),parity_checked_pairs=parity_checked,mutation_checked_pairs=mutation_checked)
        return result
    except Exception as error:
        if 'matrix' in locals():matrix.flush()
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contexts',type=Path,action='append',required=True)
    parser.add_argument('--admission',type=Path,default=producer.ADMISSION)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--full',action='store_true')
    args=parser.parse_args();report=run(args)
    print(json.dumps({key:report[key] for key in ('pairs','group_count','status_counts','role_counts','parity_checked_pairs',
        'zero_residual_native_options_mismatches','mutation_checked_pairs','whole_distribution_label_mutation_equal','elapsed_s')},indent=2))


if __name__=='__main__':main()
