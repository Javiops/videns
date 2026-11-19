"""Source-owned native decoder and separate-world audit for a versioned research distribution.

Never imports consumer runtime, observes target purchases, or edits frozen policy.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
import export_bc20_ordinary_reference_v2 as reference
producer=reference.producer;source=reference.source;require=reference.require
PROTOCOL=ROOT.parent/'winprob/docs/purchase_memory_candidates_protocol_v1.md'
ARMS=('m2','memory','shuffled')


def run(args):
    started=time.monotonic();output=args.output.resolve();require(not output.exists(),'immutable new source output required')
    entry_code=reference.runtime_code_bindings();entry_code[str(Path(__file__).resolve())]=producer.file_hash(__file__)
    memory_path=args.distribution_manifest.resolve();memory,sha=producer.load_bound(memory_path)
    require(memory['schema']=='winprob-purchase-memory-distribution-v1' and memory['state']=='complete'
            and memory['selection_count']==859 and memory['group_count']==9333 and memory['prior_scores_reproduced'] is True,'invalid consumer export')
    for flag in ('target_file_opened','test_payload_reads','confirmation_payload_reads','outcomes_analyzed','training_performed','source_runtime_imported'):
        require(memory[flag] is False,'consumer boundary failed: '+flag)
    bindings=reference.metadata_bindings(memory['bindings_sha256']);bindings[str(memory_path)]=sha
    require(Path(memory['protocol_path']).resolve()==PROTOCOL.resolve() and producer.file_hash(PROTOCOL)==memory['protocol_sha256'],'new protocol changed')
    for name,want in memory['files'].items():
        path=memory_path.parent/name;require(producer.file_hash(path)==want,'consumer artifact changed');bindings[str(path)]=want
    context_path=Path(memory['source_manifest_path']).resolve();require(producer.file_hash(context_path)==memory['source_manifest_sha256'],'context manifest changed')
    manifest,contexts,selection,native_source,context_bindings=reference.load_contexts(context_path);bindings.update(context_bindings)
    groups=producer.load_json(memory_path.parent/'group_main.json');ledger=producer.load_json(memory_path.parent/'selection.json')
    require(len(groups)==9333 and groups[0]==[] and len(ledger)==len(contexts)==859,'export dimensions changed')
    for i,(entry,context) in enumerate(zip(ledger,contexts)):
        require(context['status']=='available' and entry['ordinal']==i and all(entry[k]==context[k] for k in ('sample_id','match_id','participant_id')),'query identity changed')
    output.mkdir(parents=True,exist_ok=False)
    def status(stage,**extra):
        value={'stage':stage,'elapsed_s':time.monotonic()-started,**extra};(output/'status.json').write_text(json.dumps(value));print(json.dumps(value),flush=True)
    def guard():require(time.monotonic()-started<=600,'source stage exceeded10minutes')
    records=[];counts=Counter();status('activating',mode=args.mode)
    if args.mode=='decode':
        import app.mixture_policy as mp
        active=mp.activate_runtime(producer.PACKAGE,reference.REVIEW)
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from app.conditional_policy import purchase_sequence
        policy=mp.load_package('mixture-v4-m2-v1',device='cpu')
        actual=[[[int(i),int(n)] for i,n in group] for group in policy.group_main]
        require(policy.digest==reference.PACKAGE_DIGEST and actual==groups and policy.shortlist==128 and policy.n_options==3,'frozen M2 decoder changed')
        for p in (producer.PACKAGE/'manifest.json',producer.PACKAGE/'package_id.json',reference.REVIEW):bindings[str(p.resolve())]=producer.file_hash(p)
        for entry in policy.manifest['weights'].values():bindings[str((producer.PACKAGE/entry['file']).resolve())]=entry['sha256']
        refpath=Path(memory['reference_manifest_path']);require(producer.file_hash(refpath)==memory['reference_manifest_sha256'],'M2 reference changed')
        ref=producer.load_json(refpath);oldpath=refpath.parent/'rows.jsonl'
        require(producer.file_hash(oldpath)==ref['files']['rows.jsonl'],'M2 option receipt changed');bindings[str(oldpath)]=producer.file_hash(oldpath)
        old=reference.read_lines(oldpath)
        with np.load(memory_path.parent/'distributions.npz',allow_pickle=False) as archive:matrices={key:archive[key] for key in ARMS}
        for arm,p in matrices.items():
            require(p.shape==(859,9333) and np.isfinite(p).all() and np.all(p>=0) and np.all(np.abs(p.sum(1,dtype=np.float64)-1)<2e-6),'invalid supplied probabilities')
            require(p.dtype==(np.float32 if arm=='m2' else np.float64),'declared probability precision changed')
        for i,context in enumerate(contexts):
            guard();rows=[reference.model_row(context)];record={'sample_id':context['sample_id'],'match_id':context['match_id'],'ordinal':i,'options':{},'audits':{}}
            for arm in ARMS:
                p=torch.from_numpy(matrices[arm][i:i+1]);options=reference.options_override(policy,rows,p)[0]
                changed=[reference.mutate_ignored(rows[0])];mutated=reference.options_override(policy,changed,p)[0]
                require(options==mutated,'forbidden target fields changed native options')
                if arm=='m2':require(context['sample_id']==old[i]['sample_id'] and options==old[i]['m2_options'],'M2 native replay mismatch')
                audits=reference.audit_options(options,rows[0],groups,purchase_sequence,policy.dragon)
                require(all(a['feasible'] for a in audits),'pinned decoder produced infeasible option')
                record['options'][arm]=options;record['audits'][arm]=audits;counts[arm+'_options']+=len(options)
            records.append(record)
            if i%100==0:status('decoding',completed=i+1)
        world='16.18';dragon=policy.dragon
    else:
        require(args.decoded_manifest is not None,'native19 audit needs completed decoder artifact')
        decoded_path=args.decoded_manifest.resolve();decoded,ds=producer.load_bound(decoded_path)
        require(decoded['schema']=='source-purchase-memory-decoder-v1' and decoded['state']=='complete'
                and decoded['distribution_manifest_sha256']==sha and decoded['m2_replay_mismatches']==0,'wrong decoder artifact')
        bindings[str(decoded_path)]=ds
        rows_path=decoded_path.parent/'rows.jsonl';require(producer.file_hash(rows_path)==decoded['files']['rows.jsonl'],'decoded options changed');bindings[str(rows_path)]=producer.file_hash(rows_path)
        input_rows=reference.read_lines(rows_path);require(len(input_rows)==859,'decoder dropped queries')
        active,_,_,dragon,_=source.native.activate_native(native_source)
        require(dragon.version=='16.19.1','native audit world changed')
        from app.conditional_policy import purchase_sequence
        for i,(context,old) in enumerate(zip(contexts,input_rows)):
            guard();require(old['sample_id']==context['sample_id'] and old['ordinal']==i,'decoder/audit query identity mismatch')
            record={'sample_id':context['sample_id'],'ordinal':i,'audits':{}}
            row=reference.model_row(context)
            for arm in ARMS:
                audits=reference.audit_options(old['options'][arm],row,groups,purchase_sequence,dragon)
                record['audits'][arm]=audits;counts[arm+'_options']+=len(audits)
                counts[arm+'_infeasible']+=sum(not a['feasible'] for a in audits)
            records.append(record)
        world='16.19'
    reference.revalidate(entry_code)
    code=reference.runtime_code_bindings();code[str(Path(__file__).resolve())]=producer.file_hash(__file__)
    reference.revalidate({**bindings,**code});producer.validate_admission(manifest['admission_path'])
    with (output/'rows.jsonl').open('x',encoding='utf-8') as stream:
        for record in records:stream.write(producer.canonical(record).decode()+'\n')
    files={'rows.jsonl':source.sidecar(output/'rows.jsonl')}
    result={'schema':'source-purchase-memory-decoder-v1' if args.mode=='decode' else 'source-purchase-memory-native19-v1',
        'state':'complete','selection_count':859,'files':files,'distribution_manifest_path':str(memory_path),'distribution_manifest_sha256':sha,
        'world':world,'native_runtime':active,'static_signature':dragon.signature,'counts':dict(counts),'bindings_sha256':bindings,'code_sha256':code,
        'm2_replay_mismatches':0 if args.mode=='decode' else None,'mutation_checked':859*3 if args.mode=='decode' else None,
        'protocol_sha256':producer.file_hash(PROTOCOL),'target_file_opened':False,'test_payload_reads':False,'confirmation_payload_reads':False,
        'training_performed':False,'outcomes_analyzed':False,'consumer_runtime_imported':False,'elapsed_s':time.monotonic()-started}
    reference.write_json(output/'manifest.json',result);status('complete',counts=dict(counts))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('decode','audit19'),required=True)
    parser.add_argument('--distribution-manifest',required=True,type=Path);parser.add_argument('--decoded-manifest',type=Path)
    parser.add_argument('--output',required=True,type=Path);run(parser.parse_args())
