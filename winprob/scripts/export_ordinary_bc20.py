"""Export one frozen BC20 actor on the protocol-locked ordinary feature bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from winprob import admission
from winprob.ordinary_actor_export import (SCHEMA,PROTOCOL,CHECKPOINT_SHA256,canonical_bytes,digest,
    load_contexts,load_frozen_actor,score_contexts,revalidate)
from winprob.iql_training import RuntimeGuard,set_below_normal_priority


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    path.with_suffix('.sha256').write_text(f'{digest(path)} {path.name}\n',encoding='ascii')


def run(args):
    if not 1<=args.deadline_seconds<=900: raise ValueError('actor stage must remain within its fixed 15-minute ceiling')
    if args.output.exists() and any(args.output.iterdir()): raise ValueError('refuse to overwrite an actor export')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);set_below_normal_priority()
    guard=RuntimeGuard(args.deadline_seconds,.75);guard()
    source,rows,admitted,bindings,available=load_contexts(args.source_manifest)
    actor,compiler,catalogue,model_bindings,static_path=load_frozen_actor(args.training_run,args.dataset_manifest)
    if str(static_path)!=str(Path(source['static_export_path']).resolve()) or digest(static_path)!=source['static_export_sha256']:
        raise ValueError('source and frozen actor disagree on public static artifact')
    for path,expected in model_bindings.items():
        if path in bindings and bindings[path]!=expected: raise ValueError('source/model artifact binding conflict')
        bindings[path]=expected
    for path in (Path(__file__).resolve(),ROOT/'winprob/ordinary_actor_export.py',ROOT/'winprob/iql_training.py',
                 ROOT/'winprob/full_action_catalogue.py',ROOT/'winprob/admission.py',PROTOCOL):
        bindings[str(path)]=digest(path)
    guard.bind(bindings);revalidate(bindings);guard.sample_memory('before_actor_inference');guard()
    ledger,arrays,counts=score_contexts(actor,compiler,rows,catalogue['full_to_group'],9333,
                                      batch_size=args.batch_size,guard=guard)
    guard.sample_memory('after_actor_inference');guard()
    distribution=args.output/'distribution.npz';np.savez_compressed(distribution,**arrays)
    ledger_path=args.output/'rows.jsonl'
    with ledger_path.open('w',encoding='utf-8',newline='\n') as stream:
        for row in ledger: stream.write(canonical_bytes(row).decode('utf-8')+'\n')
    write_json(args.output/'group_main.json',catalogue['group_main'])
    write_json(args.output/'full_to_group.json',catalogue['full_to_group'])
    revalidate(bindings);admission.verify_manifest_sources(admitted);guard()
    files={name:digest(args.output/name) for name in
           ('distribution.npz','rows.jsonl','group_main.json','full_to_group.json')}
    manifest={'schema':SCHEMA,'state':'complete','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'source_manifest_path':str(Path(args.source_manifest).resolve()),
        'source_manifest_sha256':digest(args.source_manifest),'source_selection_sha256':source['files']['selection.json'],
        'source_contexts_sha256':source['files']['contexts.jsonl'],
        'opaque_targets_sha256_not_opened':source['files']['targets.jsonl'],'target_file_opened':False,
        'source_available_contexts':available,'selection_count':859,**counts,'files':files,
        'row_order_sha256':hashlib.sha256(canonical_bytes([row['sample_id'] for row in ledger])).hexdigest(),
        'probability_row_order_sha256':hashlib.sha256(canonical_bytes([row['sample_id'] for row in ledger if row['bc_status']=='scored'])).hexdigest(),
        'arrays':{key:{'dtype':str(value.dtype),'shape':list(value.shape)} for key,value in arrays.items()},
        'full_action_count':13166,'group_count':9333,'group_mapping':'sum every full variant through the frozen full_to_group map',
        'rank_field':'group_logits','rank_semantics':'finite full-logit grouped logsumexp; no log of rounded float32 probabilities',
        'decoder_field':'group_probabilities','decoder_semantics':'float32 sums of the exported full categorical masses',
        'checkpoint_sha256':CHECKPOINT_SHA256,'checkpoint_path':str(Path(args.training_run).resolve()/'actor_bc20.pt'),
        'bindings_sha256':bindings,'protocol_path':str(PROTOCOL.resolve()),'protocol_sha256':digest(PROTOCOL),
        'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'python':sys.version,'device':'cpu','threads':1,'batch_size':args.batch_size},
        'mutation_scope':'every successfully encoded context: target/outcome/legacy role/total gold/other-player wallets/future fields changed; exact full and grouped output equality',
        'raw_payload_reads':False,'test_payload_reads':False,'confirmation_payload_reads':False,
        'targets_analyzed':False,'outcomes_analyzed':False,'training_performed':False,'mixture_or_accuracy_computed':False,
        'feature_fallback_rule':'preserve ledger row; source/evaluator use unchanged M2 recommendations',
        'limits':['This export contains no observed-purchase targets or policy-accuracy calculations.',
                  'Native displayed decoding and both economic-world audits remain source-owned.',
                  'Unavailable and feature-failure rows retain selection identity and are not excluded from evaluation.',
                  'Float32 masses may underflow; ranking/pooling must use the finite group log-mass fields.',
                  'The target digest is copied from the bound source manifest without reading targets; evaluation must validate that file.'],
        'host_memory':guard.memory_report(),'elapsed_s':time.monotonic()-guard.started}
    write_json(args.output/'manifest.json',manifest)
    print(json.dumps({'manifest':str((args.output/'manifest.json').resolve()),'counts':counts,'elapsed_s':manifest['elapsed_s']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--training-run',type=Path,default=ROOT/'data/runs/iql-outcome-20260930-v1/experiment')
    parser.add_argument('--dataset-manifest',type=Path,default=ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json')
    parser.add_argument('--batch-size',type=int,choices=(32,64),default=32)
    parser.add_argument('--deadline-seconds',type=int,default=600)
    args=parser.parse_args()
    fresh_output=not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir()))
    try: run(args)
    except Exception as error:
        if fresh_output and args.output.is_dir() and not (args.output/'manifest.json').exists():
            write_json(args.output/'failure.json',{'state':'failed','error_type':type(error).__name__,'reason':str(error),
                                                'training_performed':False,'targets_analyzed':False})
        raise


if __name__=='__main__':main()
