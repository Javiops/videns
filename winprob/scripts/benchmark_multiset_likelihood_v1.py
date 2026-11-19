"""Bounded synthetic-only CUDA feasibility check; no game input or checkpoint."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from winprob.full_action_catalogue import load_catalogue
from winprob.iql_training import RuntimeGuard,file_hash,set_below_normal_priority
from winprob.multiset_likelihood_v1 import (build_lattice,build_catalogue_lattice,MultisetTransitionHead,
    catalogue_log_probabilities,catalogue_facts)


def run(output,deadline_seconds):
    if not 1<=deadline_seconds<=120:raise ValueError('synthetic benchmark must be bounded at120seconds')
    output.mkdir(parents=True,exist_ok=False);guard=RuntimeGuard(deadline_seconds);set_below_normal_priority()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths=[Path(__file__),ROOT/'winprob/multiset_likelihood_v1.py',ROOT/'docs/multiset_likelihood_design_v1.md',
           ROOT.parent/'build_tracker/data/exports/winprob-action-catalogue/full-action-catalogue-20260930-v1/catalogue.json']
    bindings={str(p.resolve()):file_hash(p) for p in paths};guard.bind(bindings)
    def idle():
        guard()
        result=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],capture_output=True,text=True,check=True)
        if 'league of legends.exe' in result.stdout.lower():raise RuntimeError('League game active; no GPU work')
    try:
        idle();catalogue,_=load_catalogue(paths[-1])
        if file_hash(paths[-1])!='14fe3b82b1d51bfae258c60f8f68771a92ec85072aab7e62e52985b600ddc1c8':raise ValueError('catalogue changed')
        bags=catalogue['group_main'];vocabulary=sorted({item for bag in bags for item,count in bag});facts=catalogue_facts(bags)
        if len(vocabulary)!=196 or facts['maximum_local_nodes']!=32 or facts['unique_prefixes']!=10037:raise ValueError('catalogue topology changed')
        device=torch.device('cuda:0');torch.manual_seed(20261001);torch.cuda.manual_seed_all(20261001)
        local=build_lattice(bags,vocabulary).to(device);shared=build_catalogue_lattice(bags,vocabulary).to(device)
        head=MultisetTransitionHead(len(vocabulary)).to(device)
        context=torch.randn(256,128,device=device,requires_grad=True)
        target_indices=torch.tensor(np.random.default_rng(20261001).integers(0,len(bags),size=256),device=device)
        lattice=local.select(target_indices);optimizer=torch.optim.Adam(head.parameters(),lr=3e-4)
        times=[]
        for step in range(40):
            if step%10==0:idle()
            torch.cuda.synchronize();started=time.monotonic()
            optimizer.zero_grad(set_to_none=True);context.grad=None
            loss=-head.log_probability(context,lattice).mean()
            loss.backward();optimizer.step();torch.cuda.synchronize()
            elapsed=time.monotonic()-started
            if step>=10:times.append(elapsed)
            if not torch.isfinite(loss) or not all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('nonfinite synthetic loss/gradient')
        del optimizer,loss,context,lattice;gc.collect();torch.cuda.empty_cache();idle()
        head.eval();contexts=torch.randn(8,128,device=device);inference=[]
        with torch.inference_mode():
            for repeat in range(5):
                guard();torch.cuda.synchronize();started=time.monotonic()
                scores=catalogue_log_probabilities(head,contexts,shared,chunk_nodes=1024)
                torch.cuda.synchronize();elapsed=time.monotonic()-started
                if repeat>=2:inference.append(elapsed)
            if not torch.isfinite(scores).all() or torch.any(scores.exp().sum(1)>1.00001):raise ValueError('invalid catalogue subdistribution')
            chosen=torch.tensor([0,1,100,9332],device=device)
            error=0.
            for row in range(8):
                exact=head.log_probability(contexts[row:row+1].expand(len(chosen),-1),local.select(chosen))
                error=max(error,float((exact-scores[row,chosen]).abs().max()))
            if error>2e-5:raise ValueError('shared/local CUDA likelihood differs')
        idle();facts['unique_main_items']=len(vocabulary)
        for path,want in bindings.items():
            if file_hash(path)!=want:raise ValueError('binding changed')
        report={'schema':'multiset-likelihood-synthetic-benchmark-v1','state':'complete','catalogue':facts,
            'head_parameters':sum(p.numel() for p in head.parameters()),'synthetic_optimizer_steps':40,'warmup_steps':10,
            'batch_size':256,'head_training_step_mean_s':float(np.mean(times)),'head_training_step_p95_s':float(np.quantile(times,.95)),
            'catalogue_batch_size':8,'catalogue_inference_batch_mean_s':float(np.mean(inference)),
            'catalogue_inference_per_query_mean_s':float(np.mean(inference)/8),'shared_local_max_logp_error':error,
            'peak_cuda_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),'host_memory':guard.memory_report(),
            'elapsed_s':time.monotonic()-guard.started,'bindings_sha256':bindings,'gradient_finite':True,
            'real_training_performed':False,'game_payload_reads':False,'development_targets_opened':False,'test_payload_reads':False,
            'confirmation_payload_reads':False,'model_saved':False,'breakthrough_established':False,
            'runtime':{'torch':str(torch.__version__),'numpy':np.__version__,'gpu':torch.cuda.get_device_name(device),'threads':1},
            'limits':['Synthetic head-only benchmark; state encoder and data gather costs are not measured here.','No purchase accuracy or M2 comparison is measured.','The synthetic optimizer does not create a reusable trained model.']}
        path=output/'report.json';path.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False))
        path.with_suffix('.sha256').write_text(file_hash(path)+' '+path.name+'\n',encoding='ascii')
        print(json.dumps({k:v for k,v in report.items() if k not in ('bindings_sha256','host_memory','limits')},indent=2))
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'state':'failed','error':str(error),'type':type(error).__name__}));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--deadline-seconds',type=int,default=120);args=parser.parse_args();run(args.output.resolve(),args.deadline_seconds)
