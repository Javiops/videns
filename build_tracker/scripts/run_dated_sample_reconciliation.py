"""Supervise the dated samples' acceptance-only reconciliation; no Riot/model calls."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256,write_json
from run_patch_development import WindowsMemoryJob

BASE=ROOT/'data/acceptance/16.18_dated_samples_20260920'
RUN=BASE/'reconciliation_run'

def main():
    if RUN.exists(): raise ValueError('Existing reconciliation attempt: inspect it before any retry')
    if shutil.disk_usage(ROOT).free < 40_000_000_000: raise ValueError('40 GB disk reserve required')
    names=[p.relative_to(ROOT).as_posix() for p in (ROOT/'app').glob('*.py')]
    names += ['scripts/'+n for n in ('run_dated_sample_reconciliation.py','run_dated_sample_reconciliation.ps1',
        'reconcile_dated_sample_acceptance.py','prepare_dated_sample_followup.py',
        'prepare_benchmark_1618.py','qualify_patch_corpus.py','corpus_rows.py','baseline.py',
        'footwear_runtime.py','reconstructible_1618_runtime.py','benchmark_1618_runtime.py',
        'build_recovered_patch_corpus.py','footwear_economics.py',
        'footwear_timer_inventory.py','run_patch_development.py')]
    code={n:sha256(ROOT/n) for n in names}
    protected={}
    for name in ('data/ml/prefix_model.pt','data/ml/deployment_manifest.json','data/ml/pipeline_hold.json',
        'data/experiments/16.18-reconstructible-development-v1/completion.json',
        'data/experiments/16.18-initial-smoke-v2/completion.json'):
        p=ROOT/name; protected[name]=sha256(p) if p.exists() else None
    RUN.mkdir()
    protocol={'purpose':'dated_sample_acceptance_only','code_sha256':code,'protected_sha256':protected,
        'preflight_sha256':sha256(BASE/'preflight.json'),'test_payload_reads':False,
        'training_or_export':False,'budget':{'wall_seconds':7200,'memory_bytes':6_000_000_000,'free_disk_bytes':40_000_000_000}}
    write_json(RUN/'protocol.json',protocol)
    for name in code:
        p=RUN/'code'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
    start=time.monotonic();job=WindowsMemoryJob(6_000_000_000)
    def status(state,**extra):
        write_json(RUN/'status.json',{'state':state,'stage':'reconciliation','time_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds':time.monotonic()-start,'peak_child_job_memory_bytes':job.peak_bytes(),**extra})
    def unchanged():
        if any(sha256(ROOT/n)!=h for n,h in code.items()):raise ValueError('Pinned code changed')
        if sha256(BASE/'preflight.json')!=protocol['preflight_sha256']:raise ValueError('Preflight changed')
        for n,h in protected.items():
            p=ROOT/n
            if (sha256(p) if p.exists() else None)!=h:raise ValueError('Protected artifact changed: '+n)
    env={k:v for k,v in os.environ.items() if not k.startswith('PREFIX_')}
    env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',HTTP_PROXY='http://127.0.0.1:9',
        HTTPS_PROXY='http://127.0.0.1:9',ALL_PROXY='http://127.0.0.1:9',NO_PROXY='')
    try:
        unchanged();status('running')
        with (RUN/'reconciliation.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'scripts/reconcile_dated_sample_acceptance.py','--stage-only'],cwd=ROOT,
                env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=4)
            job.attach(child)
            while True:
                try: result=child.wait(timeout=10);break
                except subprocess.TimeoutExpired:
                    if time.monotonic()-start>7200 or shutil.disk_usage(ROOT).free<40_000_000_000:
                        raise RuntimeError('Reconciliation resource budget reached')
                    status('running')
            if result:raise RuntimeError(f'Reconciliation failed with exit {result}; inspect reconciliation.log')
        unchanged()
        staged=BASE/'staged_final';final=BASE/'final'
        receipt=json.loads((staged/'completion.json').read_text(encoding='utf-8'))
        if (receipt.get('status')!='acceptance_reconciliation_complete'
            or receipt.get('supervisor_completion_required') is not True
            or receipt.get('training_export_ready') is not False
            or receipt.get('test_payload_reads') is not False
            or receipt.get('promotion_allowed') is not False
            or receipt.get('production_hold_cleared') is not False):
            raise ValueError('Staged acceptance receipt has the wrong scope')
        for name,binding in receipt['files'].items():
            p=(staged/name).resolve()
            if not p.is_relative_to(staged.resolve()) or p.stat().st_size!=binding['bytes'] or sha256(p)!=binding['sha256']:
                raise ValueError('Staged acceptance file binding changed')
        unchanged()
        if final.exists():raise ValueError('Final acceptance view already exists')
        staged.rename(final)
        status('complete',protected_unchanged=True,training_or_export=False,test_payload_reads=False,
               final_receipt_sha256=sha256(final/'completion.json'),final_output=str(final))
        shutil.copy2(RUN/'status.json',RUN/'completion.json')
    except Exception as exc:
        job.stop();status('failed',reason=str(exc));raise
    finally:job.close()

if __name__=='__main__':main()
