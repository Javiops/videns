"""One durable, bounded full-source footwear recovery measurement; no training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256,write_json
from run_patch_development import WindowsMemoryJob


def main():
    evidence=ROOT/'data/acceptance/footwear_repair_20260911'
    report=evidence/'timer_full_source.json'
    status=evidence/'timer_full_supervisor.json'
    if report.exists() or status.exists(): raise ValueError('Recovery run already exists')
    paths=[ROOT/'data/ml'/name for name in ('prefix_model.pt','deployment_manifest.json','pipeline_hold.json')]
    before={str(p):sha256(p) if p.exists() else None for p in paths}
    source=evidence.parent/'16.17_development_20260911/source_plan.json'
    run_args=['scripts/evaluate_footwear_timer.py','--source-plan',str(source),
              '--rune-inventory',str(source.parent/'source_rune_inventory.json'),
              '--code',str(ROOT/'data/experiments/footwear-timer-v1/reconstruct.py'),
              '--out',str(report),'--skip-prefix-probes']
    started=time.monotonic()
    job=WindowsMemoryJob(8_000_000_000)
    try:
        env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1',
             'HTTP_PROXY':'http://127.0.0.1:9','HTTPS_PROXY':'http://127.0.0.1:9'}
        with (evidence/'timer_full_source.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,*run_args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=4)
            job.attach(process)
            while True:
                write_json(status,{'state':'running','elapsed_seconds':time.monotonic()-started,
                           'source_plan_sha256':sha256(source),'pid':process.pid})
                try:
                    code=process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic()-started>10800:
                        job.stop()
                        raise RuntimeError('Three-hour recovery audit budget exceeded')
        if code: raise RuntimeError(f'Recovery audit exited {code}')
        after={str(p):sha256(p) if p.exists() else None for p in paths}
        if after!=before: raise ValueError('Protected serving/hold artifacts changed')
        write_json(status,{'state':'complete','elapsed_seconds':time.monotonic()-started,
                   'report_sha256':sha256(report),'protected_before':before,'protected_after':after,
                   'training_performed':False,'final_test_read':False,'promotion_allowed':False})
    except BaseException as error:
        job.stop()
        write_json(status,{'state':'failed','error':str(error),'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        job.close()


if __name__=='__main__': main()
