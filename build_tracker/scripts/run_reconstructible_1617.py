"""Durable offline 16.17 HISTORY reconstruction/qualification supervisor, isolated from Riot (derived from the 16.18 one, 2026-09-27)."""
from datetime import datetime,timezone
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
from process_reconstructible_1617 import BASE,CORPUS,validate_plan

WALL_SECONDS=14400


def main():
    source_plan=BASE/'source_plan.json'
    plan=validate_plan(source_plan)
    run=BASE/'run'
    if run.exists() or CORPUS.exists(): raise ValueError('Use fresh immutable processing outputs')
    run.mkdir()
    code_names={p.relative_to(ROOT).as_posix() for p in (ROOT/'app').glob('*.py')}
    code_names.update('scripts/'+name for name in ('run_reconstructible_1617.py','run_reconstructible_1617.ps1',
        'process_reconstructible_1617.py','prepare_reconstructible_1617.py','reconstructible_1617_runtime.py',
        'viego_source_gate_1617.py','audit_patch_corpus.py','qualify_patch_corpus.py','build_patch_development_corpus.py',
        'baseline.py','collect_patch_sample.py','footwear_economics.py','footwear_timer.py','footwear_timer_inventory.py',
        'magical_footwear.py','validate_footwear_timer_results.py','replay_footwear_windows.py','run_patch_development.py',
        'history_1617.py'))
    code={name:sha256(ROOT/name) for name in sorted(code_names)}
    protected_paths=[ROOT/'data/collection_recovery/20260912/plan.json',
        ROOT/'data/source_refetch/16.17-missing-20260911/plan.json',
        ROOT/'data/experiments/16.17-development-v2/completion.json']
    protected_paths += [ROOT/'data/ml'/n for n in ('prefix_model.pt','deployment_manifest.json','pipeline_hold.json')]
    protected={str(p):sha256(p) if p.exists() else None for p in protected_paths}
    protocol={'schema':1,'patch':'16.17','purpose':'offline_reconstruction_qualification',
        'source_plan_sha256':sha256(source_plan),'code_sha256':code,'protected_sha256':protected,
        'selected_sources':plan['selected_sources'],'source_snapshot_utc':plan['snapshot_utc'],
        'budget':{'max_wall_seconds':WALL_SECONDS,'max_memory_gb':6,'minimum_free_disk_gb':40},
        'collection_interruption':False,'training_or_export':False,'test_payload_reads':False}
    write_json(run/'protocol.json',protocol)
    for name in code:
        target=run/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    job=WindowsMemoryJob(6_000_000_000)
    start=time.monotonic();completed=[];current='preflight'
    def status(state,**extra):
        write_json(run/'status.json',{'state':state,'stage':current,'completed_stages':completed,
            'time_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
            'peak_child_job_memory_bytes':job.peak_bytes(),'selected_sources':plan['selected_sources'],**extra})
    def unchanged():
        if sha256(source_plan)!=protocol['source_plan_sha256']: raise ValueError('Source plan changed')
        for name,digest in code.items():
            if sha256(ROOT/name)!=digest: raise ValueError('Frozen processing code changed: '+name)
        for name,digest in protected.items():
            path=Path(name)
            if (sha256(path) if path.exists() else None)!=digest: raise ValueError('Protected artifact changed: '+name)
    def budget():
        if time.monotonic()-start>WALL_SECONDS: raise RuntimeError(f'{WALL_SECONDS}s processing budget reached')
        if shutil.disk_usage(ROOT).free<40e9: raise RuntimeError('40 GB disk reserve reached')
    env={k:v for k,v in os.environ.items() if not k.startswith('PREFIX_')}
    env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',HTTP_PROXY='http://127.0.0.1:9',
               HTTPS_PROXY='http://127.0.0.1:9',ALL_PROXY='http://127.0.0.1:9',NO_PROXY='')
    try:
        for stage in ('build','audit'):
            current=stage;unchanged();budget();status('running')
            with (run/(stage+'.log')).open('x',encoding='utf-8') as log:
                child=subprocess.Popen([sys.executable,'scripts/process_reconstructible_1617.py',stage,
                    '--source-plan',str(source_plan),'--corpus',str(CORPUS),'--out',str(BASE/'semantic_audit.json')],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=4)
                job.attach(child)
                while True:
                    try: code_result=child.wait(timeout=10);break
                    except subprocess.TimeoutExpired: budget();status('running')
                if code_result: raise RuntimeError(f'{stage} failed with exit {code_result}; inspect {stage}.log')
            unchanged();completed.append(stage);status('running')
        qualification=json.loads((BASE/'qualification.json').read_text(encoding='utf-8'))
        if qualification.get('status')!='qualified_reconstruction_snapshot': raise ValueError('Qualification missing')
        current='complete'
        status('complete',accepted_games=qualification['accepted_games'],accepted_cohorts=qualification['accepted_cohorts'],
               qualification=str(BASE/'qualification.json'),qualification_sha256=sha256(BASE/'qualification.json'),
               protected_unchanged=True,training_or_export=False,test_payload_reads=False)
        shutil.copy2(run/'status.json',run/'completion.json')
    except Exception as exc:
        job.stop();status('failed',reason=str(exc));raise
    finally: job.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=BASE,
        help='Dated acceptance directory holding source_plan.json (a new snapshot needs a new directory)')
    parser.add_argument('--corpus',type=Path,default=CORPUS,
        help='New immutable corpus output directory; refuses an existing one')
    parser.add_argument('--max-wall-seconds',type=int,default=WALL_SECONDS)
    args=parser.parse_args()
    BASE=args.base.resolve();CORPUS=args.corpus.resolve();WALL_SECONDS=args.max_wall_seconds
    main()
