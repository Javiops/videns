"""Serial daily catch-up, dated 16.18 sample, then existing archive-only refetch.

Launch only through the hidden one-shot scheduled task. The original daily
task is reused; no model, export, holdout assignment or serving file is changed.
"""
import argparse
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

RUN=ROOT/'data/collection_recovery/20260912'
SAMPLE=ROOT/'data/patch_samples/16.18_20260912'
PARENT=ROOT/'data/corpora/16.18_initial/state.json'
REFETCH=ROOT/'data/source_refetch/16.17-missing-20260911'
DAILY='BuildTracker Daily Pull'


def powershell(code):
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        "$ErrorActionPreference='Stop'; "+code],capture_output=True,text=True,timeout=40,
        creationflags=0x08000000)
    if result.returncode: raise RuntimeError('Scheduled-task inspection/action failed: '+result.stderr.strip())
    return result.stdout.strip()


def daily_state():
    return json.loads(powershell(
        "$t=Get-ScheduledTask -TaskName 'BuildTracker Daily Pull'; "
        "$i=Get-ScheduledTaskInfo -TaskName 'BuildTracker Daily Pull'; "
        "[pscustomobject]@{state=[string]$t.State;last_run_utc=$i.LastRunTime.ToUniversalTime().ToString('o');"
        "result=$i.LastTaskResult;actions=@($t.Actions|Select-Object Execute,Arguments)}|ConvertTo-Json -Depth 4"))


def verify_daily_action(snapshot):
    actions=snapshot['actions']
    if (len(actions)!=1 or '--export-patch' in (actions[0].get('Arguments') or '')
        or (actions[0].get('Arguments') or '').strip('"') != str(ROOT/'scripts/daily_pull.py')
        or Path(actions[0]['Execute']).resolve() != (ROOT/'.venv/Scripts/pythonw.exe').resolve()):
        raise ValueError('Daily task is not the reviewed collection-only invocation')


def verify_daily_finished(snapshot, started):
    last=datetime.fromisoformat(snapshot['last_run_utc'])
    if last.timestamp()<started-5 or snapshot['state']=='Running': return False
    if snapshot['result']!=0: raise RuntimeError('Daily collection failed with exit '+str(snapshot['result']))
    return True


def verify_sample_collection(sample, expected_parent_sha, expected_unresolved):
    checkpoint=sample/'state.json'
    state=json.loads(checkpoint.read_text(encoding='utf-8'))
    report=json.loads((sample/'report.json').read_text(encoding='utf-8'))
    if (state.get('patch')!='16.18' or state.get('cohort_parent_sha256')!=expected_parent_sha
        or state.get('discovery_complete') is not True or not state.get('jobs')
        or set(state.get('results',{}))!=set(state['jobs'])
        or report.get('structural_checks_passed') is not True or not report.get('games')
        or report.get('checkpoint_sha256')!=sha256(checkpoint)
        or set(report.get('unresolved_pros',[]))!=set(expected_unresolved)):
        raise ValueError('Dated sample is incomplete or differs from the accepted cohort')
    # The already accepted unresolved pro identities and individually recorded
    # source quarantines do not erase completed enumeration/collection evidence.
    return report


def prepare():
    if (RUN/'plan.json').exists(): raise ValueError('Recovery plan already exists; resume it')
    if SAMPLE.exists(): raise ValueError('A dated sample already exists; review its checkpoint before reuse')
    from refetch_missing_sources import read_plan
    refetch=read_plan(REFETCH)
    parent=json.loads(PARENT.read_text(encoding='utf-8'))
    if parent.get('patch')!='16.18' or not parent.get('discovery_complete'):
        raise ValueError('Initial 16.18 cohort is incomplete')
    RUN.mkdir(parents=True)
    names={p.relative_to(ROOT).as_posix() for p in (ROOT/'app').glob('*.py')}
    names.update('scripts/'+name for name in ('resume_morning_collection.py','run_morning_collection_recovery.ps1',
        'daily_pull.py','collect_patch_sample.py','run_patch_sample.ps1','refetch_missing_sources.py','run_patch_development.py'))
    protected=[ROOT/'data/experiments/16.17-development-v2'/n for n in ('completion.json','candidate.pt','run_plan.json')]
    protected += [ROOT/'data/ml'/n for n in ('prefix_model.pt','deployment_manifest.json','pipeline_hold.json')]
    plan={'schema':1,'purpose':'serial_collection_recovery','date':'2026-09-12',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'stages':['daily_collection','prepare_sample','collect_sample','resume_refetch'],
        'sample':str(SAMPLE),'parent':str(PARENT),'parent_sha256':sha256(PARENT),
        'accepted_unresolved_pros':sorted(k for k,v in parent['pros'].items() if not v.get('puuid')),
        'refetch':str(REFETCH),'refetch_plan_sha256':sha256(REFETCH/'plan.json'),
        'refetch_selected_games':refetch['selected_games'],'current_training_integration':False,
        'promotion_allowed':False,'test_evaluation':False,
        'code_sha256':{name:sha256(ROOT/name) for name in sorted(names)},
        'protected_sha256':{str(p):sha256(p) if p.exists() else None for p in protected}}
    for name in names:
        target=RUN/'code'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    write_json(RUN/'plan.json',plan)
    (RUN/'plan.json').chmod(0o444)
    print(json.dumps({'plan':str(RUN/'plan.json'),'stages':plan['stages'],'sample_prepared':False}),flush=True)


def main():
    from run_patch_development import WindowsMemoryJob
    plan=json.loads((RUN/'plan.json').read_text(encoding='utf-8'))
    plan_sha=sha256(RUN/'plan.json')
    if plan.get('purpose')!='serial_collection_recovery' or plan.get('current_training_integration') is not False:
        raise ValueError('Invalid collection-only plan')
    prior=json.loads((RUN/'status.json').read_text(encoding='utf-8')) if (RUN/'status.json').exists() else {}
    if prior and prior.get('plan_sha256')!=plan_sha: raise ValueError('Checkpoint belongs to another recovery')
    completed=list(prior.get('completed_stages',[]))
    if completed!=plan['stages'][:len(completed)]: raise ValueError('Invalid completed-stage checkpoint order')
    current='preflight'
    start=time.monotonic()
    job=WindowsMemoryJob(6_000_000_000)
    def status(state,**extra):
        write_json(RUN/'status.json',{'state':state,'stage':current,'completed_stages':completed,
            'plan_sha256':plan_sha,'time_utc':datetime.now(timezone.utc).isoformat(),
            'elapsed_seconds_this_attempt':time.monotonic()-start,'current_training_integration':False,**extra})
    def unchanged():
        if sha256(RUN/'plan.json')!=plan_sha: raise ValueError('Recovery plan changed')
        for name,digest in plan['code_sha256'].items():
            if sha256(ROOT/name)!=digest: raise ValueError('Recovery code changed: '+name)
        for name,digest in plan['protected_sha256'].items():
            path=Path(name)
            if (sha256(path) if path.exists() else None)!=digest: raise ValueError('Protected artifact changed: '+name)
        if sha256(PARENT)!=plan['parent_sha256'] or sha256(REFETCH/'plan.json')!=plan['refetch_plan_sha256']:
            raise ValueError('Frozen cohort or refetch plan changed')
    def budget():
        if time.monotonic()-start>12*3600: raise RuntimeError('Twelve-hour recovery budget reached')
        if shutil.disk_usage(ROOT).free<40e9: raise RuntimeError('40 GB disk reserve reached')
    def child_stage(name,command):
        nonlocal current
        current=name
        if name in completed: return
        unchanged();budget();status('running')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        env={k:v for k,v in os.environ.items() if not k.startswith('PREFIX_')}
        env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1')
        with (RUN/f'{name}_{stamp}.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=4)
            job.attach(process)
            while True:
                try: code=process.wait(timeout=10);break
                except subprocess.TimeoutExpired: budget();status('running')
            if code: raise RuntimeError(f'{name} failed with exit {code}; checkpoint retained')
        unchanged()
        completed.append(name);status('running')
    try:
        unchanged();budget()
        current='daily_collection'
        if current not in completed:
            snapshot=daily_state();verify_daily_action(snapshot)
            # A running daily collection is never restarted. Wait for its
            # actual successful completion before starting the dated sample.
            previous_started=prior.get('daily_started_epoch') if prior.get('stage')=='daily_collection' else None
            previous_complete=bool(previous_started and verify_daily_finished(snapshot,previous_started))
            daily_started=(previous_started if previous_complete else
                datetime.fromisoformat(snapshot['last_run_utc']).timestamp() if snapshot['state']=='Running' else time.time())
            if snapshot['state']!='Running' and not previous_complete:
                from app.collector_lock import CollectorLease
                with CollectorLease(): pass
                powershell("Start-ScheduledTask -TaskName 'BuildTracker Daily Pull'")
            status('running',daily_started_epoch=daily_started)
            while not verify_daily_finished(daily_state(),daily_started):
                budget();status('running',daily_started_epoch=daily_started);time.sleep(10)
            unchanged();completed.append(current);status('running')
        if not (SAMPLE/'state.json').exists():
            if datetime.now().strftime('%Y-%m-%d')!='2026-09-12':
                raise ValueError('Missed sample date; record a new actual date rather than backdate')
            child_stage('prepare_sample',[sys.executable,'scripts/collect_patch_sample.py','--sample',str(SAMPLE),
                '--reuse-cohorts-from',str(PARENT),'--prepare'])
        elif 'prepare_sample' not in completed:
            state=json.loads((SAMPLE/'state.json').read_text(encoding='utf-8'))
            if state.get('cohort_parent_sha256')!=plan['parent_sha256'] or state.get('patch')!='16.18':
                raise ValueError('Unexpected existing dated sample')
            completed.append('prepare_sample')
        child_stage('collect_sample',['powershell.exe','-NoProfile','-NonInteractive','-WindowStyle','Hidden',
            '-ExecutionPolicy','Bypass','-File',str(ROOT/'scripts/run_patch_sample.ps1'),'-Sample',str(SAMPLE)])
        report=verify_sample_collection(SAMPLE,plan['parent_sha256'],plan['accepted_unresolved_pros'])
        child_stage('resume_refetch',[sys.executable,'scripts/refetch_missing_sources.py','supervise','--out',str(REFETCH)])
        refetch_status=json.loads((REFETCH/'status.json').read_text(encoding='utf-8'))
        if refetch_status.get('state')!='complete': raise ValueError('Archive refetch remains incomplete')
        current='complete'
        status('complete',protected_inputs_unchanged=True,sample_report=str(SAMPLE/'report.json'),refetch_status=str(REFETCH/'status.json'))
        shutil.copy2(RUN/'status.json',RUN/'completion.json')
    except Exception as exc:
        job.stop()
        status('failed',reason=str(exc))
        raise
    finally: job.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else main()
