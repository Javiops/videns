"""Durable offline recovered 16.17 build/audit/train/paired-validation supervisor.

Launch only as a one-shot Windows scheduled task. No Riot requests, production
hold changes, promotion, final-test reads, or automatic model replacement.
"""
import argparse
import ctypes
from ctypes import wintypes
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
from app.dataset_artifacts import sha256, write_json
from app.dataset_generations import verify_generation, validate_development_run_plan


class WindowsMemoryJob:
    """OS-enforced aggregate committed-memory ceiling for the owned child tree."""
    def __init__(self,limit_bytes):
        class IO(ctypes.Structure):
            _fields_=[(n,ctypes.c_ulonglong) for n in ('ReadOperationCount','WriteOperationCount','OtherOperationCount',
                                                     'ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class BASIC(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
                      ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),
                      ('ActiveProcessLimit',wintypes.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),
                      ('SchedulingClass',wintypes.DWORD)]
        class EXTENDED(ctypes.Structure):
            _fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),
                      ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        self.info_type=EXTENDED
        self.kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype=wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.kernel.QueryInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.c_void_p]
        self.kernel.TerminateJobObject.argtypes=[wintypes.HANDLE,wintypes.UINT]
        self.kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        self.ntdll=ctypes.WinDLL('ntdll')
        self.ntdll.NtResumeProcess.argtypes=[wintypes.HANDLE]
        self.ntdll.NtResumeProcess.restype=wintypes.LONG
        self.handle=self.kernel.CreateJobObjectW(None,None)
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        limits=EXTENDED()
        limits.BasicLimitInformation.LimitFlags=0x200|0x2000 # job memory + kill children on close
        limits.JobMemoryLimit=limit_bytes
        if not self.kernel.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def attach(self,process):
        if not self.kernel.AssignProcessToJobObject(self.handle,wintypes.HANDLE(int(process._handle))):
            # Only our just-started child tree is terminated on attachment failure.
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
            raise ctypes.WinError(ctypes.get_last_error())
        # Children of the venv launcher inherit the job only if assignment
        # happens before that launcher runs and creates the real interpreter.
        status=self.ntdll.NtResumeProcess(wintypes.HANDLE(int(process._handle)))
        if status<0:
            self.stop()
            raise OSError(f'Could not resume owned child after job assignment: {status}')

    def peak_bytes(self):
        info=self.info_type()
        if not self.kernel.QueryInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info),None):
            raise ctypes.WinError(ctypes.get_last_error())
        return info.PeakJobMemoryUsed

    def stop(self):
        if self.handle: self.kernel.TerminateJobObject(self.handle,1)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan',type=Path,default=ROOT/'data/acceptance/16.17_development_20260911/source_plan.json')
    parser.add_argument('--prior-qualification',type=Path,default=ROOT/'data/acceptance/16.17_development_20260911/qualification.json')
    parser.add_argument('--recovery-summary',type=Path,default=ROOT/'data/acceptance/footwear_repair_20260911/recovery_summary.json')
    parser.add_argument('--name',default='16.17-development-v2')
    args=parser.parse_args()
    if args.name!='16.17-development-v2': raise ValueError('This recipe is bound to the reviewed recovered development run')
    from footwear_runtime import activate
    runtime=activate()
    source_plan=args.source_plan.resolve()
    original_evidence=source_plan.parent
    evidence=ROOT/'data/acceptance/16.17_development_recovered_20260911'
    corpus=ROOT/'data/corpora'/args.name
    dataset=ROOT/'data/dataset_generations'/args.name
    run=ROOT/'data/experiments'/args.name
    for path in (corpus,dataset,run,evidence):
        if path.exists(): raise ValueError(f'Fresh immutable output required: {path}')
    evidence.mkdir(parents=True)
    run.mkdir(parents=True)
    # Scope includes the entire planned code path so a queued subprocess can
    # never silently load a different revision from its predecessor.
    script_names=('run_recovered_development','recovered_pipeline_entry','build_recovered_patch_corpus',
                  'build_patch_development_corpus','audit_patch_corpus','audit_recovered_patch_corpus','qualify_patch_corpus',
                  'report_corpus_coverage','corpus_rows','baseline','collect_patch_sample','publish_patch_dataset','train_prefix','eval_policy',
                  'footwear_runtime','footwear_economics','footwear_timer','footwear_timer_inventory','magical_footwear',
                  'validate_footwear_timer_results','replay_footwear_windows','analyze_patch_development')
    code_names=([p.relative_to(ROOT).as_posix() for p in (ROOT/'app').glob('*.py')]
                +['scripts/'+n+'.py' for n in script_names]+['scripts/run_recovered_development.ps1'])
    code={name:sha256(ROOT/name) for name in code_names}
    for name in code:
        target=run/'code'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    protected_paths=[ROOT/'data/ml'/n for n in ('prefix_model.pt','deployment_manifest.json','pipeline_hold.json')]
    protected={str(p):sha256(p) if p.exists() else None for p in protected_paths}
    summary=json.loads(args.recovery_summary.read_text(encoding='utf-8'))
    inputs={str(p.resolve()):sha256(p) for p in (source_plan,original_evidence/'split_assignment.json',
        original_evidence/'split_metadata.json',args.prior_qualification,args.recovery_summary,Path(summary['source_report']))}
    start=time.monotonic()
    budget={'max_wall_seconds':28800,'max_train_processes':1,'max_ram_gb':10,'minimum_free_disk_gb':40,'audit_workers':4}
    fixed={'PREFIX_ML_DIR':str(dataset),'PREFIX_RUN_DIR':str(run),
           'PREFIX_DMODEL':'128','PREFIX_LAYERS':'4','PREFIX_FF':'256','PREFIX_HEADS':'8',
           'PREFIX_COSINE':'1','PREFIX_EXTRAS':'1','PREFIX_HISTORY':'0','PREFIX_GOLDEST':'1','PREFIX_GOLDX':'1',
           'PREFIX_RUNES':'1','PREFIX_POSW':'8','PREFIX_POSW_SCHED':'8:24','PREFIX_POSW_CUSTOM':'0',
           'PREFIX_TBLEND':'0','PREFIX_SPLICE':'0'}
    training={'trunk':{'PREFIX_OUT':'trunk.pt','PREFIX_EPOCHS':'16','PREFIX_SAVEW':'0','PREFIX_TARGETW':'0',
                       'PREFIX_INIT_FROM':'','PREFIX_FREEZE':'0'},
              'graft':{'PREFIX_OUT':'candidate.pt','PREFIX_EPOCHS':'6','PREFIX_SAVEW':'0.5','PREFIX_TARGETW':'0.25',
                       'PREFIX_INIT_FROM':'trunk.pt','PREFIX_FREEZE':'1'}}
    protocol={'schema':1,'purpose':'patch_development','patch':'16.17','dataset':str(dataset),'run_directory':str(run),
              'source_plan_sha256':sha256(source_plan),
              'footwear_runtime':runtime,'source_inputs_sha256':inputs,
              'question':'Does the fixed board transformer improve displayed shopping-policy imitation over a conditional lookup on the same screened 16.17 development validation games?',
              'primary_metrics':['first_action_exact','action_family_match'],
              'populations':['All scored validation visits for first_action_exact','Purchases with known next final for action_family_match'],
              'stopping_rule':'One 16-epoch trunk and six-epoch frozen-trunk graft, then one paired full-validation evaluation; no sweep or promotion. Stop on failed data/provenance/legality/resource checks. Diagnose errors after the result.',
              'resource_budget':budget,'fixed_environment':fixed,'training_stages':training,'code_sha256':code,
              'evaluation':{'split':'val','games':0,'compare_conditional':True},'evaluation_seed':16,'evaluation_bootstrap':400,
              'final_test_evaluation':False,'promotion_allowed':False,'production_hold_cleared':False,
              'protected_before':protected,'created_utc':datetime.now(timezone.utc).isoformat(),
              'limitations':['Magical Footwear delivery is causally inferred; a 1000 ms sensitivity guard is not an independently verified server-delay bound.',
                             'Viego and unresolved source/replay cases remain excluded. All source purchases remain available for target supervision when a decision input is masked.',
                             'Validation is reused development evidence; original test raw sources are unavailable and remain reserved.',
                             'Imitation under reconstructed gold and slot state does not measure exact live-wallet performance.']}
    write_json(run/'protocol.json',protocol)
    current='preflight'
    completed=[]
    env={k:v for k,v in os.environ.items() if not k.startswith('PREFIX_')}
    env.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1',HTTP_PROXY='http://127.0.0.1:9',HTTPS_PROXY='http://127.0.0.1:9',ALL_PROXY='http://127.0.0.1:9',NO_PROXY='')
    memory_job=WindowsMemoryJob(int(budget['max_ram_gb']*1e9))
    def status(state,**extra):
        write_json(run/'status.json',{'state':state,'stage':current,'completed_stages':completed,
                   'time_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
                   'peak_child_job_memory_bytes':memory_job.peak_bytes(),**extra})
    def unchanged():
        for name,digest in code.items():
            if sha256(ROOT/name)!=digest: raise ValueError('Run code changed: '+name)
        for name,digest in inputs.items():
            if sha256(Path(name))!=digest: raise ValueError('Source/split plan changed: '+name)
        for name,digest in protected.items():
            path=Path(name)
            if (sha256(path) if path.exists() else None)!=digest: raise ValueError('Protected artifact changed: '+name)
    def budget_check():
        if time.monotonic()-start>budget['max_wall_seconds']: raise RuntimeError('Global eight-hour development budget exceeded')
        if shutil.disk_usage(ROOT).free<budget['minimum_free_disk_gb']*1e9: raise RuntimeError('Development disk reserve reached')
    def stage(name,arguments,overrides=None):
        nonlocal current
        current=name
        unchanged()
        budget_check()
        status('running')
        with (run/f'{name}.log').open('x',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,*arguments],cwd=ROOT,env={**env,**(overrides or {})},
                                     stdout=log,stderr=subprocess.STDOUT,creationflags=0x00000004)
            memory_job.attach(process)
            try:
                while True:
                    try:
                        result=process.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        budget_check()
                        status('running')
                if result: raise RuntimeError(f'{name} failed with exit code {result}; see {name}.log')
            except BaseException:
                memory_job.stop()
                process.wait(timeout=30)
                raise
        unchanged()
        completed.append(name)
        status('running')
    try:
        stage('build',['scripts/build_recovered_patch_corpus.py','--source-plan',str(source_plan),
              '--prior-qualification',str(args.prior_qualification.resolve()),'--recovery-summary',str(args.recovery_summary.resolve()),'--out',str(corpus)])
        stage('audit',['scripts/audit_recovered_patch_corpus.py','--corpus',str(corpus),'--out',str(evidence/'semantic_audit.json'),'--workers','4'])
        def data_stage(name,entry,arguments):
            stage(name,['scripts/recovered_pipeline_entry.py','--entry',entry,'--',*arguments])
        data_stage('qualification','qualify_patch_corpus',['--corpus',str(corpus),'--audit',str(evidence/'semantic_audit.json'),'--out',str(evidence/'qualification.json')])
        data_stage('coverage','report_corpus_coverage',['--qualification',str(evidence/'qualification.json'),
              '--out',str(evidence/'coverage'),'--benchmark-id',args.name,'--eval-version','5','--purpose','patch_development',
              '--split-assignment',str(original_evidence/'split_assignment.json'),'--split-metadata',str(original_evidence/'split_metadata.json')])
        coverage=json.loads((evidence/'coverage/coverage.json').read_text(encoding='utf-8'))
        # The schema-bound multiset decoder supports the observed purchase count;
        # basket actions are not a claim that all purchases occupy slots at once.
        if not 1 <= coverage['max_basket_items'] <= 12:
            raise ValueError('Qualified target is outside the reviewed multiset decoder capacity')
        data_stage('export','publish_patch_dataset',['--coverage',str(evidence/'coverage'),'--name',args.name])
        generation=verify_generation(dataset)
        plan={**protocol,'export_fingerprint':generation['export_fingerprint'],
              'generation_manifest_sha256':sha256(dataset/'generation_manifest.json')}
        write_json(run/'run_plan.json',plan)
        validate_development_run_plan(dataset,run,generation,run/'run_plan.json')
        for name,settings in training.items():
            stage(name,['scripts/footwear_runtime.py','--entry','scripts/train_prefix.py'],{**fixed,**settings,'PREFIX_STAGE':name,'PREFIX_RUN_PLAN':str(run/'run_plan.json')})
        stage('policy',['scripts/footwear_runtime.py','--entry','scripts/eval_policy.py','--','--artifact','candidate.pt','--split','val','--games','0',
              '--compare-conditional','--bootstrap','400','--seed','16'],
              {**fixed,'PREFIX_RUN_PLAN':str(run/'run_plan.json')})
        policy_path=run/'policy_eval_val_candidate_vs_conditional_policy.json'
        report=json.loads(policy_path.read_text(encoding='utf-8'))
        if report['manifest']['export_fingerprint']!=generation['export_fingerprint'] or report['games']!=coverage['game_counts_by_split']['val']:
            raise ValueError('Paired evaluation used a different benchmark or incomplete validation games')
        if any(report['metrics'][tag]['unaffordable']!=0 for tag in ('A','B')):
            raise ValueError('Displayed policy produced unaffordable recommendations')
        stage('analysis',['scripts/analyze_patch_development.py','--report',str(policy_path)])
        unchanged()
        write_json(run/'completion.json',{'schema':1,'state':'complete','purpose':'patch_development',
                   'export_fingerprint':generation['export_fingerprint'],'candidate_sha256':sha256(run/'candidate.pt'),
                   'policy_report':str(policy_path),'policy_report_sha256':sha256(policy_path),
                   'completed_stages':completed,'protected_unchanged':True,'final_test_evaluation':False,
                   'promotion_allowed':False,'production_hold_cleared':False,
                   'elapsed_seconds':time.monotonic()-start,'peak_child_job_memory_bytes':memory_job.peak_bytes()})
        current='complete'
        status('complete')
    except BaseException as exc:
        memory_job.stop()
        status('failed',error=str(exc))
        raise
    finally:
        memory_job.close()


if __name__=='__main__': main()
