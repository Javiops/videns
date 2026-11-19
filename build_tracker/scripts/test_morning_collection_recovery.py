"""Offline checks that resumed collection cannot skip or broaden its gates."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import resume_morning_collection as recovery
from app.dataset_artifacts import sha256,write_json


def refused(call):
    try: call()
    except (ValueError,RuntimeError): return
    raise AssertionError('Invalid recovery gate was accepted')


action={'actions':[{'Execute':str(ROOT/'.venv/Scripts/pythonw.exe'),
                    'Arguments':str(ROOT/'scripts/daily_pull.py')}]}
recovery.verify_daily_action(action)
bad=deepcopy(action)
bad['actions'][0]['Arguments']+=' --export-patch 16.18'
refused(lambda:recovery.verify_daily_action(bad))
bad=deepcopy(action);bad['actions'][0]['Arguments']='unreviewed.py'
refused(lambda:recovery.verify_daily_action(bad))
now=datetime.now(timezone.utc)
fresh={'state':'Ready','result':0,'last_run_utc':now.isoformat()}
assert recovery.verify_daily_finished(fresh,now.timestamp()-1)
assert not recovery.verify_daily_finished({**fresh,'state':'Running'},now.timestamp()-1)
assert not recovery.verify_daily_finished(fresh,now.timestamp()+10)
refused(lambda:recovery.verify_daily_finished({**fresh,'result':1},now.timestamp()-1))

with tempfile.TemporaryDirectory(prefix='morning-recovery-test-') as temp:
    folder=Path(temp);sample=folder/'sample';sample.mkdir()
    state={'patch':'16.18','cohort_parent_sha256':'parent-sha','discovery_complete':True,
           'jobs':{'KR_A':[],'KR_B':[]},'results':{'KR_A':{'status':'stored'},'KR_B':{'status':'quarantined'}}}
    write_json(sample/'state.json',state)
    report={'structural_checks_passed':True,'scope_complete':False,'games':1,
            'checkpoint_sha256':sha256(sample/'state.json'),'unresolved_pros':['known-missing-pro']}
    write_json(sample/'report.json',report)
    recovery.verify_sample_collection(sample,'parent-sha',['known-missing-pro'])
    for mutation in ('missing_result','discovery_incomplete','changed_parent','changed_hash','new_unresolved','no_structure'):
        altered=deepcopy(state);changed=deepcopy(report)
        if mutation=='missing_result': altered['results'].pop('KR_B')
        if mutation=='discovery_incomplete': altered['discovery_complete']=False
        if mutation=='changed_parent': altered['cohort_parent_sha256']='different'
        write_json(sample/'state.json',altered)
        changed['checkpoint_sha256']=sha256(sample/'state.json')
        if mutation=='changed_hash': changed['checkpoint_sha256']='different'
        if mutation=='new_unresolved': changed['unresolved_pros'].append('new-unresolved-pro')
        if mutation=='no_structure': changed['structural_checks_passed']=False
        write_json(sample/'report.json',changed)
        refused(lambda:recovery.verify_sample_collection(sample,'parent-sha',['known-missing-pro']))

    # A failed actual daily attempt must stop the sequence before sample or
    # refetch subprocesses can be launched; this test performs no API requests.
    run=folder/'run';run.mkdir();refetch=folder/'refetch';refetch.mkdir()
    parent=folder/'parent.json';write_json(parent,{'fixture':True})
    write_json(refetch/'plan.json',{'fixture':True})
    plan={'purpose':'serial_collection_recovery','current_training_integration':False,
          'stages':['daily_collection','prepare_sample','collect_sample','resume_refetch'],
          'code_sha256':{},'protected_sha256':{},'parent_sha256':sha256(parent),
          'refetch_plan_sha256':sha256(refetch/'plan.json')}
    write_json(run/'plan.json',plan)
    class FakeJob:
        def __init__(self,*args): pass
        def stop(self): pass
        def close(self): pass
    class FakeLease:
        def __enter__(self): return self
        def __exit__(self,*args): pass
    snapshots=[{**fresh,'last_run_utc':(now-timedelta(hours=6)).isoformat()},
               {**fresh,'last_run_utc':(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat(),'result':1}]
    with patch.multiple(recovery,RUN=run,PARENT=parent,REFETCH=refetch), \
         patch('run_patch_development.WindowsMemoryJob',FakeJob), \
         patch('app.collector_lock.CollectorLease',FakeLease), \
         patch.object(recovery,'verify_daily_action'), \
         patch.object(recovery,'daily_state',side_effect=snapshots), \
         patch.object(recovery,'powershell') as shell, \
         patch.object(recovery.subprocess,'Popen') as child:
        refused(recovery.main)
        child.assert_not_called()
        shell.assert_called_once()
    failed=json.loads((run/'status.json').read_text())
    assert failed['state']=='failed' and failed['stage']=='daily_collection' and not failed['completed_stages']

print('ok morning recovery: reviewed daily invocation, current completion, exact sample scope, accepted unresolved identities, failure stops downstream jobs')
