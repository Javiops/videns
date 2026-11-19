"""Four disjoint offline workers, one frozen footwear recovery method."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
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
    base=ROOT/'data/acceptance/footwear_repair_20260911'
    folder=base/'full_parallel'
    folder.mkdir(exist_ok=False)
    plan_path=ROOT/'data/acceptance/16.17_development_20260911/source_plan.json'
    rune_path=plan_path.parent/'source_rune_inventory.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    runes=json.loads(rune_path.read_text(encoding='utf-8'))
    foot_ids={r['match_id'] for r in runes['games_metadata'] if r.get('magical_footwear') is True}
    selected=[r for r in plan['matches'] if r['match_id'] in foot_ids]
    if any(r['split'] not in ('train','val') or r['patch']!='16.17' or r.get('legacy_split')=='test' for r in selected):
        raise ValueError('Invalid source scope')
    expected={r['match_id'] for r in selected}
    if len(expected)!=len(selected): raise ValueError('Duplicate planned sources')
    protected_paths=[ROOT/'data/ml'/name for name in ('prefix_model.pt','deployment_manifest.json','pipeline_hold.json')]
    protected={str(p):sha256(p) if p.exists() else None for p in protected_paths}
    started=time.monotonic()
    memory=WindowsMemoryJob(8_000_000_000)
    processes=[]
    logs=[]
    reports=[]
    try:
        env={**os.environ,'PYTHONUTF8':'1','PYTHONUNBUFFERED':'1','HTTP_PROXY':'http://127.0.0.1:9','HTTPS_PROXY':'http://127.0.0.1:9'}
        for index in range(4):
            records=selected[index::4]
            shard_plan={**plan,'matches':records,'games':len(records),
                        'split_counts':dict(Counter(r['split'] for r in records)),
                        'parent_source_plan_sha256':sha256(plan_path),'shard_index':index,'shards':4}
            shard_path=folder/f'source_plan_{index}.json'
            report_path=folder/f'report_{index}.json'
            write_json(shard_path,shard_plan)
            log=(folder/f'worker_{index}.log').open('x',encoding='utf-8')
            logs.append(log)
            command=[sys.executable,'scripts/evaluate_footwear_timer.py','--source-plan',str(shard_path),
                     '--code',str(ROOT/'data/experiments/footwear-timer-v1/reconstruct.py'),
                     '--out',str(report_path),'--skip-prefix-probes']
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=4)
            memory.attach(process)
            processes.append(process)
            reports.append(report_path)
        while True:
            codes=[p.poll() for p in processes]
            if any(code not in (None,0) for code in codes): raise RuntimeError(f'Audit worker failed: {codes}')
            elapsed=time.monotonic()-started
            if elapsed>10800: raise RuntimeError('Parallel audit exceeded three hours')
            snapshots=[]
            for report in reports:
                status=report.with_suffix('.status.json')
                snapshots.append(json.loads(status.read_text(encoding='utf-8')) if status.exists() else {'processed':0})
            write_json(folder/'status.json',{'state':'running','elapsed_seconds':elapsed,'workers':codes,
                       'processed':sum(s.get('processed',0) for s in snapshots),'total':len(selected),
                       'source_plan_sha256':sha256(plan_path),'peak_child_job_memory_bytes':memory.peak_bytes()})
            if all(code==0 for code in codes): break
            time.sleep(15)
        all_games=[]
        totals=defaultdict(Counter)
        failures=Counter()
        shard_bindings=[]
        method=None
        for report_path in reports:
            report=json.loads(report_path.read_text(encoding='utf-8'))
            if method is None: method=report
            elif report['code_sha256']!=method['code_sha256']: raise ValueError('Workers used different code')
            all_games.extend(report['games'])
            for split,values in report['totals_by_split'].items(): totals[split].update(values)
            failures.update(report['exclusion_and_failure_counts'])
            shard_bindings.append({'path':str(report_path),'sha256':sha256(report_path),'source_plan_sha256':report['source_plan_sha256']})
        if len(all_games)!=len(expected) or {g['match_id'] for g in all_games}!=expected:
            raise ValueError('Shard membership is incomplete or overlaps')
        after={str(p):sha256(p) if p.exists() else None for p in protected_paths}
        if after!=protected: raise ValueError('Protected artifacts changed')
        final={k:v for k,v in method.items() if k not in ('games','totals_by_split','exclusion_and_failure_counts','code_snapshot')}
        final.update(source_plan_sha256=sha256(plan_path),rune_inventory_sha256=sha256(rune_path),
                     population='All 11555 selected 16.17 Magical Footwear source games; disjoint train/development-val shards',
                     finished_utc=datetime.now(timezone.utc).isoformat(),
                     totals_by_split={k:dict(v) for k,v in totals.items()},exclusion_and_failure_counts=dict(failures),
                     shard_reports=shard_bindings,protected_before=protected,protected_after=after,
                     games=sorted(all_games,key=lambda r:r['match_id']))
        out=base/'timer_full_source_parallel.json'
        if out.exists(): raise ValueError('Final recovery report exists')
        write_json(out,final)
        write_json(folder/'status.json',{'state':'complete','elapsed_seconds':time.monotonic()-started,
                   'processed':len(all_games),'total':len(selected),'report':str(out),'report_sha256':sha256(out),
                   'totals_by_split':final['totals_by_split'],'protected_unchanged':True})
    except BaseException as error:
        memory.stop()
        write_json(folder/'status.json',{'state':'failed','error':str(error),'elapsed_seconds':time.monotonic()-started})
        raise
    finally:
        memory.close()
        for log in logs: log.close()


if __name__=='__main__': main()
