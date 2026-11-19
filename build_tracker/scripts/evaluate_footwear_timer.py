"""Measure source consistency of the ordinary 16.17 boot timer/slot replay.

Outputs are recovery evidence only, never training admission. The exact delivery
tick is not claimed observed. Source contradictions fail a game independently
of the local grant-boundary row exclusion.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import canonical_sha,sha256,write_json
from app.ddragon import dragon_for_patch
from app.riot import RawMatchArchive
from app.shop_econ import payload_ids
import app.shop_econ as base_economics
from audit_patch_corpus import terminal_differences
from validate_viego_candidates import input_state
from replay_footwear_windows import load_reconstructor,receipt_queries,split_receipt_batches,state_issues
from validate_footwear_timer_results import validate_timer_result,grant_boundary_reason
import footwear_economics as economics
import baseline
from footwear_timer_inventory import FootwearTimerInventoryReplay


def replay(module,match,timeline,dragon):
    module.FOOTWEAR_ROLES={p['participantId']:p.get('teamPosition') or p.get('individualPosition') or '' for p in match['info']['participants']}
    debug={}
    with patch.object(base_economics,'net_spent',economics.net_spent):
        game,events=module.reconstruct_game(match,timeline,dragon,debug)
    return game,events,debug


def modeled_rows(events,dragon):
    result=[]
    with patch.object(baseline,'inventory_state',economics.inventory_state),patch.object(baseline,'decision_kind',economics.decision_kind):
        for index,event in enumerate(events):
            if baseline._example(event,dragon) is not None:
                result.append(index)
    return result


def analyze_record(record,module,dragon,prefix_checks):
    archive=RawMatchArchive(ROOT/'data/raw_match_v5')
    raw={}
    for kind,binding in record['files'].items():
        if sha256(ROOT/binding['path'])!=binding['sha256']: raise ValueError('Planned source bytes changed')
        raw[kind]=archive.read(kind,record['match_id'])
    match,timeline=raw['match'],raw['timeline']
    all_raw=[e for f in timeline['info']['frames'] for e in f.get('events',[])]
    result={'match_id':record['match_id'],'split':record['split'],
            'raw_sources_sha256':{k:v['sha256'] for k,v in record['files'].items()}}
    excluded=[]
    if any(p['championId']==234 for p in match['info']['participants']): excluded.append('unmodeled_viego_inventory')
    if any(e.get('type')=='ITEM_UNDO' and not e.get('beforeId') and not e.get('afterId') for e in all_raw): excluded.append('opaque_undo')
    if any(e.get('type')=='ITEM_PURCHASED' and e.get('itemId')==2422 for e in all_raw): excluded.append('explicit_grant_requires_handler')
    if split_receipt_batches(timeline,receipt_queries(match,timeline,dragon)): excluded.append('split_footwear_receipt_batch')
    if excluded:
        return {**result,'status':'excluded_before_replay','exclusions':excluded,'ordinary_replay_explains_game':False,'candidate_rows':0}
    game,events,debug=replay(module,match,timeline,dragon)
    qa=validate_timer_result(match,timeline,events,debug,dragon)
    endpoints=terminal_differences(match,debug['final_inventories'],dragon)
    inventory_issues=debug.get('footwear_inventory_failures',[])
    event_issues=Counter()
    for event in events:
        event_issues.update(state_issues(event,timeline,dragon))
    models=modeled_rows(events,dragon)
    grants=debug['footwear_grants']
    retained={}
    for guard in (0,250,1000):
        retained[guard]=[i for i in models if grant_boundary_reason(events[i],grants,guard_ms=guard) is None]
    failures=[]
    if not qa['passed']: failures.append('source_or_rune_qa')
    if endpoints: failures.append('terminal_inventory_mismatch')
    if inventory_issues: failures.append('intermediate_physical_inventory')
    if event_issues: failures.append('shop_state_or_label')
    if any(s.get('invalid') for s in debug['footwear_state']): failures.append('invalid_timer_state')
    if any(2422 in payload_ids(e.get('label_bought',[])) for e in events): failures.append('free_boot_purchase_label')
    # Whole-outcome mutation is cheap and applies to every replayed game.
    changed=deepcopy(match)
    for p in changed['info']['participants']:
        p.update({f'item{i}':3869 for i in range(7)})
        p.update(roleBoundItem=3175,kills=999,deaths=999,assists=999,win=not p['win'])
        for style in p.get('perks',{}).get('styles',[]):
            for r in style.get('selections',[]): r.update(var1=999,var2=999,var3=999)
    _game,mutated,mutated_debug=replay(module,changed,timeline,dragon)
    final_invariant=canonical_sha(events)==canonical_sha(mutated) and canonical_sha(grants)==canonical_sha(mutated_debug['footwear_grants'])
    if not final_invariant: failures.append('final_outcome_input_dependence')
    checks=[]
    if prefix_checks and not failures:
        # Test representative pre/post-grant inputs without pretending absent
        # truncated purchase-label windows establish a causal defect.
        choices=[i for i in retained[1000] if not events[i].get('is_save')]
        first=min((g['timestamp'] for g in grants),default=float('inf'))
        buckets=([i for i in choices if events[i]['ts']<first],
                 [i for i in choices if events[i]['ts']>first])
        for bucket in buckets:
            if not bucket: continue
            passed=False
            for index in bucket[:5]:
                original=events[index]
                cut=original['ts']
                trimmed=deepcopy(timeline)
                for frame in trimmed['info']['frames']:
                    frame['events']=[e for e in frame.get('events',[]) if e.get('timestamp',0)<=cut]
                    if frame['timestamp']>cut:
                        for stats in frame.get('participantFrames',{}).values():
                            stats.update(currentGold=999999,totalGold=999999,level=18,minionsKilled=999,jungleMinionsKilled=999)
                _g,short,_d=replay(module,match,trimmed,dragon)
                found=next((e for e in short if e['participant_id']==original['participant_id'] and e['ts']==cut and not e.get('is_save')),None)
                if found is None: continue
                passed=canonical_sha(input_state(found))==canonical_sha(input_state(original))
                checks.append({'participant_id':original['participant_id'],'ts':cut,'passed':passed})
                if not passed: failures.append('future_input_dependence')
                break
            if not passed and not failures: failures.append('prefix_check_inconclusive')
    clean=not failures
    return {**result,'status':'source_consistent' if clean else 'unresolved_replay',
            'ordinary_replay_explains_game':clean,'failures':failures,
            'shop_events':len(events),'model_rows':len(models),
            'candidate_rows':len(retained[1000]) if clean else 0,
            'retained_rows_by_guard_ms':{str(g):len(v) if clean else 0 for g,v in retained.items()},
            'grant_count':len(grants),'full_slot_delayed_grants':sum(g['full_slot_wait_ms']>0 for g in grants),
            'grants':grants,'timer_diagnostics':debug['footwear_diagnostics'],
            'independent_qa':qa,'endpoint_differences':endpoints,
            'physical_inventory_checks':debug['footwear_inventory_checks'],
            'physical_inventory_failure_count':len(inventory_issues),'physical_inventory_examples':inventory_issues[:6],
            'shop_state_issue_counts':dict(event_issues),'final_outcome_input_invariant':final_invariant,
            'conditional_elixir_auto_consumptions':debug.get('elixir_auto_consumptions',[]),
            'prefix_checks':checks,
            'candidate_decision_ids':[{'participant_id':events[i]['participant_id'],'ts':events[i]['ts'],'event_index':events[i]['event_index']}
                                      for i in retained[1000]] if clean else []}


def evaluate(source_plan,code,out,*,sample=None,limit=None,prefix_checks=True,rune_inventory=None):
    if out.exists(): raise ValueError('Immutable recovery report already exists')
    plan=json.loads(source_plan.read_text(encoding='utf-8'))
    records=plan['matches']
    if rune_inventory:
        inventory=json.loads(rune_inventory.read_text(encoding='utf-8'))
        if not inventory.get('completed') or inventory.get('errors'):
            raise ValueError('Rune inventory is incomplete or has errors')
        footwear_ids={r['match_id'] for r in inventory['games_metadata'] if r.get('magical_footwear') is True}
        records=[r for r in records if r['match_id'] in footwear_ids]
    if sample:
        ids={r['match_id'] for r in json.loads(sample.read_text(encoding='utf-8'))['games']}
        records=[r for r in records if r['match_id'] in ids]
    if limit is not None: records=records[:limit]
    if any(r['patch']!='16.17' or r['split'] not in ('train','val') or r.get('legacy_split')=='test' for r in records):
        raise ValueError('Source scope violation: original test remains sealed')
    module=load_reconstructor(code)
    module.InventoryReplay=lambda inventory,dragon: FootwearTimerInventoryReplay(inventory,dragon,role_for=module.FOOTWEAR_ROLES)
    dragon=economics.FootwearDataDragon(dragon_for_patch('16.17'))
    code_paths=[Path(__file__),code,*[ROOT/'scripts'/f'{name}.py' for name in
                ('footwear_timer','footwear_timer_inventory','footwear_economics','magical_footwear','validate_footwear_timer_results','replay_footwear_windows',
                 'baseline','qualify_patch_corpus','audit_patch_corpus','validate_viego_candidates')],
                *(ROOT/'app').glob('*.py')]
    bindings={str(p):sha256(p) for p in code_paths}
    code_directory=out.with_suffix('.code')
    code_directory.mkdir(parents=True,exist_ok=False)
    for path in code_paths:
        relative=path.relative_to(ROOT)
        destination=code_directory/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    totals=defaultdict(Counter)
    exclusions=Counter()
    results=[]
    started=datetime.now(timezone.utc)
    for index,record in enumerate(records,1):
        try:
            result=analyze_record(record,module,dragon,prefix_checks)
        except Exception as exc:
            result={'match_id':record['match_id'],'split':record['split'],'status':'replay_error',
                    'error':f'{type(exc).__name__}: {exc}','ordinary_replay_explains_game':False,'candidate_rows':0}
        results.append(result)
        total=totals[record['split']]
        total['sampled_games']+=1
        total['replayed_games']+=result['status'] not in ('excluded_before_replay','replay_error')
        total['source_consistent_games']+=result['ordinary_replay_explains_game']
        total['candidate_rows']+=result['candidate_rows']
        total['replayed_model_rows']+=result.get('model_rows',0)
        total['modeled_grants']+=result.get('grant_count',0)
        total['full_slot_delayed_grants']+=result.get('full_slot_delayed_grants',0)
        for key in result.get('exclusions',[])+result.get('failures',[]): exclusions[key]+=1
        if result['status']=='replay_error': exclusions['replay_error:'+result['error']]+=1
        if index%10==0:
            print(f'Checked {index}/{len(records)} games; source-consistent {sum(v["source_consistent_games"] for v in totals.values())}',flush=True)
            write_json(out.with_suffix('.status.json'),{'state':'running','processed':index,'total':len(records),
                       'totals_by_split':{k:dict(v) for k,v in totals.items()},'updated_utc':datetime.now(timezone.utc).isoformat()})
    if any(sha256(Path(path))!=digest for path,digest in bindings.items()): raise ValueError('Recovery code changed during run')
    report={'schema':1,'purpose':'ordinary_footwear_timer_slot_recovery_measurement',
            'started_utc':started.isoformat(),'finished_utc':datetime.now(timezone.utc).isoformat(),
            'source_plan_sha256':sha256(source_plan),'sample_report_sha256':sha256(sample) if sample else None,
            'rune_inventory_sha256':sha256(rune_inventory) if rune_inventory else None,
            'original_test_explicitly_excluded':True,'prefix_probes_performed':prefix_checks,
            'code_sha256':bindings,'patch':'16.17','totals_by_split':{k:dict(v) for k,v in totals.items()},
            'code_snapshot':str(code_directory),
            'exclusion_and_failure_counts':dict(exclusions),'games':results,
            'timing_guard_ms':1000,'timing_guard_is_verified_delivery_bound':False,
            'training_admission_performed':False,'final_test_payload_opened':False,
            'limits':['A pass means consistency with the checked source evidence under the timer/first-room hypothesis, not independently observed exact delivery.',
                      'One second is a conservative sensitivity guard, not a measured upper bound on the server grant tick.',
                      'Source contradictions invalidate a game even if a nearby decision is omitted.',
                      'Postgame rune summaries are QA-only and never drive reconstructed inputs.',
                      'This report alone does not publish or admit a training generation.']}
    write_json(out,report)
    write_json(out.with_suffix('.status.json'),{'state':'complete','processed':len(records),'total':len(records),'report_sha256':sha256(out)})
    print(json.dumps({'report':str(out),'totals_by_split':report['totals_by_split'],'failures':dict(exclusions)},indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--sample-report',type=Path)
    p.add_argument('--rune-inventory',type=Path,help='Restrict population to verified fixed-rune footwear users')
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--skip-prefix-probes',action='store_true')
    a=p.parse_args()
    evaluate(a.source_plan.resolve(),a.code.resolve(),a.out.resolve(),sample=a.sample_report.resolve() if a.sample_report else None,
             limit=a.limit,prefix_checks=not a.skip_prefix_probes,rune_inventory=a.rune_inventory.resolve() if a.rune_inventory else None)
