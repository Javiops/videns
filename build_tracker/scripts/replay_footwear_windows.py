"""Isolated proof of concept for boot-receipt reconciliation and safe windows.

No timer grants are inserted. A current raw 2422 sale/consumption establishes
the missing component for that transaction's accounting only. Its decision and
all ambiguous board windows remain excluded. Nothing is published for training.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import canonical_sha,sha256,write_json
from app.ddragon import dragon_for_patch
from app.inventory import InventoryReplay
from app.riot import RawMatchArchive
from app.shop_econ import payload_ids
import app.shop_econ as original_economics
import baseline
from audit_patch_corpus import terminal_inventory,slot_state_checks,quest_completion_times
from qualify_patch_corpus import inventory_issues,inventory_shape
from validate_viego_candidates import input_state
from magical_footwear import classify_decision_windows,compatible_footwear_upgrade
import footwear_economics as economics


class ReceiptReplay(InventoryReplay):
    def _consumes(self,purchase,component):
        return super()._consumes(purchase,component) or (
            purchase!=component and component in economics.components_of(purchase,self.dragon))


def load_reconstructor(path):
    spec=importlib.util.spec_from_file_location('isolated_footwear_reconstruction',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.InventoryReplay=ReceiptReplay
    module.combine_cost=economics.combine_cost
    return module


def receipt_queries(match,timeline,dragon):
    groups=defaultdict(list)
    for frame in timeline['info']['frames']:
        for e in frame.get('events',[]):
            if e.get('participantId') and e.get('type','').startswith('ITEM_'):
                groups[e['participantId'],e['timestamp']].append(e)
    receipts=[]
    for (pid,ts),batch in groups.items():
        sale=any(e['type']=='ITEM_SOLD' and e.get('itemId')==2422 for e in batch)
        consumed=any(e['type']=='ITEM_DESTROYED' and e.get('itemId')==2422 for e in batch)
        upgrade=any(e['type']=='ITEM_PURCHASED' and compatible_footwear_upgrade(e.get('itemId',0),dragon) for e in batch)
        if sale or consumed and upgrade:
            receipts.append({'participant_id':pid,'anchor_ts':ts+1,'receipt_ts':ts})
    valid=set()
    for row in classify_decision_windows(match,timeline,receipts,dragon):
        state=row['footwear_state']
        if state['phase']=='resolved' and state['resolved_at']==row['receipt_ts']:
            valid.add((row['participant_id'],row['receipt_ts']))
    return valid


def replay(module,match,timeline,dragon):
    module.FOOTWEAR_RECEIPTS=receipt_queries(match,timeline,dragon)
    debug={}
    # Only this process's diagnostic legacy spend helper is substituted.
    # Qualified model wallets still come from the causal event gold walk.
    with patch.object(original_economics,'net_spent',economics.net_spent):
        game,events=module.reconstruct_game(match,timeline,dragon,debug)
    queries=[{'participant_id':e['participant_id'],'anchor_ts':e['ts']-15000 if e.get('is_save') else e['ts'],
              'event_index':i} for i,e in enumerate(events)]
    windows=classify_decision_windows(match,timeline,queries,dragon)
    return game,events,debug,windows


def split_receipt_batches(timeline,receipts):
    """The current core prices contiguous item batches, not timestamp unions."""
    groups=Counter()
    item_types={'ITEM_PURCHASED','ITEM_DESTROYED','ITEM_SOLD','ITEM_UNDO'}
    for frame in timeline['info']['frames']:
        previous=None
        for event in frame.get('events',[]):
            key=(event.get('participantId'),event.get('timestamp'))
            if event.get('type') not in item_types:
                previous=None
                continue
            if key!=previous:
                groups[key]+=1
            previous=key
    return sorted(key for key in receipts if groups[key]>1)


def receipt_exclusion(event,receipts):
    anchor=event['ts']-15000 if event.get('is_save') else event['ts']
    if any(ts==anchor for _pid,ts in receipts):
        return 'decision_tied_to_receipt'
    if any(pid==event['participant_id'] and event['ts']<=ts<=event['ts_end'] for pid,ts in receipts):
        return 'visit_spans_receipt'
    return None


def state_issues(event,timeline,dragon):
    quests=quest_completion_times(timeline,1202)
    problems=set()
    for good,rule,_detail in slot_state_checks(event,quests):
        if not good: problems.add(rule)
    anchor=event['ts']-15000 if event.get('is_save') else event['ts']
    owner=next(m for m in event['board'] if m['participant_id']==event['participant_id'])
    states=[(m,m['items'],anchor,False) for m in event['board']]
    states.extend([(owner,event['inventory_before'],anchor,False),(owner,event['inventory_after'],event['ts_end'],True)])
    for member,items,when,inclusive in states:
        quest=quests.get(member['participant_id'],float('inf'))
        complete=quest<=when if inclusive else quest<when
        problems.update(inventory_issues(items,member['team_position'],complete,dragon))
        if inventory_shape(items,member['team_position'],complete,dragon)['regular_slots']+baseline._unmodeled_regular_slots(items,dragon)>6:
            problems.add('total_regular_slot_overflow')
    if Counter(payload_ids(event.get('label_bought',[])))-Counter(payload_ids(event['inventory_after'])):
        problems.add('label_not_retained')
    if 2422 in payload_ids(event.get('label_bought',[])): problems.add('free_boot_purchase_label')
    return sorted(problems)


def run(source_plan,sample_report,code_path,out,per_split):
    if out.exists(): raise ValueError('Evidence exists')
    plan=json.loads(source_plan.read_text(encoding='utf-8'))
    planned={r['match_id']:r for r in plan['matches']}
    sampled=json.loads(sample_report.read_text(encoding='utf-8'))['games']
    selected=[]
    counts=Counter()
    for item in sampled:
        record=planned[item['match_id']]
        split=record['split']
        if split not in ('train','val'): raise ValueError('Test or unknown source outside scope')
        if counts[split]<per_split:
            selected.append(record)
            counts[split]+=1
    module=load_reconstructor(code_path)
    dragon=economics.FootwearDataDragon(dragon_for_patch('16.17'))
    archive=RawMatchArchive(ROOT/'data/raw_match_v5')
    reports=[]
    totals=defaultdict(Counter)
    for record in selected:
        mid=record['match_id']
        raw={}
        for kind,binding in record['files'].items():
            if sha256(ROOT/binding['path'])!=binding['sha256']: raise ValueError('Planned raw source changed')
            raw[kind]=archive.read(kind,mid)
        match,timeline=raw['match'],raw['timeline']
        forbidden=[]
        if any(p['championId']==234 for p in match['info']['participants']): forbidden.append('unmodeled_viego_inventory')
        if any(e.get('type')=='ITEM_UNDO' and not e.get('beforeId') and not e.get('afterId') for f in timeline['info']['frames'] for e in f.get('events',[])):
            forbidden.append('opaque_undo')
        if any(e.get('type')=='ITEM_PURCHASED' and e.get('itemId')==2422 for f in timeline['info']['frames'] for e in f.get('events',[])):
            forbidden.append('explicit_grant_requires_separate_handler')
        if split_receipt_batches(timeline,receipt_queries(match,timeline,dragon)):
            forbidden.append('split_footwear_receipt_transaction')
        result={'match_id':mid,'split':record['split'],'excluded_game_reasons':forbidden,
                'raw_sources_sha256':{k:v['sha256'] for k,v in record['files'].items()}}
        if forbidden:
            reports.append(result)
            continue
        game,events,debug,windows=replay(module,match,timeline,dragon)
        receipt_times=sorted(module.FOOTWEAR_RECEIPTS)
        end_queries=[{'participant_id':p['participantId'],'anchor_ts':match['info']['gameDuration']*1000+1}
                     for p in match['info']['participants']]
        end_states={r['participant_id']:r['footwear_state'] for r in classify_decision_windows(match,timeline,end_queries,dragon)}
        endpoint_issues=[]
        endpoint_unknown=[]
        invalid_histories=[]
        for p in match['info']['participants']:
            pid=p['participantId']
            if end_states[pid]['phase']=='invalid':
                invalid_histories.append({'participant_id':pid,'diagnostics':end_states[pid]['diagnostics']})
                continue
            if not end_states[pid]['window_known']:
                endpoint_unknown.append(pid)
                continue
            actual=[p.get(f'item{i}',0) for i in range(7)]
            bound=p.get('roleBoundItem',0)
            cls=dragon.classify(bound)
            if bound not in actual and (bound in (2055,772043) or cls['is_boots'] or cls['is_basic_boots']): actual.append(bound)
            a,b=terminal_inventory(actual,dragon),terminal_inventory(debug['final_inventories'][pid],dragon)
            if a!=b: endpoint_issues.append({'participant_id':pid,'extra':sorted((b-a).elements()),'missing':sorted((a-b).elements())})
        candidate_indices=[]
        rejected=Counter()
        for i,(event,window) in enumerate(zip(events,windows)):
            if not window['footwear_window_known']:
                rejected['unknown_footwear_board']+=1
                continue
            # A receipt must not supply its own shop-start input, even if a
            # visit began with an earlier ordinary purchase before eligibility.
            receipt_reason=receipt_exclusion(event,receipt_times)
            if receipt_reason:
                rejected[receipt_reason]+=1
                continue
            if state_issues(event,timeline,dragon):
                rejected['inventory_or_label_check']+=1
                continue
            with patch.object(baseline,'inventory_state',economics.inventory_state),patch.object(baseline,'decision_kind',economics.decision_kind):
                row=baseline._example(event,dragon)
            if row is None:
                rejected['no_model_target']+=1
                continue
            candidate_indices.append(i)
        mutable=deepcopy(match)
        for p in mutable['info']['participants']:
            p.update({f'item{i}':3869 for i in range(7)})
            p.update(kills=999,deaths=999,assists=999,roleBoundItem=3175,win=not p['win'])
            for style in p.get('perks',{}).get('styles',[]):
                for selection in style.get('selections',[]):
                    selection.update(var1=999,var2=999,var3=999)
        _g,changed,_d,changed_windows=replay(module,mutable,timeline,dragon)
        final_invariant=canonical_sha(events)==canonical_sha(changed) and canonical_sha(windows)==canonical_sha(changed_windows)
        representatives=[]
        for phase in ('pre_grant','resolved','not_selected'):
            members=[i for i in candidate_indices if windows[i]['footwear_state']['phase']==phase and not events[i].get('is_save')]
            if members: representatives.append(members[len(members)//2])
        prefix=[]
        for i in sorted(set(representatives)):
            original=events[i]
            cutoff=original['ts']
            reduced=deepcopy(timeline)
            for frame in reduced['info']['frames']:
                frame['events']=[e for e in frame.get('events',[]) if e.get('timestamp',0)<=cutoff]
                if frame['timestamp']>cutoff:
                    for stats in frame.get('participantFrames',{}).values():
                        stats.update(currentGold=999999,totalGold=999999,level=18,minionsKilled=999,jungleMinionsKilled=999)
            _g,fresh,_d,fresh_windows=replay(module,match,reduced,dragon)
            found=next((j for j,e in enumerate(fresh) if e['participant_id']==original['participant_id'] and e['ts']==cutoff and not e.get('is_save')),None)
            same=found is not None and canonical_sha(input_state(fresh[found]))==canonical_sha(input_state(original))
            if found is not None:
                same= same and canonical_sha(fresh_windows[found]['footwear_state'])==canonical_sha(windows[i]['footwear_state'])
            prefix.append({'participant_id':original['participant_id'],'ts':cutoff,'passed':same,
                           'status':'measured' if found is not None else 'inconclusive_label_window'})
        preliminary_pass=not endpoint_issues and not invalid_histories and final_invariant and all(p['passed'] for p in prefix)
        result.update(shop_events=len(events),reconciled_receipts=[{'participant_id':p,'ts':t} for p,t in receipt_times],
                      potential_model_rows=len(candidate_indices),rejected_decisions=dict(rejected),
                      known_endpoint_issues=endpoint_issues,unresolved_endpoint_participants=endpoint_unknown,
                      invalid_footwear_histories=invalid_histories,
                      final_outcome_input_invariant=final_invariant,prefix_checks=prefix,
                      prototype_checks_passed=preliminary_pass,
                      passing_model_rows=len(candidate_indices) if preliminary_pass else 0)
        totals[record['split']].update(games=1,shop_events=len(events),potential_model_rows=len(candidate_indices),
                                     passing_model_rows=result['passing_model_rows'])
        reports.append(result)
        print(f'{mid}: candidate_rows={len(candidate_indices)}, prototype_checks={preliminary_pass}',flush=True)
    report={'schema':1,'purpose':'isolated_receipt_reconciliation_prototype','created_utc':datetime.now(timezone.utc).isoformat(),
            'source_plan_sha256':sha256(source_plan),'sample_report_sha256':sha256(sample_report),
            'code_sha256':{str(p):sha256(p) for p in (Path(__file__),code_path,ROOT/'scripts/magical_footwear.py',ROOT/'scripts/footwear_economics.py')},
            'patch':'16.17','sampled_games':len(selected),'totals_by_split':{k:dict(v) for k,v in totals.items()},
            'games':reports,'training_admission_performed':False,'final_test_payload_opened':False,
            'limits':['This is a prototype screen, not a full semantic audit or published training generation.',
                      'No delivery timestamp is inferred; all unknown board windows and receipt-crossing own visits are omitted.',
                      'Only representative retained input prefixes receive future perturbation tests here.',
                      'All receipt-tied decision anchors and split receipt batches are excluded; timestamp unions cannot change transaction pricing.',
                      'Other unobserved grants and unsupported inventory mechanics still require qualification.',
                      'A current receipt establishes accounting only for itself and later states; it never repairs its own earlier features.']}
    write_json(out,report)
    print(json.dumps(report['totals_by_split'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-plan',type=Path,required=True)
    p.add_argument('--sample-report',type=Path,required=True)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--per-split',type=int,default=10)
    a=p.parse_args()
    if not 1<=a.per_split<=10: p.error('Bounded prototype permits at most ten games per split')
    run(a.source_plan.resolve(),a.sample_report.resolve(),a.code.resolve(),a.out.resolve(),a.per_split)
