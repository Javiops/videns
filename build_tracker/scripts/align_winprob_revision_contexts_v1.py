"""Bounded pre-A alignment probe for ten existing safe 16.18 fit revisions.

Consumes only versioned winprob artifacts, never its Python modules. Source
replay/projection is the reviewed build_tracker implementation. This probe
does not load a policy, fit a model, or analyze terminal outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
import time

import export_winprob_trajectories_v1 as producer

ROOT=producer.ROOT
DEFAULT_CENSUS=ROOT.parent/'winprob/data/runs/undo-pairs-20260930-v1/experiment'
SCHEMA='winprob-revision-source-alignment-v1'
SHOP={'ITEM_PURCHASED','ITEM_SOLD','ITEM_UNDO'}
MATERIAL={'CHAMPION_KILL','BUILDING_KILL','ELITE_MONSTER_KILL','DRAGON_SOUL_GIVEN','LEVEL_UP','SKILL_LEVEL_UP','GAME_END'}


def require(condition,message):
    if not condition: raise ValueError(message)


def physical_counts(items):
    values=[item.get('item_id') if isinstance(item,dict) else item for item in items]
    require(all(type(value) is int and value>0 for value in values),'unknown physical item identity')
    return Counter(values)


def select_pairs(census,manifest,denied,limit=10):
    require(type(limit) is int and 1<=limit<=10,'Alignment probe is bounded to ten pairs')
    census=Path(census).resolve(strict=True)
    require(not any((parent/'QUARANTINED.md').exists() for parent in (census,*census.parents)),'Quarantined pair census')
    report=producer.load_json(census/'report.json')
    require(report.get('schema')=='undo-pairs-support-v1' and report.get('stage')=='outcome_blind_counts'
            and report.get('outcome_targets_extracted') is False and report.get('cache_y_accessed') is False,
            'Expected the outcome-blind frozen pair census')
    for path,digest in report['binding_sha256'].items():
        require(producer.file_hash(path)==digest,'Census source/code binding changed: '+path)
    pairs=census/'pairs.jsonl'
    require(producer.file_hash(pairs)==report['pairs_sha256'],'Pair artifact hash mismatch')
    source_paths=[Path(path) for path in report['binding_sha256'] if Path(path).name=='source_manifest.json']
    require(len(source_paths)==1,'Expected one bound clean source manifest')
    source=producer.load_json(source_paths[0])
    require(all(source.get(key) is False for key in ('test_payload_reads','confirmatory_payload_reads','historical_validation_payload_reads')),
            'Pair census source population is not clean')
    source_ids={row['match_id'] for row in source['games']}
    require(len(source_ids)==len(source['games']) and not source_ids&set(denied),'Pair file source universe intersects reserved IDs')
    admitted={row['match_id']:row for row in manifest['games']}
    source_split=producer.assignment(producer.GENERATION/'split_assignment.json')
    result=[]
    with pairs.open('r',encoding='utf-8') as handle:
        for line_number,line in enumerate(handle,1):
            pair=json.loads(line);mid=pair['match_id']
            require(mid in source_ids and mid not in denied,'Pair record lies outside allowed source universe')
            if (pair.get('role')!='fit' or pair.get('patch')!='16.18' or pair.get('eligible_revision') is not True): continue
            require(admitted[mid]['role']=='train' and admitted[mid]['historical_validation'] is False
                    and source_split.get(mid)=='train','Pair cannot be relabelled into source fit')
            result.append({**pair,'source_line_number':line_number})
            if len(result)==limit: break
    require(bool(result),'No eligible 16.18 source-fit pairs')
    return result,{str(census/'report.json'):producer.file_hash(census/'report.json'),
                   str(pairs):report['pairs_sha256'],**report['binding_sha256']}


def raw_receipt_check(pair,timeline,dragon):
    pid=pair['participant_id'];ta=pair['feature_cutoff_ms'];tu=pair['undo_ms'];tb=pair['decision_timestamp_ms']
    close=pair['retention_end_ms'];a=pair['action_a'];b=pair['action_b']
    require(ta<tu<tb and tb-ta<=10000 and close==tb+10000,'Receipt timing changed')
    require(pair['strict_transaction_pair'] is True and pair['strict_context_pair'] is True
            and pair['pair_class']=='different_item_revision' and not pair['rejection_reasons'],'Receipt is not strict')
    events=[];source_end=0
    for fi,frame in enumerate(timeline['info']['frames']):
        source_end=max(source_end,int(frame['timestamp']))
        for ei,event in enumerate(frame.get('events',[])):
            source_end=max(source_end,int(event['timestamp']))
            events.append((int(event['timestamp']),fi,ei,event))
    events.sort(key=lambda row:row[:3])
    focal=[event for ts,_,_,event in events if event.get('participantId')==pid and event.get('type') in SHOP and ta<=ts<=close]
    require(len(focal)==3,'Extra or missing focal shop operation')
    first,undo,last=focal
    require((first['type'],first.get('itemId'),first['timestamp'])==('ITEM_PURCHASED',a,ta),'A receipt mismatch')
    require((undo['type'],undo.get('beforeId'),undo.get('afterId'),undo['timestamp'])==('ITEM_UNDO',a,0,tu),'Undo receipt mismatch')
    require((last['type'],last.get('itemId'),last['timestamp'])==('ITEM_PURCHASED',b,tb),'B receipt mismatch')
    require(undo.get('goldGain')==dragon.gold_block(a)['total']==pair['certificate']['full_static_price_a'],'Refund/static receipt mismatch')
    require(dragon.gold_block(b)['total']==pair['certificate']['full_static_price_b'],'B static price mismatch')
    require(source_end>=close,'Retention endpoint not observed')
    require(not any(event.get('participantId')==pid and event.get('type')=='ITEM_DESTROYED' and ta<=ts<=close
                    for ts,_,_,event in events),'Focal destruction/transformation in receipt')
    require(not any(event.get('type') in MATERIAL and ta<=ts<=tb for ts,_,_,event in events),'Material change within pair')
    prior=[ts for ts,_,_,event in events if event.get('participantId')==pid and event.get('type') in SHOP and ts<ta]
    require(not prior or ta-max(prior)>10000,'A is not the start of the entire source visit')
    return {'three_exact_operations':True,'no_opening_or_extra_operations':True,'source_observes_retention':True}


def align_pair(pair,events,mask,players,projector,attach_gold,debug,timeline,dragon):
    checks=raw_receipt_check(pair,timeline,dragon)
    pid=pair['participant_id'];ta=pair['feature_cutoff_ms'];tb=pair['decision_timestamp_ms'];a=pair['action_a'];b=pair['action_b']
    matches=[event for event in events if event['participant_id']==pid and event['ts']==ta]
    require(len(matches)==1,'No unique qualified visit at pre-A timestamp')
    event=matches[0]
    require(event['ts_end']==tb and not event.get('is_save'),'Qualified visit bounds do not match A-through-B')
    require(not event.get('opening_sales') and event.get('opening_sale_gold')==0,'Opening-sale adjustment prevents exact pre-A alignment')
    require(physical_counts(event['inventory_before'])==physical_counts(event['decision_inventory']),'Decision inventory differs from pre-A inventory')
    require(physical_counts(event['inventory_after'])==physical_counts(event['inventory_before'])+Counter({b:1}),
            'Closed inventory contains more than retained singleton B')
    require(physical_counts(event['label_bought'])==Counter({b:1}),'Retained target is not singleton B')
    require(producer.decision_key(event) in {producer.decision_key(row) for row in mask['admitted_decision_ids']},
            'Matched visit is not admitted by the frozen decision mask')
    require(event.get('training_eligible') is True and not event.get('source_quality_issues'),'Source row is unqualified')
    require(event['gold_est_version']=='prequential-v3' and event['gold_gap']['frame_ts']==pair['feature_frame_ms']<ta,
            'Wallet context is not the matched strict pre-A frame')
    focal_frame=pair['focal_pre_frame']
    require(event['gold_left']==focal_frame['currentGold'] and event['level']==focal_frame['level'],'Strict pre-A focal snapshot mismatch')
    self_board=[player for player in event['board'] if player.get('is_self')]
    require(len(event['board'])==10 and len(self_board)==1,'Incomplete source board')
    require(self_board[0]['participant_id']==pid and self_board[0]['champion_id']==pair['champion_id']
            and self_board[0]['team_id']==pair['team_id'],'Source focal identity differs from pair')
    require(physical_counts(self_board[0]['items'])==physical_counts(event['inventory_before']),'Board inventory is not the pre-A inventory')
    require(self_board[0]['gold']==focal_frame['totalGold'],'Pre-A board gold differs from matched focal frame')
    prefix=deepcopy(event);prefix.pop('gold_est',None);prefix.pop('gold_gap',None)
    frames={owner:[value for value in series if value[0]<ta] for owner,series in debug['frame_series'].items()}
    gold={owner:[value for value in series if value[0]<ta] for owner,series in debug['gold_events'].items()}
    attach_gold([prefix],frames,gold)
    require(prefix['gold_est']==event['gold_est'] and prefix['gold_gap']==event['gold_gap']
            and prefix['gold_est_version']==event['gold_est_version'],'Wallet estimator depends on future information')
    bound={**event,'match_id':pair['match_id'],'perks':players[pid]}
    row=projector(bound)
    require(row is not None and producer.counts_of(row)=={str(b):1},'Source projected target is not singleton B')
    prestate=producer.project_state(row)
    compatibility=producer.reference_compatibility(row,prestate)['source_projection']
    mutated=deepcopy(bound)
    mutated.update(label_bought=[{'item_id':a}],bought=[{'item_id':a}],inventory_after=[],
                   ts_end=tb+99999,gold=10**9,source_target_id=123456789,win='mutation-only')
    mutated['perks']={**mutated['perks'],'win':'mutation-only','kills':99999,'deaths':99999,'assists':99999}
    changed=projector(mutated)
    require(changed is not None,'Counterfactual label mutation produced no source projection')
    changed_state=producer.project_state(changed)
    require(changed_state==prestate and producer.reference_compatibility(changed,changed_state)['source_projection']==compatibility,
            'Reference context depends on final basket, postvisit state, or outcome fields')
    checks.update({'qualified_ts_equals_pre_a':True,'qualified_ts_end_equals_b':True,'opening_sales_absent':True,
                   'physical_preinventory_coherent':True,'singleton_b_retained':True,'source_mask_admitted':True,
                   'wallet_strict_prefix_reproduced':True,'label_future_mutation_invariant':True})
    return {'schema':SCHEMA,'match_id':pair['match_id'],'participant_id':pid,'source_pair_line':pair['source_line_number'],
            'game_candidate_index':pair['game_candidate_index'],'role':pair['role'],'patch':pair['patch'],
            'context_timestamp_ms':ta,'certificate_b_timestamp_ms':tb,'action_a':a,'action_b':b,
            'event_index':event['event_index'],'status':'aligned','checks':checks,
            'source_projection':compatibility,'iql_prestate':prestate,
            'projection_sha256':producer.value_hash(compatibility),'source_event_sha256':producer.value_hash(event),
            'reference_role_warning':'Legacy final-match roles retained for exact M2 reference compatibility; not IQL prestate features.',
            'wallet_claim':'Source causal estimate, not exact wallet; both-action feasibility at B remains a separate receipt certificate.'}


def run(args):
    started=time.monotonic();output=Path(args.output).resolve()
    require(not output.exists(),'Immutable alignment output already exists')
    manifest,tests,validation,confirm,bindings=producer.validate_admission(args.admission)
    denied=tests|validation|confirm
    pairs,pair_bindings=select_pairs(args.census,manifest,denied,args.limit)
    wanted={pair['match_id'] for pair in pairs}
    restricted={**manifest,'games':[row for row in manifest['games'] if row['match_id'] in wanted]}
    source=producer.source_metadata(producer.GENERATION,restricted,tests,validation,confirm,10)
    require(set(source['selected'])==wanted,'Pair subset does not equal admitted qualified source IDs')
    bindings.update(pair_bindings);bindings.update(source['bindings'])
    from footwear_runtime import activate_for_manifest
    active=activate_for_manifest(source['corpus'],corpus_path=source['corpus_path'])
    import baseline
    from app.reconstruct import reconstruct_game
    from app.ddragon import dragon_for_patch
    from process_reconstructible_1618 import decision_mask
    dragon=dragon_for_patch('16.18')
    attach_gold=reconstruct_game.__globals__['_attach_gold_estimates']
    code={str(Path(__file__).resolve()):producer.file_hash(__file__),str(Path(producer.__file__).resolve()):producer.file_hash(producer.__file__),
          str(ROOT/'scripts/baseline.py'):producer.file_hash(ROOT/'scripts/baseline.py')}
    output.mkdir(parents=True,exist_ok=False)
    def status(state,**extra):
        (output/'status.json').write_text(json.dumps({'state':state,'elapsed_s':time.monotonic()-started,**extra},indent=2)+'\n')
    records=[];contexts={}
    try:
        db_path=source['corpus_path']/'tracker.db'
        with closing(sqlite3.connect(db_path.as_uri()+'?mode=ro&immutable=1',uri=True,timeout=2)) as conn:
            conn.execute('PRAGMA query_only=ON');conn.execute('PRAGMA cache_size=-2048')
            reader=producer.ApprovedGameReader(conn,wanted,denied)
            for index,pair in enumerate(pairs,1):
                if time.monotonic()-started>150:raise RuntimeError('Alignment smoke deadline exceeded')
                mid=pair['match_id']
                if mid not in contexts:
                    game,events=reader.game(mid)
                    raw,raw_hashes=producer.read_raw_pair(mid,source['plans'][mid],wanted,denied)
                    debug={};replayed_game,replayed_events=reconstruct_game(raw['match'],raw['timeline'],dragon,debug)
                    producer.verify_replay(game,events,replayed_game,replayed_events)
                    fresh=decision_mask(replayed_events,debug,dragon);frozen=source['masks'][mid]
                    require(all(fresh[key]==frozen[key] for key in ('admitted_decision_ids','boundary_exclusions','admitted_rows','boundary_excluded_rows')),
                            'Qualified source decision mask does not reproduce')
                    players={player['participant_id']:player for player in producer.load_participants(game)}
                    contexts[mid]=(events,players,debug,raw['timeline'],raw_hashes)
                events,players,debug,timeline,raw_hashes=contexts[mid]
                try:
                    record=align_pair(pair,events,source['masks'][mid],players,lambda event:baseline._example(event,dragon),
                                      attach_gold,debug,timeline,dragon)
                    record['raw_sources']=raw_hashes
                except (ValueError,KeyError,TypeError) as error:
                    record={'schema':SCHEMA,'match_id':mid,'participant_id':pair['participant_id'],'source_pair_line':pair['source_line_number'],
                            'context_timestamp_ms':pair['feature_cutoff_ms'],'certificate_b_timestamp_ms':pair['decision_timestamp_ms'],
                            'status':'unaligned','reason':str(error),'error_type':type(error).__name__}
                records.append(record);status('aligning',processed_pairs=index,aligned=sum(row['status']=='aligned' for row in records))
        producer.validate_admission(args.admission)
        require(all(producer.file_hash(path)==expected for path,expected in {**bindings,**code}.items()),'Frozen alignment input/code changed')
        rows_path=output/'contexts.jsonl'
        with rows_path.open('x',encoding='utf-8',newline='\n') as handle:
            for record in records:handle.write(producer.canonical(record).decode()+'\n')
        digest=producer.file_hash(rows_path);rows_path.with_suffix('.sha256').write_text(digest+'  contexts.jsonl\n',encoding='ascii')
        report={'schema':SCHEMA,'state':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                'pairs_examined':len(pairs),'games_read':len(wanted),'status_counts':dict(Counter(row['status'] for row in records)),
                'unaligned_reasons':dict(Counter(row['reason'] for row in records if row['status']!='aligned')),
                'contexts_sha256':digest,'input_metadata_sha256':bindings,'code_sha256':code,'runtime':active,
                'selection':'first at most ten eligible16.18 FIT pairs in frozen census order; no role changes',
                'policy_loaded':False,'training_performed':False,'outcomes_analyzed':False,
                'source_replay_internally_reads_final_match_fields':True,
                'test_payload_reads':False,'confirmation_payload_reads':False,'elapsed_s':time.monotonic()-started,
                'scope':'Pre-A context alignment only; no preference accuracy or winner effect estimated.'}
        report_path=output/'report.json';report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        report_path.with_suffix('.sha256').write_text(producer.file_hash(report_path)+'  report.json\n',encoding='ascii')
        status('complete',**report['status_counts'])
        return report
    except Exception as error:
        status('failed',error_type=type(error).__name__,error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--census',type=Path,default=DEFAULT_CENSUS)
    parser.add_argument('--admission',type=Path,default=producer.ADMISSION)
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=run(args)
    print(json.dumps({key:report[key] for key in ('pairs_examined','games_read','status_counts','unaligned_reasons','contexts_sha256','elapsed_s')},indent=2))


if __name__=='__main__':main()
