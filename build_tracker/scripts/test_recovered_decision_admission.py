"""Recovery admission cannot broaden frozen decisions, folds or source eligibility."""
from copy import deepcopy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import baseline
import build_recovered_patch_corpus as builder
import corpus_rows
from app.dataset_artifacts import sha256, write_json
from app.dataset_generations import validate_recovered_evidence
from app.reconstruct import SLOT_STATE_VERSION


def refused(fn):
    try:
        fn()
    except (ValueError, KeyError, TypeError):
        return
    raise AssertionError('Invalid recovery admission was accepted')


event_a = {'event_index':0, 'participant_id':1, 'ts':10000, 'ts_end':10010, 'training_eligible':True}
event_b = {'event_index':1, 'participant_id':2, 'ts':20000, 'ts_end':20010, 'training_eligible':True}
recovered = {'candidate_decision_ids':[builder.decision_id(event_a)], 'candidate_rows':1}
with patch.object(baseline, '_example', return_value={'label_counts':{'1001':1}}), patch(
        'validate_footwear_timer_results.grant_boundary_reason', side_effect=lambda e,g,guard_ms: 'all_board_anchor_near_inferred_grant' if e['ts']==20000 else None):
    mask = builder.admission_for_game([event_a,event_b], {'footwear_grants':[]}, None, recovered)
    assert mask['admitted_rows']==1 and mask['boundary_excluded_rows']==1
    refused(lambda:builder.admission_for_game([event_a,event_b], {}, None, None))
    refused(lambda:builder.admission_for_game([event_a,event_b], {}, None,
        {'candidate_decision_ids':[builder.decision_id(event_b)], 'candidate_rows':1}))
    refused(lambda:builder.admission_for_game([{**event_a,'training_eligible':False}], {}, None, recovered))
refused(lambda:builder.decision_id({**event_a,'participant_id':True}))

admission = {'schema':1,'recovery_version':builder.RECOVERY_VERSION,'patch':'16.17',
    'timing_guard_ms':1000,'purpose':'patch_development','final_test_payload_opened':False,
    'games':{'KR_A':{**mask,'split':'train'}},'games_count':1,'admitted_rows':1,'boundary_excluded_rows':1}
admission['source_bindings'] = {k:{'path':'fixture/'+k,'sha256':'a'*64} for k in
    ('source_plan','prior_qualification','recovery_summary','recovery_source_report')}
corpus = {'recovery_version':builder.RECOVERY_VERSION,'patch':'16.17','decision_admission_sha256':'mask-sha','games':1}
audit = {'decision_admission_sha256':'mask-sha'}
qualification = {'recovery_version':builder.RECOVERY_VERSION,'semantic_audit_sha256':'audit-sha',
    'decision_admission_sha256':'mask-sha','accepted_match_ids':['KR_A'],
    'footwear_qualification':{'version':builder.RECOVERY_VERSION,'passed':True,
        'semantic_audit_sha256':'audit-sha','decision_admission_sha256':'mask-sha'}}
validate_recovered_evidence(qualification,audit,corpus,admission,'mask-sha')
for mutation in ('duplicate','overlap','missing_game','changed_count','test','prior_mask','no_audit','guard'):
    changed, q = deepcopy(admission), deepcopy(qualification)
    if mutation=='duplicate': changed['games']['KR_A']['admitted_decision_ids'] *= 2
    if mutation=='overlap': changed['games']['KR_A']['boundary_exclusions'][0].update(builder.decision_id(event_a))
    if mutation=='missing_game': q['accepted_match_ids'].append('KR_MISSING')
    if mutation=='changed_count': changed['admitted_rows']=2
    if mutation=='test': changed['games']['KR_A']['split']='test'
    if mutation=='prior_mask': changed['games']['KR_A']['source_cohort']='prior_qualified'
    if mutation=='no_audit': q.pop('footwear_qualification')
    if mutation=='guard': changed['timing_guard_ms']=0
    refused(lambda:validate_recovered_evidence(q,audit,corpus,changed,'mask-sha'))

# Exercise the actual streaming exporter: even a masked source-quality failure
# must fail, and a mask cannot hide an unexpected modeled decision.
with tempfile.TemporaryDirectory(prefix='recovered-mask-',dir=ROOT/'data') as temporary:
    folder = Path(temporary)
    participants = [{'puuid':f'p{i}','participant_id':i} for i in range(1,11)]
    conn = sqlite3.connect(folder/'tracker.db')
    conn.execute('CREATE TABLE games (match_id TEXT, game_duration INTEGER, participants_json TEXT)')
    conn.execute('CREATE TABLE game_events (match_id TEXT,event_index INTEGER,payload_json TEXT)')
    conn.execute('INSERT INTO games VALUES (?,?,?)',('KR_A',900,json.dumps(participants)))
    manifest = {'reconstruction_version':'fixture-replay','inventory_version':'fixture-inventory',
        'label_source_version':'purchase-net-v1','patch':'16.17'}
    events = []
    for index in range(11):
        pid = index + 1 if index < 10 else 1
        event = {**manifest,'event_index':index,'participant_id':pid,'puuid':f'p{pid}',
            'ts':(index+1)*10000,'gold_est_version':'prequential-v3','training_eligible':True,
            'slot_state_version':SLOT_STATE_VERSION,'bot_quest_complete':False,
            'ddragon_version':'16.17.1','static_data_sha256':'fixture-static',
            'board':[{'participant_id':i,'slot_state_version':SLOT_STATE_VERSION,'bot_quest_complete':False} for i in range(1,11)]}
        if index == 10:
            event['label_bought'] = [{'item_id':3006}]
        events.append(event)
        conn.execute('INSERT INTO game_events VALUES (?,?,?)',('KR_A',index,json.dumps(event)))
    conn.commit()
    conn.close()
    game_mask = {'admitted_decision_ids':[builder.decision_id(e) for e in events[:-1]],
                 'boundary_exclusions':[{**builder.decision_id(events[-1]),'reason':'near_grant'}]}
    context = SimpleNamespace(path=folder,qualification={'accepted_match_ids':['KR_A']},manifest=manifest,
        dragon=SimpleNamespace(version='16.17.1',signature='fixture-static',classify=lambda item:{'is_boots':item==3006,'skip':False}),
        decision_admission={'games':{'KR_A':game_mask}})
    def example(event,dragon):
        return {'participant_id':event['participant_id'],'ts':event['ts'],'label_ids':[1001],'gold_arrival_true':999}
    with patch.object(baseline,'_example',side_effect=example):
        result = list(corpus_rows.iter_games(context))
        assert len(result)==1 and len(result[0][1])==10 and result[0][2] is None
        assert result[0][0]['boundary_excluded_rows']==1
        assert result[0][1][0]['source_target_id']==3006, 'Masked completion must remain the true earlier plan label'
        assert all(row['source_target_id']==0 for row in result[0][1][1:])
        assert all('gold_arrival_true' not in row for row in result[0][1])
        game_mask['boundary_exclusions']=[]
        refused(lambda:list(corpus_rows.iter_games(context)))
        game_mask['boundary_exclusions']=[{**builder.decision_id(events[-1]),'reason':'near_grant'}]
        events[-1]['training_eligible']=False
        with closing(sqlite3.connect(folder/'tracker.db')) as conn:
            conn.execute('UPDATE game_events SET payload_json=? WHERE event_index=10',(json.dumps(events[-1]),))
            conn.commit()
        refused(lambda:list(corpus_rows.iter_games(context)))

print('ok recovered admission: exact frozen IDs, audit binding, mask isolation, original source gate, test exclusion')
