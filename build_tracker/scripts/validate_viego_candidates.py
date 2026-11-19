"""Validate bounded Viego-only quarantine candidates without publishing data.

Reuses the existing semantic evidence, then checks every intermediate inventory
and input-prefix invariance. Independent source-transaction validation is a
separate evidence gate; this script alone does not admit a game to training.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import canonical_sha, sha256, write_json
from app.ddragon import dragon_for_patch
from app.reconstruct import reconstruct_game
from app.riot import RawMatchArchive
from app.shop_econ import payload_ids
from audit_patch_corpus import terminal_differences, slot_state_checks, quest_completion_times
from qualify_patch_corpus import inventory_issues, inventory_shape
from baseline import _example, _unmodeled_regular_slots


INPUT_FIELDS = ('participant_id','ts','champion_id','team_position','inventory_before',
                'board','score','kills','deaths','assists','level','cs','gold_left',
                'gold_est','gold_est_version','gold_gap','slot_state_version','bot_quest_complete')


def input_state(event):
    return {key:event.get(key) for key in INPUT_FIELDS}


def validate(qualification, match_ids, out):
    if out.exists() or not 1 <= len(match_ids) <= 5:
        raise ValueError('Use a fresh output and at most five candidate games')
    q = json.loads(qualification.read_text(encoding='utf-8'))
    corpus = Path(q['corpus'])
    manifest_path = corpus/'corpus_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if sha256(manifest_path) != q['corpus_manifest_sha256'] or sha256(corpus/'tracker.db') != q['db_sha256']:
        raise ValueError('Qualified corpus binding changed')
    audit_path = Path(q['semantic_audit'])
    if sha256(audit_path) != q['semantic_audit_sha256']:
        raise ValueError('Semantic audit binding changed')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['passed'] or audit['hard_failures']:
        raise ValueError('Source semantic audit has not passed')
    for name,digest in manifest['reconstruction_code_sha256'].items():
        if sha256(ROOT/name) != digest:
            raise ValueError(f'Reconstruction code changed: {name}')
    dragon = dragon_for_patch(manifest['patch'])
    if dragon.signature != manifest['static_data_sha256']:
        raise ValueError('Static data changed')
    archive = RawMatchArchive(corpus/'raw')
    source_path = corpus/'source_inventory.jsonl'
    if sha256(source_path) != manifest['source_inventory_sha256']:
        raise ValueError('Frozen source inventory changed')
    sources = {r['match_id']:r for r in map(json.loads,source_path.read_text().splitlines())}
    reports = []
    with sqlite3.connect((corpus/'tracker.db').as_uri()+'?mode=ro&immutable=1',uri=True) as conn:
        for mid in match_ids:
            if set(q['quarantined_match_ids'][mid]) != {'unmodeled_viego_inventory'} or mid not in audit['accepted_match_ids']:
                raise ValueError(f'{mid}: candidate has additional semantic exclusions')
            raw = {}
            for kind,binding in sources[mid]['files'].items():
                raw[kind] = archive.read(kind,mid)
                if sha256(corpus/binding['path']) != binding['sha256'] or canonical_sha(raw[kind]) != binding['payload_sha256']:
                    raise ValueError(f'{mid}: raw source changed')
            match,timeline = raw['match'],raw['timeline']
            pid = next(p['participantId'] for p in match['info']['participants'] if p['championId'] == 234)
            debug = {}
            _game,events = reconstruct_game(match,timeline,dragon,debug)
            stored = [json.loads(r[0]) for r in conn.execute('SELECT payload_json FROM game_events WHERE match_id=? ORDER BY event_index',(mid,))]
            failures = Counter()
            if canonical_sha(events) != canonical_sha(stored): failures['stored_replay_mismatch'] += 1
            endpoints = terminal_differences(match,debug['final_inventories'],dragon)
            if endpoints: failures['modeled_terminal_inventory_mismatch'] += len(endpoints)
            missing_grants = []
            for p in match['info']['participants']:
                endpoint = [p.get(f'item{i}',0) for i in range(7)]
                if 2422 in endpoint and 2422 not in debug['final_inventories'][p['participantId']]:
                    missing_grants.append({'participant_id':p['participantId'],'item_id':2422,'is_viego':p['participantId']==pid})
            if missing_grants: failures['unobserved_granted_footwear'] += len(missing_grants)
            quests = quest_completion_times(timeline,1202)
            states_checked = 0
            row_counts = Counter()
            for e in events:
                for good,rule,_detail in slot_state_checks(e,quests):
                    if not good: failures[rule] += 1
                anchor = e['ts']-15000 if e.get('is_save') else e['ts']
                owner = next(m for m in e['board'] if m['participant_id']==e['participant_id'])
                states = [(m,m['items'],anchor,False) for m in e['board']]
                states += [(owner,e['inventory_before'],anchor,False),(owner,e['inventory_after'],e['ts_end'],True)]
                for member,items,when,inclusive in states:
                    states_checked += 1
                    quest = quests.get(member['participant_id'],float('inf'))
                    complete = quest <= when if inclusive else quest < when
                    failures.update(inventory_issues(items,member['team_position'],complete,dragon))
                    shape = inventory_shape(items,member['team_position'],complete,dragon)
                    if shape['regular_slots']+_unmodeled_regular_slots(items,dragon)>6:
                        failures['total_regular_slot_overflow'] += 1
                if Counter(payload_ids(e.get('label_bought',[])))-Counter(payload_ids(e['inventory_after'])):
                    failures['label_not_retained'] += 1
                if _example(e,dragon):
                    row_counts['all_players'] += 1
                    if e['participant_id']==pid: row_counts['viego'] += 1
            mutant = deepcopy(match)
            for p in mutant['info']['participants']:
                p.update({f'item{i}':3869 for i in range(7)})
                p.update(roleBoundItem=3175,kills=999,deaths=999,assists=999,win=not p['win'])
            if canonical_sha(reconstruct_game(mutant,timeline,dragon)[1]) != canonical_sha(events):
                failures['final_dto_input_leak'] += 1
            prefix_checks = 0
            prefix_failure_examples = []
            for e in events:
                if e.get('is_save'):
                    continue
                cutoff = e['ts']
                truncated = deepcopy(timeline)
                for frame in truncated['info']['frames']:
                    frame['events'] = [r for r in frame.get('events',[]) if r.get('timestamp',0)<=cutoff]
                    if frame['timestamp']>cutoff:
                        for stats in frame.get('participantFrames',{}).values():
                            stats.update(currentGold=999999,totalGold=999999,level=18,minionsKilled=999,jungleMinionsKilled=999)
                prefix = reconstruct_game(match,truncated,dragon)[1]
                target = next((r for r in prefix if r['participant_id']==e['participant_id'] and r['ts']==cutoff and not r.get('is_save')),None)
                prefix_checks += 1
                if target is None or canonical_sha(input_state(target)) != canonical_sha(input_state(e)):
                    # A visit starting with an unmodeled purchase may only be
                    # emitted after a later modeled purchase. Removing that
                    # label window is inconclusive, not proof of input leakage.
                    failures['prefix_check_inconclusive' if target is None else 'future_input_dependence'] += 1
                    changed = [key for key in INPUT_FIELDS if target is None or canonical_sha(target.get(key)) != canonical_sha(e.get(key))]
                    prefix_failure_examples.append({'participant_id':e['participant_id'],'ts':cutoff,'changed_fields':changed,
                                                    'before':{key:e.get(key) for key in changed if key != 'board'},
                                                    'truncated':{key:target.get(key) for key in changed if key != 'board'} if target else None})
            reports.append({'match_id':mid,'duration_s':match['info']['gameDuration'],'viego_participant_id':pid,
                            'source_replay_equal':canonical_sha(events)==canonical_sha(stored),
                            'shop_events':len(events),'model_rows':dict(row_counts),'inventory_states_checked':states_checked,
                            'prefix_input_checks':prefix_checks,'modeled_terminal_differences':endpoints,
                            'prefix_failure_examples':prefix_failure_examples,
                            'unobserved_granted_footwear':missing_grants,'failures':dict(failures),
                            'checks_passed':not failures,'duration_eligible':match['info']['gameDuration']>=600,
                            'raw_sources_sha256':{k:sources[mid]['files'][k]['sha256'] for k in ('match','timeline')}})
            print(json.dumps(reports[-1],indent=2),flush=True)
    report = {'schema':1,'purpose':'candidate_source_and_input_validation','created_utc':datetime.now(timezone.utc).isoformat(),
              'qualification_sha256':sha256(qualification),'semantic_audit_sha256':sha256(audit_path),
              'corpus_manifest_sha256':sha256(manifest_path),'validator_sha256':sha256(Path(__file__)),
              'candidates':reports,'training_admission_performed':False,'final_test_payload_opened':False,
              'limits':['Requires separate strict transaction evidence; does not remove the blanket Viego exclusion by itself.',
                        'Passing games establish a narrow source-valid subset, not a general possession repair.',
                        'Modeled terminal equality is supplemented by all intermediate slot checks and event-prefix input checks.']}
    write_json(out,report)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--match',dest='matches',action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    validate(a.qualification.resolve(),a.matches,a.out.resolve())
