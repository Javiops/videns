"""Bind independent source and board checks for the reviewed Viego candidate."""
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256, write_json
from app.ddragon import dragon_for_patch
from app.shop_econ import payload_ids


def main():
    folder=ROOT/'data/acceptance/viego_recovery_20260911'
    destination=folder/'recovered_games.json'
    if destination.exists(): raise ValueError('Recovery evidence exists')
    evidence_names=('raw_scan.json','easy_candidates.json','candidate_validation_final.json',
                    'strict_transactions.json','strict_transactions_validation.json')
    evidence={name:json.loads((folder/name).read_text(encoding='utf-8')) for name in evidence_names}
    q_path=ROOT/'data/acceptance/16.18_role_slots_v1/qualification.json'
    q=json.loads(q_path.read_text(encoding='utf-8'))
    validation=evidence['candidate_validation_final.json']
    strict=evidence['strict_transactions.json']
    if validation['qualification_sha256']!=sha256(q_path): raise ValueError('Candidate evidence binding changed')
    for script,digest in (('scripts/validate_viego_candidates.py',validation['validator_sha256']),
                          ('scripts/audit_viego_transactions.py',strict['auditor_sha256'])):
        if sha256(ROOT/script)!=digest: raise ValueError('Validator changed')
    mid='KR_8375431152'
    candidate=next(r for r in validation['candidates'] if r['match_id']==mid)
    reference=next(r for r in strict['games'] if r['match_id']==mid)
    if not candidate['checks_passed'] or not candidate['duration_eligible'] or not reference['passed']:
        raise ValueError('Candidate gates did not pass')
    for kind in ('match','timeline'):
        binding=reference['sources'][kind]
        if candidate['raw_sources_sha256'][kind]!=binding['sha256'] or sha256(Path(binding['path']))!=binding['sha256']:
            raise ValueError('Independent auditors used different source bytes')
    dragon=dragon_for_patch('16.18')
    if dragon.signature!=strict['static_data_signature']: raise ValueError('Static data changed')
    times=[r['timestamp'] for r in reference['ledger']]
    def expected(when,inclusive):
        at=(bisect_right(times,when) if inclusive else bisect_left(times,when))-1
        items=reference['ledger'][at]['after_all_observed_items'] if at>=0 else []
        return Counter(i for i in items if not dragon.classify(i)['skip'])
    def modeled(items):
        return Counter(i for i in payload_ids(items) if not dragon.classify(i)['skip'])
    crosschecks=Counter()
    with sqlite3.connect((Path(q['corpus'])/'tracker.db').as_uri()+'?mode=ro&immutable=1',uri=True) as conn:
        for (payload,) in conn.execute('SELECT payload_json FROM game_events WHERE match_id=? ORDER BY event_index',(mid,)):
            row=json.loads(payload)
            anchor=row['ts']-15000 if row.get('is_save') else row['ts']
            board=next(p for p in row['board'] if p['participant_id']==reference['participant_id'])
            if modeled(board['items'])!=expected(anchor,False):
                raise ValueError(f'Independent permanent ledger disagrees with board at {anchor}')
            crosschecks['all_player_boards']+=1
            if row['participant_id']==reference['participant_id']:
                if modeled(row['inventory_before'])!=expected(anchor,False) or modeled(row['inventory_after'])!=expected(row['ts_end'],True):
                    raise ValueError(f'Independent permanent ledger disagrees with Viego shop at {anchor}')
                crosschecks['viego_shop_before_and_after']+=1
    report={'schema':1,'status':'one_validated_game_identified','created_utc':datetime.now(timezone.utc).isoformat(),
            'purpose':'quarantined_game_recovery_evidence','population':'125 quarantined Viego source games; not a model evaluation',
            'raw_games_screened':evidence['raw_scan.json']['games_scanned'],
            'fully_validated_match_ids':[mid],'validated_game':candidate,
            'independent_ownership_crosschecks':dict(crosschecks),
            'fully_explained_viego_transitions':reference['audited_transition_count'],
            'evidence':{name:{'path':str(folder/name),'sha256':sha256(folder/name)} for name in evidence_names},
            'report_builder_sha256':sha256(Path(__file__)),
            'qualification_sha256':sha256(q_path),'source_corpus':q['corpus'],
            'training_admission_performed':False,'export_modified':False,'final_test_payload_opened':False,
            'limits':['This game needs no possession-event repair under the current modeled ownership schema.',
                      'This does not demonstrate that Viego never possessed a champion, or solve corrupted possession histories.',
                      'The other 124 games are not certified by this report; repair-dependent candidates remain unreviewed.',
                      'A new qualification/export generation must explicitly admit the reviewed game before training.',
                      'Early candidate_validation and candidate_validation_v2 reports mislabeled missing truncated label windows as input dependence; candidate_validation_final records them as inconclusive. The rejected game also has unobserved Magical Footwear.']}
    write_json(destination,report)
    print(json.dumps({k:report[k] for k in ('status','fully_validated_match_ids','independent_ownership_crosschecks','fully_explained_viego_transitions')},indent=2))


if __name__=='__main__': main()
