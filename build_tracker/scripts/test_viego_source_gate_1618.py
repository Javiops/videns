"""Known Viego source controls, physical boots, and future-output independence."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from reconstructible_1618_runtime import activate
BASE=ROOT/'data/acceptance/16.18_reconstructible_20260912'
activate(BASE/'inputs/static_review.json')
from app.ddragon import dragon_for_patch
from app.reconstruct import reconstruct_game
from app.riot import RawMatchArchive
from validate_footwear_timer_results import validate_timer_result
from viego_source_gate_1618 import audit_viego_source,recipe_quote

dragon=dragon_for_patch('16.18')
plan=json.loads((BASE/'source_plan.json').read_text(encoding='utf-8'))
allowed={r['match_id'] for r in plan['matches']}
reserved=set(plan['reserved_test_ids'])
archive=RawMatchArchive(ROOT/'data/raw_match_v5')

assert recipe_quote(3020,Counter({2422:1}),dragon)==(800,Counter({2422:1}))
positive=None
for mid,expected in [('KR_8375431152',True),('EUW1_7978999931',False)]:
    assert mid in allowed and mid not in reserved
    match=archive.read('match',mid);timeline=archive.read('timeline',mid);debug={}
    _,events=reconstruct_game(match,timeline,dragon,debug)
    footwear=validate_timer_result(match,timeline,events,debug,dragon)
    result=audit_viego_source(match,timeline,dragon,replay_events=events,debug=debug,footwear_qa=footwear)
    assert result['passed'] is expected, (mid,result['failures'][:8])
    if expected:
        assert result['ownership_crosschecks']>0
        positive=(match,timeline,events,debug,footwear,result)
    else:
        assert any(f['code'] in ('unexplained_nonshop_destruction','destroy_of_unowned_item') for f in result['failures'])

match,timeline,events,debug,footwear,original=positive
mutated=deepcopy(match)
actor=next(p for p in mutated['info']['participants'] if p['championId']==234)
for slot in range(7): actor['item'+str(slot)]=0
changed=audit_viego_source(mutated,timeline,dragon,replay_events=events,debug=debug,footwear_qa=footwear)
assert changed['ledger']==original['ledger'] and not changed['passed']
bad_board=deepcopy(events)
owner=original['participant_id']
member=next(p for p in bad_board[0]['board'] if p['participant_id']==owner)
member['items'] += [{'item_id':1001},{'item_id':1001}]
assert not audit_viego_source(match,timeline,dragon,replay_events=bad_board,debug=debug,footwear_qa=footwear)['passed']
backwards=deepcopy(timeline)
backwards['info']['frames'][-1]['participantFrames'][str(owner)]['level']=0
assert any(f['code']=='nonmonotone_owner_progression' for f in audit_viego_source(
    match,backwards,dragon,replay_events=events,debug=debug,footwear_qa=footwear)['failures'])
print('ok strict16.18 Viego: known positive and possession negative, all-board source ownership, physical2422 recipe, finalDTO label-only, progression contradiction')
