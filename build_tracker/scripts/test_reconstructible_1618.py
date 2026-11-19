"""Bounded real-source smoke plus patch isolation and bootstrap label safeguards."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from app.dataset_artifacts import sha256
import app.reconstruct as production
from reconstructible_1618_runtime import activate,SOURCES,declaration

BASE=ROOT/'data/acceptance/16.18_reconstructible_20260912'
original=production.RECONSTRUCTION_VERSION
before={name:sha256(ROOT/name) for name in SOURCES.values()}
activate(BASE/'inputs/static_review.json')
from app.ddragon import dragon_for_patch
from app.reconstruct import reconstruct_game
from app.riot import RawMatchArchive
from footwear_economics import combine_cost,is_blocked
from process_reconstructible_1618 import prefix_check,decision_mask

dragon=dragon_for_patch('16.18')
assert dragon.version=='16.18.1'
assert combine_cost(3020,Counter({2422:1}),dragon)==800
assert dragon.classify(2422)['target_skip'] and not dragon.classify(2422)['skip']
assert is_blocked(1001,[],dragon,magical_footwear_selected=True,footwear_granted=False)
try: dragon_for_patch('16.17')
except ValueError: pass
else: raise AssertionError('Cross-patch runtime accepted')
try: reconstruct_game({'info':{'gameVersion':'16.17.1.1'}},{'info':{'frames':[]}},dragon)
except ValueError: pass
else: raise AssertionError('Cross-patch source accepted')
assert {name:sha256(ROOT/name) for name in SOURCES.values()}==before
child=subprocess.run([sys.executable,'-c',"import app.reconstruct; print(app.reconstruct.RECONSTRUCTION_VERSION)"],
    cwd=ROOT,capture_output=True,text=True,timeout=20)
assert child.returncode==0 and child.stdout.strip()==original,'Production process was changed'

plan=json.loads((BASE/'source_plan.json').read_text(encoding='utf-8'))
records={r['match_id']:r for r in plan['matches']}
archive=RawMatchArchive(ROOT/'data/raw_match_v5')
tested=[]
for mid in list(records)[:6]:
    assert mid not in set(plan['reserved_test_ids'])
    match=archive.read('match',mid)
    if match['info']['gameDuration']<600: continue
    timeline=archive.read('timeline',mid);debug={}
    _,events=reconstruct_game(match,timeline,dragon,debug)
    assert events and prefix_check(match,timeline,events,dragon)['passed']
    assert prefix_check(match,timeline,json.loads(json.dumps(events)),dragon)['passed']
    if any(e.get('training_eligible') is not True for e in events):
        try: decision_mask(events,debug,dragon)
        except ValueError: continue
        raise AssertionError('A decision mask erased a source-quality exclusion')
    assert all(e['reconstruction_version']==declaration()['reconstruction_version'] for e in events)
    assert all(type(e.get('footwear_granted')) is bool for e in events)
    mutated=deepcopy(match)
    for participant in mutated['info']['participants']:
        participant.update(kills=999,deaths=999,assists=999,roleBoundItem=3175)
        for slot in range(7): participant['item'+str(slot)]=3869
        for style in participant.get('perks',{}).get('styles',[]):
            for rune in style.get('selections',[]): rune.update(var1=999,var2=999,var3=999)
    assert reconstruct_game(mutated,timeline,dragon)[1]==events
    mask=decision_mask(events,debug,dragon)
    assert mask['admitted_rows']>0
    tested.append(mid)
    if len(tested)==2: break
assert len(tested)==2
print('ok16.18 runtime: pinned economics, physical2422, pregrant block, process isolation, two real-source prefixes and outcome invariance',tested)
