"""Independent semantic oracles enforce objective, combat and slot-state prefixes."""
from bisect import bisect_left, bisect_right
from copy import deepcopy
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from audit_patch_corpus import combat_context, quest_completion_times, slot_state_checks

events = [
    {'type':'BUILDING_KILL','timestamp':100,'teamId':100,'buildingType':'TOWER_BUILDING'},
    {'type':'BUILDING_KILL','timestamp':200,'teamId':100,'buildingType':'INHIBITOR_BUILDING'},
    {'type':'CHAMPION_KILL','timestamp':300,'killerId':1,'victimId':2,'assistingParticipantIds':[3]},
]
participants = [{'participantId':i,'teamId':100 if i == 2 else 200} for i in (1,2,3)]
times, states = combat_context({'info':{'frames':[{'events':events}]}}, participants)
assert states[-1][1]['200']['towers'] == 1
assert states[bisect_left(times,300)-1][0][1][0] == 0
assert states[bisect_right(times,300)-1][0][1][0] == 1
assert states[-1][0][2][1] == 1 and states[-1][0][3][2] == 1
assert states[0][0][1] == [0,0,0], 'later events must not mutate earlier snapshots'
quest = lambda ts: {'type':'ITEM_DESTROYED','itemId':1201,'participantId':8,'timestamp':ts}
assert quest_completion_times({'info':{'frames':[{'events':[quest(100),quest(200),quest(90)]}]}},1201) == {8:90}
assert quest_completion_times({'info':{'frames':[{'events':[quest(100),quest(200),quest(900)]}]}},1201) == {8:100}

def bot_marker(pid, ts):
    return {'type':'ITEM_DESTROYED','itemId':1202,'participantId':pid,'timestamp':ts}


def completion_times(raw):
    return quest_completion_times({'info':{'frames':[{'events':raw}]}},1202)


def visit(ts, complete=(), *, save=False):
    board = [{'participant_id':pid, 'slot_state_version':'role-slots-v1',
              'bot_quest_complete':pid in complete} for pid in range(1,11)]
    return {**board[7], 'ts':ts, 'is_save':save, 'board':board}


def failures(event, raw):
    return [(rule, detail['participant_id']) for valid, rule, detail
            in slot_state_checks(event, completion_times(raw)) if not valid]


# Raw ordering, later repetitions, wrong item IDs and purchase events cannot
# move the independent first-destruction timestamp or grant completion early.
raw = [bot_marker(8,200), bot_marker(8,90), bot_marker(8,100),
       bot_marker(3,150), quest(1), {**bot_marker(9,1),'type':'ITEM_PURCHASED'}]
assert completion_times(raw) == {8:90,3:150}
assert not failures(visit(151,{8,3}),raw)
assert not failures(visit(90),raw), 'a tied quest marker is not prior state'
assert failures(visit(90,{8}),raw) == [('shopper_bot_quest_causal',8),('board_bot_quest_causal',8)]
assert failures(visit(89,{8}),raw) == [('shopper_bot_quest_causal',8),('board_bot_quest_causal',8)]
assert failures(visit(91),raw) == [('shopper_bot_quest_causal',8),('board_bot_quest_causal',8)]
assert not failures(visit(101,{8}),raw), 'later repeated markers cannot erase completion'
assert not failures(visit(101,{8}),raw + [bot_marker(8,999),bot_marker(9,999)])

# A save's state is anchored at death, 15 seconds before its synthetic visit.
assert not failures(visit(15100,{8},save=True),raw)
assert failures(visit(15100,{8,3},save=True),raw) == [('board_bot_quest_causal',3)]
assert not failures(visit(15090,save=True),raw)
assert failures(visit(15090,{8},save=True),raw) == [('shopper_bot_quest_causal',8),('board_bot_quest_causal',8)]
assert not failures(visit(15091,{8},save=True),raw)

# Every shopper/board field is required, including participants with no marker.
# Mutate one field at a time so a valid self row cannot hide a broken shopper
# field, and a valid shopper cannot hide any board member's state.
valid = visit(151,{8,3})
for index in [None,*range(10)]:
    scope = 'shopper' if index is None else 'board'
    target = valid if index is None else valid['board'][index]
    pid = target['participant_id']
    expected = target['bot_quest_complete']
    for value in (None,0,1,'false',not expected):
        mutated = deepcopy(valid)
        state = mutated if index is None else mutated['board'][index]
        state['bot_quest_complete'] = value
        assert failures(mutated,raw) == [(scope + '_bot_quest_causal',pid)], (index,value)
    for field, rule in (('bot_quest_complete','_bot_quest_causal'),('slot_state_version','_slot_state_version')):
        mutated = deepcopy(valid)
        state = mutated if index is None else mutated['board'][index]
        del state[field]
        assert failures(mutated,raw) == [(scope + rule,pid)], (index,field)
    for version in (None,'role-slots-v0','',1):
        mutated = deepcopy(valid)
        state = mutated if index is None else mutated['board'][index]
        state['slot_state_version'] = version
        assert failures(mutated,raw) == [(scope + '_slot_state_version',pid)], (index,version)
assert not failures(visit(1),[]), 'literal False is valid when no prior quest exists'
print('ok independent corpus oracle: objectives, KDA, repeated quest markers, shopper/board slot versions, strict purchase/death anchors, and boolean state mutation rejection')
