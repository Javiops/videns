"""Compile source-owned episode nodes into explicit IQL transitions.

Source rewards live on arrival nodes. The terminal node has no action: its
outcome must be attached to the preceding transition, not silently discarded.
This module validates data/accounting only; it neither fits nor evaluates a
policy and does not turn observational sequences into identified interventions.
"""
from __future__ import annotations

import math

SCHEMA = 'winprob-retained-basket-trajectories-v1'
FORBIDDEN_STATE_KEYS = frozenset({
    'win', 'win_team', 'winner', 'outcome', 'game_duration', 'source_target_id',
    'label_id', 'label_ids', 'label_counts', 'decision', 'save_kind',
    'gold_arrival_true', 'role', 'team_position',
})


def assert_state_boundary(state):
    if not isinstance(state, dict):
        raise ValueError('pre-action state must be a mapping')
    def visit(value):
        if isinstance(value, dict):
            if set(value) & FORBIDDEN_STATE_KEYS:
                raise ValueError('target/future/role field entered state')
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite state value')
    visit(state)


def compile_transitions(record):
    if record.get('schema') != SCHEMA:
        raise ValueError('unknown trajectory schema')
    if record.get('whole_player_game_eligible') is not True:
        raise ValueError('ineligible player-game cannot be silently stitched')
    if record.get('ineligibility_reasons'):
        raise ValueError('eligibility disagrees with reasons')
    decisions = record.get('decisions')
    terminal = record.get('terminal')
    if not isinstance(decisions, list) or not decisions or not isinstance(terminal, dict):
        raise ValueError('complete decision chain and terminal required')
    if (terminal.get('kind') != 'terminal' or terminal.get('done') is not True
            or terminal.get('action') is not None or terminal.get('prestate') is not None
            or terminal.get('next_ref') is not None or type(terminal.get('reward')) is not int
            or terminal['reward'] not in (0, 1)):
        raise ValueError('invalid absorbing terminal')
    all_nodes = decisions + [terminal]
    ids = [node.get('node_id') for node in all_nodes]
    prefix = str(record['match_id']) + ':' + str(record['participant_id']) + ':'
    if len(set(ids)) != len(ids) or any(not isinstance(n, str) or not n.startswith(prefix) for n in ids):
        raise ValueError('duplicate or cross-player node identity')
    transitions = []
    for node, nxt in zip(decisions, all_nodes[1:]):
        if (node.get('kind') != 'decision' or node.get('done') is not False
                or node.get('reward') != 0 or node.get('next_ref') != nxt['node_id']):
            raise ValueError('missing, gapped or reward-contaminated decision chain')
        t, end = node.get('timestamp_ms'), nxt.get('timestamp_ms')
        if type(t) is not int or type(end) is not int or t >= end:
            raise ValueError('non-increasing transition time')
        if node.get('visit_end_ms', t) > end:
            raise ValueError('overlapping decision interval')
        assert_state_boundary(node['prestate'])
        action = node.get('action')
        if not isinstance(action, dict) or action.get('encoding') != 'item-counts-v1':
            raise ValueError('unknown retained-basket action')
        counts = action.get('label_counts')
        if (not isinstance(counts, dict) or not counts
                or any(not isinstance(k, str) or not k.isdecimal() or int(k) <= 0
                       or str(int(k)) != k or type(v) is not int or v <= 0 for k, v in counts.items())):
            raise ValueError('invalid purchase multiset')
        done = nxt['kind'] == 'terminal'
        transitions.append({
            'state_ref': node['node_id'], 'next_state_ref': None if done else nxt['node_id'],
            'action': action, 'reward': nxt['reward'], 'terminated': done,
            'bootstrap_multiplier': 0.0 if done else 1.0,
            'elapsed_ms': end-t,
        })
    if sum(row['reward'] for row in transitions) != terminal['reward']:
        raise ValueError('terminal return was not paid exactly once')
    if sum(row['terminated'] for row in transitions) != 1 or not transitions[-1]['terminated']:
        raise ValueError('invalid terminal transition placement')
    return transitions
