"""Read a versioned full-basket catalogue without importing source inference."""
from __future__ import annotations

import json
from pathlib import Path

from winprob.preference_reference import M2_IDENTITY, file_hash
from winprob.trajectory_contract import compile_transitions

WARDS = {2055, 772043}
SAVE = 999999


def counts_key(counts):
    if (not isinstance(counts, dict) or not counts
            or any(not isinstance(k, str) or not k.isdecimal() or str(int(k)) != k
                   or int(k) <= 0 or type(v) is not int or v <= 0 for k, v in counts.items())):
        raise ValueError('positive canonical retained item counts required')
    if str(SAVE) in counts:
        if counts != {str(SAVE): 1}:
            raise ValueError('observed no-buy sentinel cannot mix with a purchase')
        return ('observed_no_buy_death', ())
    return ('retained_net_purchase', tuple(sorted((int(k), v) for k, v in counts.items())))


def validate_catalogue(payload):
    if (payload.get('schema') != 'winprob-full-action-catalogue-v1'
            or payload.get('package_identity_sha256') != M2_IDENTITY
            or payload.get('full_action_count') != 13166 or payload.get('group_count') != 9333
            or payload.get('save_action_id') != 0 or payload.get('source_save_sentinel') != SAVE
            or payload.get('source_save_label_counts') != {str(SAVE): 1}
            or any(payload.get(k) is not False for k in ('game_payload_reads', 'policy_loaded', 'training_performed'))):
        raise ValueError('wrong full-action source contract')
    full, kinds, mapping, groups = (payload[k] for k in
                                    ('full_actions', 'full_action_kinds', 'full_to_group', 'group_main'))
    if not len(full) == len(kinds) == len(mapping) == 13166 or len(groups) != 9333 or groups[0] != []:
        raise ValueError('catalogue dimensions differ from frozen M2')
    lookup = {}
    for action_id, (basket, kind, group) in enumerate(zip(full, kinds, mapping, strict=True)):
        if type(group) is not int or not 0 <= group < len(groups):
            raise ValueError('full action has no canonical main group')
        if action_id == 0:
            if basket != [] or kind != 'observed_no_buy_death' or group != 0:
                raise ValueError('no-buy identity collapsed into ward-only action')
            key = counts_key({str(SAVE): 1})
        else:
            if kind != 'retained_net_purchase' or not basket:
                raise ValueError('non-save action must retain a nonempty purchase')
            counts = {}
            for entry in basket:
                if (not isinstance(entry, list) or len(entry) != 2 or type(entry[0]) is not int
                        or type(entry[1]) is not int or entry[0] <= 0 or entry[1] <= 0
                        or str(entry[0]) in counts or entry[0] == SAVE):
                    raise ValueError('invalid full-action multiset')
                counts[str(entry[0])] = entry[1]
            key = counts_key(counts)
            if list(map(list, key[1])) != basket:
                raise ValueError('full action is not in canonical sorted order')
        if key in lookup:
            raise ValueError('duplicate full action')
        lookup[key] = action_id
        projected = [entry for entry in basket if entry[0] not in WARDS]
        if projected != groups[group]:
            raise ValueError('ward-stripped projection differs from bound M2 mapping')
    if set(mapping) != set(range(9333)):
        raise ValueError('some native M2 groups lack full-action support')
    return lookup


def load_catalogue(path):
    path = Path(path).resolve(strict=True)
    if path.with_suffix('.sha256').read_text(encoding='ascii').split() != [file_hash(path), path.name]:
        raise ValueError('full-action catalogue digest mismatch')
    payload = json.loads(path.read_text(encoding='utf-8'))
    for key in ('source_files_sha256', 'static_files_sha256', 'code_sha256'):
        for source, expected in payload[key].items():
            if file_hash(source) != expected:
                raise ValueError('bound catalogue source changed')
    return payload, validate_catalogue(payload)


def encode_complete_episode(record, lookup):
    """Do not bridge over an unknown action: the entire player-game is excluded."""
    transitions = compile_transitions(record)
    keys = [counts_key(t['action']['label_counts']) for t in transitions]
    missing = sorted(set(key for key in keys if key not in lookup))
    if missing:
        return {'eligible': False, 'reason': 'full_action_outside_frozen_catalogue',
                'missing_actions': missing, 'original_transitions': len(transitions)}
    return {'eligible': True, 'transitions': [{**t, 'full_action_id': lookup[key]}
                                             for t, key in zip(transitions, keys, strict=True)]}
