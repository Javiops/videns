"""Join certified FIT prestates to source role receipts; separate target output.

No DB/raw reads, native model loading or consumer runtime imports. Selection
was frozen before this script's first actual purchase access.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import export_recipe_roles_v1 as scope
producer = scope.producer; require = scope.require; write = scope.write
ROLES = ROOT/'data/exports/winprob-recipe-transition/recipe-roles-20261001-v1/manifest.json'


def attach_reference_roles(prestate, role_receipt):
    """Preserve every original state field/order, adding ONLY legacy M2 roles."""
    roles = role_receipt['reference_roles']
    require(prestate['champion_id'] == roles['champion_id'] and prestate['team_id'] == roles['team_id'], 'focal role/prestate mismatch')
    observed = [(p['team_id'], p['champion_id']) for p in prestate['others']]
    lookup = {(p['team_id'], p['champion_id']): p['role'] for p in roles['others']}
    require(len(observed) == len(set(observed)) == len(lookup) == 9 and set(observed) == lookup.keys(), 'other-player roster mismatch')
    require(isinstance(roles['role'], str) and all(isinstance(v, str) for v in lookup.values()), 'invalid role values')
    result = deepcopy(prestate); result['role'] = roles['role']
    for p in result['others']:
        p['role'] = lookup[(p['team_id'], p['champion_id'])]
    return result


def residual_public(prestate):
    """New residual receives only the explicit seven-field public context."""
    fields = ('inventory', 'champion_id', 'team_id', 'level', 'ts', 'gold_est')
    result = {key: deepcopy(prestate[key]) for key in fields}
    result['others'] = [{'champion_id': p['champion_id'], 'team_id': p['team_id']} for p in prestate['others']]
    return result


def target_projection(action):
    """A retained observed multiset, never a hypothetical preference label."""
    keys = ('kind', 'encoding', 'label_counts', 'full_action_id', 'main_group_id', 'wards_preserved')
    require(all(k in action for k in keys) and action['encoding'] == 'item-counts-v1' and action['wards_preserved'] is True,
            'original full-action contract changed')
    require(type(action['full_action_id']) is int and 0 <= action['full_action_id'] < 13166
            and type(action['main_group_id']) is int and 0 <= action['main_group_id'] < 9333, 'invalid frozen action index')
    counts = producer.counts_of({'label_counts': action['label_counts']})
    if action['kind'] == 'observed_no_buy_death':
        require(counts == {'999999': 1} and action['full_action_id'] == action['main_group_id'] == 0, 'invalid observed no-buy')
    else:
        require(action['kind'] == 'retained_net_purchase' and '999999' not in counts, 'invalid observed purchase')
    return {**{key: action[key] for key in keys}, 'label_counts': counts}


def run(output):
    started = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'new immutable context release required')
    meta, sh = producer.load_bound(scope.SELECTION)
    require(sh == scope.SELECTION_SHA and producer.file_hash(scope.PROTOCOL) == scope.PROTOCOL_SHA, 'selection/protocol changed')
    bindings = {str(scope.SELECTION): sh, str(scope.PROTOCOL): scope.PROTOCOL_SHA}
    roles_meta, rh = producer.load_bound(ROLES); bindings[str(ROLES)] = rh
    require(roles_meta['schema'] == 'recipe-transition-reference-roles-v1' and roles_meta['state'] == 'complete'
            and roles_meta['selection_manifest_sha256'] == sh and roles_meta['states'] == 56288
            and roles_meta['whole_indexed_game_receipts_verified'] == 3518 and roles_meta['mutation_checked_games'] == 3518,
            'incomplete/mismatched source role release')
    for raw, want in {**roles_meta['bindings_sha256'], **roles_meta['code_sha256']}.items():
        require(producer.file_hash(raw) == want, 'role release lineage changed'); bindings[raw] = want
    selection_path = scope.SELECTION.parent/'selection.json'
    require(producer.file_hash(selection_path) == meta['files']['selection.json'], 'selection data changed')
    selection = producer.load_json(selection_path); entries = selection['entries']; n = len(entries)
    require(n == 56288, 'fixed selection size changed')
    selected = {(e['match_id'], e['participant_id'], e['source_event_index']): e for e in entries}
    ids = {e['match_id'] for e in entries}
    require(len(selected) == n and len(ids) == 3518, 'duplicate selection identity')
    base, bs = producer.load_bound(scope.BASE); require(bs == scope.BASE_SHA, 'base metadata changed')
    admission_path = Path(base['source_admission_path'])
    admitted, tests, validation, confirm, ab = producer.validate_admission(admission_path); bindings.update(ab)
    require(not ids & (tests | validation | confirm), 'selected game became reserved')
    source, srcsha = producer.load_bound(scope.SOURCE); bindings[str(scope.SOURCE)] = srcsha
    require(srcsha == roles_meta['source_manifest_sha256'] and set(source['eligible_game_ids']) == ids, 'source game population changed')
    path = scope.SOURCE.parent/source['rows_file']
    require(path.resolve() == (scope.SOURCE.parent/'trajectories.jsonl.gz').resolve()
            and producer.file_hash(path) == source['rows_sha256'] == '1b5a999c5ac41bb4e86088e1c6653763e4c847e725a8d76dd3f27b722aee9e48', 'certified trajectory shard changed')
    bindings[str(path)] = source['rows_sha256']
    roles_path = ROLES.parent/'roles.jsonl'
    require(producer.file_hash(roles_path) == roles_meta['files']['roles.jsonl'], 'role rows changed'); bindings[str(roles_path)] = roles_meta['files']['roles.jsonl']
    roles = [None]*n
    with roles_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); i = row['ordinal']; e = entries[i]
            require(roles[i] is None and all(row[k] == e[k] for k in ('ordinal', 'sample_id', 'match_id', 'participant_id', 'source_event_index', 'timestamp_ms')),
                    'role receipt selection mismatch')
            roles[i] = row
    require(all(r is not None for r in roles), 'role receipt missing')
    code = {str(p.resolve()): producer.file_hash(p) for p in (Path(__file__), Path(scope.__file__), Path(producer.__file__))}
    output.mkdir(parents=True)
    write(output/'selection_receipt.json', {'schema': 'recipe-context-selection-receipt-v1', 'states': n,
        'selection_manifest_sha256': sh, 'role_manifest_sha256': rh, 'before_trajectory_payload_reads': True})
    uncompressed = hashlib.sha256(); episodes = 0; seen = set(); mutated = 0
    contexts_path = output/'contexts.jsonl.gz'; targets_path = output/'targets.jsonl'
    with gzip.open(path, 'rb') as stream, contexts_path.open('xb') as destination, gzip.GzipFile(filename='', mode='wb', fileobj=destination, compresslevel=1, mtime=0) as contexts, targets_path.open('x', encoding='utf-8') as targets:
        for line in stream:
            require(time.monotonic()-started < 1800, 'fixed30minute context preparation bound reached')
            uncompressed.update(line); record = json.loads(line)
            mid = record['match_id']; pid = record['participant_id']
            require(mid in ids and record['patch'] == '16.17' and record['role'] == 'fit'
                    and record['whole_player_game_eligible'] is True, 'nonadmitted trajectory encountered')
            episodes += 1
            for local, node in enumerate(record['decisions']):
                entry = selected.get((mid, pid, node['event_index']))
                if entry is None:
                    continue
                i = entry['ordinal']; role = roles[i]
                require(i not in seen and entry['source_node_ordinal'] == local and entry['trajectory_id'] == record['trajectory_id']
                        and entry['timestamp_ms'] == node['timestamp_ms'] and role['source_event_sha256'] == node['source_event_sha256'], 'trajectory/role/selection identity differs')
                prestate = node['prestate']
                require(producer.project_state(prestate) == prestate, 'certified prestate whitelist changed')
                reference = attach_reference_roles(prestate, role); public = residual_public(prestate)
                poisoned = deepcopy(prestate); poisoned.update(win='poison', label_counts={'999999': 1}, future={'role': 'poison'}, role='poison', total_gold=1e9)
                for other in poisoned['others']:
                    other.update(role='poison', gold=1e9, future={'item': 99999})
                require(residual_public(poisoned) == public, 'forbidden fields changed new residual context'); mutated += 1
                context = {**entry, 'source_event_sha256': role['source_event_sha256'], 'source_projection': reference,
                           'residual_public': public, 'reference_roles_not_certified_predecision': True}
                contexts.write(producer.canonical(context)+b'\n')
                target = {'ordinal': i, 'sample_id': entry['sample_id'], 'match_id': mid, 'participant_id': pid,
                          'action': target_projection(node['action'])}
                targets.write(producer.canonical(target).decode()+'\n'); seen.add(i); roles[i] = None
            if episodes % 1000 == 0:
                producer.validate_admission(admission_path)
                print(json.dumps({'episodes': episodes, 'selected_states': len(seen), 'elapsed_s': time.monotonic()-started}), flush=True)
    require(episodes == source['eligible_player_games'] == 28821 and len(seen) == mutated == n
            and all(r is None for r in roles), 'incomplete source context coverage')
    require(uncompressed.hexdigest() == source['uncompressed_rows_sha256'], 'uncompressed trajectory identity changed')
    for raw, want in {**bindings, **code}.items():
        require(producer.file_hash(raw) == want, 'bound input/code changed during preparation')
    producer.validate_admission(admission_path)
    result = {'schema': 'recipe-transition-source-contexts-v1', 'state': 'complete', 'games': 3518, 'states': n,
        'selection_manifest_path': str(scope.SELECTION), 'selection_manifest_sha256': sh, 'role_manifest_path': str(ROLES),
        'role_manifest_sha256': rh, 'protocol_sha256': scope.PROTOCOL_SHA,
        'bindings_sha256': bindings, 'code_sha256': code,
        'files': {p.name: producer.file_hash(p) for p in (output/'selection_receipt.json', contexts_path, targets_path)},
        'source_episodes_streamed': episodes, 'uncompressed_trajectory_sha256': uncompressed.hexdigest(),
        'residual_mutation_checked_rows': mutated, 'raw_archive_reads': False, 'source_database_reads': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'consumer_runtime_imported': False,
        'training_performed': False, 'model_loaded': False, 'terminal_metadata_deserialized': True,
        'outcomes_used_or_analyzed': False, 'selected_purchase_targets_exported_separately': True,
        'elapsed_s': time.monotonic()-started,
        'limits': ['Legacy roles occur only in source_projection for frozen M2; absent from residual_public.',
                   'Original catalogue-supported complete-episode restriction is inherited.']}
    write(output/'manifest.json', result)
    print(json.dumps({'state': 'complete', 'games': 3518, 'states': n, 'elapsed_s': result['elapsed_s']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
