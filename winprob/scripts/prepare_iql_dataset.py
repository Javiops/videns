"""Two-pass, bounded-memory compilation of source-owned FIT trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from winprob import admission
from winprob.iql_features import PublicFeatureCompiler
from winprob.iql_dataset_contract import validate_sources, iter_episodes
from winprob.preference_reference import file_hash
from boots_support import load_static
from setback_experiment import write_json, set_below_normal_priority

PATCHES = ('16.17', '16.18', '16.19')
STATE_ARRAYS = {
    'champion_indices': ('int32', (10,)), 'inventory_static': ('float32', (10, 22)),
    'visible_level': ('float32', (10, 1)), 'global_public': ('float32', (37,)),
    'game_index': ('int32', ()), 'episode_index': ('int32', ()), 'full_action_id': ('int32', ()),
    'main_group_id': ('int32', ()), 'patch_index': ('uint8', ()), 'fold': ('uint8', ()),
    'next_state_index': ('int64', ()), 'reward': ('float32', ()), 'terminated': ('bool', ()),
    'timestamp_ms': ('int64', ()), 'elapsed_ms': ('int64', ()),
}


def game_fold(match_id):
    return int.from_bytes(hashlib.sha256(('iql-v1-fold:'+match_id).encode()).digest()[:8], 'big') % 2


def budget(started, deadline):
    if time.monotonic()-started > deadline:
        raise RuntimeError('IQL cache deadline reached')
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (key, ctypes.c_ulonglong) for key in ('total', 'available', 'page_total', 'page_available', 'virtual_total', 'virtual_available', 'extended')]
        state = Memory(); state.length = ctypes.sizeof(state)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)) or state.available < .75*1024**3:
            raise RuntimeError('IQL cache memory guard: less than0.75GiB available')


def action_tables(compiler, catalogue):
    k = len(catalogue['full_actions'])
    features = np.zeros((3, k, 23), dtype=np.float32)
    valid = np.zeros((3, k), dtype=bool)
    offsets, items, counts = [0], [], []
    for action, (basket, kind) in enumerate(zip(catalogue['full_actions'], catalogue['full_action_kinds'], strict=True)):
        bag, _ = compiler.full_action(basket, kind, patch='16.18')
        items.extend(item for item, count in bag); counts.extend(count for item, count in bag)
        offsets.append(len(items))
        for pi, patch in enumerate(PATCHES):
            try:
                _, vector = compiler.full_action(basket, kind, patch=patch)
            except (KeyError, ValueError):
                continue
            features[pi, action] = vector; valid[pi, action] = True
    return {'action_static': features, 'action_static_valid': valid,
            'action_offsets': np.asarray(offsets, dtype=np.int64),
            'action_item_indices': np.asarray(items, dtype=np.int32),
            'action_item_counts': np.asarray(counts, dtype=np.int32),
            'full_to_group': np.asarray(catalogue['full_to_group'], dtype=np.int32)}


def encode_episode(record, transitions, compiler, tables):
    patch_index = PATCHES.index(record['patch'])
    actions = [node['action']['full_action_id'] for node in record['decisions']]
    if len(actions) != len(transitions) or not np.all(tables['action_static_valid'][patch_index, actions]):
        raise ValueError('observed full action lacks native-patch public attributes')
    states = [compiler.state(node['prestate'], record['patch']) for node in record['decisions']]
    return states, actions, patch_index


def population_gate(game_metadata, episodes, state_count, source_complete_player_games,
                    action_counts, fold_transition_counts, fold_terminal_counts):
    denominator = sum(source_complete_player_games.values())
    coverage = len(episodes)/denominator if denominator else 0.
    folds = {}
    for fold in (0, 1):
        n = int(fold_transition_counts[fold])
        supported = int(action_counts[fold][action_counts[1-fold] > 0].sum())
        folds[str(fold)] = {'games': sum(row['fold'] == fold for row in game_metadata),
                            'transitions': n, 'terminal_classes': fold_terminal_counts[fold].tolist(),
                            'opposite_fold_supported_transitions': supported,
                            'opposite_fold_action_coverage': supported/n if n else 0.}
    checks = {'minimum_games': len(game_metadata) >= 1000, 'minimum_episodes': len(episodes) >= 5000,
              'minimum_transitions': state_count >= 100000, 'source_complete_chain_survival': coverage >= .7,
              'fold_game_minimum': all(row['games'] >= 450 for row in folds.values()),
              'fold_transition_minimum': all(row['transitions'] >= 40000 for row in folds.values()),
              'opposite_fold_action_coverage': all(row['opposite_fold_action_coverage'] >= .7 for row in folds.values()),
              'both_terminal_classes_per_fold': bool(np.all(fold_terminal_counts > 0))}
    return {'passed': all(checks.values()), 'checks': checks, 'eligible_games': len(game_metadata),
            'eligible_player_games': len(episodes), 'transitions': state_count,
            'otherwise_source_complete_player_games': denominator, 'complete_chain_survival': coverage,
            'folds': folds}


def run(args):
    started = time.monotonic(); set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(stage, **extra):
        write_json(args.output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra})
    try:
        status('validating_bound_source_artifacts')
        budget(started, args.deadline_seconds)
        bundle = validate_sources(args.source, args.admission, args.catalogue, smoke=args.smoke)
        source_validation_failures = sum(source['manifest'].get('exclusion_reason_counts', {}).get(
            'source_chain_validation_failed', 0) for source in bundle['sources'])
        if source_validation_failures:
            raise ValueError('source-chain validation failures require review before changing the coverage denominator')
        static, bindings = load_static(args.static_export)
        bindings.update(bundle['bindings'])
        for path in (args.admission, args.admission.with_suffix('.sha256'), args.catalogue,
                     Path(__file__), ROOT/'winprob/iql_features.py', ROOT/'winprob/iql_dataset_contract.py',
                     ROOT/'winprob/full_action_catalogue.py', ROOT/'winprob/trajectory_contract.py',
                     ROOT/'winprob/ipo_residual.py', ROOT/'winprob/preference_reference.py', ROOT/'winprob/admission.py',
                     ROOT/'scripts/boots_support.py', ROOT/'docs/iql_experiment_protocol_v1.md'):
            bindings[str(path.resolve())] = file_hash(path)
        compiler = PublicFeatureCompiler(static)
        tables = action_tables(compiler, bundle['catalogue'])
        episodes, selected, games, game_metadata = [], {}, {}, []
        states_total = tokens_total = max_bag = 0
        encoding_exclusions = Counter(); fold_transition_counts = np.zeros(2, dtype=np.int64)
        fold_terminal_counts = np.zeros((2, 2), dtype=np.int64)
        action_counts = np.zeros((2, 13166), dtype=np.int64)
        with (args.output/'encoding_exclusions.jsonl').open('x', encoding='utf-8') as excluded:
            for ordinal, (source_index, record, transitions) in enumerate(iter_episodes(bundle), 1):
                if ordinal % 250 == 1:
                    budget(started, args.deadline_seconds)
                    status('counting_encodable_complete_chains', episodes=ordinal-1, transitions=states_total)
                try:
                    features, action_ids, pi = encode_episode(record, transitions, compiler, tables)
                except (ValueError, KeyError, TypeError, IndexError) as error:
                    reason = str(error); encoding_exclusions[reason] += 1
                    excluded.write(json.dumps({'trajectory_id': record['trajectory_id'], 'match_id': record['match_id'],
                                               'patch': record['patch'], 'transitions': len(transitions), 'reason': reason})+'\n')
                    continue
                mid, trajectory_id = record['match_id'], record['trajectory_id']
                if trajectory_id in selected:
                    raise ValueError('duplicate compiled trajectory')
                if mid not in games:
                    games[mid] = len(game_metadata)
                    game_metadata.append({'match_id': mid, 'patch': record['patch'], 'fold': game_fold(mid)})
                fold = game_fold(mid); n = len(transitions)
                episode = {'trajectory_id': trajectory_id, 'match_id': mid, 'participant_id': record['participant_id'],
                           'game_index': games[mid], 'patch': record['patch'], 'fold': fold, 'source_index': source_index,
                           'first_state': states_total, 'states': n}
                selected[trajectory_id] = len(episodes); episodes.append(episode)
                states_total += n; fold_transition_counts[fold] += n
                fold_terminal_counts[fold, int(transitions[-1]['reward'])] += 1
                action_counts[fold] += np.bincount(action_ids, minlength=13166)
                for feature in features:
                    for bag in feature['inventory_bags']:
                        tokens_total += len(bag); max_bag = max(max_bag, len(bag))
        if bundle.get('verified_complete') is not True:
            raise ValueError('first source pass did not finish verification')
        gates = population_gate(game_metadata, episodes, states_total, bundle['source_complete_player_games'],
                                action_counts, fold_transition_counts, fold_terminal_counts)
        write_json(args.output/'population_gate.json', gates)
        if not args.smoke and not gates['passed']:
            raise ValueError('fixed IQL population/support gate failed; no tensor cache authorized')
        if not states_total:
            raise ValueError('no encodable complete trajectories')
        write_json(args.output/'feature_metadata.json', compiler.metadata())
        write_json(args.output/'episodes.json', episodes)
        arrays, specs = {}, {}
        for name, (dtype, tail) in STATE_ARRAYS.items():
            specs[name] = {'file': name+'.npy', 'dtype': dtype, 'shape': [states_total, *tail]}
        specs.update({'inventory_offsets': {'file': 'inventory_offsets.npy', 'dtype': 'int64', 'shape': [states_total*10+1]},
                      'inventory_item_indices': {'file': 'inventory_item_indices.npy', 'dtype': 'int32', 'shape': [tokens_total]},
                      'inventory_item_counts': {'file': 'inventory_item_counts.npy', 'dtype': 'int32', 'shape': [tokens_total]}})
        for name, spec in specs.items():
            arrays[name] = np.lib.format.open_memmap(args.output/spec['file'], mode='w+', dtype=spec['dtype'], shape=tuple(spec['shape']))
        written_states = token_cursor = completed_episodes = 0
        for source_index, record, transitions in iter_episodes(bundle):
            if record['trajectory_id'] not in selected:
                continue
            ei = selected[record['trajectory_id']]; episode = episodes[ei]
            if ei != completed_episodes or episode['first_state'] != written_states or episode['source_index'] != source_index:
                raise ValueError('source episode order changed between passes')
            features, actions, pi = encode_episode(record, transitions, compiler, tables)
            for feature, action, transition, node in zip(features, actions, transitions, record['decisions'], strict=True):
                row = written_states
                for name in ('champion_indices', 'inventory_static', 'visible_level', 'global_public'):
                    arrays[name][row] = feature[name]
                for player, bag in enumerate(feature['inventory_bags']):
                    bag_index = row*10+player; arrays['inventory_offsets'][bag_index] = token_cursor
                    for item, count in bag:
                        if count > np.iinfo(np.int32).max:
                            raise ValueError('inventory multiplicity exceeds exact cache integer range')
                        arrays['inventory_item_indices'][token_cursor] = item
                        arrays['inventory_item_counts'][token_cursor] = count
                        token_cursor += 1
                    arrays['inventory_offsets'][bag_index+1] = token_cursor
                values = {'game_index': episode['game_index'], 'episode_index': ei, 'patch_index': pi,
                          'fold': episode['fold'], 'full_action_id': action,
                          'main_group_id': tables['full_to_group'][action], 'reward': transition['reward'],
                          'terminated': transition['terminated'], 'timestamp_ms': node['timestamp_ms'],
                          'elapsed_ms': transition['elapsed_ms'],
                          'next_state_index': -1 if transition['terminated'] else row+1}
                for name, value in values.items():
                    arrays[name][row] = value
                written_states += 1
            if written_states != episode['first_state']+episode['states']:
                raise ValueError('episode length changed between passes')
            completed_episodes += 1
            if completed_episodes % 250 == 0:
                budget(started, args.deadline_seconds)
                status('writing_memory_mapped_tensors', episodes=completed_episodes, transitions=written_states)
        if (bundle.get('verified_complete') is not True or written_states != states_total or token_cursor != tokens_total
                or completed_episodes != len(episodes) or int(arrays['terminated'].sum()) != len(episodes)):
            raise ValueError('compiled tensor/terminal accounting mismatch')
        for value in arrays.values():
            value.flush()
        arrays.clear()
        for name, value in tables.items():
            filename = name+'.npy'; np.save(args.output/filename, value, allow_pickle=False)
            specs[name] = {'file': filename, 'dtype': str(value.dtype), 'shape': list(value.shape)}
        np.save(args.output/'fold_action_counts.npy', action_counts, allow_pickle=False)
        specs['fold_action_counts'] = {'file': 'fold_action_counts.npy', 'dtype': 'int64', 'shape': [2, 13166]}
        budget(started, args.deadline_seconds)
        if any(file_hash(path) != expected for path, expected in bindings.items()):
            raise RuntimeError('source/static/protocol/compiler changed during cache creation')
        admission.verify_manifest_sources(json.loads(args.admission.read_text(encoding='utf-8')))
        filenames = [spec['file'] for spec in specs.values()] + ['feature_metadata.json', 'episodes.json',
                                                               'encoding_exclusions.jsonl', 'population_gate.json']
        files = {name: file_hash(args.output/name) for name in filenames}
        manifest = {'schema': 'iql-dataset-v1', 'state': 'complete', 'smoke': args.smoke,
                    'training_authorized': bool(gates['passed'] and not args.smoke), 'state_count': states_total,
                    'episode_count': len(episodes), 'game_count': len(game_metadata), 'game_metadata': game_metadata,
                    'feature_metadata_file': 'feature_metadata.json', 'episode_metadata_file': 'episodes.json',
                    'arrays': specs, 'files': files, 'bindings': bindings,
                    'source_admission_path': str(args.admission), 'source_admission_sha256': file_hash(args.admission),
                    'static_export_path': str(args.static_export), 'static_export_sha256': file_hash(args.static_export),
                    'catalogue_path': str(args.catalogue), 'catalogue_sha256': file_hash(args.catalogue),
                    'source_directories': [str(path) for path in args.source], 'gates': gates,
                    'source_coverage': {
                        'considered_player_games': sum(source['manifest']['counts']['considered_player_games'] for source in bundle['sources']),
                        'validated_source_complete_player_games': sum(bundle['source_complete_player_games'].values()),
                        'source_flag_complete_player_games': sum(bundle['source_complete_player_games'].values())+source_validation_failures,
                        'source_chain_validation_failed_player_games': source_validation_failures,
                        'source_exported_player_games': sum(source['manifest']['trajectory_count'] for source in bundle['sources']),
                        'feature_encoded_player_games': len(episodes),
                        'encoded_by_patch': dict(Counter(row['patch'] for row in episodes))},
                    'encoding_exclusions': dict(encoding_exclusions), 'inventory_unique_items_max': max_bag,
                    'inventory_ragged_tokens': tokens_total, 'patches': list(PATCHES),
                    'terminal_reward_convention': 'Y on transition into absorbing terminal exactly once; no terminal state tensor',
                    'fold_rule': 'big-endian first8bytes SHA256(iql-v1-fold:+match_id) modulo2',
                    'test_payload_reads': False, 'confirmation_payload_reads': False,
                    'development_terminal_targets_read': False, 'raw_sources_reopened': False,
                    'elapsed_s': time.monotonic()-started}
        write_json(args.output/'manifest.json', manifest)
        (args.output/'manifest.sha256').write_text(file_hash(args.output/'manifest.json')+'  manifest.json\n')
        status('completed', transitions=states_total, games=len(game_metadata), training_authorized=manifest['training_authorized'])
        return manifest
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--catalogue', type=Path, required=True)
    parser.add_argument('--static-export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=2400)
    args = parser.parse_args()
    if args.deadline_seconds <= 0:
        parser.error('positive deadline required')
    args.source = [path.resolve(strict=True) for path in args.source]
    for name in ('admission', 'catalogue', 'static_export', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(json.dumps({key: result[key] for key in ('state_count', 'episode_count', 'game_count', 'training_authorized', 'elapsed_s')}, indent=2))


if __name__ == '__main__':
    main()
