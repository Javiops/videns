"""Publish the fixed recipe-probe selection using identity metadata only."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.history_probe_data import BaseCache, canonical, make_selection, require, write_json
from winprob.iql_dataset_contract import bound_json, file_hash, json_file

BASE = ROOT/'data/runs/iql-cache-20260930-v1/cache'
HISTORY = ROOT/'data/runs/history-cache-20260930-v1/cache'
PROTOCOL = ROOT/'docs/recipe_transition_probe_protocol_v1.md'
PACKAGE = ROOT.parent/'build_tracker/data/packages/mixture-v4-m2-v1'
SOURCE_SPLIT = ROOT.parent/'build_tracker/data/dataset_generations/16.18-reconstructible-development-v4/split_assignment.json'


def run(output):
    start = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'new immutable metadata-only selection required')
    # Existing BaseCache reads identity/time arrays only; current targets stay locked.
    cache = BaseCache(BASE, ROOT/'docs/history_probe_protocol_v1.md')
    bindings = cache.bindings
    bindings[str(PROTOCOL)] = file_hash(PROTOCOL)
    history, hs = bound_json(HISTORY/'manifest.json', bindings)
    require(hs == '1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b', 'original history manifest changed')
    old, ss = bound_json(HISTORY/'selection.json', bindings)
    require(ss == 'fcc372f7598ea87d0d1e6e9456ea5cd143b21198f870234ac55306a2b353e3f7', 'original focal selection changed')
    reproduced = make_selection(cache.games, cache.episodes, cache.arrays['timestamp_ms'])
    require(reproduced == old['entries'], 'original label-blind focal selection failed reproduction')
    events_path = HISTORY/history['arrays']['source_event_index']['file']
    event_sha = file_hash(events_path)
    require(event_sha == history['files'][events_path.name], 'source event index changed')
    bindings[str(events_path)] = event_sha
    events = np.load(events_path, mmap_mode='r', allow_pickle=False)
    require(events.shape == (392448,) and events.dtype == np.int64 and np.all(events >= 0), 'invalid event metadata')

    package = json_file(PACKAGE/'manifest.json'); identity = json_file(PACKAGE/'package_id.json')
    package_sha = hashlib.sha256(json.dumps(package, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(package_sha == identity['manifest_sha256'] == 'ab2b4cc49874bd652f5cd716c5e597971cb158fa784e140ec27dc7806e64fdd9', 'M2 package identity changed')
    for path in (PACKAGE/'manifest.json', PACKAGE/'package_id.json', SOURCE_SPLIT):
        bindings[str(path)] = file_hash(path)
    assignment = json_file(SOURCE_SPLIT)
    train_ids = sorted(mid for mid, role in assignment.items() if role == 'train')
    train_digest = hashlib.sha256(('\n'.join(train_ids)+'\n').encode()).hexdigest()
    require(len(train_ids) == package['generation']['train_games'] == 6354
            and train_digest == package['generation']['train_games_sha256'], 'M2 training membership differs')
    games = {g['match_id']: g for g in cache.games if g['patch'] == '16.17'}
    require(len(games) == 3518 and not games.keys() & set(train_ids), 'M2/FIT overlap or scope change')
    focal = {r['match_id']: r for r in reproduced if r['patch'] == '16.17'}
    require(sum(r['status'] == 'selected' for r in focal.values()) == 2900, 'focal denominator changed')
    episodes = defaultdict(list)
    for ei, ep in enumerate(cache.episodes):
        if ep['match_id'] in games:
            episodes[ep['match_id']].append((ei, ep))
    entries = []; per_game = []; focal_count = 0
    for mid, game in sorted(games.items()):
        candidates = []
        for ei, ep in episodes[mid]:
            for local in range(ep['states']):
                index = ep['first_state'] + local; event = int(events[index]); pid = ep['participant_id']
                sample = hashlib.sha256(f'recipe-transition-v1/sample/{mid}|{pid}|{event}'.encode()).hexdigest()
                candidates.append((sample, index, ei, local, pid, event))
        require(candidates and len({r[0] for r in candidates}) == len(candidates), 'missing/duplicate source state')
        chosen = []; focal_index = focal[mid]['selected_state_index']
        if focal_index is not None:
            matches = [c for c in candidates if c[1] == focal_index]
            require(len(matches) == 1, 'focal state missing from original cache')
            chosen = matches; focal_count += 1
        chosen += [c for c in sorted(candidates) if c[1] != focal_index][:16-len(chosen)]
        require(len(chosen) == min(16, len(candidates)), 'per-game sample size changed')
        per_game.append({'match_id': mid, 'fold': game['fold'], 'eligible_states': len(candidates), 'selected_states': len(chosen)})
        for sample, index, ei, local, pid, event in sorted(chosen):
            entries.append({'ordinal': len(entries), 'sample_id': sample, 'match_id': mid,
                'participant_id': pid, 'source_event_index': event, 'source_node_ordinal': local,
                'state_index': index, 'episode_index': ei, 'trajectory_id': f'{mid}:{pid}',
                'timestamp_ms': int(cache.arrays['timestamp_ms'][index]), 'fold': game['fold'],
                'patch': '16.17', 'is_original_focal': index == focal_index})
    require(focal_count == 2900 and len({e['state_index'] for e in entries}) == len(entries), 'selection identity failure')
    cache.revalidate()
    require(not cache.selection_frozen and not {'full_action_id', 'main_group_id'} & cache.arrays.keys(), 'targets were opened')
    output.mkdir(parents=True)
    selection = {'schema': 'recipe-transition-selection-v1', 'state': 'complete', 'entries': entries, 'games': per_game,
                 'pilot_sample_ids': sorted(e['sample_id'] for e in entries)[:200]}
    write_json(output/'selection.json', selection)
    paths = [Path(__file__), ROOT/'winprob/history_probe_data.py', ROOT/'winprob/iql_dataset_contract.py', ROOT/'winprob/admission.py']
    manifest = {'schema': 'recipe-transition-selection-manifest-v1', 'state': 'complete',
        'games': len(games), 'states': len(entries), 'focal_decisions': focal_count, 'unavailable_focal_games': 618,
        'fold_games': dict(Counter(str(g['fold']) for g in games.values())),
        'fold_states': dict(Counter(str(e['fold']) for e in entries)),
        'fold_focals': dict(Counter(str(e['fold']) for e in entries if e['is_original_focal'])),
        'm2_train_games': len(train_ids), 'm2_train_digest': train_digest, 'm2_train_overlap': 0,
        'files': {'selection.json': file_hash(output/'selection.json')}, 'bindings_sha256': bindings,
        'code_sha256': {str(p.resolve()): file_hash(p) for p in paths},
        'protocol_sha256': bindings[str(PROTOCOL)], 'current_purchase_arrays_opened': False,
        'source_trajectories_opened': False, 'source_databases_opened': False, 'source_runtime_imported': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'outcomes_analyzed': False,
        'training_performed': False, 'elapsed_s': time.monotonic()-start}
    write_json(output/'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('state', 'games', 'states', 'focal_decisions', 'fold_states', 'm2_train_overlap', 'elapsed_s')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
