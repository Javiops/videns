"""Metadata-first census of strictly closed REAL purchase prefixes only."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.history_probe_data import BaseCache, write_json, require
from winprob.iql_dataset_contract import bound_json, file_hash
from winprob.bayes_proxy_ratio_v1 import closed_real_prefix

SELECTION = ROOT/'data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json'
HISTORY = ROOT/'data/runs/history-cache-20260930-v1/cache/manifest.json'
PROTOCOL = ROOT/'docs/bayes_proxy_prefix_protocol_v1.md'


def run(output):
    started = time.monotonic(); output = output.resolve(); require(not output.exists(), 'new immutable prefix census required')
    cache = BaseCache(ROOT/'data/runs/iql-cache-20260930-v1/cache', ROOT/'docs/history_probe_protocol_v1.md')
    bindings = dict(cache.bindings)
    selected, sh = bound_json(SELECTION, bindings)
    require(sh == '979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6', 'original query selection changed')
    selection_path = SELECTION.parent/'selection.json'
    require(file_hash(selection_path) == selected['files']['selection.json'], 'query ledger changed')
    bindings[str(selection_path)] = selected['files']['selection.json']
    entries = json.loads(selection_path.read_text(encoding='utf-8'))['entries']
    queries = [e for e in entries if e['is_original_focal']]
    require(len(entries) == 56288 and len(queries) == 2900 and len({e['match_id'] for e in entries}) == 3518
            and len({e['match_id'] for e in queries}) == 2900 and all(e['patch'] == '16.17' for e in entries), 'fixed16.17focals required')
    history, hh = bound_json(HISTORY, bindings)
    require(hh == '1bae0e5e58c752e23a82e461c072b6e5e64162732c98322a8c1ac11b9c79831b', 'certified closure artifact changed')
    arrays = {}
    for name in ('visit_end_ms', 'source_event_index'):
        spec = history['arrays'][name]; path = HISTORY.parent/spec['file']
        require(file_hash(path) == history['files'][path.name], 'closure/event sidecar changed')
        bindings[str(path)] = history['files'][path.name]
        arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        require(arrays[name].shape == (392448,) and arrays[name].dtype == np.int64, 'closure/event axes changed')
    starts = cache.array('timestamp_ms'); ends = arrays['visit_end_ms']; source_events = arrays['source_event_index']
    require(np.all(ends >= starts), 'invalid certified closures')
    for e in queries:
        index = e['state_index']; episode = cache.episodes[e['episode_index']]
        require(episode['first_state']+e['source_node_ordinal'] == index
                and episode['match_id'] == e['match_id'] and episode['participant_id'] == e['participant_id']
                and episode['fold'] == e['fold'] and episode['trajectory_id'] == e['trajectory_id']
                and int(starts[index]) == e['timestamp_ms'] and int(source_events[index]) == e['source_event_index'],
                'fixed focal identity/cutoff changed')
    code = {str(p.resolve()): file_hash(p) for p in (Path(__file__), PROTOCOL,
        ROOT/'docs/bayes_proxy_ratio_design_v1.md', ROOT/'winprob/bayes_proxy_ratio_v1.py',
        ROOT/'winprob/history_probe_data.py', ROOT/'winprob/iql_dataset_contract.py')}
    output.mkdir(parents=True)
    write_json(output/'query_selection.json', {'state': 'queries_frozen_before_past_action_kind_access',
        'queries': queries, 'games': 2900, 'source_fit_games': 3518, 'unavailable_original_focals': 618,
        'bindings_sha256': bindings, 'code_sha256': code, 'action_array_opened': False})
    cache.selection_frozen = True
    full = cache.array('full_action_id')  # Hashes file; index ONLY strictly past rows below.
    prefix = []; offsets = [0]; sizes = Counter(); fold_counts = Counter(); synthetic = ties = overlap = 0
    for position, entry in enumerate(queries):
        require(time.monotonic()-started < 900, 'fixed15minute census limit')
        episode = cache.episodes[entry['episode_index']]; current = entry['state_index']; cutoff = entry['timestamp_ms']
        events = []; index_for_start = {}
        for index in range(episode['first_state'], current):
            require(int(starts[index]) < cutoff, 'current/future action index would be touched')
            real = int(full[index]) != 0
            if not real:
                synthetic += 1
                continue
            end = int(ends[index]); ties += int(end == cutoff); overlap += int(end > cutoff)
            start = int(starts[index]); require(start not in index_for_start, 'real physical visit start repeated')
            index_for_start[start] = index
            events.append({'id': start, 'start': start, 'end': end, 'real_purchase': True})
        chosen = tuple(index_for_start[t] for t in closed_real_prefix(events, cutoff))
        require(all(episode['first_state'] <= i < current and ends[i] < cutoff for i in chosen), 'prefix crosses identity/cutoff')
        prefix.extend(chosen); offsets.append(len(prefix)); sizes[len(chosen)] += 1
        fold_counts[str(entry['fold'])+'_queries'] += 1
        fold_counts[str(entry['fold'])+'_past_visits'] += len(chosen)
        if (position+1) % 500 == 0:
            print(json.dumps({'stage': 'closed_prefix_census', 'queries': position+1, 'elapsed_s': time.monotonic()-started}), flush=True)
    values = np.asarray(prefix, dtype=np.int64); pointer = np.asarray(offsets, dtype=np.int64)
    require(len(np.unique(values)) == len(values), 'one-focal-per-game prefixes unexpectedly share states')
    np.save(output/'prefix_state_indices.npy', values, allow_pickle=False)
    np.save(output/'prefix_offsets.npy', pointer, allow_pickle=False)
    cache.revalidate()
    for path, sha in {**bindings, **code}.items(): require(file_hash(path) == sha, 'census input/code changed')
    report = {'schema': 'bayes-proxy-prefix-census-v1', 'state': 'complete', 'query_games': 2900,
        'original_fit_games': 3518, 'unavailable_original_focals': 618,
        'past_real_closed_visits': len(prefix), 'causal_forecast_rows_needed_including_queries': len(prefix)+2900,
        'zero_history_queries': sizes.get(0, 0), 'prefix_size_histogram': dict(sorted(sizes.items())),
        'fold_counts': dict(fold_counts), 'synthetic_nodes_excluded_before_closure': synthetic,
        'real_visit_closure_ties_masked': ties, 'real_visits_not_closed_at_cutoff_masked': overlap,
        'past_full_action_kind_values_read': True, 'complete_full_action_file_hashed': True,
        'current_or_future_action_values_indexed': False, 'main_action_labels_read': False,
        'model_or_prediction_payloads_read': False, 'raw_or_database_reads': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'accuracy_computed': False,
        'training_performed': False, 'bindings_sha256': {**bindings, **cache.bindings}, 'code_sha256': code,
        'files': {name: file_hash(output/name) for name in ('query_selection.json', 'prefix_state_indices.npy', 'prefix_offsets.npy')},
        'elapsed_s': time.monotonic()-started,
        'limits': ['Past kind/closure availability only; not a model, evidence of persistence, or an expected improvement.']}
    write_json(output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('state', 'query_games', 'past_real_closed_visits', 'zero_history_queries', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); run(p.parse_args().output)
