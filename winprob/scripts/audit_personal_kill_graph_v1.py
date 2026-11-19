"""Independently verify source kill prefixes and prepare fixed degree controls.

Information-only: no purchase target file, fitted model or prediction score.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
import os
from pathlib import Path
import sys
import time
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from winprob.iql_dataset_contract import bound_json, current_reservations, file_hash, json_file
from winprob.personal_kill_graph_v1 import Kill, graph_for_observation, degree_control, control_seed, edge_excess

SOURCE_ROOT = ROOT.parent/'build_tracker'
SELECTION = ROOT/'data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json'
SELECTION_SHA = '979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6'
SCORES = SOURCE_ROOT/'data/exports/winprob-scoreboard/scoreboard-export-20261001-v1/manifest.json'
SCORES_SHA = 'fee70f9d7be28235e07b22b409471833bd4674d33a10abfc9939ae046fa2eeb2'
ROLES = SOURCE_ROOT/'data/exports/winprob-recipe-transition/recipe-roles-20261001-v1/manifest.json'
ROLES_SHA = 'b5537e3e4bb521e724212ae9787a8351c8233cd2bfcbe65a5128847e6a195805'
BASE = ROOT/'data/runs/iql-cache-20260930-v1/cache/manifest.json'
PROTOCOL = ROOT/'docs/personal_kill_graph_information_v1.md'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def run(source, output):
    started = time.monotonic(); source = source.resolve(); output = output.resolve()
    require(source.is_relative_to(SOURCE_ROOT/'data/exports/winprob-personal-kill-graph') and not output.exists(), 'artifact/output scope invalid')
    bindings = {}; m, source_sha = bound_json(source/'manifest.json', bindings)
    require(m['schema'] == 'personal-kill-graph-source-v1' and m['state'] == 'complete' and (m['games'], m['states']) == (3518, 56288)
            and m['selection_manifest_sha256'] == SELECTION_SHA and m['all_kda_prefixes_exact'] and m['included_future_events'] == 0
            and m['future_suffix_mutation_games'] == 3518, 'source graph integrity gates incomplete')
    for flag in ('selected_purchase_target_file_opened', 'raw_match_reads', 'database_reads', 'new_collection',
                 'purchase_labels_used_or_analyzed', 'outcomes_used_or_analyzed', 'test_payload_reads',
                 'confirmation_payload_reads', 'consumer_runtime_imported', 'training_performed', 'accuracy_computed'):
        require(m[flag] is False, 'source boundary failed: '+flag)
    require(file_hash(PROTOCOL) == m['protocol_sha256'], 'consumed graph protocol changed'); bindings[str(PROTOCOL)] = m['protocol_sha256']
    sm, sh = bound_json(SELECTION, bindings); require(sh == SELECTION_SHA, 'selection manifest changed')
    sp = SELECTION.parent/'selection.json'; require(file_hash(sp) == sm['files'][sp.name], 'selection data changed'); bindings[str(sp)] = sm['files'][sp.name]
    entries = json_file(sp)['entries']; by_game = defaultdict(list)
    require(len(entries) == 56288 and all(e['ordinal'] == i for i, e in enumerate(entries)), 'selection axis changed')
    for e in entries:
        by_game[e['match_id']].append(e)
    base, bs = bound_json(BASE, bindings)
    require(bs == '1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea', 'base metadata changed')
    admitted, ad = bound_json(base['source_admission_path'], bindings)
    require(ad == base['source_admission_sha256'], 'admission changed')
    lookup = {e['match_id']: e for e in admitted['games']}; ids = set(by_game)
    require(len(ids) == 3518 and not ids & current_reservations(admitted) and all(lookup[mid]['role'] == 'train'
            and lookup[mid]['patch'] == '16.17' and lookup[mid]['historical_validation'] is False for mid in ids), 'reserved/non-FIT graph input')
    for raw, expected in {**m['bindings_sha256'], **m['code_sha256']}.items():
        p = Path(raw)
        require(p.name != 'targets.jsonl' and p.suffix.lower() not in ('.db', '.gz', '.sqlite'), 'source lineage would reopen forbidden payload')
        require(file_hash(p) == expected, 'bound source input/code changed'); bindings[str(p)] = expected
    for name, expected in m['files'].items():
        p = source/name; require(p.parent.resolve() == source and file_hash(p) == expected, 'source member changed/escaped')
        bindings[str(p)] = expected
    arrays = {name: np.load(source/(name+'.npy'), mmap_mode='r', allow_pickle=False) for name in (
        'counts', 'environment', 'kda', 'teams', 'champions', 'maximum_included_ts')}
    require(arrays['counts'].shape == (56288, 10, 10) and arrays['kda'].shape == (56288, 10, 3)
            and all(arrays[k].shape == (56288, 10) for k in ('environment', 'teams', 'champions'))
            and arrays['maximum_included_ts'].shape == (56288,), 'source array axes changed')
    scores, ksha = bound_json(SCORES, bindings); require(ksha == SCORES_SHA, 'original KDA manifest changed')
    for name in ('player_kda.npy', 'participant_ids.npy'):
        p = SCORES.parent/name; require(file_hash(p) == scores['files'][name], 'original KDA file changed'); bindings[str(p)] = scores['files'][name]
    old_kda = np.load(SCORES.parent/'player_kda.npy', mmap_mode='r', allow_pickle=False)
    old_ids = np.load(SCORES.parent/'participant_ids.npy', mmap_mode='r', allow_pickle=False)
    roles, rsha = bound_json(ROLES, bindings); require(rsha == ROLES_SHA, 'original public roster manifest changed')
    rp = ROLES.parent/'roles.jsonl'; require(file_hash(rp) == roles['files'][rp.name], 'original public rosters changed'); bindings[str(rp)] = roles['files'][rp.name]
    roster_checked = np.zeros(56288, dtype=bool)
    with rp.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); i = row['ordinal']; e = entries[i]; r = row['reference_roles']
            require(not roster_checked[i] and row['sample_id'] == e['sample_id'] and row['match_id'] == e['match_id'], 'roster identity changed')
            players = [{'participant_id': e['participant_id'], 'team_id': r['team_id'], 'champion_id': r['champion_id']}, *r['others']]
            ordered = sorted(players, key=lambda p: p['participant_id'])
            require([p['participant_id'] for p in ordered] == list(range(1, 11))
                    and np.array_equal([p['team_id'] for p in ordered], arrays['teams'][i])
                    and np.array_equal([p['champion_id'] for p in ordered], arrays['champions'][i]), 'public roster parity failed')
            roster_checked[i] = True
    require(bool(roster_checked.all()), 'roster coverage incomplete')
    output.mkdir(parents=True)
    controls = np.lib.format.open_memmap(output/'null_counts.npy', mode='w+', dtype=np.int32, shape=(56288, 10, 10))
    active = np.zeros(56288, dtype=bool); null_personal_changed = np.zeros(56288, dtype=bool)
    seen = np.zeros(56288, dtype=bool); games_seen = set(); graph_changed = 0; count = 0
    with gzip.open(source/'events.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            require(time.monotonic()-started < 1800, 'fixed30minute information audit bound reached')
            game = json.loads(line); mid = game['match_id']
            require(mid in ids and mid not in games_seen, 'unexpected/duplicate source game')
            events = [Kill(e['timestamp_ms'], e['killer'], e['victim'], tuple(e['assisters'])) for e in game['events']]
            require(len(events) == game['maximum_needed_deaths'], 'source prefix ledger changed')
            for e in by_game[mid]:
                i = e['ordinal']; si = e['state_index']; order = old_ids[si]
                require(np.array_equal(np.sort(order), np.arange(1, 11)), 'original KDA player IDs changed')
                expected = old_kda[si, np.argsort(order)]
                require(np.array_equal(expected, arrays['kda'][i]), 'source KDA differs from original stored snapshot')
                graph, environment, audit = graph_for_observation(events, expected, e['timestamp_ms'], arrays['teams'][i])
                last = -1 if audit['maximum_included_timestamp_ms'] is None else audit['maximum_included_timestamp_ms']
                require(np.array_equal(graph, arrays['counts'][i]) and np.array_equal(environment, arrays['environment'][i])
                        and last == arrays['maximum_included_ts'][i], 'independent source graph replay differs')
                control = degree_control(graph, arrays['teams'][i], control_seed(e)); controls[i] = control
                _, numerator, _ = edge_excess(graph, arrays['teams'][i], e['participant_id'])
                active[i] = bool(np.any(numerator)); b = e['participant_id']-1
                null_personal_changed[i] = not np.array_equal(graph[:, b], control[:, b])
                graph_changed += int(not np.array_equal(graph, control)); seen[i] = True; count += 1
            games_seen.add(mid)
            if len(games_seen) % 250 == 0:
                print(json.dumps({'games': len(games_seen), 'states': count, 'elapsed_s': time.monotonic()-started}), flush=True)
    require(bool(seen.all()) and games_seen == ids, 'incomplete independent graph verification')
    controls.flush(); del controls
    focal = np.asarray([e['is_original_focal'] for e in entries], dtype=bool)
    require(int(focal.sum()) == 2900, 'original focal population changed')
    for raw, expected in bindings.items():
        require(file_hash(raw) == expected, 'bound input changed during graph audit')
    require(not ids & current_reservations(admitted), 'selected game became reserved')
    np.save(output/'active_excess.npy', active, allow_pickle=False)
    result = {'schema': 'personal-kill-graph-information-audit-v1', 'state': 'complete', 'passed': True,
        'source_manifest_path': str(source/'manifest.json'), 'source_manifest_sha256': source_sha,
        'selection_manifest_sha256': sh, 'games': 3518, 'states': count, 'focal_games': 2900,
        'exact_KDA_graph_roster_temporal_replay': True, 'all_null_row_column_degrees_preserved': True,
        'states_with_nonzero_exact_edge_excess': int(active.sum()), 'focals_with_nonzero_exact_edge_excess': int(active[focal].sum()),
        'outcome_blind_maximum_first_gain_bound': float(active[focal].mean()),
        'necessary_three_pp_capacity_gate_passed': bool(active[focal].sum() >= 87),
        'states_with_changed_null_graph': graph_changed, 'states_with_changed_null_personal_edges': int(null_personal_changed.sum()),
        'focals_with_changed_null_personal_edges': int(null_personal_changed[focal].sum()),
        'files': {name: file_hash(output/name) for name in ('null_counts.npy', 'active_excess.npy')},
        'bindings_sha256': bindings, 'code_sha256': {str(p): file_hash(p) for p in (Path(__file__), ROOT/'winprob/personal_kill_graph_v1.py')},
        'purchase_target_file_opened': False, 'raw_game_payload_reads': False, 'source_runtime_imported': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'training_performed': False, 'accuracy_computed': False,
        'elapsed_s': time.monotonic()-started,
        'limits': ['Only information availability and a loose outcome-blind capacity bound; no measured prediction benefit.',
                   'Kill attribution is not damage or pressure, and temporal/live-client parity is not measured.']}
    path = output/'report.json'; path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'report.sha256').write_text(file_hash(path)+' report.json\n', encoding='ascii')
    print(json.dumps({k: result[k] for k in ('state', 'passed', 'states', 'focals_with_nonzero_exact_edge_excess',
        'necessary_three_pp_capacity_gate_passed', 'elapsed_s')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); run(a.source, a.output)
