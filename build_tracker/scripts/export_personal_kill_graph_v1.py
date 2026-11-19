"""Source-owned past kill graphs for the exact previously selected FIT states.

Only certified timeline envelopes and existing metadata/scoreboard artifacts.
No match payload, target ledger, SQL, source-model inference or consumer imports.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
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
import numpy as np
import export_winprob_trajectories_v1 as producer

OWN = ROOT.parent/'winprob'
SELECTION = OWN/'data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json'
SELECTION_SHA = '979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6'
SOURCE = ROOT/'data/exports/winprob-fit-trajectories/16.17-iql-trajectories-20260930-v1/manifest.json'
SCORES = ROOT/'data/exports/winprob-scoreboard/scoreboard-export-20261001-v1/manifest.json'
SCORES_SHA = 'fee70f9d7be28235e07b22b409471833bd4674d33a10abfc9939ae046fa2eeb2'
ROLES = ROOT/'data/exports/winprob-recipe-transition/recipe-roles-20261001-v1/manifest.json'
ROLES_SHA = 'b5537e3e4bb521e724212ae9787a8351c8233cd2bfcbe65a5128847e6a195805'
BASE = OWN/'data/runs/iql-cache-20260930-v1/cache/manifest.json'
PROTOCOL = OWN/'docs/personal_kill_graph_information_v1.md'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    path.with_suffix('.sha256').write_text(producer.file_hash(path)+' '+path.name+'\n', encoding='ascii')


def needed_events(payload, maximum):
    """Independent source projection in original archive order; unused suffix ignored."""
    require(type(maximum) is int and 0 <= maximum <= 10000, 'bounded observed death prefix required')
    result = []
    if maximum == 0:
        return result
    frames = payload.get('info', {}).get('frames')
    require(isinstance(frames, list), 'timeline frames missing')
    for frame in frames:
        for raw in frame.get('events') or []:
            if raw.get('type') != 'CHAMPION_KILL':
                continue
            timestamp = raw.get('timestamp'); killer = raw.get('killerId') or 0; victim = raw.get('victimId')
            assists = raw.get('assistingParticipantIds') or []
            require(type(timestamp) is int and timestamp >= 0 and type(killer) is int and 0 <= killer <= 10
                    and type(victim) is int and 1 <= victim <= 10, 'invalid raw kill identity/time')
            require(isinstance(assists, list) and all(type(a) is int and 1 <= a <= 10 for a in assists)
                    and len(assists) == len(set(assists)) and killer not in assists and victim not in assists, 'invalid raw assist identity')
            result.append({'timestamp_ms': timestamp, 'killer': killer, 'victim': victim, 'assisters': list(assists)})
            if len(result) == maximum:
                return result
    raise ValueError('raw timeline lacks required observed prefix')


def source_snapshot(events, expected, teams, cutoff):
    """Python-list implementation, independent of the own NumPy projector."""
    require(len(expected) == 10 and all(len(r) == 3 and all(type(x) is int and x >= 0 for x in r) for r in expected), 'complete KDA witness required')
    require(len(teams) == 10 and teams.count(100) == teams.count(200) == 5, 'complete teams required')
    count = sum(r[1] for r in expected); require(count <= len(events), 'missing prefix')
    kda = [[0, 0, 0] for _ in range(10)]; graph = [[0]*10 for _ in range(10)]; environment = [0]*10
    last = -1
    for event in events[:count]:
        require(event['timestamp_ms'] <= cutoff, 'future kill entered the source snapshot')
        killer, victim = event['killer'], event['victim']; v = victim-1
        if killer:
            require(teams[killer-1] != teams[v], 'unsupported friendly/self kill')
            kda[killer-1][0] += 1; graph[killer-1][v] += 1
        else:
            environment[v] += 1
        kda[v][1] += 1
        for assister in event['assisters']:
            require(teams[assister-1] != teams[v], 'unsupported friendly assist')
            kda[assister-1][2] += 1
        last = max(last, event['timestamp_ms'])
    require(kda == expected, 'original all-player KDA does not match exact raw prefix')
    require([sum(row) for row in graph] == [r[0] for r in expected]
            and [sum(graph[j][i] for j in range(10))+environment[i] for i in range(10)] == [r[1] for r in expected], 'source graph count conservation failed')
    return graph, environment, last


def public_roster(row, entry):
    r = row['reference_roles']
    players = [{'participant_id': entry['participant_id'], 'champion_id': r['champion_id'], 'team_id': r['team_id']}]
    players += [{k: p[k] for k in ('participant_id', 'champion_id', 'team_id')} for p in r['others']]
    players.sort(key=lambda p: p['participant_id'])
    require([p['participant_id'] for p in players] == list(range(1, 11))
            and all(type(p['champion_id']) is int and p['champion_id'] > 0 for p in players), 'public roster identity failed')
    require(sum(p['team_id'] == 100 for p in players) == sum(p['team_id'] == 200 for p in players) == 5, 'public teams failed')
    return players


def run(output):
    started = time.monotonic(); output = output.resolve(); require(not output.exists(), 'new immutable graph release required')
    selection_meta, sh = producer.load_bound(SELECTION)
    require(sh == SELECTION_SHA and selection_meta['states'] == 56288 and selection_meta['games'] == 3518, 'selection changed')
    bindings = {str(SELECTION): sh, str(PROTOCOL): producer.file_hash(PROTOCOL)}
    for raw, want in {**selection_meta['bindings_sha256'], **selection_meta['code_sha256']}.items():
        require(Path(raw).suffix.lower() not in ('.db', '.gz', '.sqlite') and Path(raw).name != 'targets.jsonl', 'selection contains forbidden payload binding')
        require(producer.file_hash(raw) == want, 'selection code/metadata changed'); bindings[raw] = want
    sp = SELECTION.parent/'selection.json'; require(producer.file_hash(sp) == selection_meta['files'][sp.name], 'selection rows changed')
    bindings[str(sp)] = selection_meta['files'][sp.name]; selected = producer.load_json(sp); entries = selected['entries']; n = len(entries)
    require([r['ordinal'] for r in entries] == list(range(n)) and n == 56288, 'selection axis changed')
    by_game = defaultdict(list)
    for entry in entries:
        require(entry['patch'] == '16.17', 'wrong selected patch'); by_game[entry['match_id']].append(entry)
    ids = set(by_game); require(len(ids) == 3518 and all(len(v) == 16 for v in by_game.values()), 'fixed population changed')
    base, bs = producer.load_bound(BASE)
    require(bs == '1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea', 'original base metadata changed')
    bindings[str(BASE)] = bs; admission = Path(base['source_admission_path'])
    admitted, tests, validation, confirm, ab = producer.validate_admission(admission); bindings.update(ab)
    denied = tests | validation | confirm; lookup = {r['match_id']: r for r in admitted['games']}
    require(not ids & denied and all(lookup[mid]['role'] == 'train' and lookup[mid]['patch'] == '16.17'
            and lookup[mid]['historical_validation'] is False for mid in ids), 'reserved/non-FIT game before raw reads')
    source, srcsha = producer.load_bound(SOURCE); bindings[str(SOURCE)] = srcsha
    require(source['state'] == 'complete' and source['schema'] == 'winprob-fit-trajectories-v2' and set(source['eligible_game_ids']) == ids, 'wrong certified timeline population')
    raw_bindings = {r['match_id']: r['raw_sources']['timeline'] for r in source['game_bindings'] if r['match_id'] in ids}
    require(set(raw_bindings) == ids, 'missing certified timeline binding')
    scores, scsha = producer.load_bound(SCORES); roles, rsha = producer.load_bound(ROLES)
    require(scsha == SCORES_SHA and rsha == ROLES_SHA and scores['base_manifest_sha256'] == bs
            and scores['score_columns'] == ['kills', 'deaths', 'assists'] and roles['selection_manifest_sha256'] == sh, 'scoreboard/roster lineage changed')
    bindings.update({str(SCORES): scsha, str(ROLES): rsha})
    for name in ('player_kda.npy', 'participant_ids.npy'):
        p = SCORES.parent/name; require(producer.file_hash(p) == scores['files'][name], 'scoreboard member changed'); bindings[str(p)] = scores['files'][name]
    kdas = np.load(SCORES.parent/'player_kda.npy', mmap_mode='r', allow_pickle=False)
    orders = np.load(SCORES.parent/'participant_ids.npy', mmap_mode='r', allow_pickle=False)
    require(kdas.shape == (392448, 10, 3) and kdas.dtype == np.int32 and orders.shape == (392448, 10), 'scoreboard axes/dtype changed')
    rp = ROLES.parent/'roles.jsonl'; require(producer.file_hash(rp) == roles['files'][rp.name], 'public roster source changed'); bindings[str(rp)] = roles['files'][rp.name]
    rosters = {}; role_seen = set()
    with rp.open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line); i = r['ordinal']; e = entries[i]
            require(i not in role_seen and all(r[k] == e[k] for k in ('sample_id', 'match_id', 'participant_id', 'source_event_index', 'timestamp_ms')), 'roster query identity changed')
            roster = public_roster(r, e); mid = e['match_id']
            if mid in rosters:
                require(rosters[mid] == roster, 'roster changed within a selected game')
            rosters[mid] = roster; role_seen.add(i)
    require(len(role_seen) == n and set(rosters) == ids, 'public roster coverage incomplete')
    code = {str(p.resolve()): producer.file_hash(p) for p in (Path(__file__), Path(producer.__file__))}
    output.mkdir(parents=True)
    write(output/'selection_receipt.json', {'selection_manifest_sha256': sh, 'games': sorted(ids), 'states': n,
        'before_raw_timeline_reads': True, 'protocol_sha256': bindings[str(PROTOCOL)]})
    shapes = {'counts': ((n, 10, 10), np.int32), 'environment': ((n, 10), np.int32), 'kda': ((n, 10, 3), np.int32),
              'teams': ((n, 10), np.int16), 'champions': ((n, 10), np.int32), 'maximum_included_ts': ((n,), np.int64)}
    arrays = {name: np.lib.format.open_memmap(output/(name+'.npy'), mode='w+', dtype=dtype, shape=shape) for name, (shape, dtype) in shapes.items()}
    for value in arrays.values():
        value[:] = -1
    ledger = []; count = 0; compressed_bytes = 0; projected_events = 0; mutation_games = 0
    with (output/'events.jsonl.gz').open('xb') as file, gzip.GzipFile(filename='', fileobj=file, mode='wb', compresslevel=1, mtime=0) as event_output:
        for mid in sorted(ids):
            require(time.monotonic()-started < 1800, 'fixed30minute graph extraction bound reached')
            if len(ledger) % 100 == 0:
                producer.validate_admission(admission)
            queries = by_game[mid]; expected = {}
            for e in queries:
                si = e['state_index']; order = orders[si].tolist()
                require(sorted(order) == list(range(1, 11)) and order[0] == e['participant_id'], 'scoreboard participant mapping changed')
                value = np.zeros((10, 3), dtype=np.int32)
                for column, pid in enumerate(order):
                    value[pid-1] = kdas[si, column]
                require(np.all(value >= 0), 'negative scoreboard count'); expected[e['ordinal']] = value
            maximum = max(int(k[:, 1].sum()) for k in expected.values())
            raw = raw_bindings[mid]; path = Path(raw['path']).resolve(strict=True)
            require(mid not in denied and path == (ROOT/'data/raw_match_v5/timeline'/(mid+'.json.gz')).resolve(strict=True), 'raw path escaped selected timeline')
            compressed = path.read_bytes(); digest = hashlib.sha256(compressed).hexdigest()
            require(len(compressed) == raw['bytes'] and digest == raw['sha256'], 'certified raw timeline changed')
            envelope = json.loads(gzip.decompress(compressed))
            require(envelope.get('schema') == 1 and envelope.get('endpoint') == 'timeline' and envelope.get('match_id') == mid
                    and envelope.get('payload', {}).get('metadata', {}).get('matchId') == mid, 'timeline envelope identity mismatch')
            events = needed_events(envelope['payload'], maximum)
            teams = [p['team_id'] for p in rosters[mid]]; champions = [p['champion_id'] for p in rosters[mid]]
            for qindex, e in enumerate(queries):
                i = e['ordinal']; witness = expected[i].tolist()
                graph, env, last = source_snapshot(events, witness, teams, e['timestamp_ms'])
                arrays['counts'][i] = graph; arrays['environment'][i] = env; arrays['kda'][i] = witness
                arrays['teams'][i] = teams; arrays['champions'][i] = champions; arrays['maximum_included_ts'][i] = last
                if qindex == 0:
                    prefix = sum(r[1] for r in witness)
                    poisoned = events[:prefix]+[{'timestamp_ms': 10**12, 'killer': 99, 'victim': 99, 'assisters': ['poison']}]
                    require(source_snapshot(poisoned, witness, teams, e['timestamp_ms']) == (graph, env, last), 'unused future suffix changed source graph')
                    mutation_games += 1
                count += 1
            event_output.write(producer.canonical({'match_id': mid, 'roster': rosters[mid], 'events': events,
                'maximum_needed_deaths': maximum, 'timeline_sha256': digest})+b'\n')
            ledger.append({'match_id': mid, 'timeline_path': str(path), 'timeline_sha256': digest,
                           'compressed_bytes': len(compressed), 'exported_kill_events': len(events), 'states': len(queries)})
            compressed_bytes += len(compressed); projected_events += len(events)
            if len(ledger) % 100 == 0:
                print(json.dumps({'games': len(ledger), 'states': count, 'kill_events': projected_events, 'elapsed_s': time.monotonic()-started}), flush=True)
            del envelope, compressed, events
    require(count == n and len(ledger) == mutation_games == 3518, 'incomplete graph release')
    for name, value in arrays.items():
        if name != 'maximum_included_ts':
            require(np.all(value >= 0), 'unfilled graph output: '+name)
        value.flush()
    arrays.clear()
    write(output/'game_ledger.json', ledger)
    for raw, want in {**bindings, **code}.items():
        require(producer.file_hash(raw) == want, 'bound metadata/code changed during graph export')
    producer.validate_admission(admission)
    names = ['selection_receipt.json', 'events.jsonl.gz', 'game_ledger.json', *[name+'.npy' for name in shapes]]
    manifest = {'schema': 'personal-kill-graph-source-v1', 'state': 'complete', 'games': 3518, 'states': n,
        'selection_manifest_path': str(SELECTION), 'selection_manifest_sha256': sh, 'scoreboard_manifest_sha256': scsha,
        'role_manifest_sha256': rsha, 'source_fit_manifest_sha256': srcsha, 'protocol_path': str(PROTOCOL),
        'protocol_sha256': bindings[str(PROTOCOL)], 'bindings_sha256': bindings, 'code_sha256': code,
        'files': {name: producer.file_hash(output/name) for name in names}, 'participant_axis': 'participantId1..10',
        'kda_columns': ['kills', 'deaths', 'assists'], 'all_kda_prefixes_exact': True, 'included_future_events': 0,
        'future_suffix_mutation_games': mutation_games, 'projected_kill_events': projected_events, 'compressed_timeline_bytes': compressed_bytes,
        'timestamp_arrays_audit_only': True, 'selected_purchase_target_file_opened': False,
        'raw_timeline_reads': True, 'raw_match_reads': False, 'database_reads': False, 'new_collection': False,
        'raw_other_event_fields_deserialized': True, 'purchase_labels_used_or_analyzed': False, 'outcomes_used_or_analyzed': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'consumer_runtime_imported': False,
        'training_performed': False, 'accuracy_computed': False, 'elapsed_s': time.monotonic()-started,
        'limits': ['Past last-hit attribution, not damage or pressure.', 'Source KDA aligns the original stored observation, including stale synthetic no-buy snapshots.',
                   'This release establishes information availability only, not purchase benefit or live parity.']}
    write(output/'manifest.json', manifest)
    print(json.dumps({'state': 'complete', 'games': 3518, 'states': n, 'kill_events': projected_events, 'elapsed_s': manifest['elapsed_s']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
