"""Recover legacy M2 roles for the frozen recipe probe, indexed and read-only.

No consumer runtime, trajectory payloads, model, raw archive or outcome analysis.
The indexed envelopes are hashed against the original certified FIT receipts.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import export_winprob_trajectories_v1 as producer

CONSUMER = ROOT.parent/'winprob'
SELECTION = CONSUMER/'data/runs/recipe-transition-selection-20261001-v1/selection/manifest.json'
SELECTION_SHA = '979c206cb4e98946e43a7c902a6dd8afc79254b02184d7c9a1151a881660a9a6'
PROTOCOL = CONSUMER/'docs/recipe_transition_probe_protocol_v1.md'
PROTOCOL_SHA = 'e87e91493727335f867adfef5f2401ffcd1daea82596c35a1f55636545d8b46e'
SOURCE = ROOT/'data/exports/winprob-fit-trajectories/16.17-iql-trajectories-20260930-v1/manifest.json'
BASE = CONSUMER/'data/runs/iql-cache-20260930-v1/cache/manifest.json'
BASE_SHA = '1d54caf78e8f5186453b0e38ecbb11ff8da55a41115cb773e2216c0431faafea'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    path.with_suffix('.sha256').write_text(producer.file_hash(path)+' '+path.name+'\n', encoding='ascii')


def role_projection(event, entry):
    """Match baseline role semantics, preserving the source other-player order."""
    require(event.get('event_index') == entry['source_event_index'] and event.get('ts') == entry['timestamp_ms']
            and event.get('participant_id') == entry['participant_id'], 'selected event identity differs')
    board = event.get('board')
    require(isinstance(board, list) and len(board) == 10, 'complete source roster required')
    pids = [p.get('participant_id') for p in board]
    require(all(type(p) is int for p in pids) and set(pids) == set(range(1, 11)), 'participant roster invalid')
    own = [p for p in board if p.get('is_self') is True]
    require(len(own) == 1 and own[0]['participant_id'] == entry['participant_id'], 'source buyer mismatch')
    require(event.get('team_id') == own[0].get('team_id') and event.get('champion_id') == own[0].get('champion_id'), 'focal event/board mismatch')
    roles = []
    for p in board:
        cid = p.get('champion_id'); side = p.get('team_id'); role = p.get('team_position') or ''
        require(type(cid) is int and cid > 0 and type(side) is int and side in (100, 200)
                and isinstance(role, str), 'invalid source role/roster field')
        if not p.get('is_self'):
            roles.append({'participant_id': p['participant_id'], 'champion_id': cid, 'team_id': side, 'role': role})
    require(len({(p['team_id'], p['champion_id']) for p in board}) == 10
            and sum(p['team_id'] == own[0]['team_id'] for p in board) == 5, 'ambiguous/incomplete champion roster')
    role = event.get('team_position') or ''
    require(isinstance(role, str), 'invalid focal role')
    return {'champion_id': own[0]['champion_id'], 'team_id': own[0]['team_id'], 'role': role, 'others': roles}


def poison_non_roles(event):
    value = deepcopy(event)
    value.update(win=True, label_counts={'999999': 1}, label_bought=[], gold=10**9,
                 future_state={'role': 'poison'}, inventory_before=[], purchases='poison')
    for p in value.get('board', []):
        p.update(gold=10**9, items=[], kills=999, deaths=999, outcome='poison')
    return value


def run(output):
    started = time.monotonic(); output = output.resolve()
    require(not output.exists(), 'new immutable role release required')
    selection_meta, sh = producer.load_bound(SELECTION)
    require(sh == SELECTION_SHA and selection_meta['states'] == 56288 and selection_meta['games'] == 3518
            and selection_meta['m2_train_overlap'] == 0 and selection_meta['current_purchase_arrays_opened'] is False,
            'frozen metadata selection changed')
    require(producer.file_hash(PROTOCOL) == PROTOCOL_SHA == selection_meta['protocol_sha256'], 'protocol changed')
    bindings = {str(SELECTION): sh, str(PROTOCOL): PROTOCOL_SHA}
    for raw, want in {**selection_meta['bindings_sha256'], **selection_meta['code_sha256']}.items():
        path = Path(raw)
        require(path.suffix.lower() not in ('.db', '.gz', '.sqlite'), 'selection bound a payload')
        require(producer.file_hash(path) == want, 'selection metadata/code changed')
        bindings[str(path)] = want
    selection_path = SELECTION.parent/'selection.json'
    require(producer.file_hash(selection_path) == selection_meta['files']['selection.json'], 'selection entries changed')
    bindings[str(selection_path)] = selection_meta['files']['selection.json']
    selection = producer.load_json(selection_path); entries = selection['entries']
    require(len(entries) == 56288 and [e['ordinal'] for e in entries] == list(range(56288)), 'invalid selection order')
    by_game = defaultdict(list)
    for e in entries:
        require(e['patch'] == '16.17', 'wrong selected patch')
        by_game[e['match_id']].append(e)
    ids = set(by_game)
    require(len(ids) == 3518 and all(len(v) == 16 for v in by_game.values()), 'fixed per-game selection changed')
    base, bs = producer.load_bound(BASE); require(bs == BASE_SHA, 'original base metadata changed'); bindings[str(BASE)] = bs
    admission_path = Path(base['source_admission_path'])
    admitted, tests, validation, confirm, ab = producer.validate_admission(admission_path)
    bindings.update(ab); denied = tests | validation | confirm
    lookup = {e['match_id']: e for e in admitted['games']}
    require(not ids & denied and all(lookup[mid]['role'] == 'train' and lookup[mid]['patch'] == '16.17'
            and lookup[mid]['historical_validation'] is False for mid in ids), 'non-FIT or reserved ID before SQL')
    source, source_sha = producer.load_bound(SOURCE); bindings[str(SOURCE)] = source_sha
    require(source['schema'] == 'winprob-fit-trajectories-v2' and source['state'] == 'complete'
            and source['patch'] == '16.17' and set(source['eligible_game_ids']) == ids, 'wrong certified source population')
    for raw, want in {**source['input_metadata_sha256'], **source['source_code_sha256']}.items():
        require(producer.file_hash(raw) == want, 'original source metadata/code changed'); bindings[raw] = want
    originals = {r['match_id']: r for r in source['game_bindings']}
    database = Path(source['source_database']).resolve(strict=True)
    require(database == (ROOT/'data/corpora/16.17-history-v1/tracker.db').resolve(), 'unexpected database')
    before = (database.stat().st_size, database.stat().st_mtime_ns)
    code = {str(p.resolve()): producer.file_hash(p) for p in (Path(__file__), Path(producer.__file__))}
    output.mkdir(parents=True)
    write(output/'selection_receipt.json', {'selection_manifest_sha256': sh, 'game_ids': sorted(ids),
        'states': len(entries), 'before_sql_payload_reads': True, 'protocol_sha256': PROTOCOL_SHA})
    ledger = []; count = 0; mutations = 0
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1', uri=True, timeout=2)) as conn, (output/'roles.jsonl').open('x', encoding='utf-8') as stream:
        conn.execute('PRAGMA query_only=ON'); conn.execute('PRAGMA cache_size=-2048')
        plan = conn.execute('EXPLAIN QUERY PLAN SELECT event_index,ts,ts_end,type,payload_json FROM game_events WHERE match_id=? ORDER BY event_index', (sorted(ids)[0],)).fetchall()
        require(any('SEARCH' in str(p).upper() for p in plan) and not any('SCAN GAME_EVENTS' in str(p).upper() for p in plan), 'indexed query required')
        reader = producer.ApprovedGameReader(conn, ids, denied)
        for mid in sorted(ids):
            require(time.monotonic()-started < 1800, 'fixed30minute role extraction bound reached')
            if len(ledger) % 100 == 0:
                producer.validate_admission(admission_path)
            game, events = reader.game(mid); receipt = originals[mid]
            require(receipt['raw_replay_matches_indexed_rows'] is True and receipt['admitted_mask_reproduced'] is True
                    and producer.value_hash({'game': game, 'events': events}) == receipt['indexed_rows_sha256'], 'certified indexed source changed')
            event_map = {e['event_index']: e for e in events}; require(len(event_map) == len(events), 'duplicate event index')
            for i, entry in enumerate(by_game[mid]):
                event = event_map[entry['source_event_index']]
                projected = role_projection(event, entry)
                if i == 0:
                    require(projected == role_projection(poison_non_roles(event), entry), 'forbidden fields changed role projection'); mutations += 1
                row = {'ordinal': entry['ordinal'], 'sample_id': entry['sample_id'], 'match_id': mid,
                    'participant_id': entry['participant_id'], 'source_event_index': entry['source_event_index'],
                    'timestamp_ms': entry['timestamp_ms'], 'source_event_sha256': producer.value_hash(event), 'reference_roles': projected}
                stream.write(producer.canonical(row).decode()+'\n'); count += 1
            ledger.append({'match_id': mid, 'states': 16, 'indexed_rows_sha256': receipt['indexed_rows_sha256']})
            if len(ledger) % 100 == 0:
                print(json.dumps({'games': len(ledger), 'states': count, 'elapsed_s': time.monotonic()-started}), flush=True)
            del game, events, event_map
    require(count == 56288 and len(ledger) == 3518 and mutations == 3518, 'incomplete role release')
    require(before == (database.stat().st_size, database.stat().st_mtime_ns), 'source database changed')
    for raw, want in {**bindings, **code}.items():
        require(producer.file_hash(raw) == want, 'bound input/code changed during role export')
    producer.validate_admission(admission_path); write(output/'game_ledger.json', ledger)
    report = {'schema': 'recipe-transition-reference-roles-v1', 'state': 'complete', 'games': 3518, 'states': count,
        'selection_manifest_path': str(SELECTION), 'selection_manifest_sha256': sh, 'protocol_sha256': PROTOCOL_SHA,
        'source_manifest_path': str(SOURCE), 'source_manifest_sha256': source_sha,
        'bindings_sha256': bindings, 'code_sha256': code, 'sql_query_plan': [list(p) for p in plan],
        'files': {name: producer.file_hash(output/name) for name in ('selection_receipt.json', 'roles.jsonl', 'game_ledger.json')},
        'mutation_checked_games': mutations, 'whole_indexed_game_receipts_verified': len(ledger),
        'raw_archive_reads': False, 'whole_database_scanned': False, 'source_trajectory_payload_reads': False,
        'test_payload_reads': False, 'confirmation_payload_reads': False, 'consumer_runtime_imported': False,
        'training_performed': False, 'model_loaded': False, 'indexed_purchase_fields_deserialized_for_identity': True,
        'indexed_outcomes_hashed_for_identity': True, 'purchase_labels_used_or_analyzed': False, 'outcomes_used_or_analyzed': False,
        'elapsed_s': time.monotonic()-started, 'role_scope': 'legacy frozen-reference replay only; not certified predecision residual features'}
    write(output/'manifest.json', report)
    print(json.dumps({'state': 'complete', 'games': 3518, 'states': count, 'elapsed_s': report['elapsed_s']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
