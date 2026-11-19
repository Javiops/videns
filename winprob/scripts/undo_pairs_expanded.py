"""Expanded 5,861-game counts runner; the v1 receipt certificate is unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import undo_pairs_support as v1
import numpy as np
from winprob import admission

SCHEMA = 'winprob-revision-context-v1'
ROLE_COUNTS = {'fit': 4002, 'calibration': 1000, 'development': 859}
ARRAY_NAMES = {'x', 'game', 'role', 'timestamp_ms', 'start_ms', 'split_time_ms', 'patch', 'feature_names'}


def validate_scope(metadata, source, array_names):
    if metadata.get('schema') != SCHEMA or source.get('schema') != SCHEMA:
        raise ValueError('expanded runner requires the outcome-free revision-context schema')
    if (metadata.get('contains_outcome_array') is not False or metadata.get('outcome_fields_accessed') is not False
            or metadata.get('cache_y_accessed') is not False or set(array_names) != ARRAY_NAMES):
        raise ValueError('cache is not the declared outcome-free array schema')
    rows = source['games']
    actual = dict(Counter(row['role'] for row in rows))
    if actual != ROLE_COUNTS or metadata.get('role_game_counts') != ROLE_COUNTS or len(rows) != 5861:
        raise ValueError('expanded source role counts differ from fixed 5861-game scope')
    if len({row['match_id'] for row in rows}) != len(rows) or len({row['game_index'] for row in rows}) != len(rows):
        raise ValueError('duplicate expanded source identity')
    if any(row['patch'] != '16.19' for row in rows if row['role'] == 'development'):
        raise ValueError('primary development contains M2-trained or wrong-patch games')
    if source.get('exclusions'):
        raise ValueError('expanded source must not silently exclude selected games')


def stable_pair_id(match_id, row):
    return '|'.join(map(str, (match_id, row['participant_id'], row['feature_cutoff_ms'], row['undo_ms'],
                             row['decision_timestamp_ms'], row['action_a'], row['action_b'])))


def analyze_with_ids(match_id, match, timeline, patch_static, cache_rows, feature_names):
    rows, counts = v1.analyze_game(match, timeline, patch_static, cache_rows, feature_names)
    for row in rows:
        row['pair_id'] = stable_pair_id(match_id, row)
    if len({row['pair_id'] for row in rows}) != len(rows):
        raise ValueError('ambiguous duplicate candidate receipt identity')
    return rows, counts


def run(args):
    started = time.monotonic()
    v1.set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(stage, **extra):
        v1.write_json(args.output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra})
    ledger_count = 0
    def ledger(entry):
        nonlocal ledger_count
        ledger_count += 1
        with (args.output/'access_ledger.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'number': ledger_count, 'utc': datetime.now(timezone.utc).isoformat(), **entry}, sort_keys=True)+'\n')
            handle.flush()
    try:
        status('verifying_expanded_outcome_free_inputs')
        v1.check_resources(started, args.deadline_seconds)
        for path in (args.cache, args.source_manifest, args.static_export):
            v1.reject_quarantined(path)
        meta_path = args.cache.parent/'cache_metadata.json'
        v1.verify_sidecar(meta_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if (metadata.get('state') != 'complete' or v1.digest(args.cache) != metadata['npz_sha256']
                or v1.digest(args.source_manifest) != metadata['source_manifest_sha256']):
            raise ValueError('expanded cache/source binding mismatch')
        source = json.loads(args.source_manifest.read_text(encoding='utf-8'))
        if any(source.get(key) is not False for key in ('test_payload_reads', 'confirmatory_payload_reads', 'historical_validation_payload_reads')):
            raise ValueError('forbidden source population')
        manifest_path, manifest_hash = v1.verify_admitted_sources(source, metadata)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        cache = np.load(args.cache, allow_pickle=False)
        validate_scope(metadata, source, cache.files)
        static, files = v1.load_static(args.static_export)
        for path in (Path(__file__).resolve(), ROOT/'docs/undo_pairs_protocol_v2.md',
                     ROOT/'scripts/undo_pairs_support.py', ROOT/'docs/undo_pairs_protocol_v1.md',
                     ROOT/'scripts/boots_support.py', ROOT/'scripts/setback_experiment.py',
                     ROOT/'winprob/admission.py', args.cache, args.source_manifest, meta_path, manifest_path):
            files[str(path.resolve())] = v1.digest(path)
        x, games, roles, timestamps = (cache[key] for key in ('x', 'game', 'role', 'timestamp_ms'))
        names = cache['feature_names'].tolist()
        if set(map(int, np.unique(games))) != {row['game_index'] for row in source['games']}:
            raise ValueError('cache/source game indices differ')
        raw_sources = {row['match_id']: row for row in source['compressed_sources']}
        if len(raw_sources) != len(source['compressed_sources']):
            raise ValueError('duplicate compressed-source binding')
        archive = (Path(manifest['source_data_root'])/'raw_match_v5').resolve(strict=True)
        report = {'schema': 'undo-pairs-support-v2', 'stage': 'outcome_blind_expanded_counts',
                  'report_path': str((args.output/'report.json').resolve()),
                  'source_scope': '5861 games: 4002 fit,1000 calibration,859 development16.19',
                  'certificate_implementation': str(Path(v1.__file__).resolve()),
                  'certificate_sha256': v1.digest(v1.__file__), 'certificate_changed_from_v1': False,
                  'outcome_targets_extracted': False, 'cache_y_accessed': False,
                  'source_deserialization_includes_outcome_fields': True, 'causal_claim': False,
                  'inventory_claim': 'conditional inverse-transaction equivalence, not complete reconstruction',
                  'admission_manifest_sha256': manifest_hash, 'binding_sha256': files,
                  'overall': Counter(), 'by_role': {}, 'by_patch': {}, 'eligible_pair_categories': Counter(),
                  'feature_names': names, 'primary_weight': '1/(10 * eligible revisions for player/game)'}
        v1.write_json(args.output/'binding.json', {'files': files, 'source_game_count': 5861,
            'source_ids_sha256': admission.id_digest(row['match_id'] for row in source['games']),
            'certificate_reused_unchanged': True, 'mode': 'counts only; no outcome array exists'})
        line = 0
        with (args.output/'pairs.jsonl').open('x', encoding='utf-8') as handle:
            for number, entry in enumerate(source['games'], 1):
                if number == 1 or number % 100 == 0:
                    v1.check_resources(started, args.deadline_seconds)
                    admission.verify_manifest_sources(manifest)
                    status('counting', processed_games=number-1, source_games=5861)
                mid, gid = entry['match_id'], int(entry['game_index'])
                if mid not in raw_sources or entry['patch'] not in static:
                    raise ValueError('missing source or static binding')
                idx = np.flatnonzero(games == gid)
                if (len(idx) != entry['rows'] or set(roles[idx].tolist()) != {entry['role']}
                        or len(np.unique(timestamps[idx])) != len(idx)):
                    raise ValueError('cache frame/role identity mismatch')
                contexts = {int(timestamps[i]): x[i] for i in idx}
                match = v1.read_bound_payload(raw_sources[mid]['match'], 'match', mid, archive, ledger)
                timeline = v1.read_bound_payload(raw_sources[mid]['timeline'], 'timeline', mid, archive, ledger)
                if (match['info'].get('queueId') != 420
                        or '.'.join(str(match['info'].get('gameVersion', '')).split('.')[:2]) != entry['patch']):
                    raise ValueError('match queue/patch mismatch')
                rows, counts = analyze_with_ids(mid, match, timeline, static[entry['patch']], contexts, names)
                for index, row in enumerate(rows):
                    line += 1
                    handle.write(json.dumps({'match_id': mid, 'game_index': gid, 'role': entry['role'], 'patch': entry['patch'],
                                             'game_candidate_index': index, 'source_pair_line': line, **row},
                                            sort_keys=True, allow_nan=False)+'\n')
                v1.add_counts(report, rows, counts, entry['role'], entry['patch'])
                del match, timeline
        v1.check_resources(started, args.deadline_seconds)
        admission.verify_manifest_sources(manifest)
        if any(v1.digest(path) != expected for path, expected in files.items()):
            raise RuntimeError('expanded source/protocol/implementation changed during run')
        for path in (args.cache, args.source_manifest, args.static_export):
            v1.reject_quarantined(path)
        fit, dev = report['by_role'].get('fit', {}), report['by_role'].get('development', {})
        report['challenge_count_minimum_met'] = bool(fit.get('games_with_eligible_revision', 0) >= 100
            and dev.get('games_with_eligible_revision', 0) >= 50 and dev.get('eligible_revision', 0) >= 100)
        report['count_gate_interpretation'] = 'size only; canonical M2 mapping/context exclusions still apply'
        report['pairs_sha256'] = v1.digest(args.output/'pairs.jsonl')
        report['access_ledger_sha256'] = v1.digest(args.output/'access_ledger.jsonl')
        report['elapsed_s'] = time.monotonic()-started
        v1.write_json(args.output/'report.json', report)
        status('completed', report=report['report_path'])
        return report
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--static-export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=int, default=1200)
    args = parser.parse_args()
    if args.deadline_seconds <= 0:
        parser.error('deadline must be positive')
    for name in ('cache', 'source_manifest', 'static_export', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    print(json.dumps({'report': report['report_path'], 'challenge_count_minimum_met': report['challenge_count_minimum_met'],
                      'elapsed_s': report['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
