"""Count outcome-blind isolated purchase/undo/replacement sequences."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from winprob import admission
from boots_support import load_static, read_bound_payload, recipe_descendants, item_price, finite
from setback_experiment import (digest, verify_sidecar, verify_admitted_sources,
    reject_quarantined, set_below_normal_priority, check_resources, write_json)

SHOP = {'ITEM_PURCHASED', 'ITEM_SOLD', 'ITEM_UNDO'}
ITEM = SHOP | {'ITEM_DESTROYED'}
MATERIAL = {'CHAMPION_KILL', 'BUILDING_KILL', 'ELITE_MONSTER_KILL', 'DRAGON_SOUL_GIVEN',
            'LEVEL_UP', 'SKILL_LEVEL_UP', 'GAME_END'}
IDLE = 10000


def equipment_allowed(items, item):
    obj = items.get(str(item), {})
    price = item_price(items, item)
    return bool(obj.get('maps', {}).get('11') is True and obj.get('gold', {}).get('purchasable') is True
                and price is not None and price > 0
                and not set(obj.get('tags', [])) & {'Consumable', 'Trinket', 'GoldPer'}
                and not obj.get('specialRecipe') and not obj.get('requiredChampion')
                and not obj.get('requiredAlly'))


def roster_fields(match):
    """Immutable roster only; no winner, final build or postmatch role access."""
    roster = {int(p['participantId']): {'champion_id': int(p['championId']), 'team_id': int(p['teamId'])}
              for p in match['info']['participants']}
    if set(roster) != set(range(1, 11)) or Counter(p['team_id'] for p in roster.values()) != {100: 5, 200: 5}:
        raise ValueError('invalid ten-player roster')
    return roster


def collect_events(timeline):
    events, end = [], 0
    for fi, frame in enumerate(timeline['info']['frames']):
        if finite(frame.get('timestamp')):
            end = max(end, int(frame['timestamp']))
        for ei, source in enumerate(frame.get('events', [])):
            timestamp = source.get('timestamp')
            if not finite(timestamp) or int(timestamp) != timestamp:
                raise ValueError('invalid timeline event timestamp')
            end = max(end, int(timestamp))
            kind = source.get('type')
            if kind not in ITEM | MATERIAL:
                continue
            row = {key: source.get(key) for key in ('type', 'participantId', 'itemId', 'beforeId', 'afterId', 'goldGain')}
            row.update(timestamp=int(timestamp), order=(fi, ei))
            events.append(row)
    return sorted(events, key=lambda e: (e['timestamp'], e['order'])), end


def participant_visits(events, pid):
    visits = []
    for event in events:
        if event.get('participantId') != pid or event['type'] not in SHOP:
            continue
        if not visits or event['timestamp'] - visits[-1][-1]['timestamp'] > IDLE:
            visits.append([])
        visits[-1].append(event)
    return visits


def candidate_pair(visit, undo_index, events, source_end, items, pid):
    undo = visit[undo_index]
    if undo_index == 0 or undo_index+1 == len(visit):
        return None
    first, final = visit[undo_index-1], visit[undo_index+1]
    if first['type'] != 'ITEM_PURCHASED' or final['type'] != 'ITEM_PURCHASED':
        return None
    a, b = int(first.get('itemId') or 0), int(final.get('itemId') or 0)
    ta, tu, tb = first['timestamp'], undo['timestamp'], final['timestamp']
    close = visit[-1]['timestamp']+IDLE
    reasons = []
    if len(visit) != 3 or undo_index != 1:
        reasons.append('not_an_isolated_three_operation_visit')
    if not ta < tu < tb:
        reasons.append('non_strict_timestamp_order')
    if tb-ta > IDLE:
        reasons.append('pair_longer_than_ten_seconds')
    before_id, after_id = undo.get('beforeId'), undo.get('afterId')
    if type(before_id) is not int:
        reasons.append('undo_before_id_not_explicit_integer')
    elif before_id != a:
        reasons.append('undo_before_id_does_not_match_purchase')
    if type(after_id) is not int:
        reasons.append('undo_after_id_not_explicit_integer')
    elif after_id != 0:
        reasons.append('returned_component_requires_inventory_reconstruction')
    refund = undo.get('goldGain')
    price_a, price_b = item_price(items, a), item_price(items, b)
    if not finite(refund) or refund <= 0:
        reasons.append('not_a_known_positive_purchase_refund')
    elif price_a is None or refund != price_a:
        reasons.append('refund_not_equal_to_full_pinned_price')
    if not equipment_allowed(items, a) or not equipment_allowed(items, b):
        reasons.append('non_primary_equipment_or_static_restriction')
    interval = [e for e in events if ta <= e['timestamp'] <= close]
    if any(e.get('participantId') == pid and e['type'] == 'ITEM_DESTROYED' for e in interval):
        reasons.append('inventory_destruction_or_transformation_in_window')
    expected_orders = {first['order'], undo['order'], final['order']}
    if any(e.get('participantId') == pid and e['type'] in SHOP and e['order'] not in expected_orders for e in interval):
        reasons.append('extra_focal_shop_operation_in_window')
    if source_end < close:
        reasons.append('retention_window_not_observed_to_closure')
    material = [e['type'] for e in events if ta <= e['timestamp'] <= tb and e['type'] in MATERIAL]
    if material:
        reasons.append('material_recorded_context_change')
    nested = a != b and (a in recipe_descendants(items, b) or b in recipe_descendants(items, a))
    if nested:
        reasons.append('component_upgrade_pair_not_primary_revision')
    nonfocal_items = sum(e['type'] in ITEM and e.get('participantId') != pid for e in events if ta <= e['timestamp'] <= tb)
    return {'participant_id': pid, 'action_a': a, 'action_b': b,
            'pair_class': 'same_item_roundtrip_control' if a == b else 'different_item_revision',
            'feature_cutoff_ms': ta, 'undo_ms': tu, 'decision_timestamp_ms': tb,
            'retention_end_ms': close, 'elapsed_ms': tb-ta, 'recipe_nested': nested,
            'certificate': {'undo_before_id': undo.get('beforeId'), 'undo_after_id': undo.get('afterId'),
                            'refund': float(refund) if finite(refund) else None,
                            'full_static_price_a': price_a, 'full_static_price_b': price_b,
                            'both_feasible_at_b_conditional': not reasons,
                            'b_already_affordable_at_a_by_price_dominance': not reasons and price_b <= price_a,
                            'scope': 'conditional transactional certificate, not inferred wallet or causal value'},
            'audit': {'material_event_types': sorted(set(material)), 'nonfocal_item_events_during_pair': int(nonfocal_items)},
            'rejection_reasons': reasons, 'strict_transaction_pair': not reasons}


def analyze_game(match, timeline, patch_static, cache_rows, feature_names):
    roster = roster_fields(match)
    events, source_end = collect_events(timeline)
    pairs = []
    counts = Counter()
    for pid in range(1, 11):
        visits = participant_visits(events, pid)
        for visit in visits:
            for index, event in enumerate(visit):
                if event['type'] != 'ITEM_UNDO':
                    continue
                counts['undo_events'] += 1
                row = candidate_pair(visit, index, events, source_end, patch_static['items'], pid)
                if row is None:
                    counts['undo_without_adjacent_purchase_pair'] += 1
                    continue
                counts['purchase_undo_purchase_candidates'] += 1
                row.update(roster[pid])
                row['allied_champion_ids'] = sorted(p['champion_id'] for other, p in roster.items()
                                                   if other != pid and p['team_id'] == roster[pid]['team_id'])
                row['enemy_champion_ids'] = sorted(p['champion_id'] for p in roster.values()
                                                  if p['team_id'] != roster[pid]['team_id'])
                cutoff = row['feature_cutoff_ms']
                preceding = [f for f in timeline['info']['frames'] if finite(f.get('timestamp')) and f['timestamp'] < cutoff]
                frame = max(preceding, key=lambda f: f['timestamp']) if preceding else None
                if frame is None or int(frame['timestamp']) not in cache_rows:
                    row['feature_status'] = 'no_matching_strict_prior_cache_frame'
                elif cutoff-int(frame['timestamp']) < 1000:
                    row['feature_status'] = 'frame_event_boundary_under_one_second'
                else:
                    focal_frame = frame.get('participantFrames', {}).get(str(pid))
                    if not isinstance(focal_frame, dict):
                        row['feature_status'] = 'missing_focal_participant_frame'
                    else:
                        row['feature_status'] = 'matched_pre_a_frame'
                        row['pre_pair_features'] = dict(zip(feature_names, map(float, cache_rows[int(frame['timestamp'])])))
                        row['focal_pre_frame'] = {key: focal_frame.get(key) for key in ('currentGold', 'totalGold', 'xp', 'level')}
                        row['feature_frame_ms'] = int(frame['timestamp'])
                        row['feature_age_at_b_ms'] = row['decision_timestamp_ms']-int(frame['timestamp'])
                row['strict_context_pair'] = row['strict_transaction_pair'] and row['feature_status'] == 'matched_pre_a_frame'
                row['eligible_revision'] = row['strict_context_pair'] and row['pair_class'] == 'different_item_revision'
                row['eligible_roundtrip_control'] = row['strict_context_pair'] and row['pair_class'] == 'same_item_roundtrip_control'
                pairs.append(row)
    # Integrate over a uniformly sampled one-of-ten focal player, then one
    # eligible pair for that player. Noneligible players represent abstention.
    per_player = Counter(p['participant_id'] for p in pairs if p['eligible_revision'])
    for row in pairs:
        row['random_focal_pair_weight'] = 1.0/(10*per_player[row['participant_id']]) if row['eligible_revision'] else 0.0
    counts['players_with_eligible_revision'] = len(per_player)
    counts['random_focal_revision_coverage_mass'] = len(per_player)/10
    return pairs, dict(counts)


def add_counts(report, rows, base_counts, role, patch):
    count = Counter(base_counts)
    count['source_games'] = 1
    count['games_with_eligible_revision'] = int(any(row['eligible_revision'] for row in rows))
    count['games_with_eligible_roundtrip_control'] = int(any(row['eligible_roundtrip_control'] for row in rows))
    for row in rows:
        count['class:'+row['pair_class']] += 1
        count['features:'+row['feature_status']] += 1
        for key in ('strict_transaction_pair', 'strict_context_pair', 'eligible_revision', 'eligible_roundtrip_control'):
            count[key] += int(row[key])
        for reason in row['rejection_reasons']:
            count['reject:'+reason] += 1
        if row['eligible_revision']:
            count['revision_B_more_expensive'] += int(row['certificate']['full_static_price_b'] > row['certificate']['full_static_price_a'])
            count['revision_B_cheaper'] += int(row['certificate']['full_static_price_b'] < row['certificate']['full_static_price_a'])
            count['revision_equal_price'] += int(row['certificate']['full_static_price_b'] == row['certificate']['full_static_price_a'])
            key = f"{role}|{patch}|{row['action_a']}->{row['action_b']}"
            report['eligible_pair_categories'][key] += 1
    report['overall'].update(count)
    report['by_role'].setdefault(role, Counter()).update(count)
    report['by_patch'].setdefault(patch, Counter()).update(count)


def run(args):
    started = time.monotonic()
    set_below_normal_priority()
    args.output.mkdir(parents=True, exist_ok=False)
    def status(stage, **extra):
        write_json(args.output/'status.json', {'stage': stage, 'elapsed_s': time.monotonic()-started, **extra})
    ledger_count = 0
    def ledger(entry):
        nonlocal ledger_count
        ledger_count += 1
        with (args.output/'access_ledger.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'number': ledger_count, 'utc': datetime.now(timezone.utc).isoformat(), **entry}, sort_keys=True)+'\n')
            handle.flush()
    try:
        status('verifying_outcome_blind_sources')
        check_resources(started, args.deadline_seconds)
        for path in (args.cache, args.source_manifest, args.static_export):
            reject_quarantined(path)
        meta_path = args.cache.parent/'cache_metadata.json'
        verify_sidecar(meta_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if metadata.get('state') != 'complete' or digest(args.cache) != metadata['npz_sha256'] or digest(args.source_manifest) != metadata['source_manifest_sha256']:
            raise ValueError('clean cache/source hash mismatch')
        source = json.loads(args.source_manifest.read_text(encoding='utf-8'))
        if any(source.get(k) is not False for k in ('test_payload_reads', 'confirmatory_payload_reads', 'historical_validation_payload_reads')):
            raise ValueError('forbidden source population')
        manifest_path, manifest_hash = verify_admitted_sources(source, metadata)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not 1 <= len(source['games']) <= 6000:
            raise ValueError('source count exceeds bounded population')
        static, files = load_static(args.static_export)
        for path in (Path(__file__).resolve(), ROOT/'docs/undo_pairs_protocol_v1.md', ROOT/'scripts/boots_support.py',
                     ROOT/'scripts/setback_experiment.py', ROOT/'winprob/admission.py', args.cache, args.source_manifest, meta_path, manifest_path):
            files[str(path.resolve())] = digest(path)
        raw_sources = {row['match_id']: row for row in source['compressed_sources']}
        if len(raw_sources) != len(source['compressed_sources']):
            raise ValueError('duplicate compressed-source ID')
        archive = (Path(manifest['source_data_root'])/'raw_match_v5').resolve(strict=True)
        cache = np.load(args.cache, allow_pickle=False)
        x, games, roles, timestamps = (cache[k] for k in ('x', 'game', 'role', 'timestamp_ms'))
        names = cache['feature_names'].tolist()  # cache y is never accessed.
        if set(map(int, np.unique(games))) != {row['game_index'] for row in source['games']}:
            raise ValueError('cache/source population mismatch')
        report = {'schema': 'undo-pairs-support-v1', 'stage': 'outcome_blind_counts',
                  'report_path': str((args.output/'report.json').resolve()), 'outcome_targets_extracted': False,
                  'cache_y_accessed': False, 'source_deserialization_includes_outcome_fields': True,
                  'causal_claim': False, 'inventory_claim': 'conditional inverse-transaction equivalence, not complete inventory reconstruction',
                  'admission_manifest_sha256': manifest_hash, 'binding_sha256': files,
                  'overall': Counter(), 'by_role': {}, 'by_patch': {}, 'eligible_pair_categories': Counter(),
                  'feature_names': names, 'primary_weight': '1/(10 * eligible revision count for this player in this game)'}
        write_json(args.output/'binding.json', {'files': files, 'source_game_count': len(source['games']),
            'source_ids_sha256': admission.id_digest(row['match_id'] for row in source['games']), 'mode': 'counts only, no winner/terminal targets'})
        with (args.output/'pairs.jsonl').open('x', encoding='utf-8') as handle:
            for number, entry in enumerate(source['games'], 1):
                if number == 1 or number % 100 == 0:
                    check_resources(started, args.deadline_seconds)
                    admission.verify_manifest_sources(manifest)
                    status('counting', processed_games=number-1, source_games=len(source['games']))
                mid, gid = entry['match_id'], int(entry['game_index'])
                if mid not in raw_sources or entry['patch'] not in static:
                    raise ValueError('source/static binding missing')
                idx = np.flatnonzero(games == gid)
                if len(idx) != entry['rows'] or set(roles[idx].tolist()) != {entry['role']} or len(np.unique(timestamps[idx])) != len(idx):
                    raise ValueError('cache row identity/role/timestamp mismatch')
                cache_rows = {int(timestamps[i]): x[i] for i in idx}
                match = read_bound_payload(raw_sources[mid]['match'], 'match', mid, archive, ledger)
                timeline = read_bound_payload(raw_sources[mid]['timeline'], 'timeline', mid, archive, ledger)
                if match['info'].get('queueId') != 420 or '.'.join(str(match['info'].get('gameVersion', '')).split('.')[:2]) != entry['patch']:
                    raise ValueError('match queue/patch mismatch')
                rows, base_counts = analyze_game(match, timeline, static[entry['patch']], cache_rows, names)
                for sequence, row in enumerate(rows):
                    handle.write(json.dumps({'match_id': mid, 'game_index': gid, 'role': entry['role'], 'patch': entry['patch'],
                                             'game_candidate_index': sequence, **row}, sort_keys=True, allow_nan=False)+'\n')
                add_counts(report, rows, base_counts, entry['role'], entry['patch'])
                del match, timeline
        check_resources(started, args.deadline_seconds)
        admission.verify_manifest_sources(manifest)
        if any(digest(path) != value for path, value in files.items()):
            raise RuntimeError('frozen input or implementation changed during run')
        for path in (args.cache, args.source_manifest, args.static_export):
            reject_quarantined(path)
        fit, dev = report['by_role'].get('fit', {}), report['by_role'].get('development', {})
        report['challenge_count_minimum_met'] = bool(fit.get('games_with_eligible_revision', 0) >= 100
            and dev.get('games_with_eligible_revision', 0) >= 50 and dev.get('eligible_revision', 0) >= 100)
        report['count_gate_interpretation'] = 'challenge-size minimum only; no pair-prediction or win-benefit evidence'
        report['pairs_sha256'] = digest(args.output/'pairs.jsonl')
        report['access_ledger_sha256'] = digest(args.output/'access_ledger.jsonl')
        report['elapsed_s'] = time.monotonic()-started
        write_json(args.output/'report.json', report)
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
    print(json.dumps({'report': report['report_path'], 'counts_only': True,
                      'challenge_count_minimum_met': report['challenge_count_minimum_met'], 'elapsed_s': report['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
