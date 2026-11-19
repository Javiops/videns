"""Outcome-blind first-boot support counts on a supplied clean source-bound set."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
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
from setback_experiment import (digest, verify_sidecar, verify_admitted_sources,
    reject_quarantined, set_below_normal_priority, check_resources, write_json)

PAIR = (3047, 3111)
SHOP_TYPES = {'ITEM_PURCHASED', 'ITEM_SOLD', 'ITEM_UNDO'}
ITEM_TYPES = SHOP_TYPES | {'ITEM_DESTROYED'}
MAX_IDLE_MS = 10000


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def item_price(items, item):
    value = items.get(str(item), {}).get('gold', {}).get('total')
    return float(value) if finite(value) and value >= 0 else None


def recipe_descendants(items, item, seen=None):
    seen = set() if seen is None else seen
    if item in seen:
        return set()
    seen.add(item)
    result = set()
    for raw in items.get(str(item), {}).get('from', []):
        child = int(raw)
        result.add(child)
        result.update(recipe_descendants(items, child, seen))
    return result


def boot_categories(items):
    direct = {int(i) for i in items.get('1001', {}).get('into', [])}
    tier2 = {i for i in direct if items.get(str(i), {}).get('maps', {}).get('11') is True
             and items.get(str(i), {}).get('gold', {}).get('purchasable') is True}
    if not set(PAIR) <= tier2:
        raise ValueError('static catalogue lacks the declared defensive action pair')
    descendants, pending = set(), list(tier2)
    while pending:
        parent = pending.pop()
        for raw in items.get(str(parent), {}).get('into', []):
            child = int(raw)
            if child not in descendants and child not in tier2:
                descendants.add(child)
                pending.append(child)
    return tier2, descendants


def select_focal(match, champion_tags):
    """Read only immutable roster fields/perks; never access winner or final stats."""
    mid = match['metadata']['matchId']
    roster = []
    for raw in match['info']['participants']:
        pid, cid, side = int(raw['participantId']), int(raw['championId']), int(raw['teamId'])
        if str(cid) not in champion_tags:
            raise ValueError('champion missing from pinned static tags')
        perks = raw.get('perks', {}).get('styles', [])
        rune_ids = [selection.get('perk') for style in perks for selection in style.get('selections', [])]
        magical = (8304 in rune_ids) if rune_ids and all(isinstance(rune, int) for rune in rune_ids) else None
        roster.append({'participant_id': pid, 'champion_id': cid, 'team_id': side,
                       'magical_footwear_selected': magical})
    if (len(roster) != 10 or {p['participant_id'] for p in roster} != set(range(1, 11))
            or Counter(p['team_id'] for p in roster) != {100: 5, 200: 5}):
        raise ValueError('invalid ten-player roster')
    candidates = [p for p in roster if set(champion_tags[str(p['champion_id'])]) & {'Fighter', 'Tank'}]
    if not candidates:
        return None, roster
    focal = min(candidates, key=lambda p: hashlib.sha256(
        f"{mid}:boots-focal:{p['participant_id']}".encode()).digest())
    return focal, roster


def focal_events(timeline, pid):
    result = []
    for frame_index, frame in enumerate(timeline['info']['frames']):
        for event_index, event in enumerate(frame.get('events', [])):
            if event.get('participantId') != pid or event.get('type') not in ITEM_TYPES:
                continue
            timestamp = event.get('timestamp')
            if not finite(timestamp) or int(timestamp) != timestamp or timestamp < 0:
                raise ValueError('invalid item-event timestamp')
            copy = {key: event.get(key) for key in ('type', 'itemId', 'beforeId', 'afterId')}
            copy.update(timestamp=int(timestamp), order=(frame_index, event_index))
            result.append(copy)
    return sorted(result, key=lambda e: (e['timestamp'], e['order']))


def find_first_choice(events, items):
    tier2, later = boot_categories(items)
    visits = []
    for event in events:
        if event['type'] not in SHOP_TYPES:
            continue
        if not visits or event['timestamp'] - visits[-1][-1]['timestamp'] > MAX_IDLE_MS:
            visits.append([])
        visits[-1].append(event)
    for visit in visits:
        for event in visit:
            if event['type'] != 'ITEM_PURCHASED':
                continue
            item = int(event.get('itemId') or 0)
            if item in later:
                return {'status': 'later_boot_upgrade_before_tier2', 'action_item_id': item,
                        'cutoff_ms': visit[0]['timestamp'], 'purchase_ms': event['timestamp'], 'visit': visit}
            if item in tier2:
                boot_events = [e for e in visit if e['type'] == 'ITEM_PURCHASED' and int(e.get('itemId') or 0) in tier2]
                recipe = recipe_descendants(items, item) | {item}
                # Keep target-resolution evidence separate from pre-visit X.
                # Destroy/quest events are not SHOP_TYPES, yet can transform
                # boots at the same timestamp or during the final idle window.
                audit_end = visit[-1]['timestamp'] + MAX_IDLE_MS
                full_visit_events = [e for e in events if visit[0]['timestamp'] <= e['timestamp'] <= audit_end]
                quest_transform = any(e['type'] == 'ITEM_DESTROYED' and int(e.get('itemId') or 0) in (1201, 1202)
                                      for e in full_visit_events)
                later_upgrade = any(e['type'] == 'ITEM_PURCHASED' and int(e.get('itemId') or 0) in later
                                    for e in full_visit_events)
                tier2_destroy = any(e['type'] == 'ITEM_DESTROYED' and int(e.get('itemId') or 0) in tier2
                                    for e in full_visit_events)
                other = [int(e.get('itemId') or 0) for e in visit
                         if e['type'] == 'ITEM_PURCHASED' and int(e.get('itemId') or 0) not in recipe]
                return {'status': 'first_tier2_purchase', 'action_item_id': item,
                        'cutoff_ms': visit[0]['timestamp'], 'purchase_ms': event['timestamp'], 'visit': visit,
                        'label_audit': {'undo_in_visit': any(e['type'] == 'ITEM_UNDO' for e in visit),
                                        'multiple_tier2_in_visit': len(boot_events) != 1,
                                        'tier2_attempt_items': [int(e['itemId']) for e in boot_events],
                                        'audit_end_ms': audit_end,
                                        'role_quest_transform_in_visit': quest_transform,
                                        'later_boot_upgrade_in_visit': later_upgrade,
                                        'tier2_destroy_in_visit': tier2_destroy,
                                        'sales_in_visit': any(e['type'] == 'ITEM_SOLD' for e in visit),
                                        'other_purchase_items': other,
                                        'mixed_basket': bool(other) or any(e['type'] == 'ITEM_SOLD' for e in visit)}}
    return {'status': 'no_explicit_tier2_purchase'}


def prior_boot_evidence(events, cutoff, magical):
    """Minimal explicit evidence; ambiguous undo/transformation is never repaired."""
    held = Counter()
    unknown = []
    for event in events:
        if event['timestamp'] >= cutoff:
            break
        kind, item = event['type'], int(event.get('itemId') or 0)
        if kind == 'ITEM_UNDO':
            unknown.append('prior_undo_inventory_unresolved')
        elif kind == 'ITEM_PURCHASED':
            held[item] += 1
        elif kind in ('ITEM_SOLD', 'ITEM_DESTROYED'):
            if item in (1201, 1202):
                unknown.append('prior_role_quest_transform')
            if item in (1001, 2422) and kind == 'ITEM_DESTROYED':
                unknown.append('prior_boot_destroy_transform_unresolved')
            held[item] = max(0, held[item] - 1)
    if magical is None:
        unknown.append('magical_footwear_selection_unknown')
    elif magical:
        unknown.append('magical_footwear_grant_not_reconstructed')
    return {'ordinary_boots_explicit_count': held[1001], 'magical_boots_explicit_count': held[2422],
            'ordinary_boots_proven': held[1001] == 1 and not unknown,
            'ambiguities': sorted(set(unknown))}


def wallet_lower_bound(wallet, frame_time, cutoff, events, items):
    reasons, spend = [], 0.0
    if not finite(wallet) or wallet < 0:
        reasons.append('invalid_prior_wallet')
    if frame_time >= cutoff:
        reasons.append('frame_not_strictly_prior')
    if cutoff - frame_time < 1000:
        reasons.append('frame_event_boundary_under_one_second')
    for event in events:
        if not frame_time < event['timestamp'] < cutoff:
            continue
        if event['type'] == 'ITEM_UNDO':
            reasons.append('undo_in_wallet_interval')
        if event['type'] == 'ITEM_PURCHASED':
            price = item_price(items, int(event.get('itemId') or 0))
            if price is None:
                reasons.append('unknown_intervening_purchase_price')
            else:
                spend += price
    # Same-timestamp frame/events have ambiguous ordering; do not infer which
    # spend the cached wallet already contains.
    if any(e['timestamp'] == frame_time and e['type'] in SHOP_TYPES for e in events):
        reasons.append('shop_event_at_wallet_frame_timestamp')
    return {'prior_current_gold': float(wallet) if finite(wallet) else None,
            'known_full_price_spend_upper_bound': spend,
            'lower_bound': max(0.0, float(wallet)-spend) if not reasons else None,
            'confidence': 'conditional_lower_bound' if not reasons else 'unknown',
            'unknown_reasons': sorted(set(reasons)),
            'assumption': 'all spending visible; prior wallet truly before the visit; no unobserved gold drain'}


def analyze_game(match, timeline, patch_static, cache_rows, feature_names):
    focal, roster = select_focal(match, patch_static['champion_tags'])
    if focal is None:
        return {'status': 'no_fighter_or_tank_in_roster'}
    result = {'focal': focal,
              'allied_champion_ids': sorted(p['champion_id'] for p in roster if p['team_id'] == focal['team_id'] and p != focal),
              'enemy_champion_ids': sorted(p['champion_id'] for p in roster if p['team_id'] != focal['team_id'])}
    items = patch_static['items']
    events = focal_events(timeline, focal['participant_id'])
    choice = find_first_choice(events, items)
    result.update({key: value for key, value in choice.items() if key != 'visit'})
    if choice['status'] != 'first_tier2_purchase':
        return result
    cutoff = choice['cutoff_ms']
    inventory = prior_boot_evidence(events, cutoff, focal['magical_footwear_selected'])
    result['pre_visit_boot_evidence'] = inventory
    prior_frames = [frame for frame in timeline['info']['frames'] if finite(frame.get('timestamp')) and frame['timestamp'] < cutoff]
    if not prior_frames:
        result['feature_status'] = 'no_strictly_prior_frame'
        return result
    frame = max(prior_frames, key=lambda f: f['timestamp'])
    frame_time = int(frame['timestamp'])
    result.update(prior_frame_ms=frame_time, frame_age_ms=cutoff-frame_time)
    if frame_time not in cache_rows:
        result['feature_status'] = 'prior_frame_not_in_clean_cache'
        return result
    participant = frame.get('participantFrames', {}).get(str(focal['participant_id']))
    if not isinstance(participant, dict):
        result['feature_status'] = 'missing_focal_participant_frame'
        return result
    result['feature_status'] = 'matched_clean_prior_frame'
    result['pre_visit_features'] = dict(zip(feature_names, map(float, cache_rows[frame_time])))
    result['focal_pre_frame'] = {key: participant.get(key) for key in ('currentGold', 'totalGold', 'xp', 'level')}
    wallet = wallet_lower_bound(participant.get('currentGold'), frame_time, cutoff, events, items)
    result['wallet_bound'] = wallet
    prices = {str(item): item_price(items, item) for item in PAIR}
    if any(price is None for price in prices.values()):
        raise ValueError('declared boot pair has missing total price')
    lb = wallet['lower_bound']
    full = lb is not None and lb >= max(prices.values())
    ordinary_price = item_price(items, 1001)
    discounted = (lb is not None and inventory['ordinary_boots_proven'] and ordinary_price is not None
                  and lb >= max(prices.values())-ordinary_price)
    audit = result['label_audit']
    clean = (not inventory['ambiguities'] and not audit['undo_in_visit'] and not audit['multiple_tier2_in_visit']
             and not audit['mixed_basket'] and not audit['role_quest_transform_in_visit']
             and not audit['later_boot_upgrade_in_visit'] and not audit['tier2_destroy_in_visit']
             and cutoff-frame_time >= 1000)
    result['affordability'] = {'full_total_prices': prices, 'full_price_dual_money_proven': bool(full),
        'ordinary_common_component_dual_money_proven': bool(discounted), 'slot_legality': 'unknown',
        'reliable_unmixed_pair': bool(clean and result['action_item_id'] in PAIR),
        'primary_full_price_support': bool(clean and full and result['action_item_id'] in PAIR),
        'secondary_ordinary_boots_support': bool(clean and discounted and result['action_item_id'] in PAIR)}
    return result


def load_static(path):
    verify_sidecar(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema') != 'winprob-boots-static-v1':
        raise ValueError('unknown static export schema')
    bindings = {str(path.resolve()): digest(path), str(path.with_suffix('.sha256').resolve()): digest(path.with_suffix('.sha256'))}
    for patch, data in payload['patches'].items():
        if data['version'].rsplit('.', 1)[0] != patch:
            raise ValueError('static patch/version mismatch')
        boot_categories(data['items'])
        for source in data['source_files']:
            source_path = Path(source['path']).resolve(strict=True)
            if digest(source_path) != source['sha256']:
                raise ValueError('static source digest changed')
            bindings[str(source_path)] = source['sha256']
    return payload['patches'], bindings


def read_bound_payload(binding, endpoint, mid, archive, ledger):
    path = Path(binding['path']).resolve(strict=True)
    if not path.is_relative_to(archive) or path.parent.name != endpoint or path.name != mid+'.json.gz':
        raise ValueError('source path escaped its bound archive endpoint/identity')
    reject_quarantined(path)
    ledger({'event': 'read_started', 'match_id': mid, 'endpoint': endpoint, 'path': str(path)})
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != binding['sha256']:
        raise ValueError('compressed source digest differs from clean W1 binding')
    envelope = json.loads(gzip.decompress(raw).decode('utf-8'))
    if (envelope.get('schema') != 1 or envelope.get('endpoint') != endpoint
            or envelope.get('match_id') != mid or envelope.get('payload', {}).get('metadata', {}).get('matchId') != mid):
        raise ValueError('archive envelope identity/schema mismatch')
    ledger({'event': 'read_complete', 'match_id': mid, 'endpoint': endpoint, 'sha256': actual})
    return envelope['payload']


def accumulate(report, row):
    role, patch = row['role'], row['patch']
    for group in (report['overall'], report['by_role'].setdefault(role, Counter()),
                  report['by_patch'].setdefault(patch, Counter())):
        group['source_games'] += 1
        group['status:'+row['status']] += 1
        if 'action_item_id' in row:
            group['first_boot_item:'+str(row['action_item_id'])] += 1
        if 'feature_status' in row:
            group['features:'+row['feature_status']] += 1
        for key, value in row.get('affordability', {}).items():
            if isinstance(value, bool) and value:
                group[key] += 1
                if key.endswith('_support'):
                    group[key+':'+str(row['action_item_id'])] += 1
        for reason in row.get('wallet_bound', {}).get('unknown_reasons', []):
            group['wallet_unknown:'+reason] += 1
        for reason in row.get('pre_visit_boot_evidence', {}).get('ambiguities', []):
            group['inventory_unknown:'+reason] += 1
        if row.get('label_audit', {}).get('mixed_basket'):
            group['mixed_basket'] += 1
        if row.get('label_audit', {}).get('undo_in_visit'):
            group['undo_in_choice_visit'] += 1
        for key in ('role_quest_transform_in_visit', 'later_boot_upgrade_in_visit', 'tier2_destroy_in_visit'):
            if row.get('label_audit', {}).get(key):
                group[key] += 1
    if row.get('action_item_id') in PAIR:
        key = f"{role}|{patch}|{row['focal']['champion_id']}"
        cell = report['champion_pair_counts'].setdefault(key, Counter())
        cell[str(row['action_item_id'])] += 1
        if row.get('affordability', {}).get('primary_full_price_support'):
            cell['supported:'+str(row['action_item_id'])] += 1


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
        status('verifying_outcome_blind_inputs')
        check_resources(started, args.deadline_seconds)
        for path in (args.cache, args.source_manifest, args.static_export):
            reject_quarantined(path)
        meta_path = args.cache.parent/'cache_metadata.json'
        verify_sidecar(meta_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        if (metadata.get('state') != 'complete' or digest(args.cache) != metadata['npz_sha256']
                or digest(args.source_manifest) != metadata['source_manifest_sha256']):
            raise ValueError('clean cache/source binding mismatch')
        source = json.loads(args.source_manifest.read_text(encoding='utf-8'))
        if any(source.get(key) is not False for key in ('test_payload_reads', 'confirmatory_payload_reads', 'historical_validation_payload_reads')):
            raise ValueError('forbidden source population')
        manifest_path, manifest_hash = verify_admitted_sources(source, metadata)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if len(source['games']) > 6000:
            raise ValueError('source population exceeds first-stage cap')
        static, files = load_static(args.static_export)
        for path in (Path(__file__).resolve(), ROOT/'docs/boots_value_protocol_v1.md', ROOT/'scripts/setback_experiment.py',
                     ROOT/'winprob/admission.py', args.cache, args.source_manifest, meta_path, manifest_path):
            files[str(path.resolve())] = digest(path)
        raw_sources = {entry['match_id']: entry for entry in source['compressed_sources']}
        if len(raw_sources) != len(source['compressed_sources']):
            raise ValueError('duplicate compressed source identity')
        cache = np.load(args.cache, allow_pickle=False)
        # Deliberately do not access cache['y'], match participant win fields,
        # match teams/outcomes, or final performance counters.
        x, games, roles, timestamps = (cache[key] for key in ('x', 'game', 'role', 'timestamp_ms'))
        feature_names = cache['feature_names'].tolist()
        if set(map(int, np.unique(games))) != {row['game_index'] for row in source['games']}:
            raise ValueError('cache/source game-index mismatch')
        archive = (Path(manifest['source_data_root'])/'raw_match_v5').resolve(strict=True)
        report = {'schema': 'boots-support-v1', 'stage': 'outcome_blind_counts',
                  'report_path': str((args.output/'report.json').resolve()), 'outcome_targets_extracted': False,
                  'cache_y_array_accessed': False, 'source_deserialization_includes_outcome_fields': True,
                  'raw_source_scope': 'only compressed paths bound in the supplied clean W1 cache source manifest',
                  'admission_manifest_sha256': manifest_hash, 'binding_sha256': files,
                  'overall': Counter(), 'by_role': {}, 'by_patch': {}, 'champion_pair_counts': {},
                  'feature_names': feature_names, 'causal_claim': False, 'slot_legality_established': False}
        write_json(args.output/'binding.json', {'files': files, 'source_game_count': len(source['games']),
            'mode': 'counts only; no outcome extraction', 'source_ids_sha256': admission.id_digest(r['match_id'] for r in source['games'])})
        output_rows = args.output/'choices.jsonl'
        with output_rows.open('x', encoding='utf-8') as handle:
            for number, entry in enumerate(source['games'], 1):
                if number == 1 or number % 100 == 0:
                    check_resources(started, args.deadline_seconds)
                    admission.verify_manifest_sources(manifest)
                    status('counting', processed_games=number-1, total_games=len(source['games']))
                mid, gid = entry['match_id'], int(entry['game_index'])
                if mid not in raw_sources or entry['patch'] not in static:
                    raise ValueError('missing exact source or patch-static binding')
                indices = np.flatnonzero(games == gid)
                if len(indices) != entry['rows'] or set(roles[indices].tolist()) != {entry['role']}:
                    raise ValueError('cache row count/role mismatch')
                if len(np.unique(timestamps[indices])) != len(indices):
                    raise ValueError('duplicate cached frame timestamp')
                cache_rows = {int(timestamps[i]): x[i] for i in indices}
                match = read_bound_payload(raw_sources[mid]['match'], 'match', mid, archive, ledger)
                timeline = read_bound_payload(raw_sources[mid]['timeline'], 'timeline', mid, archive, ledger)
                info = match['info']
                if info.get('queueId') != 420 or '.'.join(str(info.get('gameVersion', '')).split('.')[:2]) != entry['patch']:
                    raise ValueError('match queue/patch differs from source manifest')
                row = {'match_id': mid, 'game_index': gid, 'role': entry['role'], 'patch': entry['patch'],
                       **analyze_game(match, timeline, static[entry['patch']], cache_rows, feature_names)}
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
                accumulate(report, row)
                del match, timeline
        check_resources(started, args.deadline_seconds)
        admission.verify_manifest_sources(manifest)
        if any(digest(path) != expected for path, expected in files.items()):
            raise RuntimeError('input/code/protocol/static digest changed during support audit')
        for path in (args.cache, args.source_manifest, args.static_export):
            reject_quarantined(path)
        thresholds = {'fit': 100, 'calibration': 30, 'development': 50}
        report['support_minimum_checks'] = {role: {str(a): report['by_role'].get(role, {}).get('primary_full_price_support:'+str(a), 0) >= minimum
                                                   for a in PAIR} for role, minimum in thresholds.items()}
        report['support_minima_met'] = all(all(arms.values()) for arms in report['support_minimum_checks'].values())
        report['support_minimum_interpretation'] = 'count/engineering minima only; no terminal-effect power or positivity claim'
        report['choices_sha256'] = digest(output_rows)
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
    for name in ('cache', 'source_manifest', 'static_export', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.deadline_seconds <= 0:
        parser.error('deadline must be positive')
    report = run(args)
    print(json.dumps({'report': report['report_path'], 'counts_only': True,
                      'support_minima_met': report['support_minima_met'], 'elapsed_s': report['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
