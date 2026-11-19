"""Bounded raw-source Magical Footwear QA; never writes training data.

Only reads source-plan train/development-val games. Rune outcome variables are
reported as post-game QA anchors; they are never used by the causal predictor.
"""
from __future__ import annotations

import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/acceptance/16.17_development_20260911/source_plan.json'
RUNES = SOURCE.with_name('source_rune_inventory.json')
OUT = ROOT / 'data/acceptance/footwear_repair_20260911'
BOOT_IDS = {1001, 2422, 3006, 3009, 3020, 3047, 3111, 3117, 3158, 3170, 3171,
            3172, 3173, 3174, 3175}


def raw_read(path, endpoint, match_id):
    with gzip.open(ROOT / path, 'rt', encoding='utf-8') as handle:
        envelope = json.load(handle)
    assert envelope['endpoint'] == endpoint and envelope['match_id'] == match_id
    return envelope['payload']


def causal_grant(kills, participant_id):
    """Returns threshold crossing using only kills prior to the predicted grant."""
    deadline = 720_000
    counted = []
    for event in kills:
        stamp = event['timestamp']
        if stamp >= deadline:
            return deadline, counted, 'clock'
        if (event.get('killerId') == participant_id
                or participant_id in event.get('assistingParticipantIds', [])):
            counted.append(stamp)
            deadline -= 45_000
            if stamp >= deadline:
                return stamp, counted, 'takedown_crossing'
    return deadline, counted, 'clock'


def brief(event):
    return {key: event[key] for key in ('timestamp', 'type', 'participantId',
            'itemId', 'beforeId', 'afterId', 'goldGain', 'killerId',
            'victimId', 'assistingParticipantIds') if key in event}


def main():
    started = time.monotonic()
    plan = json.loads(SOURCE.read_text(encoding='utf-8'))
    metadata = json.loads(RUNES.read_text(encoding='utf-8'))
    footwear_ids = {r['match_id'] for r in metadata['games_metadata'] if r['magical_footwear']}
    selected = []
    for split in ('train', 'val'):
        candidates = [r for r in plan['matches'] if r['split'] == split
                      and r['legacy_split'] != 'test' and r['match_id'] in footwear_ids]
        selected.extend(candidates[i * len(candidates) // min(50, len(candidates))]
                        for i in range(min(50, len(candidates))))
    records, errors = [], []
    summary = collections.Counter()
    for row in selected:
        if time.monotonic() - started > 130:
            break
        mid = row['match_id']
        try:
            assert row['split'] in ('train', 'val') and row['legacy_split'] != 'test'
            match = raw_read(row['files']['match']['path'], 'match', mid)
            timeline = raw_read(row['files']['timeline']['path'], 'timeline', mid)
            assert '.'.join(match['info']['gameVersion'].split('.')[:2]) == '16.17'
            assert match['metadata']['matchId'] == mid
            events = sorted([e for f in timeline['info']['frames'] for e in f['events']],
                            key=lambda event: event['timestamp'])
            kills = [e for e in events if e['type'] == 'CHAMPION_KILL']
            duration = match['info']['gameDuration'] * 1000
            game = {'match_id': mid, 'split': row['split'], 'legacy_split': row['legacy_split'],
                    'duration_ms': duration, 'players': []}
            for participant in match['info']['participants']:
                selections = [r for s in participant.get('perks', {}).get('styles', [])
                              for r in s.get('selections', []) if r['perk'] == 8304]
                if not selections:
                    continue
                pid = participant['participantId']
                grant, counted, trigger = causal_grant(kills, pid)
                rune = selections[0]
                qa_second = rune['var1'] * 60 + rune['var2'] * 10 + rune['var3']
                item_events = [e for e in events if e.get('participantId') == pid
                               and e['type'].startswith('ITEM_')]
                boots = [brief(e) for e in item_events if e.get('itemId') in BOOT_IDS
                         or e.get('beforeId') in BOOT_IDS or e.get('afterId') in BOOT_IDS]
                nearby = [brief(e) for e in item_events if abs(e['timestamp'] - grant) <= 2000]
                rune_grant_events = [e for e in boots if e.get('itemId') == 2422
                                     and e['type'] == 'ITEM_PURCHASED']
                destroyed = [e for e in boots if e.get('itemId') == 2422 and e['type'] == 'ITEM_DESTROYED']
                paired = [{'destroy': e, 'same_timestamp_item_events': [brief(other)
                          for other in item_events if other['timestamp'] == e['timestamp']]}
                          for e in destroyed]
                predicted_second = grant // 1000
                granted_before_end = grant <= duration
                qa_match = qa_second == predicted_second if granted_before_end else qa_second == 0
                qa_deadline_match = qa_second * 1000 == 720_000 - 45_000 * len(counted)
                key = row['split'] + ':'
                summary[key + 'footwear_players'] += 1
                summary[key + 'qa_mss_matches_prefix_prediction'] += qa_match
                summary[key + 'qa_mss_differs_prefix_prediction'] += not qa_match
                summary[key + 'qa_mss_matches_reduced_deadline'] += qa_deadline_match
                summary[key + 'qa_mss_differs_reduced_deadline'] += not qa_deadline_match
                summary[key + 'no_grant_purchase_event'] += not rune_grant_events
                summary[key + 'grant_purchase_event_present'] += bool(rune_grant_events)
                summary[key + 'takedown_crossing'] += trigger == 'takedown_crossing'
                summary[key + 'not_due_before_end'] += not granted_before_end
                summary[key + 'boots_sale_players'] += any(e['type'] == 'ITEM_SOLD' for e in boots)
                summary[key + 'boots_undo_players'] += any(e['type'] == 'ITEM_UNDO' for e in boots)
                summary[key + '2422_destroy_players'] += bool(destroyed)
                game['players'].append({'participant_id': pid, 'champion': participant['championName'],
                    'postgame_rune_qa_only': rune,
                    'postgame_rune_hypothesis_m_ss_seconds': qa_second,
                    'prefix_predicted_grant_ms': grant, 'trigger': trigger,
                    'grant_due_before_game_end': granted_before_end,
                    'prefix_counted_takedowns_ms': counted, 'qa_match_to_integer_seconds': qa_match,
                    'qa_match_to_reduced_deadline': qa_deadline_match,
                    'events_within_2s_predicted_grant': nearby, 'boot_events': boots,
                    '2422_destroy_pairs': paired,
                    'inventory_capacity_at_grant': 'not established by this diagnostic',
                    'timer_independently_observed_in_timeline': bool(rune_grant_events)})
                if not qa_deadline_match:
                    game['players'][-1]['deadline_mismatch_item_trace'] = [brief(e) for e in item_events
                        if e['timestamp'] <= (boots[0]['timestamp'] if boots else grant + 300_000)]
                    game['players'][-1]['deadline_mismatch_takedown_trace'] = [brief(e) for e in kills
                        if (e.get('killerId') == pid or pid in e.get('assistingParticipantIds', []))
                        and e['timestamp'] <= (boots[0]['timestamp'] if boots else grant + 300_000)]
            records.append(game)
            summary[row['split'] + ':games'] += 1
        except Exception as exc:
            errors.append({'match_id': mid, 'error': str(exc)})
    report = {'schema': 1, 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': '100 systematic sampled footwear games (50 train,50 development val) selected from frozen 16.17 source plan; no final-test raw payload or model data reads',
        'source_plan_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'causal_rule_hypothesis': '12 minutes minus 45 seconds for each prefix takedown; crossing at max(reduced deadline, causal takedown timestamp)',
        'rune_qa_hypothesis': 'var1:minutes,var2:tens-of-seconds,var3:seconds; postgame QA ONLY, never model input',
        'timer_validation_limit': 'A matching final rune summary is consistent with the clock hypothesis, but does not independently prove inventory grant availability. No full-inventory or death-delay handling is inferred.',
        'summary': dict(sorted(summary.items())), 'games': records, 'errors': errors,
        'elapsed_seconds': time.monotonic() - started}
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'raw_footwear_audit.json'
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'report_path': str(output), 'summary': report['summary'], 'errors': errors,
                      'elapsed_seconds': report['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
