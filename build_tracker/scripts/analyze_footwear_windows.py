"""Potential-row accounting only; no repaired/accepted datasets are produced."""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_footwear_sources import causal_grant, raw_read, brief
import app.ddragon as dd
from app.reconstruct import reconstruct_game

BASE = ROOT / 'data/acceptance/footwear_repair_20260911'
SOURCE = ROOT / 'data/acceptance/16.17_development_20260911/source_plan.json'


def cache_only(path, url):
    if not path.exists():
        raise RuntimeError(f'Missing local cache {path}; networking forbidden')
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    started = time.monotonic()
    source_bytes = SOURCE.read_bytes()
    original = json.loads((BASE / 'raw_footwear_audit.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(source_bytes).hexdigest() == original['source_plan_sha256']
    plan = {r['match_id']: r for r in json.loads(source_bytes)['matches']}
    dd.load_json = cache_only
    dragon = dd.DataDragon('16.17.1')
    allowed_boot_receipts = {int(i) for i in dragon.item(2422).get('into', [])}
    totals = collections.Counter()
    records, examples, errors = [], [], []
    for original_game in original['games']:
        if time.monotonic() - started > 140:
            break
        row = plan[original_game['match_id']]
        mid = row['match_id']
        assert row['split'] in ('train', 'val') and row['legacy_split'] != 'test'
        try:
            match = raw_read(row['files']['match']['path'], 'match', mid)
            timeline = raw_read(row['files']['timeline']['path'], 'timeline', mid)
            assert match['info']['gameVersion'].startswith('16.17.')
            raw = sorted([e for f in timeline['info']['frames'] for e in f['events']], key=lambda e:e['timestamp'])
            kills = [e for e in raw if e['type'] == 'CHAMPION_KILL']
            intervals = []
            for p in match['info']['participants']:
                pid = p['participantId']
                runes = [r['perk'] for s in p['perks']['styles'] for r in s['selections']]
                if 8304 not in runes:
                    continue
                eligible, _, _ = causal_grant(kills, pid)
                groups = collections.defaultdict(list)
                for e in raw:
                    if e.get('participantId') == pid and e['type'].startswith('ITEM_'):
                        groups[e['timestamp']].append(e)
                receipt = None
                candidates = []
                for timestamp, batch in sorted(groups.items()):
                    if timestamp < eligible:
                        continue
                    consumed = any(e['type'] == 'ITEM_DESTROYED' and e.get('itemId') == 2422 for e in batch)
                    sold = any(e['type'] == 'ITEM_SOLD' and e.get('itemId') == 2422 for e in batch)
                    purchases = {e.get('itemId') for e in batch if e['type'] == 'ITEM_PURCHASED'}
                    if sold or (consumed and purchases & allowed_boot_receipts):
                        candidates.append({'timestamp':timestamp, 'batch':[brief(e) for e in batch]})
                        if receipt is None:
                            receipt = timestamp
                intervals.append({'participant_id':pid, 'champion':p['championName'],
                    'role':p.get('teamPosition'), 'runes':runes,
                    'eligible_ms':eligible, 'receipt_ms':receipt, 'receipt_candidates':candidates})
            _, visits = reconstruct_game(match, timeline, dragon)
            counts = collections.Counter()
            spanning = []
            for event in visits:
                timestamp, end = event['ts'], event.get('ts_end', event['ts'])
                unknown_board = [w for w in intervals if timestamp >= w['eligible_ms']
                                 and (w['receipt_ms'] is None or timestamp <= w['receipt_ms'])]
                crosses_own_unknown = [w for w in intervals if w['participant_id']==event['participant_id']
                    and end >= w['eligible_ms'] and (w['receipt_ms'] is None or timestamp <= w['receipt_ms'])]
                if unknown_board or crosses_own_unknown:
                    phase = 'MIDDLE_omit'
                elif timestamp < min(w['eligible_ms'] for w in intervals):
                    phase = 'BEFORE_potential'
                else:
                    phase = 'AFTER_potential'
                counts[phase] += 1
                if any(w['receipt_ms'] is not None and timestamp <= w['receipt_ms'] <= end
                       and w['participant_id']==event['participant_id'] for w in intervals):
                    spanning.append({'participant_id':event['participant_id'],'ts':timestamp,'ts_end':end,
                                     'phase':phase,'purchase_actions':event.get('purchase_actions')})
            key = row['split']+':'
            totals[key+'games'] += 1
            totals[key+'reconstructed_decisions'] += len(visits)
            totals.update({key+k:v for k,v in counts.items()})
            totals[key+'receipt_players'] += sum(w['receipt_ms'] is not None for w in intervals)
            totals[key+'footwear_players'] += len(intervals)
            totals[key+'receipt_spanning_own_decisions'] += len(spanning)
            raw_biscuit_players = sorted({e['participantId'] for e in raw if e.get('itemId')==2010})
            totals[key+'games_with_biscuit_events'] += bool(raw_biscuit_players)
            records.append({'match_id':mid,'split':row['split'],'counts':dict(counts),
                            'intervals':intervals,'receipt_spanning_own_decisions':spanning,
                            'players_with_raw_biscuit_events':raw_biscuit_players})
            if mid in ('EUW1_7970798064','EUW1_7971784474','EUW1_7968191358','EUW1_7978947600'):
                examples.append({'match_id':mid,'split':row['split'],
                    'relevant_raw_events':[brief(e) for e in raw if e['type'].startswith('ITEM_')
                        and (e.get('itemId') in (1201,1202,2010,2422,3009,3047,3111,3158,3171)
                             or e.get('beforeId') in (2422,3047,3111,3158,3171))]})
        except Exception as exc:
            errors.append({'match_id':mid,'error':repr(exc)})
    report = {'schema':1,'scope':original['scope'],
        'metric':'Potential existing reconstructed decisions outside causal footwear uncertainty intervals; economic and semantic correctness NOT established; zero admitted rows',
        'definition':'BEFORE: start before every eligibility and own outcome avoids uncertainty. MIDDLE: any player uncertain at decision start, or own outcome intersects own uncertainty. AFTER: remaining known starts after at least one eligibility. Every receipt timestamp remains excluded.',
        'limitation':'This computes potential coverage using existing reconstruction only. It does not repair missing components, gold, boot legality, secondary grants or prove receipt batch compatibility beyond direct static into relation.',
        'source_plan_sha256':original['source_plan_sha256'],'summary':dict(sorted(totals.items())),
        'games':records,'examples':examples,'errors':errors,'elapsed_seconds':time.monotonic()-started}
    out=BASE/'potential_window_coverage.json'
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(out),'summary':report['summary'],'errors':errors,
                     'elapsed_seconds':report['elapsed_seconds']},indent=2))


if __name__=='__main__':
    main()
