"""Gate 2: source-bound replay plus independent whole-corpus semantic checks."""
import argparse
from contextlib import closing
from copy import deepcopy
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from app.dataset_artifacts import canonical_sha, sha256, write_json
from app.ddragon import dragon_for_patch
from app.reconstruct import GOLD_EST_VERSION, SLOT_STATE_VERSION, reconstruct_game
from app.riot import RawMatchArchive
from app.shop_econ import SAVE_ITEM, combine_cost, payload_ids, feature_inventory
from collect_patch_sample import validate_pair


OBJECTIVES = ('towers', 'dragons', 'barons', 'heralds', 'voidgrubs')


def quest_completion_times(timeline, item_id):
    result = {}
    for frame in timeline['info']['frames']:
        for event in frame.get('events', []):
            if event.get('type') == 'ITEM_DESTROYED' and event.get('itemId') == item_id:
                pid, ts = event['participantId'], event['timestamp']
                result[pid] = min(result.get(pid, ts), ts)
    return result


def slot_state_checks(event, bot_quests):
    """Check persisted slot inputs against raw quest times, not replay output."""
    anchor = event['ts'] - 15000 if event.get('is_save') else event['ts']
    for scope, state in [('shopper', event), *(('board', member) for member in event['board'])]:
        pid = state['participant_id']
        version = state.get('slot_state_version')
        yield version == SLOT_STATE_VERSION, scope + '_slot_state_version', {
            'participant_id':pid, 'actual':version, 'expected':SLOT_STATE_VERSION}
        completed_at = bot_quests.get(pid)
        expected = completed_at is not None and completed_at < anchor
        actual = state.get('bot_quest_complete')
        # Equality alone admits 0/1; unknown state is not observed incompletion.
        yield type(actual) is bool and actual is expected, scope + '_bot_quest_causal', {
            'participant_id':pid, 'anchor_ts':anchor, 'quest_completed_at':completed_at,
            'actual':actual, 'expected':expected}


def decision_state_binding(event, before=None):
    """The decision inventory is exactly the pre-visit inventory minus the
    declared opening sales, and the credited proceeds are non-negative and
    only ever present with sales. Returns (ok, detail)."""
    if before is None:
        before = Counter(payload_ids(event['inventory_before']))
    # An EMPTY decision inventory is a real state, not a missing field: on
    # 16.18 v4 one visit in 1,249,675 (KR_8388382904, ts 1368998) sold all
    # six items for 5255 g before buying. `payload or fallback` read that []
    # as absent, substituted the pre-sale inventory and failed correct data.
    # Only a payload predating the field falls back to the pre-sale snapshot.
    decision_payload = event.get('decision_inventory')
    decision = Counter(payload_ids(event['inventory_before'] if decision_payload is None else decision_payload))
    opening = Counter(int(i) for i in event.get('opening_sales') or [])
    sale_gold = event.get('opening_sale_gold') or 0
    # Proceeds WITHOUT a freed modeled slot are legitimate and common
    # (425/636,936 visits on 16.18 v3): selling an unmodeled consumable —
    # Biscuit 2010 at 5 g, potions — pays real gold the player held when
    # deciding, while the modeled inventory never carried the item, so no
    # slot opens. The converse is the real invariant: a freed slot must
    # have been paid for.
    ok = (decision == before - opening and opening <= before
          and isinstance(sale_gold, int) and sale_gold >= 0
          and (sum(opening.values()) == 0 or sale_gold > 0))
    detail = None if ok else {'opening_sales': sorted(opening.elements()), 'opening_sale_gold': sale_gold,
                              'inventory_before': sorted(before.elements()),
                              'decision_inventory': sorted(decision.elements())}
    return ok, detail


def terminal_inventory(ids, dragon):
    """DTO has no ward stack sizes; compare presence for wards, counts otherwise."""
    result = Counter(feature_inventory([i for i in ids if i and not dragon.classify(i)['skip']]))
    for item in (2055,772043):
        if result[item]: result[item] = 1
    return +result


def terminal_differences(match, inventories, dragon):
    differences = []
    for participant in match['info']['participants']:
        actual = [participant.get(f'item{i}') or 0 for i in range(7)]
        bound = participant.get('roleBoundItem') or 0
        cls = dragon.classify(bound)
        if bound not in actual and (bound in (2055,772043) or cls['is_boots'] or cls['is_basic_boots']):
            actual.append(bound)
        expected = terminal_inventory(actual,dragon)
        replayed = terminal_inventory(inventories[participant['participantId']],dragon)
        if expected != replayed:
            differences.append({'participant_id':participant['participantId'],
                'extra':sorted((replayed-expected).elements()),'missing':sorted((expected-replayed).elements())})
    return differences


def combat_context(timeline, participants):
    """Independent prefix counts: inhibitors must never increment tower count."""
    teams = {p['participantId']: p['teamId'] for p in participants}
    kda = {p: [0, 0, 0] for p in teams}
    objectives = {str(team): dict.fromkeys(OBJECTIVES, 0) for team in (100, 200)}
    times, snapshots = [-1], [(dict(kda), {k: dict(v) for k, v in objectives.items()})]
    for event in sorted((e for f in timeline['info']['frames'] for e in f.get('events', [])), key=lambda e: e.get('timestamp', 0)):
        kind = event.get('type')
        if kind not in ('CHAMPION_KILL', 'BUILDING_KILL', 'ELITE_MONSTER_KILL'):
            continue
        kda = {k: list(v) for k, v in kda.items()}
        if kind == 'CHAMPION_KILL':
            for pid, column in ((event.get('killerId'), 0), (event.get('victimId'), 1)):
                if pid in kda:
                    kda[pid][column] += 1
            for pid in event.get('assistingParticipantIds', []):
                if pid in kda:
                    kda[pid][2] += 1
        elif kind == 'BUILDING_KILL' and event.get('buildingType') == 'TOWER_BUILDING':
            defending = event.get('teamId')
            if defending in (100, 200):
                objectives[str(300 - defending)]['towers'] += 1
        elif kind == 'ELITE_MONSTER_KILL':
            team = teams.get(event.get('killerId')) or event.get('killerTeamId')
            metric = {'DRAGON': 'dragons', 'BARON_NASHOR': 'barons', 'RIFTHERALD': 'heralds', 'HORDE': 'voidgrubs'}.get(event.get('monsterType'))
            if team in (100, 200) and metric:
                objectives[str(team)][metric] += 1
        times.append(event.get('timestamp', 0))
        snapshots.append((kda, {k: dict(v) for k, v in objectives.items()}))
    return times, snapshots


def audit(corpus: Path, out: Path, *, recovered_checker=None, preflight=None, shard_index=None):
    if out.exists():
        raise ValueError('Immutable semantic audit already exists')
    audit_digest = sha256(Path(__file__))
    manifest = json.loads((corpus / 'corpus_manifest.json').read_text(encoding='utf-8'))
    state = json.loads((corpus / 'state.json').read_text(encoding='utf-8'))
    selected = None
    if preflight is not None:
        from audit_recovered_patch_corpus import validate_preflight
        receipt = validate_preflight(corpus, preflight)
        if recovered_checker is None or type(shard_index) is not int or not 0 <= shard_index < len(receipt['shards']):
            raise ValueError('A recovered audit shard needs a verified parent scope and checker')
        selected = set(receipt['shards'][shard_index])
    else:
        if shard_index is not None:
            raise ValueError('Audit sharding requires a verified parent scope')
        if sha256(corpus / 'tracker.db') != manifest['db_sha256']:
            raise ValueError('Frozen database hash changed')
        for relative, info in manifest['files'].items():
            if sha256(corpus / relative) != info['sha256']:
                raise ValueError(f'Frozen file changed: {relative}')
    if bool(manifest.get('recovery_version')) != bool(recovered_checker):
        raise ValueError('Recovered corpus requires the explicit recovered semantic auditor')
    for relative in ('app/reconstruct.py', 'app/inventory.py', 'app/shop_econ.py', 'app/ddragon.py', 'app/config.py'):
        if sha256(ROOT / relative) != manifest.get('reconstruction_code_sha256', manifest['collection_code_sha256'])[relative]:
            raise ValueError(f'Replay code differs from frozen code: {relative}')
    dragon = dragon_for_patch(manifest['patch'])
    if dragon.signature != manifest['static_data_sha256']:
        raise ValueError('Static data mismatch')
    inventory = {r['match_id']: r for r in (json.loads(line) for line in (corpus / 'source_inventory.jsonl').read_text().splitlines())}
    archive = RawMatchArchive(corpus / 'raw')
    counts, diagnostic = Counter(), Counter()
    examples = defaultdict(list)
    negative_budget_gaps = []
    accepted, exclusions = [], {}
    started = time.monotonic()
    def check(condition, rule, mid, event=None, detail=None):
        counts['checks:' + rule] += 1
        if not condition:
            counts['failed:' + rule] += 1
            if len(examples[rule]) < 12:
                examples[rule].append({'match_id': mid, 'event_index': (event or {}).get('event_index'),
                    'ts': (event or {}).get('ts'), 'detail': detail})
    if 'slot_state_version' in manifest:
        check(manifest['slot_state_version'] == SLOT_STATE_VERSION, 'manifest_slot_state_version', '',
            detail={'actual':manifest['slot_state_version'], 'expected':SLOT_STATE_VERSION})
    with closing(sqlite3.connect((corpus / 'tracker.db').as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        check(receipt['sqlite_integrity'] if preflight else [r[0] for r in conn.execute('PRAGMA integrity_check')] == ['ok'], 'sqlite_integrity', '')
        check(receipt['foreign_keys'] if preflight else not list(conn.execute('PRAGMA foreign_key_check')), 'foreign_keys', '')
        games = list(conn.execute('SELECT * FROM games ORDER BY match_id'))
        check({r['match_id'] for r in games} == set(inventory), 'source_inventory_ids', '')
        if selected is not None:
            if not selected or not selected <= set(inventory):
                raise ValueError('Audit shard contains empty or unknown membership')
            games = [game for game in games if game['match_id'] in selected]
        for index, game in enumerate(games, 1):
            mid = game['match_id']
            counts['games'] += 1
            entry = inventory[mid]
            sources = {}
            for kind, meta in entry['files'].items():
                check(sha256(corpus / meta['path']) == meta['sha256'], 'raw_file_hash', mid)
                sources[kind] = archive.read(kind, mid)
                check(canonical_sha(sources[kind]) == meta['payload_sha256'], 'raw_payload_hash', mid)
            match, timeline = sources['match'], sources['timeline']
            validate_pair(mid, match, timeline, manifest['patch'])
            participants = match['info']['participants']
            ids = {p['participantId'] for p in participants}
            actual_perspectives = {r[0] for r in conn.execute('SELECT puuid FROM matches WHERE match_id=?', (mid,))}
            check(actual_perspectives == {p['puuid'] for p in participants}, 'ten_perspectives', mid)
            rows = list(conn.execute('SELECT event_index,ts,ts_end,payload_json FROM game_events WHERE match_id=? ORDER BY event_index', (mid,)))
            stored = [json.loads(r['payload_json']) for r in rows]
            debug = {}
            replay_game, fresh = reconstruct_game(match, timeline, dragon, debug)
            check(canonical_sha(stored) == canonical_sha(fresh), 'source_replay_equal', mid)
            check(all(game[k] == replay_game[k] for k in ('patch','game_version','game_creation','game_duration','participants_json')), 'game_source_equal', mid)
            # Final DTO data is valid as a QA anchor, never as an input. Mutate
            # it and prove every reconstructed decision remains unchanged.
            mutated = deepcopy(match)
            for p in mutated['info']['participants']:
                p.update({f'item{i}':3869 for i in range(7)})
                p.update(kills=999,deaths=999,assists=999,win=not p['win'],roleBoundItem=3175)
                if recovered_checker:
                    for style in p.get('perks', {}).get('styles', []):
                        for rune in style.get('selections', []):
                            rune.update(var1=999,var2=999,var3=999)
            check(canonical_sha(fresh) == canonical_sha(reconstruct_game(mutated,timeline,dragon)[1]), 'final_dto_invariance',mid)
            if recovered_checker:
                for condition, rule, detail in recovered_checker(match, timeline, stored, fresh, debug, dragon):
                    check(condition, rule, mid, detail=detail)
            opaque = any(e.get('type') == 'ITEM_UNDO' and not e.get('beforeId') and not e.get('afterId')
                for f in timeline['info']['frames'] for e in f.get('events',[]))
            differences = terminal_differences(match,debug['final_inventories'],dragon)
            reasons = []
            if opaque: reasons.append('undo_without_item_ids')
            if differences:
                reasons.append('unreconciled_terminal_inventory')
                diagnostic['terminal_inventory_mismatched_games'] += 1
                if len(examples['terminal_inventory_mismatch']) < 12:
                    examples['terminal_inventory_mismatch'].append({'match_id':mid,'differences':differences})
            if reasons: exclusions[mid] = reasons
            else: accepted.append(mid)
            combat_times, snapshots = combat_context(timeline, participants)
            raw_items = defaultdict(list)
            mid_quests = quest_completion_times(timeline, 1201)
            bot_quests = quest_completion_times(timeline, 1202)
            for frame in timeline['info']['frames']:
                for item in frame.get('events', []):
                    if item.get('type') == 'ITEM_PURCHASED':
                        raw_items[item.get('participantId')].append(item)
            for event in stored:
                counts['shop_events'] += 1
                check(event.get('training_eligible') == (not opaque)
                    and event.get('source_quality_issues') == (['undo_without_item_ids'] if opaque else [])
                    and event.get('inventory_version') == manifest['inventory_version'], 'source_quality_binding',mid,event)
                check(event.get('patch') == manifest['patch'] and event.get('ddragon_version') == dragon.version
                    and event.get('static_data_sha256') == dragon.signature and event.get('reconstruction_version') == manifest['reconstruction_version'], 'event_provenance', mid, event)
                t, end, pid = event['ts'], event['ts_end'], event['participant_id']
                check(0 <= t <= end <= match['info']['gameDuration'] * 1000 + 2000, 'visit_time_range', mid, event)
                check(event.get('gold_est_version') == GOLD_EST_VERSION and isinstance(event.get('gold_est'), (int,float))
                    and math.isfinite(event['gold_est']) and event['gold_est'] >= 0
                    and event.get('gold_gap', {}).get('frame_ts', t + 1) < t, 'causal_gold_binding', mid, event)
                board = event['board']
                check(len(board) == 10 and {p['participant_id'] for p in board} == ids and sum(bool(p.get('is_self')) for p in board) == 1, 'board_membership', mid, event)
                for condition, rule, detail in slot_state_checks(event, bot_quests):
                    check(condition, rule, mid, event, detail)
                self_row = next(p for p in board if p.get('is_self'))
                before, after = Counter(payload_ids(event['inventory_before'])), Counter(payload_ids(event['inventory_after']))
                bought, consumed = Counter(payload_ids(event['bought'])), Counter(payload_ids(event['consumed']))
                check(Counter(payload_ids(self_row['items'])) == before and self_row['participant_id'] == pid, 'board_self_inventory', mid, event)
                check(bought == after - before and consumed == before - after, 'inventory_label_algebra', mid, event)
                # Decision-state binding (the transition the old snapshot gate
                # missed); the rule and its two documented exceptions live in
                # decision_state_binding() so a test can pin them.
                ok, detail = decision_state_binding(event, before)
                check(ok, 'decision_state_binding', mid, event, detail)
                check(all(bool(dragon.item(i)) for i in before | after), 'known_item_ids', mid, event)
                # Historic saves intentionally use the death snapshot, 15s before
                # their synthetic observation time. They are not manual recalls.
                anchor = t - 15000 if event.get('is_save') else t
                low = snapshots[bisect_left(combat_times, anchor) - 1]
                high = snapshots[bisect_right(combat_times, anchor) - 1]
                for member in board:
                    member_id = member['participant_id']
                    check(all(low[0][member_id][k] <= (member.get(field) or 0) <= high[0][member_id][k]
                              for k, field in enumerate(('kills','deaths','assists'))), 'board_kda_prefix', mid, event)
                check(all(low[0][pid][k] <= (event.get(field) or 0) <= high[0][pid][k]
                          for k, field in enumerate(('kills','deaths','assists'))), 'shopper_kda_prefix', mid, event)
                score = event.get('score', {})
                for team in ('100', '200'):
                    for objective in OBJECTIVES:
                        actual = score.get(team, {}).get(objective, 0)
                        check(low[1][team][objective] <= actual <= high[1][team][objective], 'objective_' + objective, mid, event,
                            {'team':team, 'actual':actual, 'expected_before':low[1][team][objective], 'expected_inclusive':high[1][team][objective]})
                if event.get('is_save'):
                    counts['no_buy_death'] += 1
                    check(not bought and before == after and not any(anchor < e['timestamp'] <= anchor + 90000 for e in raw_items[pid]), 'save_label_window', mid, event)
                else:
                    source_buys = [e['itemId'] for e in raw_items[pid] if t <= e['timestamp'] <= end]
                    if mid_quests.get(pid, float('inf')) <= end:
                        source_buys = [dragon.free_boot_upgrades().get(i,i) for i in source_buys]
                    purchases = Counter(source_buys)
                    if 'label_bought' in event:
                        actions = Counter(event.get('purchase_actions',[]))
                        check(not (actions-purchases) and Counter(payload_ids(event['label_bought'])) == bought & actions
                            and event.get('label_source_version') == 'purchase-net-v1', 'purchase_backed_labels', mid, event)
                    inv = before.copy()
                    cost = 0
                    for item in payload_ids(event.get('label_bought',event['bought'])):
                        cost += combine_cost(item, inv, dragon, consume=True)
                        inv[item] += 1
                    counts['purchase_context_events'] += 1
                    if cost > event['gold_est']:
                        diagnostic['basket_cost_above_estimated_wallet'] += 1
                        negative_budget_gaps.append(cost - event['gold_est'])
                    purchased = {e['itemId'] for e in raw_items[pid] if t <= e['timestamp'] <= end}
                    gained_unpurchased = set(bought) - purchased
                    if gained_unpurchased:
                        diagnostic['net_inventory_gain_without_purchase_event'] += 1
                        if len(examples['net_inventory_gain_without_purchase_event']) < 12:
                            examples['net_inventory_gain_without_purchase_event'].append({'match_id':mid,'ts':t,'items':sorted(gained_unpurchased)})
            if index % 100 == 0:
                print(f'Audited {index}/{len(games)} games, {counts["shop_events"]} shops', flush=True)
    failures = {k[7:]: v for k, v in counts.items() if k.startswith('failed:')}
    counts['accepted_games_before_duration_filter'] = len(accepted)
    counts['quarantined_games'] = len(exclusions)
    if sha256(Path(__file__)) != audit_digest:
        raise ValueError('Validator changed while the audit was running')
    passed = not failures and bool(accepted)
    report = {'schema':1, 'corpus':str(corpus), 'corpus_manifest_sha256':sha256(corpus / 'corpus_manifest.json'),
        'db_sha256':manifest['db_sha256'], 'audit_code_sha256':audit_digest,
        'slot_state_version':SLOT_STATE_VERSION,
        'finished_utc':datetime.now(timezone.utc).isoformat(), 'elapsed_seconds':time.monotonic()-started,
        'counts':dict(counts), 'hard_failures':failures, 'diagnostics':dict(diagnostic), 'examples':dict(examples),
        'passed':passed, 'training_clearance':passed and selected is None,
        'clearance_scope':'accepted_match_ids only; the source collection as a whole is not certified',
        'accepted_match_ids':sorted(accepted), 'quarantined_match_ids':exclusions,
        'limits':['Replay agreement proves reproducibility, not independent correctness.',
                  'Basket cost above the approximate wallet is diagnostic, not proof of an impossible historic buy.',
                  'Combat/objective timestamp ties permit either boundary count; slot completion requires a strictly earlier raw quest marker.',
                  'Terminal inventory compares modeled items, role-slot boots, and ward presence; the DTO omits ward stack sizes.',
                  'Unreconciled final inventories are excluded, not silently repaired from future DTO slots.',
                  'Exact live-wallet accuracy and unobserved manual recalls are not measured.']}
    if preflight:
        report.update(audit_scope='recovered_shard', shard_index=shard_index,
            shard_match_ids=sorted(selected), preflight_sha256=sha256(preflight),
            recovery_version=manifest['recovery_version'], decision_admission_sha256=manifest['decision_admission_sha256'],
            recovered_audit_code_sha256=receipt['recovered_audit_code_sha256'])
    if negative_budget_gaps:
        negative_budget_gaps.sort()
        report['budget_shortfall_diagnostic'] = {'n':len(negative_budget_gaps),
            'median_gold':negative_budget_gaps[len(negative_budget_gaps)//2],
            'p95_gold':negative_budget_gaps[int(len(negative_budget_gaps)*.95)]}
    write_json(out, report)
    # Preserve the exact validator used, independently of future repository edits.
    import shutil
    shutil.copy2(Path(__file__), out.parent / 'audit_code.py')
    print(json.dumps({'report':str(out), 'games':counts['games'], 'shops':counts['shop_events'], 'passed':passed,
        'accepted_games':len(accepted),'quarantined_games':len(exclusions), 'hard_failures':failures, 'diagnostics':dict(diagnostic)}, indent=2), flush=True)
    return passed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(0 if audit(args.corpus.resolve(), args.out.resolve()) else 2)
