"""Bounded independent source transaction audit for quarantined Viego candidates.

This deliberately does not import reconstruction or InventoryReplay. It accepts
only directly observed, reconciled transactions plus narrow documented non-shop
events. Unsupported rules fail closed; it is not a general inventory rebuilder.
Final DTO bags are QA outputs only, never ledger inputs. No corpus is modified.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.ddragon import dragon_for_patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bag_list(bag):
    return sorted((+bag).elements())


def load_source(raw, endpoint, mid):
    path = raw / endpoint / (mid + '.json.gz')
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        envelope = json.load(handle)
    if envelope['endpoint'] != endpoint or envelope['match_id'] != mid:
        raise ValueError('Raw envelope identity mismatch')
    return envelope['payload'], {'path': str(path), 'sha256': digest(path)}


def expected_recipe(item, inventory, dragon):
    """Independent recursive payment/consumption calculation using pinned data."""
    available = inventory.copy()
    used = Counter()

    def acquire(component, root=False, ancestors=()):
        if component in ancestors:
            raise ValueError('Cyclic recipe')
        if not root and available[component]:
            available[component] -= 1
            used[component] += 1
            return 0
        data = dragon.item(component)
        if not data:
            raise ValueError('Unknown item ' + str(component))
        children = [int(x) for x in data.get('from', [])]
        if not children:
            return int(data['gold']['total'])
        return int(data['gold']['base']) + sum(
            acquire(child, ancestors=ancestors + (component,)) for child in children)

    return acquire(item, root=True), used


def audit_one(raw, mid, dragon):
    match, match_source = load_source(raw, 'match', mid)
    timeline, timeline_source = load_source(raw, 'timeline', mid)
    viegos = [p for p in match['info']['participants'] if p['championId'] == 234]
    if len(viegos) != 1:
        raise ValueError('Expected one Viego')
    owner = viegos[0]['participantId']
    role = viegos[0].get('teamPosition') or viegos[0].get('individualPosition')
    events = []
    for fi, frame in enumerate(timeline['info']['frames']):
        for ei, event in enumerate(frame.get('events', [])):
            if event.get('participantId') == owner and event['type'].startswith('ITEM_'):
                events.append((event['timestamp'], fi, ei, event))
            elif event.get('creatorId') == owner and event['type'] == 'WARD_PLACED':
                events.append((event['timestamp'], fi, ei, event))
    events.sort(key=lambda row: row[:3])
    batches = {}
    for ts, fi, ei, event in events:
        batches.setdefault(ts, []).append((fi, ei, event))
    inventory = Counter()
    paid_history = {}
    ledger, failures, rule_counts = [], [], Counter()
    top_quest_observed = False

    def issue(ts, code, **detail):
        failures.append({'timestamp': ts, 'code': code, **detail})

    for ts, entries in batches.items():
        purchases = [e for _, _, e in entries if e['type'] == 'ITEM_PURCHASED']
        destroys = Counter(e.get('itemId') for _, _, e in entries if e['type'] == 'ITEM_DESTROYED')
        pink_placements = [e for _, _, e in entries
                           if e['type'] == 'WARD_PLACED' and e.get('wardType') == 'CONTROL_WARD']
        batch_before = inventory.copy()
        recipe_used, purchase_cost = Counter(), None
        if len(purchases) > 1:
            issue(ts, 'unsupported_multiple_simultaneous_purchases')
        elif purchases:
            purchase = purchases[0]['itemId']
            try:
                purchase_cost, recipe_used = expected_recipe(purchase, batch_before, dragon)
            except ValueError as exc:
                issue(ts, 'recipe_error', detail=str(exc))
            if destroys != recipe_used:
                issue(ts, 'purchase_destroy_multiset_mismatch',
                      expected=bag_list(recipe_used), observed=bag_list(destroys), purchased=purchase)
        if len(pink_placements) > 1:
            issue(ts, 'multiple_same_timestamp_control_ward_placements')

        for fi, ei, event in entries:
            before = inventory.copy()
            kind = event['type']
            item = event.get('itemId') or 0
            reason = 'unsupported_event'
            detail = {}
            if kind == 'ITEM_PURCHASED':
                inventory[item] += 1
                reason = 'observed_purchase'
                detail = {'expected_paid_gold': purchase_cost,
                          'expected_owned_components': bag_list(recipe_used)}
                paid_history.setdefault(item, []).append({
                    'timestamp': ts, 'components': recipe_used.copy(), 'price': purchase_cost})
            elif kind == 'ITEM_SOLD':
                if inventory[item] <= 0:
                    issue(ts, 'sale_of_unowned_item', item_id=item)
                else:
                    inventory[item] -= 1
                reason = 'observed_sale'
            elif kind == 'ITEM_UNDO':
                removed, restored = event.get('beforeId') or 0, event.get('afterId') or 0
                histories = paid_history.get(removed) or []
                transaction = histories.pop() if histories else None
                if not removed or transaction is None:
                    issue(ts, 'unsupported_or_unlinked_undo', before_id=removed, after_id=restored)
                if removed and inventory[removed] <= 0:
                    issue(ts, 'undo_of_unowned_item', item_id=removed)
                elif removed:
                    inventory[removed] -= 1
                components = transaction['components'] if transaction else Counter()
                if restored and not components[restored]:
                    issue(ts, 'undo_after_id_not_in_original_components', item_id=restored)
                if transaction and event.get('goldGain') != transaction['price']:
                    issue(ts, 'undo_refund_mismatch', expected=transaction['price'], observed=event.get('goldGain'))
                inventory.update(components)
                reason = 'undo_linked_observed_purchase'
                detail = {'purchase_timestamp': transaction['timestamp'] if transaction else None,
                          'restored_components': bag_list(components), 'gold_gain': event.get('goldGain')}
            elif kind == 'ITEM_DESTROYED':
                data = dragon.item(item) or {}
                if item == 1200 and role == 'TOP' and not top_quest_observed and not purchases:
                    # The role quest is an implicit role-bound marker, not a
                    # permanently owned modeled build item or a missing grant.
                    top_quest_observed = True
                    reason = 'own_top_role_quest_marker_outside_build_inventory'
                else:
                    if inventory[item] <= 0:
                        issue(ts, 'destroy_of_unowned_item_or_missing_grant', item_id=item)
                    else:
                        inventory[item] -= 1
                    if purchases and recipe_used[item]:
                        reason = 'observed_component_consumption'
                    elif item in (2055, 772043) and len(pink_placements) == 1:
                        reason = 'control_ward_consumption_with_paired_placement'
                    elif item not in (2055, 772043) and data.get('consumed') and 'Consumable' in data.get('tags', []):
                        reason = 'observed_consumable_use'
                    else:
                        issue(ts, 'unexplained_nonshop_destruction', item_id=item,
                              name=data.get('name'), modeled=not dragon.classify(item)['skip'])
                        reason = 'unexplained_nonshop_destruction'
            elif kind == 'WARD_PLACED':
                if event.get('wardType') != 'CONTROL_WARD':
                    reason = 'trinket_placement_does_not_consume_build_item'
                elif destroys[2055] + destroys[772043]:
                    reason = 'placement_already_accounted_by_same_timestamp_destroy'
                else:
                    held = next((ward for ward in (2055, 772043) if inventory[ward] > 0), None)
                    if held is None:
                        issue(ts, 'control_ward_placement_without_owned_ward')
                    else:
                        inventory[held] -= 1
                    reason = 'control_ward_consumption_from_placement'
            else:
                issue(ts, 'unsupported_event_type', event_type=kind)
            rule_counts[reason] += 1
            ledger.append({'timestamp': ts, 'frame_index': fi, 'frame_event_index': ei,
                           'raw_event': event, 'before_all_observed_items': bag_list(before),
                           'after_all_observed_items': bag_list(inventory), 'rule': reason, **detail})

    # Outcome fields are consulted only after completing every causal ledger
    # transition; no outcome-dependent corrections can enter the ledger above.
    terminal = [viegos[0].get('item' + str(slot)) or 0 for slot in range(7)]
    reference_modeled = Counter(i for i in bag_list(inventory) if not dragon.classify(i)['skip'])
    terminal_modeled = Counter(i for i in terminal if i and not dragon.classify(i)['skip'])
    if reference_modeled != terminal_modeled:
        issue(events[-1][0] if events else 0, 'terminal_modeled_inventory_mismatch',
              extra=bag_list(reference_modeled - terminal_modeled),
              missing=bag_list(terminal_modeled - reference_modeled))
    return {
        'match_id': mid, 'participant_id': owner, 'role': role,
        'sources': {'match': match_source, 'timeline': timeline_source},
        'raw_item_event_count': sum(e['type'].startswith('ITEM_') for _, _, _, e in events),
        'ward_placement_event_count': sum(e['type'] == 'WARD_PLACED' for _, _, _, e in events),
        'audited_transition_count': len(ledger), 'rule_counts': dict(rule_counts),
        'passed': not failures, 'failures': failures,
        'final_reference_modeled_items': bag_list(reference_modeled),
        'terminal_dto_modeled_items_qa_only': bag_list(terminal_modeled),
        'ledger': ledger,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--match-id', action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(args.match_id) > 4:
        parser.error('This is a bounded audit: at most four source games')
    if args.out.exists():
        raise ValueError('Evidence already exists; choose a new output path')
    dragon = dragon_for_patch('16.18')
    report = {
        'schema': 1, 'purpose': 'independent_candidate_diagnostic_only',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'split': 'quarantined_source_games_not_in_frozen_export',
        'metric': 'fully_explained_causal_viego_inventory_transactions',
        'auditor_source': str(Path(__file__).resolve()), 'auditor_sha256': digest(Path(__file__)),
        'static_data_version': dragon.version, 'static_data_signature': dragon.signature,
        'n_games': len(args.match_id),
        'production_hold_cleared': False, 'training_clearance': False,
        'method_limits': [
            'This is a deliberately narrow rejecting reference ledger, not a possession repair.',
            'No reconstruction, InventoryReplay, combine_cost, or terminal-based ledger inference is used.',
            'Unsupported implicit grants, automatic transforms, sale undos, quest types, or simultaneous multi-purchases fail closed.',
            'Top quest1200 is accepted only once for a TOP participant as an unmodeled role marker.',
            'One Viego inventory audit does not certify the other nine players or freeze a benchmark.',
        ],
        'games': [audit_one(args.raw.resolve(), mid, dragon) for mid in args.match_id],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.out.resolve()), 'n_games': report['n_games'],
                      'games': [{k: game[k] for k in ('match_id', 'passed', 'audited_transition_count', 'rule_counts')}
                                | {'failure_count': len(game['failures'])} for game in report['games']]}, indent=2))


if __name__ == '__main__':
    main()
