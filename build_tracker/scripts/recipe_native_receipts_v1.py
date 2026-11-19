"""Source-native receipt packaging with the same optional-field semantics as M2."""
from collections import Counter


def execution_arguments(row):
    """Mirror purchase_sequence/econ_features, including their absent-field defaults."""
    return {'role': row.get('role') or '', 'bot_quest_complete': row.get('bot_quest_complete'),
            'unmodeled_regular_slots': row.get('unmodeled_regular_slots') or 0,
            'role_slot_inventory_unknown': bool(row.get('role_slot_inventory_unknown'))}


def replay_candidate(row, action, dragon):
    from app.conditional_policy import footwear_arguments, purchase_sequence
    from app.shop_econ import try_purchase
    before = list(row['inventory']); wallet = float(row['gold_est'])
    seq = purchase_sequence(tuple(action), row, wallet, dragon) if action else []
    if seq is None:
        raise ValueError('candidate must be source-native feasible')
    target = Counter(action); pending = target.copy(); inventory = Counter(before); steps = []
    footwear = footwear_arguments(row, dragon, required_version=dragon.footwear_state_version)
    for native in seq:
        iid = native['item_id']
        result = try_purchase(iid, list(inventory.elements()), wallet, dragon,
                              protected=target-pending, **execution_arguments(row), **footwear)
        if result is None:
            raise ValueError('native replay diverged from search')
        cost, after = result
        if abs(cost-native['cost']) >= 1e-8:
            raise ValueError('native replay cost changed')
        steps.append({'item_id': iid, 'cost': float(cost), 'inventory_after': sorted(after.elements())})
        inventory = after; wallet -= cost; pending[iid] -= 1
        if not pending[iid]:
            del pending[iid]
    if pending:
        raise ValueError('native replay dropped a retained purchase')
    return {'before': before, 'action': list(action), 'wallet': float(row['gold_est']), 'steps': steps}
