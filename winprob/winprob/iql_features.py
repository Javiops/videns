"""Public-input feature compiler for full-basket IQL; no source imports.

Only the explicit fields below are consumed. Legacy roles, total gold,
other-player wallets, target metadata and quest-derived fields are ignored.
Inventories are count-preserving ragged bags, never fixed-slot truncations.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

from winprob.ipo_residual import ACTION_NAMES, action_features

OBJECTIVES = ('towers', 'dragons', 'barons', 'heralds', 'voidgrubs')
INVENTORY_NAMES = ACTION_NAMES[:-1]
GLOBAL_NAMES = ('time_over_40min', 'is_blue', 'own_gold_est_over_3000',
                'own_cs_over_500', 'own_kills_over_20', 'own_deaths_over_20', 'own_assists_over_20') + tuple(
    f'{side}_{key}_over_10' for side in ('ally', 'enemy') for key in OBJECTIVES) + tuple(
    f'{side}_{key}_present' for side in ('ally', 'enemy') for key in OBJECTIVES) + tuple(
    f'{side}_item_value_{key}' for side in ('ally', 'enemy')
    for key in ('sum_over_30000', 'maximum_share', 'top_two_share', 'available')) + (
    'focal_allied_item_value_share', 'focal_share_available')


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < minimum:
        raise ValueError('invalid public scalar: '+name)
    return float(value)


class PublicFeatureCompiler:
    def __init__(self, patch_static):
        if set(patch_static) != {'16.17', '16.18', '16.19'}:
            raise ValueError('fixed three-patch public static union required')
        self.static = patch_static
        champion_ids = sorted({int(key) for data in patch_static.values() for key in data['champion_tags']})
        item_ids = sorted({int(key) for data in patch_static.values() for key in data['items']})
        self.champion_index = {item: i+1 for i, item in enumerate(champion_ids)}
        self.item_index = {item: i+1 for i, item in enumerate(item_ids)}
        self.item_attributes = {}
        for patch, data in patch_static.items():
            ids = sorted(map(int, data['items']))
            vectors = action_features([[[item, 1]] for item in ids], data['items'])[:, :-1]
            self.item_attributes[patch] = dict(zip(ids, vectors, strict=True))

    def metadata(self):
        return {'schema': 'iql-public-features-v1', 'patches': sorted(self.static),
                'champion_index': self.champion_index, 'item_index': self.item_index,
                'inventory_names': list(INVENTORY_NAMES), 'global_names': list(GLOBAL_NAMES),
                'vocabulary_source': 'fixed public static union only; no game labels or holdouts',
                'padding_index': 0, 'inventory_truncation': False}

    def inventory(self, items, patch):
        if not isinstance(items, list) or any(type(item) is not int or item <= 0 for item in items):
            raise ValueError('observed inventory must preserve positive item IDs')
        counts = Counter(items)
        if any(item not in self.item_attributes[patch] for item in counts):
            raise ValueError('inventory item absent from its native public static world')
        attrs = np.zeros(22, dtype=np.float64)
        for item, count in counts.items():
            attrs += count*self.item_attributes[patch][item]
        bag = tuple((self.item_index[item], count) for item, count in sorted(counts.items()))
        return bag, attrs

    def state(self, state, patch):
        if patch not in self.static or state.get('gold_est_version') != 'prequential-v3':
            raise ValueError('unbound native patch or wallet estimator')
        side = state['team_id']
        if type(side) is not int or side not in (100, 200):
            raise ValueError('invalid focal team')
        others = state['others']
        if not isinstance(others, list) or len(others) != 9:
            raise ValueError('complete ten-player roster required')
        allies = [row for row in others if row['team_id'] == side]
        enemies = [row for row in others if row['team_id'] in (100, 200) and row['team_id'] != side]
        if len(allies) != 4 or len(enemies) != 5:
            raise ValueError('four allies and five enemies required')
        ordered = [dict(champion_id=state['champion_id'], items=state['inventory'], level=state['level'])]
        ordered += sorted(allies, key=lambda row: row['champion_id'])
        ordered += sorted(enemies, key=lambda row: row['champion_id'])
        champions, bags, attributes, levels = [], [], [], []
        for player in ordered:
            champion = player['champion_id']
            if type(champion) is not int or champion not in self.champion_index:
                raise ValueError('champion absent from fixed public static union')
            level = number(player['level'], 'level', minimum=1)
            if level > 30:
                raise ValueError('invalid recorded level')
            bag, attrs = self.inventory(player['items'], patch)
            champions.append(self.champion_index[champion]); bags.append(bag)
            attributes.append(attrs); levels.append([level/18.])
        globals_ = [number(state['ts'], 'ts')/2400000., float(side == 100),
                    number(state['gold_est'], 'gold_est', minimum=-np.inf)/3000., number(state['cs'], 'cs')/500.] + [
                    number(state[key], key)/20. for key in ('kills', 'deaths', 'assists')]
        objectives = [state['ally_obj'], state['enemy_obj']]
        for obj in objectives:
            if not isinstance(obj, dict):
                raise ValueError('objective fields must be a mapping')
            globals_.extend(number(obj[key], key)/10. if key in obj else 0. for key in OBJECTIVES)
        for obj in objectives:
            globals_.extend(float(key in obj) for key in OBJECTIVES)
        values = np.asarray(attributes)[:, 0]*3000.
        for team_values in (values[:5], values[5:]):
            total = float(team_values.sum()); available = total > 0
            ordered_value = np.sort(team_values)
            globals_.extend([total/30000., float(ordered_value[-1]/total) if available else 0.,
                             float(ordered_value[-2:].sum()/total) if available else 0., float(available)])
        total = float(values[:5].sum())
        globals_.extend([float(values[0]/total) if total > 0 else 0., float(total > 0)])
        if len(globals_) != 37:
            raise AssertionError('public global schema changed')
        return {'champion_indices': np.asarray(champions, dtype=np.int32), 'inventory_bags': bags,
                'inventory_static': np.asarray(attributes, dtype=np.float32),
                'visible_level': np.asarray(levels, dtype=np.float32),
                'ally_mask': np.asarray([False]+[True]*4+[False]*5),
                'enemy_mask': np.asarray([False]*5+[True]*5),
                'global_public': np.asarray(globals_, dtype=np.float32)}

    def full_action(self, basket, kind, patch='16.18'):
        if patch not in self.static:
            raise ValueError('unbound native action patch')
        if kind == 'observed_no_buy_death':
            if basket:
                raise ValueError('no-buy catalogue action must have an empty purchase multiset')
            features = np.zeros(23, dtype=np.float32); features[-1] = 1.
            return (), features
        if kind != 'retained_net_purchase' or not basket:
            raise ValueError('invalid full-action semantic tag')
        counts = {str(item): count for item, count in basket}
        if len(counts) != len(basket) or any(type(count) is not int or count <= 0 for count in counts.values()):
            raise ValueError('noncanonical full-action counts')
        features = action_features([basket], self.static[patch]['items'])[0].astype(np.float32)
        bag = tuple((self.item_index[item], count) for item, count in basket)
        return bag, features
