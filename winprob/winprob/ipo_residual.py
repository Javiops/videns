"""Small finite-action residual learners; no source loading or inference imports.

Real callers must establish canonical M2/context and population gates first.
The functions also support synthetic tests, which make no claim about M2.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

CHAMPION_TAGS = ('Assassin', 'Fighter', 'Mage', 'Marksman', 'Support', 'Tank')
ACTION_TAGS = ('AttackSpeed', 'Armor', 'SpellBlock', 'Health', 'Damage', 'SpellDamage',
               'Boots', 'CooldownReduction', 'LifeSteal', 'MagicPenetration')
ACTION_STATS = (
    ('FlatHPPoolMod', 1000.), ('FlatMPPoolMod', 1000.), ('FlatPhysicalDamageMod', 100.),
    ('FlatMagicDamageMod', 100.), ('FlatArmorMod', 100.), ('FlatSpellBlockMod', 100.),
    ('FlatMovementSpeedMod', 100.), ('PercentAttackSpeedMod', .5),
    ('FlatCritChanceMod', .5), ('PercentLifeStealMod', .3),
)
CONTEXT_NAMES = ('intercept', 'time_min', 'is_blue', 'own_level') + tuple(
    f'{side}_{tag}' for side in ('own', 'allies', 'enemies') for tag in CHAMPION_TAGS)
ACTION_NAMES = ('full_static_price_over_3000',) + tuple(key for key, _ in ACTION_STATS) + tuple(
    'tag_'+tag for tag in ACTION_TAGS) + ('item_count_over_6', 'save')


def finite_matrix(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError(name+' must be a finite nonempty matrix')
    return array


def normalized_weights(values, n):
    weights = np.asarray(values, dtype=float)
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('strictly positive finite row weights required')
    return weights/weights.sum()


def indices(values, n, k, name):
    raw = np.asarray(values)
    if raw.shape != (n,) or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw >= k):
        raise ValueError(name+' must contain valid integer action indices')
    return raw.astype(int, copy=False)


def context_features(rows, patch_champion_tags):
    result = []
    for row in rows:
        projection = row['source_projection']
        side = projection['team_id']
        if type(side) is not int or side not in (100, 200):
            raise ValueError('canonical team_id required')
        time_ms, level = row['context_timestamp_ms'], projection['level']
        if (isinstance(time_ms, bool) or not isinstance(time_ms, (int, float)) or not np.isfinite(time_ms) or time_ms < 0
                or isinstance(level, bool) or not isinstance(level, (int, float)) or not np.isfinite(level) or not 1 <= level <= 30):
            raise ValueError('valid pre-A timestamp and own level required')
        tags = patch_champion_tags[row['patch']]
        own = int(projection['champion_id'])
        others = projection['others']
        allies = [int(p['champion_id']) for p in others if p['team_id'] == side]
        enemies = [int(p['champion_id']) for p in others if p['team_id'] in (100, 200) and p['team_id'] != side]
        if len(others) != 9 or len(allies) != 4 or len(enemies) != 5:
            raise ValueError('canonical four-allies/five-enemies roster required')
        values = [1., time_ms/60000., float(side == 100), float(level)]
        for players in ([own], allies, enemies):
            if any(str(cid) not in tags for cid in players):
                raise ValueError('champion absent from pinned immutable tags')
            values.extend(sum(tag in tags[str(cid)] for cid in players) for tag in CHAMPION_TAGS)
        result.append(values)
    return finite_matrix(result, 'context')


def action_features(group_main, pinned_items):
    result = []
    for group in group_main:
        vector = np.zeros(len(ACTION_NAMES), dtype=float)
        total_count = 0
        for raw_item, raw_count in group:
            item, count = int(raw_item), int(raw_count)
            if isinstance(raw_count, bool) or count != raw_count or count <= 0 or str(item) not in pinned_items:
                raise ValueError('canonical positive multiset count and known item required')
            obj = pinned_items[str(item)]
            price = obj.get('gold', {}).get('total')
            if isinstance(price, bool) or not isinstance(price, (int, float)) or not np.isfinite(price) or price < 0:
                raise ValueError('valid pinned static item price required')
            vector[0] += count*price/3000.
            for j, (key, scale) in enumerate(ACTION_STATS, start=1):
                value = obj.get('stats', {}).get(key, 0)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError('nonfinite or nonnumeric declared item stat')
                vector[j] += count*value/scale
            offset = 1+len(ACTION_STATS)
            for j, tag in enumerate(ACTION_TAGS):
                vector[offset+j] += count*(tag in obj.get('tags', []))/6.
            total_count += count
        vector[-2] = total_count/6.
        vector[-1] = float(total_count == 0)
        result.append(vector)
    return finite_matrix(result, 'actions')


@dataclass(frozen=True)
class ContextStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, context, weights):
        context = finite_matrix(context, 'context')
        if not np.array_equal(context[:, 0], np.ones(len(context))):
            raise ValueError('first context coordinate must be the fixed intercept')
        weight = normalized_weights(weights, len(context))
        mean = weight @ context
        variance = weight @ ((context-mean)**2)
        scale = np.sqrt(np.maximum(variance, 0))
        scale[scale < 1e-12] = 1.
        mean[0], scale[0] = 0., 1.
        return cls(mean, scale)

    def transform(self, context):
        context = finite_matrix(context, 'context')
        if context.shape[1] != len(self.mean) or not np.array_equal(context[:, 0], np.ones(len(context))):
            raise ValueError('context schema/intercept mismatch')
        return (context-self.mean)/self.scale


def pair_design(context, actions, preferred, rejected):
    context, actions = finite_matrix(context, 'context'), finite_matrix(actions, 'actions')
    plus = indices(preferred, len(context), len(actions), 'preferred')
    minus = indices(rejected, len(context), len(actions), 'rejected')
    # No centering, learned offset or orientation-dependent scaling of D.
    return np.einsum('ni,nj->nij', context, actions[plus]-actions[minus]).reshape(len(context), -1)


def residual_scores(context, actions, theta):
    context, actions = finite_matrix(context, 'context'), finite_matrix(actions, 'actions')
    theta = finite_matrix(theta, 'theta')
    if theta.shape != (context.shape[1], actions.shape[1]):
        raise ValueError('bilinear parameter shape mismatch')
    return context @ theta @ actions.T


def canonical_reference(reference):
    reference = finite_matrix(reference, 'reference')
    if np.any(reference < 0) or np.any(reference.sum(axis=1) <= 0):
        raise ValueError('nonnegative reference with positive row mass required')
    if not np.allclose(reference.sum(axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError('reference rows must already be normalized within numeric tolerance')
    return reference/reference.sum(axis=1, keepdims=True)


def residual_policy(context, actions, theta, reference):
    original = np.asarray(reference)
    base = canonical_reference(reference)
    score = residual_scores(context, actions, theta)
    if base.shape != score.shape:
        raise ValueError('reference/action shape mismatch')
    if not np.any(theta):
        # Source native decoder parity is bound to these exact float32 masses.
        # A mathematically harmless renormalization can alter finite-precision
        # ties, so the zero residual must not perform any numeric roundtrip.
        return original.copy()
    logits = np.full(base.shape, -np.inf)
    positive = base > 0
    logits[positive] = np.log(base[positive])+score[positive]
    return np.exp(logits-logsumexp(logits, axis=1, keepdims=True))


def fit_ipo_ridge(context, actions, preferred, rejected, weights, tau=.5, ridge=.01, target_sign=1.):
    context, actions = finite_matrix(context, 'context'), finite_matrix(actions, 'actions')
    if context.shape[1]*actions.shape[1] > 768:
        raise ValueError('bounded adapter exceeds 768 parameters')
    if not np.isfinite(tau) or tau <= 0 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('positive finite tau and ridge required')
    design = pair_design(context, actions, preferred, rejected)
    weight = normalized_weights(weights, len(context))
    signs = np.broadcast_to(np.asarray(target_sign, dtype=float), (len(context),))
    if not np.isin(signs, [-1, 1]).all():
        raise ValueError('target signs must be +1 or -1')
    normal = tau*(design.T @ (weight[:, None]*design))+ridge*np.eye(design.shape[1])
    rhs = .5*(design.T @ (weight*signs))
    flat = np.linalg.solve(normal, rhs)
    error = design @ flat-signs/(2*tau)
    gradient = 2*(tau*design.T @ (weight*error)+ridge*flat)
    theta = flat.reshape(context.shape[1], actions.shape[1])
    return theta, {'objective': float(tau*np.sum(weight*error**2)+ridge*np.sum(flat**2)),
                   'gradient_max_abs': float(np.max(np.abs(gradient))), 'parameters': len(flat),
                   'design_rank': int(np.linalg.matrix_rank(design)), 'design_rows': len(design),
                   'tau': float(tau), 'ridge': float(ridge), 'solver': 'exact_weighted_ridge',
                   'pair_difference_centered': False}


def winner_objective(flat, context, actions, reference, winners, weights, ridge=.01, batch_size=128):
    """True full-action retained-only CE; no rejected-action argument exists."""
    context, actions = finite_matrix(context, 'context'), finite_matrix(actions, 'actions')
    base = canonical_reference(reference)
    if base.shape != (len(context), len(actions)):
        raise ValueError('reference does not cover the full declared action catalogue')
    target = indices(winners, len(context), len(actions), 'winner')
    if np.any(base[np.arange(len(context)), target] <= 0):
        raise ValueError('winner outside reference support')
    weight = normalized_weights(weights, len(context))
    theta = np.asarray(flat, dtype=float).reshape(context.shape[1], actions.shape[1])
    if not np.isfinite(theta).all() or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('finite parameters and positive ridge required')
    loss = ridge*np.sum(theta**2)
    gradient = 2*ridge*theta
    for lo in range(0, len(context), batch_size):
        hi = min(lo+batch_size, len(context))
        p0 = base[lo:hi]
        logits = np.full(p0.shape, -np.inf)
        mask = p0 > 0
        logits[mask] = np.log(p0[mask])
        logits += context[lo:hi] @ theta @ actions.T
        normalizer = logsumexp(logits, axis=1)
        local_target = target[lo:hi]
        loss += np.sum(weight[lo:hi]*(normalizer-logits[np.arange(hi-lo), local_target]))
        error = np.exp(logits-normalizer[:, None])
        error[np.arange(hi-lo), local_target] -= 1
        error *= weight[lo:hi, None]
        gradient += context[lo:hi].T @ (error @ actions)
    return float(loss), gradient.ravel()


def fit_winner_only(context, actions, reference, winners, weights, ridge=.01, max_iter=250):
    context, actions = finite_matrix(context, 'context'), finite_matrix(actions, 'actions')
    n_parameters = context.shape[1]*actions.shape[1]
    if n_parameters > 768:
        raise ValueError('bounded adapter exceeds 768 parameters')
    result = minimize(winner_objective, np.zeros(n_parameters),
                      args=(context, actions, reference, winners, weights, ridge), jac=True,
                      method='L-BFGS-B', options={'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 1e-12, 'maxls': 50})
    return result.x.reshape(context.shape[1], actions.shape[1]), {
        'objective': float(result.fun), 'gradient_max_abs': float(np.max(np.abs(result.jac))),
        'converged': bool(result.success), 'iterations': int(result.nit), 'message': str(result.message),
        'parameters': n_parameters, 'ridge': float(ridge), 'normalizer_actions': len(actions),
        'rejected_labels_consumed': False}
