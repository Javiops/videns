"""Native-receipt descriptors and a convex correction to a frozen choice prior.

No datasets, source runtime, model package or labels are loaded on import.
Receipts describe hypothetical candidates, never an observed future inventory.
Conservation validation does not replace the source native legality audit.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from winprob.ipo_residual import ACTION_STATS, CHAMPION_TAGS

CONTEXT_NAMES = ('intercept', 'time_over_40min', 'level_over_18', 'wallet_over_3000') + tuple(
    'owned_' + key for key, _ in ACTION_STATS) + tuple(
    side + '_' + tag for side in ('own', 'enemies') for tag in CHAMPION_TAGS)
CANDIDATE_NAMES = ('empty_main', 'cost_over_3000', 'remaining_gold_over_3000',
                   'bought_copies_over_6', 'consumed_copies_over_6', 'item_count_delta_over_6') + tuple(
    part + '_' + key for part in ('bought', 'net') for key, _ in ACTION_STATS)
MAX_PARAMETERS = 768


def _number(value, name, minimum=-math.inf):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError('invalid finite scalar: ' + name)
    return float(value)


def _bag(items, name):
    if not isinstance(items, (list, tuple)) or any(type(i) is not int or i <= 0 for i in items):
        raise ValueError('positive item IDs with multiplicities required: ' + name)
    return Counter(items)


def _attributes(bag, items):
    result = np.zeros(len(ACTION_STATS), dtype=np.float64)
    for iid, count in bag.items():
        if str(iid) not in items:
            raise ValueError('item absent from pinned native statics')
        stats = items[str(iid)].get('stats', {})
        if not isinstance(stats, dict):
            raise ValueError('item stats must be a mapping')
        for j, (key, scale) in enumerate(ACTION_STATS):
            result[j] += count * _number(stats.get(key, 0), key) / scale
    return result


def context_features(state, pinned_items, champion_tags):
    """Explicit predecision allowlist; labels, roles and future fields ignored."""
    inventory = _bag(state['inventory'], 'owned inventory')
    side = state['team_id']
    cid = state['champion_id']
    if type(side) is not int or side not in (100, 200) or type(cid) is not int:
        raise ValueError('invalid focal identity')
    others = state['others']
    if not isinstance(others, list) or len(others) != 9:
        raise ValueError('complete ten-player roster required')
    if any(type(p.get('team_id')) is not int or p['team_id'] not in (100, 200)
           or type(p.get('champion_id')) is not int for p in others):
        raise ValueError('invalid other-player identity')
    enemy_ids = [p['champion_id'] for p in others if p['team_id'] != side]
    if len(enemy_ids) != 5 or any(str(i) not in champion_tags for i in [cid, *enemy_ids]):
        raise ValueError('known focal champion and five enemies required')
    level = _number(state['level'], 'level', 1)
    if level > 30:
        raise ValueError('invalid observed level')
    values = [1., _number(state['ts'], 'timestamp', 0) / 2400000., level / 18.,
              _number(state['gold_est'], 'wallet') / 3000.]
    values.extend(_attributes(inventory, pinned_items))
    values.extend(float(tag in champion_tags[str(cid)]) for tag in CHAMPION_TAGS)
    values.extend(sum(tag in champion_tags[str(i)] for i in enemy_ids) / 5. for tag in CHAMPION_TAGS)
    return np.asarray(values, dtype=np.float64)


def transition_features(before, action, wallet, steps, pinned_items):
    """Verify count conservation, then describe a native hypothetical execution.

    action is the complete expanded MAIN multiset. steps contain item_id, cost,
    and inventory_after from native try_purchase, in purchase_sequence order.
    Descriptors sum static item attributes; they do not simulate passives.
    """
    inventory = _bag(before, 'before'); original = inventory.copy()
    target = _bag(action, 'main action')
    wallet = _number(wallet, 'wallet')
    if not isinstance(steps, list) or len(steps) != len(action):
        raise ValueError('one native step per retained purchased copy required')
    executed = Counter(); consumed = Counter(); spent = 0.
    for step in steps:
        iid = step.get('item_id')
        if type(iid) is not int or iid not in target or executed[iid] >= target[iid]:
            raise ValueError('step outside the candidate multiset')
        cost = _number(step.get('cost'), 'step cost', 0)
        if spent + cost > wallet + 1e-7:
            raise ValueError('receipt exceeds its fixed wallet')
        after = _bag(step.get('inventory_after'), 'after')
        available = inventory + Counter([iid])
        if after - available:
            raise ValueError('receipt creates an unpurchased item or copy')
        executed[iid] += 1
        if executed - after:
            raise ValueError('receipt consumes a protected retained purchase')
        consumed += available - after
        inventory = after; spent += cost
    if executed != target or original + target != inventory + consumed:
        raise ValueError('purchase multiset conservation failed')
    # Check every referenced static item, including consumed components.
    owned_attrs = _attributes(original, pinned_items)
    bought_attrs = _attributes(target, pinned_items)
    net_attrs = _attributes(inventory, pinned_items) - owned_attrs
    if not np.allclose(net_attrs, bought_attrs - _attributes(consumed, pinned_items), atol=1e-12, rtol=1e-12):
        raise ValueError('attribute conservation failed')
    values = [float(not target), spent / 3000., (wallet - spent) / 3000.,
              sum(target.values()) / 6., sum(consumed.values()) / 6.,
              (sum(inventory.values()) - sum(original.values())) / 6.]
    return np.asarray([*values, *bought_attrs, *net_attrs], dtype=np.float64)


def coarsen_reference(full_probability, candidate_ids):
    """Group the frozen catalogue into selected candidates and an untouched tail.

    -1 is padding, never a label or a candidate. The caller keeps native option
    order and binds group IDs to those exact multiset options independently.
    """
    p = np.asarray(full_probability, dtype=np.float64)
    ids = np.asarray(candidate_ids)
    if p.ndim != 2 or not all(p.shape) or ids.shape != (len(p), 3) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('full distribution and up-to-three candidate IDs required')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(np.abs(p.sum(1) - 1) > 2e-6):
        raise ValueError('invalid full-catalogue reference mass')
    if np.any(ids < -1) or np.any(ids >= p.shape[1]):
        raise ValueError('candidate outside full reference catalogue')
    for row in ids:
        selected = row[row >= 0]
        if not len(selected) or len(set(selected.tolist())) != len(selected) or np.any(row[:len(selected)] < 0):
            raise ValueError('unique candidates followed only by padding required')
    p = p / p.sum(1, keepdims=True)
    result = np.zeros((len(p), 4), dtype=np.float64)
    for i, row in enumerate(ids):
        selected = row[row >= 0]
        result[i, :len(selected)] = p[i, selected]
        keep = np.ones(p.shape[1], dtype=bool); keep[selected] = False
        result[i, 3] = p[i, keep].sum()  # no cancellation from 1 - selected mass
        if np.any(result[i, :len(selected)] <= 0):
            raise ValueError('displayed candidate outside reference support')
    return result


@dataclass(frozen=True)
class ChoiceInputs:
    reference: np.ndarray            # [N,4]; OTHER is column3
    context: np.ndarray              # [N,H]
    candidates: np.ndarray           # [N,3,F]; padding features exactly0

    def validate(self):
        p, h, f = self.reference, self.context, self.candidates
        if not all(isinstance(a, np.ndarray) and a.dtype == np.float64 and np.isfinite(a).all() for a in (p, h, f)):
            raise ValueError('finite float64 choice inputs required')
        if p.ndim != 2 or p.shape[1] != 4 or not len(p) or h.ndim != 2 or h.shape[0] != len(p) or f.ndim != 3 or f.shape[:2] != (len(p), 3):
            raise ValueError('choice input dimensions disagree')
        if not 1 <= h.shape[1] * f.shape[2] <= MAX_PARAMETERS:
            raise ValueError('correction exceeds bounded parameter count')
        if np.any(p < 0) or np.any(np.abs(p.sum(1) - 1) > 1e-12) or np.any(p[:, :3].sum(1) <= 0):
            raise ValueError('normalized prior with a displayed candidate required')
        present = p[:, :3] > 0
        if np.any(present[:, 1:] & ~present[:, :-1]):
            raise ValueError('native options must precede padding')
        if np.any(f[p[:, :3] == 0] != 0):
            raise ValueError('absent candidates must have zero features')
        return self

    @property
    def parameter_shape(self):
        return self.context.shape[1], self.candidates.shape[2]


def _logits(inputs, theta, sl):
    p = inputs.reference[sl]
    logits = np.full(p.shape, -np.inf)
    np.log(p, out=logits, where=p > 0)
    logits[:, :3] += np.einsum('nf,nkf->nk', inputs.context[sl] @ theta, inputs.candidates[sl])
    return logits


def _supervision(inputs, target, weights, ridge):
    inputs.validate(); y = np.asarray(target); w = np.asarray(weights, dtype=np.float64)
    if y.shape != (len(inputs.reference),) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y > 3)):
        raise ValueError('candidate-or-OTHER labels required')
    if np.any(inputs.reference[np.arange(len(y)), y] <= 0):
        raise ValueError('observed category outside prior support')
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('positive game-balanced row weights required')
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError('positive ridge needed for strict convexity')
    return y, w / w.sum()


def _objective(flat, inputs, y, w, ridge, batch_size, direction=None):
    theta = np.asarray(flat, dtype=np.float64).reshape(inputs.parameter_shape)
    if not np.isfinite(theta).all():
        raise ValueError('finite parameters required')
    loss = ridge * np.sum(theta ** 2); gradient = 2 * ridge * theta
    if direction is not None:
        vector = np.asarray(direction, dtype=np.float64).reshape(theta.shape)
        if not np.isfinite(vector).all():
            raise ValueError('finite curvature direction required')
        hessian = 2 * ridge * vector
    for lo in range(0, len(y), batch_size):
        sl = slice(lo, min(lo + batch_size, len(y)))
        h, f = inputs.context[sl], inputs.candidates[sl]
        logits = _logits(inputs, theta, sl); normalizer = logsumexp(logits, axis=1)
        p = np.exp(logits - normalizer[:, None]); local_y = y[sl]
        loss += np.sum(w[sl] * (normalizer - logits[np.arange(len(local_y)), local_y]))
        error = p.copy(); error[np.arange(len(local_y)), local_y] -= 1
        feature_error = np.einsum('nk,nkf->nf', error[:, :3], f)
        gradient += h.T @ (w[sl, None] * feature_error)
        if direction is not None:
            directional = np.einsum('nf,nkf->nk', h @ vector, f)
            mean = np.sum(p[:, :3] * directional, axis=1)  # OTHER's direction=0
            centered = p[:, :3] * (directional - mean[:, None])
            hessian += h.T @ (w[sl, None] * np.einsum('nk,nkf->nf', centered, f))
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise ValueError('nonfinite residual objective')
    return (float(loss), gradient.ravel()) if direction is None else hessian.ravel()


def objective(flat, inputs, target, weights, ridge=.01, batch_size=512):
    y, w = _supervision(inputs, target, weights, ridge)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('positive batch size required')
    return _objective(flat, inputs, y, w, ridge, batch_size)


def hessian_vector(flat, direction, inputs, target, weights, ridge=.01):
    y, w = _supervision(inputs, target, weights, ridge)
    return _objective(flat, inputs, y, w, ridge, 512, direction)


def fit(inputs, target, weights, ridge=.01, max_iter=300):
    y, w = _supervision(inputs, target, weights, ridge)
    if type(max_iter) is not int or max_iter <= 0:
        raise ValueError('positive fixed optimizer bound required')
    result = minimize(_objective, np.zeros(np.prod(inputs.parameter_shape)),
                      args=(inputs, y, w, ridge, 512), jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': 1e-7, 'ftol': 1e-12, 'maxls': 50})
    return result.x.reshape(inputs.parameter_shape), {
        'converged': bool(result.success), 'iterations': int(result.nit),
        'objective': float(result.fun), 'gradient_max_abs': float(np.max(np.abs(result.jac))),
        'message': str(result.message), 'parameters': len(result.x), 'ridge': ridge,
        'rows': len(y), 'outside_shortlist_rows': int(np.sum(y == 3)),
        'likelihood': 'exact_coarsened_candidates_plus_OTHER', 'source_runtime_imported': False}


def predict(inputs, theta):
    """Target-free probabilities and a stable permutation of the native options."""
    inputs.validate(); theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != inputs.parameter_shape or not np.isfinite(theta).all():
        raise ValueError('aligned finite coefficient matrix required')
    if not np.any(theta):
        p = inputs.reference.copy()
        order = np.broadcast_to(np.arange(3), (len(p), 3)).copy()
    else:
        logits = _logits(inputs, theta, slice(None))
        p = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        order = np.argsort(-logits[:, :3], axis=1, kind='stable')
    return p, order
