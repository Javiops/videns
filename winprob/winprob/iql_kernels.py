"""Framework-neutral numerical reference for full-action IQL updates.

These NumPy primitives validate accounting and losses. They do not train a
critic, provide automatic differentiation, or identify causal action values.
Torch implementations must detach targets and match these synthetic checks.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
import numpy as np


def _vector(values, name, n=None, finite=True):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or (n is not None and len(result) != n):
        raise ValueError(name+' must be an aligned nonempty vector, not an action-candidate matrix')
    if finite and not np.isfinite(result).all():
        raise ValueError(name+' must be finite')
    return result


def _terminal(values, n):
    result = np.asarray(values)
    if result.shape != (n,) or result.dtype != np.bool_:
        raise ValueError('terminated must be an aligned explicit boolean vector')
    return result


def _weights(values, n):
    if values is None:
        return np.full(n, 1/n)
    result = _vector(values, 'sampling weights', n)
    if np.any(result <= 0):
        raise ValueError('sampling weights must be strictly positive')
    return result/result.sum()


def full_basket_key(action):
    """Preserve complete item multiplicities, including wards; never group-strip."""
    if not isinstance(action, Mapping) or action.get('encoding') != 'item-counts-v1':
        raise ValueError('explicit full item-counts action required')
    counts = action.get('label_counts')
    if not isinstance(counts, Mapping):
        raise ValueError('full basket counts required')
    kind = action.get('kind')
    if '999999' in counts:
        if kind != 'observed_no_buy_death' or dict(counts) != {'999999': 1}:
            raise ValueError('SAVE sentinel requires its sole explicit observed-no-buy tag')
        return ()
    if not counts:
        if kind != 'observed_no_buy_death':
            raise ValueError('empty action requires explicit observed-no-buy semantics')
        return ()
    if kind == 'observed_no_buy_death':
        raise ValueError('observed no-buy cannot contain physical purchases')
    pairs = []
    for key, count in counts.items():
        if (not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key or int(key) <= 0
                or type(count) is not int or count <= 0):
            raise ValueError('canonical positive item IDs and integer multiplicities required')
        pairs.append((int(key), count))
    return tuple(sorted(pairs))


def observed_target_q(target_q_evaluator, state_refs, observed_full_actions):
    """One Q query on recorded full actions only; no argmax or candidate proposals."""
    if len(state_refs) == 0 or len(state_refs) != len(observed_full_actions):
        raise ValueError('aligned observed state/action records required')
    keys = tuple(full_basket_key(action) for action in observed_full_actions)
    result = target_q_evaluator(tuple(state_refs), keys)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError('two target critics are required')
    q1 = _vector(result[0], 'target Q1 on observed actions', len(state_refs))
    q2 = _vector(result[1], 'target Q2 on observed actions', len(state_refs))
    return np.minimum(q1, q2).copy()


def bellman_targets(rewards, terminated, next_values, discount=1.):
    rewards = _vector(rewards, 'rewards')
    done = _terminal(terminated, len(rewards))
    values = _vector(next_values, 'next values', len(rewards), finite=False)
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError('discount must be in [0,1]')
    if not np.isfinite(values[~done]).all():
        raise ValueError('nonterminal next values must be finite')
    target = rewards.copy()
    # Do not multiply a nonexistent terminal value by zero: 0*NaN is NaN.
    target[~done] += discount*values[~done]
    return target


def targets_from_next_states(value_evaluator, next_state_refs, rewards, terminated, discount=1.):
    rewards = _vector(rewards, 'rewards')
    done = _terminal(terminated, len(rewards))
    if len(next_state_refs) != len(rewards) or any((ref is None) != bool(end) for ref, end in zip(next_state_refs, done)):
        raise ValueError('terminal must have no next state; nonterminal must have a real next reference')
    values = np.full(len(rewards), np.nan)
    keep = np.flatnonzero(~done)
    if len(keep):
        values[keep] = _vector(value_evaluator(tuple(next_state_refs[i] for i in keep)), 'evaluated next V', len(keep))
    return bellman_targets(rewards, done, values, discount)


def expectile_value_loss(observed_q_targets, values, expectile=.7, sampling_weights=None):
    """Upper expectile over predicted observed-action Q, not Bernoulli outcomes."""
    q = _vector(observed_q_targets, 'detached observed-action Q targets')
    value = _vector(values, 'online V', len(q))
    if not math.isfinite(expectile) or not 0 < expectile < 1:
        raise ValueError('expectile must be strictly between zero and one')
    residual = q-value
    asymmetry = np.where(residual > 0, expectile, 1-expectile)
    weight = _weights(sampling_weights, len(q))
    return {'loss': float(np.sum(weight*asymmetry*residual**2)),
            'per_row_loss': asymmetry*residual**2,
            'gradient_wrt_online_v': -2*weight*asymmetry*residual,
            'target_scope': 'detached minimum of target Q1/Q2 on observed full action'}


def observed_critic_loss(q1_observed, q2_observed, detached_targets, sampling_weights=None):
    target = _vector(detached_targets, 'detached Bellman targets')
    q1 = _vector(q1_observed, 'online Q1 observed', len(target))
    q2 = _vector(q2_observed, 'online Q2 observed', len(target))
    weight = _weights(sampling_weights, len(target))
    residual1, residual2 = q1-target, q2-target
    per_row = residual1**2+residual2**2
    return {'loss': float(weight@per_row), 'per_row_loss': per_row,
            'gradient_wrt_online_q1': 2*weight*residual1,
            'gradient_wrt_online_q2': 2*weight*residual2}


def actor_advantage_weights(observed_q_targets, values, temperature=.1, cap=20.):
    q = _vector(observed_q_targets, 'frozen out-of-game observed Q')
    value = _vector(values, 'frozen out-of-game V', len(q))
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(cap) or cap <= 0:
        raise ValueError('positive finite temperature and cap required')
    advantage = q-value
    with np.errstate(over='ignore', under='ignore'):
        log_weight = advantage/temperature
        result = np.exp(np.minimum(log_weight, math.log(cap)))
    return {'advantage': advantage, 'weights': result,
            'clipped': log_weight > math.log(cap),
            'temperature_convention': 'exp(advantage / temperature)', 'cap': cap}


def awr_actor_loss(observed_action_log_probabilities, advantage_weights, sampling_weights=None):
    log_probability = _vector(observed_action_log_probabilities, 'observed full-action log probabilities')
    importance = _vector(advantage_weights, 'detached advantage weights', len(log_probability))
    if np.any(log_probability > 1e-12) or np.any(importance < 0):
        raise ValueError('categorical log probabilities must be <=0; advantage weights must be nonnegative')
    weight = _weights(sampling_weights, len(log_probability))
    # Normalize sampling mass only. Renormalizing AWR weights changes the recipe.
    return {'loss': float(-np.sum(weight*importance*log_probability)),
            'gradient_wrt_observed_log_probability': -weight*importance}


def normalize_oof_actor_weights(q1_targets, q2_targets, values, action_supported,
                                temperature=.1, cap=20., minimum_support=.7):
    """Application-only: one global uniform-transition FIT rescale, never per batch.

    Unsupported opposite-fold actions receive raw weight one. Their Q/V may
    be NaN because the learner need not query an unsupported action at all.
    Exposure provenance must be checked separately before calling this.
    """
    q1 = _vector(q1_targets, 'OOF Q1', finite=False)
    q2 = _vector(q2_targets, 'OOF Q2', len(q1), finite=False)
    value = _vector(values, 'OOF V', len(q1), finite=False)
    supported = _terminal(action_supported, len(q1))
    if not math.isfinite(minimum_support) or not 0 < minimum_support <= 1:
        raise ValueError('positive support threshold at most one required')
    support_fraction = float(supported.mean())
    if support_fraction < minimum_support:
        raise ValueError('opposite-fold complete-action support is below the declared gate')
    raw = np.ones(len(q1))
    computed = actor_advantage_weights(np.minimum(q1[supported], q2[supported]), value[supported], temperature, cap)
    raw[supported] = computed['weights']
    mean = float(raw.mean())
    if not math.isfinite(mean) or mean <= 0:
        raise ValueError('invalid global FIT advantage-weight mean')
    normalized = raw/mean
    return {'raw_weights': raw, 'weights': normalized, 'global_fit_mean_raw_weight': mean,
            'support_fraction': support_fraction, 'supported_transitions': int(supported.sum()),
            'fallback_transitions': int((~supported).sum()),
            'raw_cap_fraction_supported': float(np.mean(computed['clipped'])),
            'raw_cap_fraction_all': float(np.sum(computed['clipped'])/len(q1)),
            'effective_sample_size': float(normalized.sum()**2/np.sum(normalized**2)),
            'normalization_scope': 'one complete FIT population, uniform base transition mass',
            'fallback_rule': 'raw one before the shared global FIT rescale'}


def initialize_target(online_parameters):
    if not isinstance(online_parameters, Mapping) or not online_parameters:
        raise ValueError('nonempty named parameter mapping required')
    target = {}
    for name, value in online_parameters.items():
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
            raise ValueError('finite floating model parameters required')
        target[name] = array.copy()
    return target


def polyak_target(online_parameters, old_target_parameters, online_fraction=.005):
    if not math.isfinite(online_fraction) or not 0 <= online_fraction <= 1:
        raise ValueError('online fraction must be in [0,1]')
    online, target = initialize_target(online_parameters), initialize_target(old_target_parameters)
    if set(online) != set(target):
        raise ValueError('online/target named parameters differ')
    result = {}
    for name in online:
        if online[name].shape != target[name].shape or online[name].dtype != target[name].dtype:
            raise ValueError('online/target parameter shape or dtype differs')
        result[name] = ((1-online_fraction)*target[name]+online_fraction*online[name]).astype(target[name].dtype)
    return result


def episodic_returns(rewards, terminated, discount=1.):
    rewards = _vector(rewards, 'rewards')
    done = _terminal(terminated, len(rewards))
    if not done[-1] or not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError('complete terminal-ended chains and valid discount required')
    result = np.empty_like(rewards)
    future = 0.
    for index in range(len(rewards)-1, -1, -1):
        future = rewards[index]+(0. if done[index] else discount*future)
        result[index] = future
    return result


def potential_shaped_rewards(rewards, terminated, current_potential, next_potential, discount=1.):
    rewards = _vector(rewards, 'original rewards')
    done = _terminal(terminated, len(rewards))
    current = _vector(current_potential, 'current potential', len(rewards))
    future = _vector(next_potential, 'next potential', len(rewards), finite=False)
    # The terminal potential is ZERO. The terminal outcome is already r.
    return bellman_targets(rewards, done, future, discount)-current


def game_folds(game_ids, folds=2, seed=20260929):
    if type(folds) is not int or folds < 2 or len(game_ids) == 0:
        raise ValueError('at least two whole-game folds and a nonempty ID list required')
    if any(not isinstance(gid, str) or not gid for gid in game_ids):
        raise ValueError('nonempty match IDs required')
    return {gid: int.from_bytes(hashlib.sha256(f'{seed}|{gid}'.encode()).digest()[:8], 'big') % folds
            for gid in sorted(set(game_ids))}


def assert_out_of_game_predictions(scored_game_ids, learner_ids, seen_games_by_learner):
    """Seen includes training, calibration and any data-dependent selection IDs."""
    if len(scored_game_ids) != len(learner_ids) or len(scored_game_ids) == 0:
        raise ValueError('aligned nonempty prediction provenance required')
    for gid, learner in zip(scored_game_ids, learner_ids):
        if learner not in seen_games_by_learner:
            raise ValueError('missing learner exposure provenance')
        if gid in set(seen_games_by_learner[learner]):
            raise ValueError('scored game was seen by its critic/value learner')
    return {'predictions': len(scored_game_ids), 'games': len(set(scored_game_ids)), 'exposure_overlap': 0}
