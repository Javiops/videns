"""Fixed prediction/scoring semantics for the native-M2 residual screen."""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from winprob.recipe_transition_v1 import predict, _logits
from winprob.recipe_transition_probe_v1 import ARMS

SEED = 20261001


def require(ok, message):
    if not ok:
        raise ValueError(message)


def predict_correction(inputs, permutations, standardizer, coefficients, arm):
    """No purchase labels or target objects are accepted by prediction."""
    transformed = standardizer.transform(inputs, arm, permutations if arm == 'shuffled_net' else None)
    p, order = predict(transformed, coefficients)
    logits = _logits(transformed, coefficients, slice(None))
    logp = logits if not np.any(coefficients) else logits-logsumexp(logits, axis=1, keepdims=True)
    require(np.allclose(np.exp(logp), p, atol=1e-14, rtol=1e-12), 'stable log probabilities differ from prediction')
    return {'probabilities': p, 'log_probabilities': logp, 'candidate_order': order}


def compact_prediction(prediction, candidate_groups, coarsened_targets):
    """Scoring-only label access, after prediction is frozen."""
    groups = np.asarray(candidate_groups); order = np.asarray(prediction['candidate_order'])
    y = np.asarray(coarsened_targets); logp = np.asarray(prediction['log_probabilities'])
    n = len(groups)
    require(groups.shape == order.shape == (n, 3) and y.shape == (n,) and logp.shape == (n, 4)
            and np.issubdtype(order.dtype, np.integer) and np.issubdtype(y.dtype, np.integer)
            and np.all((y >= 0) & (y <= 3)) and np.array_equal(np.sort(order, axis=1), np.tile([0, 1, 2], (n, 1))), 'invalid prediction/label axes')
    ranked = np.take_along_axis(groups, order, axis=1)
    require(np.all(ranked[:, 0] >= 0), 'prediction promoted padding')
    target_logp = logp[np.arange(n), y]
    require(np.isfinite(target_logp).all(), 'observed coarsened class has invalid log probability')
    return {'top1': ranked[:, 0], 'top3': ranked, 'coarsened_target_logp': target_logp}


def evaluate(labels, scores, candidate_groups, game_ids, folds):
    y = np.asarray(labels['main']); n = len(y)
    require(set(scores) == {'m2', *ARMS} and np.shape(candidate_groups) == (n, 3)
            and len(game_ids) == len(set(game_ids)) == n and np.shape(folds) == (n,), 'paired distinct-game population required')
    require(all(np.shape(labels[k]) == (n,) for k in ('contains_wards', 'is_no_buy', 'main_size')), 'complete target semantics required')
    require(np.all(y >= 0) and np.all(np.asarray(labels['main_size']) >= 0)
            and not np.any(np.asarray(labels['is_no_buy']) & ((y != 0) | np.asarray(labels['contains_wards']))), 'invalid no-buy or target semantics')
    draws = np.random.default_rng(SEED).integers(0, n, size=(2000, n))
    values = {}; summaries = {}
    for arm, s in scores.items():
        top = np.asarray(s['top3']); first = np.asarray(s['top1']); lp = np.asarray(s['coarsened_target_logp'])
        require(top.shape == (n, 3) and first.shape == lp.shape == (n,) and np.isfinite(lp).all()
                and np.array_equal(first, top[:, 0]) and np.all(first >= 0), 'incomplete prediction')
        require(np.array_equal(np.sort(top, axis=1), np.sort(candidate_groups, axis=1)), 'correction changed original option membership')
        exact = (first == y).astype(float)
        v = {'main_top1': exact, 'main_top3': (top == y[:, None]).any(1).astype(float),
             'complete_top1': exact * ~np.asarray(labels['contains_wards'], dtype=bool),
             'typed_main_top1': exact * ((y != 0) | np.asarray(labels['is_no_buy'], dtype=bool)),
             'coarsened_nll': -lp}
        values[arm] = v
        summaries[arm] = {key: {'mean': float(a.mean()), 'interval95': np.quantile(a[draws].mean(1), [.025, .975]).tolist(),
                                **({'correct': int(a.sum())} if key != 'coarsened_nll' else {})}
                          for key, a in v.items()}
    require(all(np.array_equal(v['main_top3'], values['m2']['main_top3']) for v in values.values()), 'top3 coverage unexpectedly changed')
    paired = {}
    for control in ('m2', 'raw', 'shuffled_net'):
        paired[control] = {}
        for key in values['net']:
            d = values['net'][key] - values[control][key]
            paired[control][key] = {'difference': float(d.mean()), 'interval95': np.quantile(d[draws].mean(1), [.025, .975]).tolist()}
        d = values['net']['main_top1'] - values[control]['main_top1']
        paired[control].update(corrected=int((d > 0).sum()), broken=int((d < 0).sum()))
    nonempty = y != 0; d = values['net']['main_top1'] - values['m2']['main_top1']
    require(nonempty.sum() >= 30 and np.all(nonempty[draws].sum(1) > 0), 'nonempty guard population insufficient')
    guard = {'games': int(nonempty.sum()), 'difference': float(d[nonempty].mean()),
             'interval95': np.quantile((d*nonempty)[draws].sum(1)/nonempty[draws].sum(1), [.025, .975]).tolist()}
    gate = (paired['m2']['main_top1']['difference'] >= .03 and paired['m2']['main_top1']['interval95'][0] > 0
            and paired['raw']['main_top1']['interval95'][0] > 0 and guard['interval95'][0] > 0)
    kinds = np.where(labels['is_no_buy'], 'observed_no_buy_death', np.where(y == 0, 'ward_only', 'purchase'))
    masks = {**{'kind_'+k: kinds == k for k in sorted(set(kinds))},
             **{'size_'+str(k): np.asarray(labels['main_size']) == k for k in sorted(set(labels['main_size']))},
             **{'fold_'+str(f): np.asarray(folds) == f for f in sorted(set(folds))}}
    slices = {}
    for name, mask in masks.items():
        slices[name] = {'games': int(mask.sum()), 'status': 'descriptive' if mask.sum() >= 30 else 'too_small_for_rates'}
        if mask.sum() >= 30:
            slices[name]['arms'] = {arm: {key: float(a[mask].mean()) for key, a in v.items()} for arm, v in values.items()}
    return {'games': n, 'arms': summaries, 'net_minus_control': paired, 'nonempty_purchase_guard': guard,
            'native_option_membership_unchanged': True, 'top3_coverage_unchanged': True,
            'metric_gate_passed': bool(gate), 'descriptive_slices': slices,
            'bootstrap': {'draws': 2000, 'seed': SEED, 'unit': 'whole game, one fixed focal per game',
                          'limits': 'Conditional on fitted cross-fit models; does not include training-set uncertainty.'}}
