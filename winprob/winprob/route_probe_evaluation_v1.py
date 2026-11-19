"""Paired scoring of frozen, natively decoded full-catalogue route policies."""
from __future__ import annotations
import numpy as np
from winprob.route_marginal_v1 import require

ARMS = ('m2', 'recipe', 'permuted')


def evaluate(labels, scores, game_ids, folds, legality_failures):
    y = np.asarray(labels['main']); n = len(y)
    require(set(scores) == set(ARMS) and len(game_ids) == len(set(game_ids)) == n
            and np.shape(folds) == (n,), 'paired one-focal-per-game population required')
    wards = np.asarray(labels['contains_wards'], dtype=bool); no_buy = np.asarray(labels['is_no_buy'], dtype=bool)
    size = np.asarray(labels['main_size'])
    require(wards.shape == no_buy.shape == size.shape == (n,) and np.all(y >= 0)
            and np.all(size >= 0) and not np.any(no_buy & ((y != 0) | wards)), 'complete target semantics required')
    require(set(legality_failures) == {'16.17', '16.18'} and all(set(v) == set(ARMS)
            and all(type(c) is int and c >= 0 for c in v.values()) for v in legality_failures.values()),
            'legality audit counts required for every world/arm')
    draws = np.random.default_rng(20261003).integers(0, n, size=(2000, n))
    values = {}; summaries = {}; changes = {}
    for arm, prediction in scores.items():
        first = np.asarray(prediction['top1']); top = np.asarray(prediction['top3'])
        lp = np.asarray(prediction['catalogue_target_logp'])
        require(first.shape == lp.shape == (n,) and top.shape == (n, 3)
                and np.isfinite(lp).all() and np.all(lp <= 1e-12)
                and np.array_equal(first, top[:, 0]) and np.all(first >= 0)
                and np.all(top >= -1), 'incomplete native predictions or target probabilities')
        require(all(len(set(row[row >= 0])) == int((row >= 0).sum()) for row in top), 'duplicate native option')
        exact = (first == y).astype(float)
        v = {'main_top1': exact, 'main_top3': (top == y[:, None]).any(1).astype(float),
             'complete_top1': exact * ~wards, 'typed_main_top1': exact * ((y != 0) | no_buy),
             'catalogue_nll': -lp}
        values[arm] = v
        summaries[arm] = {k: {'mean': float(a.mean()), 'interval95': np.quantile(a[draws].mean(1), [.025, .975]).tolist(),
                             **({'correct': int(a.sum())} if k != 'catalogue_nll' else {})} for k, a in v.items()}
        if arm != 'm2':
            base = np.asarray(scores['m2']['top3'])
            changes[arm] = {'changed_first': int(np.sum(first != base[:, 0])),
                'changed_option_set': int(sum(set(a[a >= 0]) != set(b[b >= 0]) for a, b in zip(top, base, strict=True)))}
    contrasts = {}
    for control in ('m2', 'permuted'):
        contrasts[control] = {}
        for key in values['recipe']:
            delta = values['recipe'][key]-values[control][key]
            contrasts[control][key] = {'difference': float(delta.mean()),
                'interval95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist()}
        delta = values['recipe']['main_top1']-values[control]['main_top1']
        contrasts[control].update(corrected=int((delta > 0).sum()), broken=int((delta < 0).sum()))
    nonempty = y != 0; delta = values['recipe']['main_top1']-values['m2']['main_top1']
    require(nonempty.sum() >= 30 and np.all(nonempty[draws].sum(1) > 0), 'nonempty guard population insufficient')
    nonempty_guard = {'games': int(nonempty.sum()), 'difference': float(delta[nonempty].mean()),
        'interval95': np.quantile((delta*nonempty)[draws].sum(1)/nonempty[draws].sum(1), [.025, .975]).tolist()}
    components = {'first_gain_at_least_3pp': contrasts['m2']['main_top1']['difference'] >= .03,
        'first_positive_interval_vs_m2': contrasts['m2']['main_top1']['interval95'][0] > 0,
        'first_positive_interval_vs_permuted': contrasts['permuted']['main_top1']['interval95'][0] > 0,
        'nonempty_positive_interval': nonempty_guard['interval95'][0] > 0,
        'displayed_top3_not_worse': contrasts['m2']['main_top3']['difference'] >= 0,
        'catalogue_nll_not_worse': contrasts['m2']['catalogue_nll']['difference'] <= 0,
        'zero_legality_failures_all_arms_worlds': all(c == 0 for v in legality_failures.values() for c in v.values())}
    kinds = np.where(no_buy, 'observed_no_buy_death', np.where(y == 0, 'ward_only', 'purchase'))
    masks = {**{'kind_'+k: kinds == k for k in sorted(set(kinds))},
             **{'size_'+str(k): size == k for k in sorted(set(size))},
             **{'fold_'+str(f): np.asarray(folds) == f for f in sorted(set(folds))}}
    slices = {}
    for name, mask in masks.items():
        slices[name] = {'games': int(mask.sum()), 'status': 'descriptive' if mask.sum() >= 30 else 'too_small_for_rates'}
        if mask.sum() >= 30:
            slices[name]['arms'] = {arm: {key: float(a[mask].mean()) for key, a in v.items()} for arm, v in values.items()}
    return {'games': n, 'arms': summaries, 'recipe_minus_control': contrasts,
        'nonempty_purchase_guard': nonempty_guard, 'legality_failures': legality_failures,
        'gate_components': {k: bool(v) for k, v in components.items()},
        'metric_gate_passed': bool(all(components.values())), 'candidate_changes': changes,
        'descriptive_slices': slices, 'bootstrap': {'draws': 2000, 'seed': 20261003,
            'unit': 'whole game, one original focal per game',
            'limits': 'Exposed internal screen; conditional on fitted cross-fit models, no training-set uncertainty or multiplicity correction.'}}
