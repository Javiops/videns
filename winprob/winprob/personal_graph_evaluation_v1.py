"""Fixed paired endpoints for the personal-relation information test."""
from __future__ import annotations
import numpy as np
from winprob.recipe_transition_evaluation_v1 import compact_prediction

SEED = 20261003


def require(ok, message):
    if not ok:
        raise ValueError(message)


def evaluate(labels, scores, groups, game_ids, folds, zero_profiles):
    y = np.asarray(labels['main']); n = len(y)
    require(set(scores) == {'m2', 'personal', 'degree_null'} and len(game_ids) == len(set(game_ids)) == n
            and np.shape(groups) == (n, 3) and np.shape(folds) == (n,), 'fixed distinct-game paired population required')
    require(np.all(y >= 0) and all(np.shape(labels[k]) == (n,) for k in ('contains_wards', 'is_no_buy'))
            and not np.any(np.asarray(labels['is_no_buy']) & ((y != 0) | np.asarray(labels['contains_wards']))), 'invalid target semantics')
    draws = np.random.default_rng(SEED).integers(0, n, size=(2000, n)); values = {}; summaries = {}
    for arm, s in scores.items():
        require(np.shape(s['top1']) == np.shape(s['coarsened_target_logp']) == (n,) and np.shape(s['top3']) == (n, 3)
                and np.isfinite(s['coarsened_target_logp']).all() and np.all(s['top1'] >= 0)
                and np.array_equal(s['top1'], s['top3'][:, 0]), 'incomplete or invalid scoring rows')
        require(np.array_equal(np.sort(s['top3'], axis=1), np.sort(groups, axis=1)), 'original options changed')
        exact = (s['top1'] == y).astype(float)
        v = {'main_top1': exact, 'main_top3': (s['top3'] == y[:, None]).any(1).astype(float),
             'complete_top1': exact * ~np.asarray(labels['contains_wards'], dtype=bool),
             'typed_main_top1': exact * ((y != 0) | np.asarray(labels['is_no_buy'], dtype=bool)),
             'coarsened_nll': -np.asarray(s['coarsened_target_logp'])}
        values[arm] = v
        summaries[arm] = {key: {'mean': float(a.mean()), 'interval95': np.quantile(a[draws].mean(1), [.025, .975]).tolist(),
                                **({'correct': int(a.sum())} if key != 'coarsened_nll' else {})} for key, a in v.items()}
    for arm in ('personal', 'degree_null'):
        mask = np.asarray(zero_profiles[arm])
        require(mask.shape == (n,) and mask.dtype == bool and np.array_equal(scores[arm]['top3'][mask], scores['m2']['top3'][mask]), 'zero-profile native order changed')
        require(np.array_equal(values[arm]['main_top3'], values['m2']['main_top3']), 'top3 coverage changed')
    paired = {}
    for control in ('m2', 'degree_null'):
        paired[control] = {}
        for key in values['personal']:
            d = values['personal'][key]-values[control][key]
            paired[control][key] = {'difference': float(d.mean()), 'interval95': np.quantile(d[draws].mean(1), [.025, .975]).tolist()}
        d = values['personal']['main_top1']-values[control]['main_top1']
        paired[control].update(corrected=int((d > 0).sum()), broken=int((d < 0).sum()))
    mask = y != 0; difference = values['personal']['main_top1']-values['m2']['main_top1']
    require(mask.sum() >= 30 and np.all(mask[draws].sum(1) > 0), 'nonempty purchase guard has inadequate coverage')
    guard = {'games': int(mask.sum()), 'difference': float(difference[mask].mean()),
             'interval95': np.quantile((difference*mask)[draws].sum(1)/mask[draws].sum(1), [.025, .975]).tolist()}
    gate = (paired['m2']['main_top1']['difference'] >= .03 and paired['m2']['main_top1']['interval95'][0] > 0
            and paired['degree_null']['main_top1']['interval95'][0] > 0 and guard['interval95'][0] > 0)
    kinds = np.where(labels['is_no_buy'], 'observed_no_buy_death', np.where(y == 0, 'ward_only', 'purchase'))
    masks = {**{'kind_'+k: kinds == k for k in sorted(set(kinds))},
             **{'fold_'+str(f): np.asarray(folds) == f for f in sorted(set(folds))},
             'personal_profile_zero': zero_profiles['personal'], 'personal_profile_nonzero': ~zero_profiles['personal']}
    slices = {}
    for key, mask in masks.items():
        slices[key] = {'games': int(mask.sum()), 'status': 'descriptive' if mask.sum() >= 30 else 'too_small_for_rates'}
        if mask.sum() >= 30:
            slices[key]['arms'] = {arm: {name: float(a[mask].mean()) for name, a in v.items()} for arm, v in values.items()}
    return {'games': n, 'arms': summaries, 'personal_minus_control': paired, 'nonempty_purchase_guard': guard,
            'metric_gate_passed': bool(gate), 'candidate_membership_unchanged': True, 'top3_coverage_unchanged': True,
            'zero_profile_order_parity': True, 'descriptive_slices': slices,
            'bootstrap': {'seed': SEED, 'draws': 2000, 'unit': 'whole game, one original focal per game',
                          'limits': 'Conditional on fitted models; no training-set uncertainty.'}}
