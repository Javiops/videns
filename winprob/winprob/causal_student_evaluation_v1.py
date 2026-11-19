"""Fixed two-candidate native scoring, adjusted primary intervals and selection."""
from __future__ import annotations
import numpy as np

def require(ok, message):
    if not ok: raise ValueError(message)

ARMS = ('m2', 'hard', 'distilled', 'shuffled_teacher')
CANDIDATES = ('hard', 'distilled')


def evaluate(labels, scores, game_ids, folds, legality_failures):
    y = np.asarray(labels['main']); n = len(y)
    require(set(scores) == set(ARMS) and len(game_ids) == len(set(game_ids)) == n and np.shape(folds) == (n,), 'paired one-focal-per-game scores required')
    wards = np.asarray(labels['contains_wards'], dtype=bool); no_buy = np.asarray(labels['is_no_buy'], dtype=bool); size = np.asarray(labels['main_size'])
    require(wards.shape == no_buy.shape == size.shape == (n,) and np.all(y >= 0) and np.all(size >= 0)
            and not np.any(no_buy & ((y != 0) | wards)), 'counted target semantics changed')
    require(set(legality_failures) == {'16.17', '16.18'} and all(set(v) == set(ARMS)
            and all(type(c) is int and c >= 0 for c in v.values()) for v in legality_failures.values()), 'both-world legality counts required')
    draws = np.random.default_rng(20261004).integers(0, n, size=(2000, n))
    values = {}; arms = {}; changes = {}
    for arm, prediction in scores.items():
        top = np.asarray(prediction['top3']); first = np.asarray(prediction['top1']); logp = np.asarray(prediction['catalogue_target_logp'])
        require(top.shape == (n, 3) and first.shape == logp.shape == (n,) and np.all(top >= -1) and np.all(first >= 0)
                and np.array_equal(first, top[:, 0]) and np.isfinite(logp).all() and np.all(logp <= 1e-12)
                and all(len(set(row[row >= 0])) == int((row >= 0).sum()) for row in top), 'invalid native recommendation/probability')
        correct = (first == y).astype(float)
        v = {'main_top1': correct, 'main_top3': (top == y[:, None]).any(1).astype(float),
             'complete_top1': correct*~wards, 'typed_main_top1': correct*((y != 0) | no_buy), 'catalogue_nll': -logp}
        values[arm] = v
        arms[arm] = {k: {'mean': float(a.mean()), 'interval95': np.quantile(a[draws].mean(1), [.025, .975]).tolist(),
                         **({'correct': int(a.sum())} if k != 'catalogue_nll' else {})} for k, a in v.items()}
        if arm != 'm2':
            base = scores['m2']['top3']
            changes[arm] = {'changed_first': int((first != base[:, 0]).sum()),
                'changed_option_set': int(sum(set(a[a >= 0]) != set(b[b >= 0]) for a, b in zip(top, base, strict=True)))}
    contrasts = {}
    pairs = [('hard', 'm2'), ('distilled', 'm2'), ('shuffled_teacher', 'm2'), ('distilled', 'hard'), ('distilled', 'shuffled_teacher')]
    for candidate, control in pairs:
        key = candidate+'__minus__'+control; contrast = {}
        for metric in values[candidate]:
            d = values[candidate][metric]-values[control][metric]
            contrast[metric] = {'difference': float(d.mean()), 'interval95': np.quantile(d[draws].mean(1), [.025, .975]).tolist()}
            if candidate in CANDIDATES and control == 'm2' and metric == 'main_top1':
                contrast[metric]['primary_interval97_5'] = np.quantile(d[draws].mean(1), [.0125, .9875]).tolist()
        delta = values[candidate]['main_top1']-values[control]['main_top1']
        contrast.update(corrected=int((delta > 0).sum()), broken=int((delta < 0).sum())); contrasts[key] = contrast
    nonempty = y != 0; require(nonempty.sum() >= 30 and np.all(nonempty[draws].sum(1) > 0), 'nonempty guard population insufficient')
    nonempty_guards = {}; gates = {}; legal = all(c == 0 for counts in legality_failures.values() for c in counts.values())
    for candidate in CANDIDATES:
        d = values[candidate]['main_top1']-values['m2']['main_top1']
        nonempty_guards[candidate] = {'games': int(nonempty.sum()), 'difference': float(d[nonempty].mean()),
            'interval95': np.quantile((d*nonempty)[draws].sum(1)/nonempty[draws].sum(1), [.025, .975]).tolist()}
        c = contrasts[candidate+'__minus__m2']
        components = {'first_gain_at_least_3pp': c['main_top1']['difference'] >= .03,
            'adjusted_primary_lower_positive': c['main_top1']['primary_interval97_5'][0] > 0,
            'nonempty_lower_positive': nonempty_guards[candidate]['interval95'][0] > 0,
            'top3_mean_not_worse': c['main_top3']['difference'] >= 0, 'catalogue_nll_mean_not_worse': c['catalogue_nll']['difference'] <= 0,
            'all_native_legality_passed': legal}
        gates[candidate] = {'components': {k: bool(v) for k, v in components.items()}, 'qualified': bool(all(components.values()))}
    qualified = [a for a in CANDIDATES if gates[a]['qualified']]
    selected = min(qualified, key=lambda a: (-arms[a]['main_top1']['mean'], arms[a]['catalogue_nll']['mean'], a != 'hard')) if qualified else None
    teacher_signal = (contrasts['distilled__minus__hard']['main_top1']['interval95'][0] > 0
                      and contrasts['distilled__minus__shuffled_teacher']['main_top1']['interval95'][0] > 0)
    return {'games': n, 'arms': arms, 'paired_contrasts': contrasts, 'nonempty_purchase_guards': nonempty_guards,
        'candidate_gates': gates, 'selected_candidate': selected, 'metric_gate_passed': bool(qualified),
        'teacher_specific_first_signal': bool(teacher_signal), 'teacher_signal_scope': 'Within-student diagnostic only; not an M2 benefit by itself.',
        'candidate_changes': changes, 'legality_failures': legality_failures,
        'bootstrap': {'draws': 2000, 'seed': 20261004, 'unit': 'whole game, one original focal',
            'ordinary_quantiles': [.025, .975], 'two_candidate_primary_quantiles': [.0125, .9875],
            'limits': 'Conditional on fitted models; exposed internal cohort, no training uncertainty or broader-program multiplicity correction.'}}
