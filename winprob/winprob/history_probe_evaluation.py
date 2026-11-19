"""Fixed whole-game OOF history-probe metrics and sparse frequency control.

No source payload access, neural fitting, legality decoder, or outcome input.
Fold keys identify the scored fold; each learner sees every opposite-fold game.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
import numpy as np

ARMS = ('state', 'history', 'shuffled', 'frequency')
CLASS_COUNT = 9333
LEDGER_GAMES = 4001
SEED = 20260930
BOOTSTRAPS = 2000
CHAMPION_MASS = 100.
INVENTORY_MASS = 20.
TARGET_KIND_MAP = {'no_buy': 'no_buy', 'ward_only': 'ward_only', 'purchase': 'purchase', 'mixed': 'mixed',
                   'observed_no_buy_death': 'no_buy', 'ward_only_purchase': 'ward_only',
                   'other_purchase': 'purchase', 'mixed_purchase_with_wards': 'mixed'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fold_for_game(game_id):
    return int.from_bytes(hashlib.sha256(('iql-v1-fold:'+game_id).encode()).digest()[:8], 'big') % 2


def integer(value, label, minimum=0):
    require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
            and int(value) >= minimum, 'Invalid '+label)
    return int(value)


def canonical_inventory_signature(item_ids, item_counts):
    """Canonical multiset of the focal predecision inventory; never a set."""
    result = Counter()
    for item, count in zip(item_ids, item_counts, strict=True):
        result[integer(item, 'inventory item', 1)] += integer(count, 'inventory count', 1)
    return tuple(sorted(result.items()))


def checked_signature(signature):
    require(isinstance(signature, tuple) and all(isinstance(pair, tuple) and len(pair) == 2 for pair in signature),
            'Use canonical_inventory_signature for immutable inventory keys')
    parsed = canonical_inventory_signature([pair[0] for pair in signature], [pair[1] for pair in signature])
    require(signature == parsed, 'Inventory signature is not canonical/count-preserving')
    return parsed


@dataclass
class FrequencyModel:
    class_count: int
    global_counts: np.ndarray
    champion_counts: dict
    inventory_counts: dict
    training_game_ids: tuple
    training_rows: int


def fit_frequency(champions, inventory_signatures, main_labels, *, game_ids,
                  allowed_training_game_ids, class_count=CLASS_COUNT):
    """One fold only; row identities must belong to its explicit training set."""
    class_count = integer(class_count, 'class count', 3)
    allowed = set(allowed_training_game_ids)
    require(allowed and all(isinstance(g, str) and g for g in allowed), 'Explicit training-game whitelist required')
    global_counts = np.zeros(class_count, dtype=np.int64)
    by_champion, by_inventory, seen = {}, {}, set()
    n = 0
    for champion, inventory, label, game in zip(champions, inventory_signatures, main_labels, game_ids, strict=True):
        require(game in allowed, 'Frequency fitting received a non-training game')
        champ = integer(champion, 'champion index', 1)
        label = integer(label, 'main action label')
        require(label < class_count, 'Main action outside fixed catalogue')
        key = (champ, checked_signature(inventory))
        global_counts[label] += 1
        by_champion.setdefault(champ, Counter())[label] += 1
        by_inventory.setdefault(key, Counter())[label] += 1
        seen.add(game); n += 1
    require(n > 0 and seen == allowed, 'Frequency training rows do not cover the declared opposite-fold games')
    return FrequencyModel(class_count, global_counts, by_champion, by_inventory, tuple(sorted(seen)), n)


def score_frequency(model, champions, inventory_signatures, labels):
    """Fixed Dirichlet backoff; selected targets only extract their log scores."""
    require(isinstance(model, FrequencyModel), 'Expected frequency model')
    prior = (model.global_counts.astype(np.float64)+1.)/(model.training_rows+model.class_count)
    champion_cache, target_logp, top_rows = {}, [], []
    for champion, inventory, label in zip(champions, inventory_signatures, labels, strict=True):
        champ = integer(champion, 'champion index', 1)
        signature = checked_signature(inventory)
        label = integer(label, 'score label')
        require(label < model.class_count, 'Score label outside fixed catalogue')
        if champ not in champion_cache:
            counts = model.champion_counts.get(champ, {})
            probability = CHAMPION_MASS*prior
            for action, count in counts.items():
                probability[action] += count
            champion_cache[champ] = probability/(sum(counts.values())+CHAMPION_MASS)
        counts = model.inventory_counts.get((champ, signature), {})
        probability = INVENTORY_MASS*champion_cache[champ]
        for action, count in counts.items():
            probability[action] += count
        probability /= sum(counts.values())+INVENTORY_MASS
        require(np.all(probability > 0) and np.isfinite(probability).all()
                and math.isclose(float(probability.sum()), 1., rel_tol=0, abs_tol=1e-12), 'Invalid backoff distribution')
        target_logp.append(math.log(float(probability[label])))
        top_rows.append(np.argsort(-probability, kind='stable')[:3])
    require(target_logp, 'Empty frequency scoring population')
    top3 = np.asarray(top_rows, dtype=np.int64)
    return {'target_log_probability': np.asarray(target_logp, dtype=np.float64), 'top1': top3[:, 0].copy(), 'top3': top3}


def serialize_frequency(model):
    """Compact JSON-safe sparse counts; callers may gzip the resulting artifact."""
    return {'schema': 'history-probe-frequency-v1', 'class_count': model.class_count,
            'training_rows': model.training_rows, 'training_game_ids': list(model.training_game_ids),
            'recipe': {'global_count_per_class': 1., 'champion_backoff_mass': CHAMPION_MASS,
                       'champion_inventory_backoff_mass': INVENTORY_MASS, 'tie_rule': 'ascending main class ID'},
            'global_counts': model.global_counts.tolist(),
            'champion_counts': [[champion, sorted(counts.items())] for champion, counts in sorted(model.champion_counts.items())],
            'inventory_counts': [[champion, list(inventory), sorted(counts.items())]
                                 for (champion, inventory), counts in sorted(model.inventory_counts.items())]}


def summarize_logits(logits, labels):
    """Compact sufficient metrics from complete logits, using stable logsumexp."""
    values = np.asarray(logits, dtype=np.float64); target = np.asarray(labels)
    require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] >= 3 and np.isfinite(values).all(),
            'Finite nonempty complete main-class logits required')
    require(target.shape == (len(values),) and np.issubdtype(target.dtype, np.integer)
            and np.all((target >= 0) & (target < values.shape[1])), 'Aligned integer targets required')
    shifted = values-values.max(axis=1, keepdims=True)
    logp = shifted[np.arange(len(values)), target]-np.log(np.exp(shifted).sum(axis=1))
    top3 = np.argsort(-values, axis=1, kind='stable')[:, :3].astype(np.int64)
    return {'target_log_probability': logp.astype(np.float64), 'top1': top3[:, 0].copy(), 'top3': top3}


def validate_summary(value, n, class_count):
    require(isinstance(value, dict) and set(value) == {'target_log_probability', 'top1', 'top3'}, 'Exact compact score fields required')
    logp = np.asarray(value['target_log_probability'], dtype=np.float64)
    top1, top3 = np.asarray(value['top1']), np.asarray(value['top3'])
    require(logp.shape == (n,) and np.isfinite(logp).all() and np.all(logp <= 0), 'Finite nonpositive target log probability required')
    require(top1.shape == (n,) and top3.shape == (n, 3)
            and np.issubdtype(top1.dtype, np.integer) and np.issubdtype(top3.dtype, np.integer)
            and np.all((top3 >= 0) & (top3 < class_count))
            and np.array_equal(top1, top3[:, 0])
            and all(len(set(row)) == 3 for row in top3.tolist()), 'Aligned unique top-three class ranking required')
    return logp, top1, top3


def bootstrap_metrics(loss, accuracy, top3, indexes):
    result = {}
    for arm_index, name in enumerate(ARMS):
        row = {}
        for metric, values in (('nll', loss[:, arm_index]), ('top1_accuracy', accuracy[:, arm_index]), ('top3_accuracy', top3[:, arm_index])):
            sampled = values[indexes].mean(axis=1)
            row[metric] = {'value': float(values.mean()), 'interval95': np.quantile(sampled, [.025, .975]).tolist()}
        result[name] = row
    return result


def evaluate_history_probe(game_ids, labels, scores_by_arm, *, selected_state_indices, prediction_fold,
                           training_game_ids_by_fold, full_selection_game_ids, selection_statuses,
                           target_kinds, class_count=CLASS_COUNT):
    """All selected whole games, without post-selection support/label exclusions."""
    class_count = integer(class_count, 'class count', 3)
    full = list(full_selection_game_ids); statuses = list(selection_statuses); ids = list(game_ids)
    require(len(full) == len(statuses) == LEDGER_GAMES and len(set(full)) == LEDGER_GAMES
            and all(isinstance(g, str) and g for g in full), 'Complete fixed4001-game selection ledger required')
    require(set(statuses) <= {'selected', 'no_eligible_focal_episode'}, 'Selection status cannot hide post-selection exclusions')
    selected = [g for g, status in zip(full, statuses, strict=True) if status == 'selected']
    require(ids == selected and len(ids) >= 2 and len(set(ids)) == len(ids), 'All selected games must be scored once in ledger order')
    n = len(ids); y = np.asarray(labels); states = np.asarray(selected_state_indices); fold = np.asarray(prediction_fold)
    require(y.shape == (n,) and np.issubdtype(y.dtype, np.integer) and np.all((y >= 0) & (y < class_count)), 'All selected targets must be retained')
    require(states.shape == (n,) and np.issubdtype(states.dtype, np.integer) and np.all(states >= 0)
            and len(set(states.tolist())) == n, 'Unique selected state indices required')
    require(fold.shape == (n,) and np.issubdtype(fold.dtype, np.integer)
            and np.array_equal(fold, [fold_for_game(g) for g in ids]), 'Prediction folds differ from original whole-game folds')
    require(isinstance(training_game_ids_by_fold, dict), 'Fold training exposure required')
    training = {str(key): set(value) for key, value in training_game_ids_by_fold.items()}
    require(set(training) == {'0', '1'} and len(training_game_ids_by_fold) == 2, 'Both held-out-fold training receipts required')
    for scored_fold in (0, 1):
        expected = {g for g in full if fold_for_game(g) != scored_fold}
        require(training[str(scored_fold)] == expected, 'Learner must fit exactly the opposite whole-game fold')
        require(not training[str(scored_fold)] & {g for g, f in zip(ids, fold, strict=True) if f == scored_fold}, 'Scored game leaked into learner')
    raw_kinds = list(target_kinds)
    require(len(raw_kinds) == n and set(raw_kinds) <= set(TARGET_KIND_MAP), 'Explicit complete target-kind partition required')
    kinds = [TARGET_KIND_MAP[kind] for kind in raw_kinds]
    require(np.array_equal(np.asarray([kind in ('no_buy', 'ward_only') for kind in kinds]), y == 0),
            'No-buy/ward-only kinds must coincide with the empty main group0')
    require(set(scores_by_arm) == set(ARMS), 'All four frozen arms required; no partial-arm publication')
    loss = np.empty((n, 4), dtype=np.float64); accuracy = np.empty((n, 4), dtype=np.float64); hit3 = np.empty((n, 4), dtype=np.float64)
    for column, arm in enumerate(ARMS):
        logp, top1, top = validate_summary(scores_by_arm[arm], n, class_count)
        loss[:, column] = -logp; accuracy[:, column] = top1 == y
        hit3[:, column] = (top == y[:, None]).any(axis=1)
    indexes = np.random.default_rng(SEED).integers(0, n, size=(BOOTSTRAPS, n), dtype=np.int32)
    arms = bootstrap_metrics(loss, accuracy, hit3, indexes)
    contrasts = {}
    for name in ('state', 'shuffled', 'frequency'):
        column = ARMS.index(name); improvement = loss[:, column]-loss[:, 1]
        top1_gain = accuracy[:, 1]-accuracy[:, column]
        base = float(loss[:, column].mean())
        improvement_draws = improvement[indexes].mean(axis=1)
        baseline_draws = loss[:, column][indexes].mean(axis=1)
        relative_interval = (np.quantile(improvement_draws/baseline_draws, [.025, .975]).tolist()
                             if np.all(baseline_draws > 0) else None)
        contrasts[name] = {
            'nll_reduction': float(improvement.mean()),
            'nll_reduction_interval95': np.quantile(improvement_draws, [.025, .975]).tolist(),
            'relative_nll_reduction': float(improvement.mean()/base) if base > 0 else None,
            'relative_nll_reduction_interval95': relative_interval,
            'top1_gain': float(top1_gain.mean()),
            'top1_gain_interval95': np.quantile(top1_gain[indexes].mean(axis=1), [.025, .975]).tolist(),
            'top1_improved_games': int((top1_gain > 0).sum()), 'top1_regressed_games': int((top1_gain < 0).sum())}
    primary = contrasts['state']
    gates = {'minimum_selected_games': n >= 3000, 'complete_selected_prediction_coverage': True,
             'two_percent_relative_nll_reduction_vs_state': bool(primary['relative_nll_reduction'] is not None and primary['relative_nll_reduction'] >= .02),
             'two_pp_top1_gain_vs_state': 100*(primary['top1_improved_games']-primary['top1_regressed_games']) >= 2*n,
             'positive_paired_relative_nll_reduction_vs_state': bool(primary['relative_nll_reduction_interval95'] is not None
                                                                  and primary['relative_nll_reduction_interval95'][0] > 0),
             'positive_paired_top1_gain_vs_state': primary['top1_gain_interval95'][0] > 0,
             'positive_paired_nll_reduction_vs_shuffled': contrasts['shuffled']['nll_reduction_interval95'][0] > 0}
    slices = {}
    for label, allowed in (('purchase', {'purchase', 'mixed', 'ward_only'}),
                           ('observed_no_buy', {'no_buy'}), ('ward_only', {'ward_only'})):
        chosen = np.flatnonzero([kind in allowed for kind in kinds])
        if len(chosen) < 30:
            slices[label] = {'games': len(chosen), 'status': 'not_measured_below_30_games'}
            continue
        local = np.random.default_rng(SEED).integers(0, len(chosen), size=(BOOTSTRAPS, len(chosen)), dtype=np.int32)
        slice_contrasts = {}
        for name in ('state', 'shuffled', 'frequency'):
            column = ARMS.index(name)
            ll = loss[chosen, column]-loss[chosen, 1]; aa = accuracy[chosen, 1]-accuracy[chosen, column]
            slice_contrasts[name] = {'nll_reduction': float(ll.mean()),
                'nll_reduction_interval95': np.quantile(ll[local].mean(axis=1), [.025, .975]).tolist(),
                'top1_gain': float(aa.mean()), 'top1_gain_interval95': np.quantile(aa[local].mean(axis=1), [.025, .975]).tolist()}
        slices[label] = {'games': len(chosen), 'scope': 'exploratory target-kind slice',
                         'arms': bootstrap_metrics(loss[chosen], accuracy[chosen], hit3[chosen], local),
                         'history_contrasts': slice_contrasts}
    per_fold = {}
    for value in (0, 1):
        chosen = np.flatnonzero(fold == value)
        if len(chosen) < 30:
            per_fold[str(value)] = {'games': len(chosen), 'status': 'not_measured_below_30_games'}
        else:
            local = np.random.default_rng(SEED).integers(0, len(chosen), size=(BOOTSTRAPS, len(chosen)), dtype=np.int32)
            per_fold[str(value)] = {'games': len(chosen),
                                   'arms': bootstrap_metrics(loss[chosen], accuracy[chosen], hit3[chosen], local)}
    return {'schema': 'history-information-probe-metrics-v1', 'selection_ledger_games': LEDGER_GAMES,
            'selected_games': n, 'no_eligible_focal_episode_games': LEDGER_GAMES-n, 'prediction_coverage': 1.,
            'arms': arms, 'history_contrasts': contrasts, 'gates': gates, 'metric_gate_passed': all(gates.values()),
            'target_kind_counts': dict(Counter(kinds)), 'target_kind_slices': slices, 'per_fold': per_fold,
            'selected_state_indices_sha256': hashlib.sha256(states.astype('<i8').tobytes()).hexdigest(),
            'selected_game_order_sha256': hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
            'prediction_fold_games': {str(f): int((fold == f).sum()) for f in (0, 1)},
            'training_fold_games': {key: len(value) for key, value in training.items()},
            'bootstrap': {'unit': 'whole selected game', 'samples': BOOTSTRAPS, 'seed': SEED,
                          'shared_games_all_arms': True, 'interval': 'paired percentile; fold learners held fixed'},
            'endpoint': 'unfiltered main-class NLL and exact top1; top3 diagnostic; no-buy and ward-only share main0',
            'scope': 'whole-game OOF information probe on existing FIT; not external confirmation',
            'remaining_gates': ['Bound source selection, strict prior-visit closure, fixed basket-only shuffle and fresh learner provenance must pass upstream.',
                                'This does not assess displayed legality, causal action value, win rate, or serving benefit.']}
