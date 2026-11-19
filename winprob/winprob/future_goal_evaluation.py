"""Immediate-action evaluation for fixed future-goal auxiliary supervision.

Only immediate main-action labels and compact OOF predictions enter this
module. Future-goal labels are neither an evaluator input nor an inference
input. The existing frozen paired-metric arithmetic is reused explicitly;
all public arm names and interpretation describe this new experiment.
"""
from __future__ import annotations

from .history_probe_evaluation import (
    CLASS_COUNT, canonical_inventory_signature, fit_frequency, score_frequency,
    serialize_frequency, summarize_logits, evaluate_history_probe,
)

ARMS = ('mainonly', 'future', 'shuffled', 'frequency')
_INTERNAL = {'mainonly': 'state', 'future': 'history', 'shuffled': 'shuffled', 'frequency': 'frequency'}
_PUBLIC = {value: key for key, value in _INTERNAL.items()}


def _public_report(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = 'future_goal_contrasts' if key == 'history_contrasts' else _PUBLIC.get(key, key)
            name = name.replace('_vs_state', '_vs_mainonly')
            result[name] = _public_report(item)
        return result
    if isinstance(value, list):
        return [_public_report(item) for item in value]
    return value


def evaluate_future_goal_probe(game_ids, labels, scores_by_arm, *, selected_state_indices, prediction_fold,
                               training_game_ids_by_fold, full_selection_game_ids, selection_statuses,
                               target_kinds, class_count=CLASS_COUNT):
    """Exactly the frozen paired gates, using the honest new experiment arms.

Models must already be fixed. The caller proves fresh matched learners,
strict-later supervision, target-blind inference and source provenance.
Frequency is a reported fixed control; it cannot replace either mainonly
or shuffled in the primary gate.
"""
    if not isinstance(scores_by_arm, dict) or set(scores_by_arm) != set(ARMS):
        raise ValueError('Exactly mainonly/future/shuffled/frequency compact predictions required')
    internal = {_INTERNAL[arm]: scores_by_arm[arm] for arm in ARMS}
    result = _public_report(evaluate_history_probe(
        game_ids, labels, internal, selected_state_indices=selected_state_indices,
        prediction_fold=prediction_fold, training_game_ids_by_fold=training_game_ids_by_fold,
        full_selection_game_ids=full_selection_game_ids, selection_statuses=selection_statuses,
        target_kinds=target_kinds, class_count=class_count))
    result.update(
        schema='future-goal-auxiliary-main-metrics-v1',
        scope='whole-game OOF immediate-action prediction with future-goal auxiliary supervision on existing FIT; not external confirmation',
        endpoint='unfiltered9333 main-action NLL and exact top1; top3 diagnostic; future goal is never an inference input',
        future_goal_labels_used_by_evaluator=False,
        mathematical_reuse='Frozen history_probe_evaluation paired arithmetic; explicit arm mapping only, no reused fitted model or encoder',
        supervision={'mainonly_goal_loss_weight': 0., 'future_goal_loss_weight': 1.,
                     'shuffled_goal_loss_weight': 1., 'goal_classes': 258,
                     'goal_definition': 'strictly later completed goal within360000ms; NONE statuses merge to one EMPTY class'},
        primary_comparisons={'mainonly': 'at least2% relative NLL reduction and2pp top1 gain; both paired95% lower bounds positive',
                             'shuffled': 'positive paired95% lower bound for NLL reduction; accuracy remains reported only'},
        remaining_gates=[
            'Bind the fixed strict-later goal census, exact258-class EMPTY merge, population and selected rows before fitting.',
            'Goal labels must never enter main inference; current/future label mutation must preserve main logits.',
            'Fresh matched initialization, full-class capacity, epoch/batch order, optimizer recipe and all completed models require independent receipts.',
            'Goal-shuffle donors come from other training-fold games within current-action/patch/EMPTY strata; unchanged-label and no-donor fallback coverage must remain visible.',
            'Native-unavailable candidate metadata zeros only its compositional contribution; free-class scores remain and invalid observed targets fail.',
            'The caller must AND these integrity/runtime proofs with metric_gate_passed; no secondary can rescue the primary.',
            'Auxiliary future supervision is not new observed information, verified private intention, causal action value, win benefit or serving approval.',
        ])
    return result
