"""Synthetic arm/provenance/gate regressions; no future labels or cache reads."""
from copy import deepcopy
import json
import unittest
import numpy as np

from winprob.future_goal_evaluation import ARMS, evaluate_future_goal_probe
from winprob.history_probe_evaluation import evaluate_history_probe, fold_for_game


def arguments(n):
    full = ['future-goal-synthetic-'+str(i) for i in range(4001)]
    return full[:n], dict(selected_state_indices=np.arange(n, dtype=np.int64),
        prediction_fold=np.array([fold_for_game(g) for g in full[:n]], dtype=np.int64),
        training_game_ids_by_fold={str(f): [g for g in full if fold_for_game(g) != f] for f in (0, 1)},
        full_selection_game_ids=full, selection_statuses=['selected']*n+['no_eligible_focal_episode']*(4001-n),
        target_kinds=['other_purchase']*n, class_count=6)


def scores(n, loss, correct=True):
    ranking = [1, 2, 3] if correct else [2, 3, 4]
    top = np.tile(np.array(ranking, dtype=np.int64), (n, 1))
    return {'target_log_probability': -np.full(n, loss, dtype=np.float64), 'top1': top[:, 0].copy(), 'top3': top}


class FutureGoalEvaluationTests(unittest.TestCase):
    def test_metric_reuse_is_exact_with_new_names_in_every_report_level(self):
        n = 80; games, kwargs = arguments(n); labels = np.ones(n, dtype=np.int64)
        arms = {'mainonly': scores(n, 1., False), 'future': scores(n, .8),
                'shuffled': scores(n, 1.1), 'frequency': scores(n, 1.3, False)}
        actual = evaluate_future_goal_probe(games, labels, arms, **kwargs)
        reference = evaluate_history_probe(games, labels,
            {'state': arms['mainonly'], 'history': arms['future'], 'shuffled': arms['shuffled'], 'frequency': arms['frequency']}, **kwargs)
        self.assertEqual(actual['arms']['future'], reference['arms']['history'])
        self.assertEqual(actual['future_goal_contrasts']['mainonly'], reference['history_contrasts']['state'])
        self.assertEqual(actual['metric_gate_passed'], reference['metric_gate_passed'])
        self.assertEqual(set(actual['target_kind_slices']['purchase']['arms']), set(ARMS))
        self.assertIn('future_goal_contrasts', actual['target_kind_slices']['purchase'])
        for fold in actual['per_fold'].values():
            if 'arms' in fold:
                self.assertEqual(set(fold['arms']), set(ARMS))
        encoded = json.dumps(actual)
        self.assertNotIn('"history_contrasts"', encoded)
        self.assertNotIn('"history":', encoded)
        self.assertNotIn('"state":', encoded)
        self.assertNotIn('_vs_state', encoded)
        self.assertFalse(actual['future_goal_labels_used_by_evaluator'])

    def test_shuffled_accuracy_is_reported_but_is_not_a_success_gate(self):
        n = 40; games, kwargs = arguments(n); labels = np.ones(n, dtype=np.int64)
        arms = {'mainonly': scores(n, 1.5, False), 'future': scores(n, 1., False),
                'shuffled': scores(n, 1.2, True), 'frequency': scores(n, 2., False)}
        result = evaluate_future_goal_probe(games, labels, arms, **kwargs)
        self.assertEqual(result['future_goal_contrasts']['shuffled']['top1_gain'], -1.)
        self.assertTrue(result['gates']['positive_paired_nll_reduction_vs_shuffled'])
        self.assertFalse(any('top1' in key and 'shuffled' in key for key in result['gates']))
        self.assertFalse(result['gates']['two_pp_top1_gain_vs_mainonly'])
        self.assertFalse(result['metric_gate_passed'])

    def test_partial_scores_and_old_arm_aliases_cannot_be_published(self):
        n = 3; games, kwargs = arguments(n); labels = np.ones(n, dtype=np.int64)
        arms = {name: scores(n, 1.) for name in ARMS}
        wrong = deepcopy(arms); wrong['history'] = wrong.pop('future')
        with self.assertRaisesRegex(ValueError, 'Exactly mainonly'):
            evaluate_future_goal_probe(games, labels, wrong, **kwargs)
        broken = deepcopy(arms); broken['future']['target_log_probability'] = np.array([-1., -1.])
        with self.assertRaisesRegex(ValueError, 'target log probability'):
            evaluate_future_goal_probe(games, labels, broken, **kwargs)

    def test_whole_game_exposure_guard_is_preserved(self):
        n = 3; games, kwargs = arguments(n); labels = np.ones(n, dtype=np.int64)
        arms = {name: scores(n, 1.) for name in ARMS}
        kwargs['training_game_ids_by_fold'][str(fold_for_game(games[0]))].append(games[0])
        with self.assertRaisesRegex(ValueError, 'opposite whole-game'):
            evaluate_future_goal_probe(games, labels, arms, **kwargs)


if __name__ == '__main__':
    unittest.main()
