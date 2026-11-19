"""Synthetic kernels only; no cache, source history or research targets opened."""
from copy import deepcopy
import json
import unittest
import numpy as np

from winprob.history_probe_evaluation import (ARMS, canonical_inventory_signature,
    fit_frequency, score_frequency, serialize_frequency, summarize_logits,
    evaluate_history_probe, fold_for_game)


def compact(losses, labels, correct=True, classes=6):
    rows = []
    for label in labels:
        first = int(label) if correct else (int(label)+1) % classes
        others = [value for value in range(classes) if value != first]
        rows.append([first, *others[:2]])
    top = np.asarray(rows, dtype=np.int64)
    return {'target_log_probability': -np.asarray(losses, dtype=np.float64),
            'top1': top[:, 0].copy(), 'top3': top}


def provenance(n):
    full = ['synthetic-game-'+str(i) for i in range(4001)]
    selected = full[:n]
    training = {str(f): [game for game in full if fold_for_game(game) != f] for f in (0, 1)}
    return selected, dict(selected_state_indices=np.arange(n, dtype=np.int64),
        prediction_fold=np.asarray([fold_for_game(game) for game in selected]),
        training_game_ids_by_fold=training, full_selection_game_ids=full,
        selection_statuses=['selected']*n+['no_eligible_focal_episode']*(4001-n), class_count=6)


class HistoryFrequencyTests(unittest.TestCase):
    def test_fixed_backoff_formula_and_unseen_keys(self):
        inv = canonical_inventory_signature([10], [1])
        model = fit_frequency([1, 1, 2], [inv, inv, ()], [0, 0, 1], game_ids=['a', 'a', 'b'],
                              allowed_training_game_ids={'a', 'b'}, class_count=3)
        global_p = np.array([3., 2., 1.])/6.
        champion_p = (np.array([2., 0., 0.])+100*global_p)/102
        inventory_p = (np.array([2., 0., 0.])+20*champion_p)/22
        result = score_frequency(model, [1, 1, 3], [inv, (), ()], [0, 0, 0])
        np.testing.assert_allclose(np.exp(result['target_log_probability']),
                                   [inventory_p[0], champion_p[0], global_p[0]], rtol=0, atol=1e-14)
        self.assertEqual(result['top1'].tolist(), [0, 0, 0])
        receipt = serialize_frequency(model)
        self.assertEqual(receipt['training_rows'], 3)
        self.assertEqual(receipt['recipe']['champion_inventory_backoff_mass'], 20)
        json.dumps(receipt)

    def test_inventory_multiplicities_change_the_conditioning_key(self):
        once = canonical_inventory_signature([10], [1])
        twice = canonical_inventory_signature([10, 10], [1, 1])
        self.assertEqual(twice, ((10, 2),))
        self.assertNotEqual(once, twice)
        model = fit_frequency([1, 1], [once, twice], [0, 1], game_ids=['a', 'b'],
                              allowed_training_game_ids={'a', 'b'}, class_count=3)
        result = score_frequency(model, [1, 1], [once, twice], [0, 1])
        self.assertEqual(result['top1'].tolist(), [0, 1])
        with self.assertRaisesRegex(ValueError, 'count'):
            canonical_inventory_signature([10], [True])

    def test_frequency_fit_cannot_see_a_scored_or_missing_training_game(self):
        with self.assertRaisesRegex(ValueError, 'non-training'):
            fit_frequency([1], [()], [0], game_ids=['heldout'], allowed_training_game_ids={'fit'}, class_count=3)
        with self.assertRaisesRegex(ValueError, 'cover'):
            fit_frequency([1], [()], [0], game_ids=['fit1'], allowed_training_game_ids={'fit1', 'fit2'}, class_count=3)

    def test_logits_are_stable_under_probability_underflow_and_target_mutation(self):
        values = np.array([[10000., 0., -10000.], [1., 1., 1.]])
        original = summarize_logits(values, np.array([2, 0]))
        changed = summarize_logits(values, np.array([0, 2]))
        self.assertTrue(np.isfinite(original['target_log_probability']).all())
        self.assertEqual(original['target_log_probability'][0], -20000.)
        np.testing.assert_array_equal(original['top3'], changed['top3'])
        self.assertEqual(original['top3'][1].tolist(), [0, 1, 2])


class HistoryMetricTests(unittest.TestCase):
    def evaluate(self, labels, scores, kinds=None, **updates):
        games, kwargs = provenance(len(labels))
        kwargs['target_kinds'] = kinds or ['observed_no_buy_death' if y == 0 else 'other_purchase' for y in labels]
        kwargs.update(updates)
        return evaluate_history_probe(games, np.asarray(labels), scores, **kwargs)

    def test_identical_arms_have_zero_paired_gains_and_fixed_shared_bootstrap(self):
        labels = [0, 1, 2, 3, 4, 5]
        score = compact([1.]*6, labels)
        result = self.evaluate(labels, {arm: deepcopy(score) for arm in ARMS})
        self.assertFalse(result['metric_gate_passed'])
        self.assertEqual(result['history_contrasts']['state']['nll_reduction_interval95'], [0., 0.])
        self.assertEqual(result['history_contrasts']['state']['top1_gain_interval95'], [0., 0.])
        self.assertEqual(result['bootstrap']['samples'], 2000)
        self.assertEqual(result['bootstrap']['seed'], 20260930)
        self.assertEqual(result['prediction_coverage'], 1.)

    def test_relative_nll_bootstrap_uses_ratio_of_resampled_means(self):
        labels = [1, 2]
        scores = {arm: compact([1., 4.], labels) for arm in ARMS}
        scores['history'] = compact([.5, 3.], labels)
        result = self.evaluate(labels, scores)
        contrast = result['history_contrasts']['state']
        self.assertEqual(contrast['relative_nll_reduction'], .3)
        draws = np.random.default_rng(20260930).integers(0, 2, size=(2000, 2), dtype=np.int32)
        baseline = np.array([1., 4.]); reduction = np.array([.5, 1.])
        expected = np.quantile(reduction[draws].mean(1)/baseline[draws].mean(1), [.025, .975])
        np.testing.assert_allclose(contrast['relative_nll_reduction_interval95'], expected)
        self.assertNotEqual(contrast['relative_nll_reduction'], .375)

    def test_no_selected_prediction_or_bad_fold_can_be_dropped(self):
        labels = [0, 1, 2]
        scores = {arm: compact([1.]*3, labels) for arm in ARMS}
        bad = deepcopy(scores); bad['history']['top1'] = bad['history']['top1'][:-1]
        with self.assertRaisesRegex(ValueError, 'top-three'):
            self.evaluate(labels, bad)
        games, kwargs = provenance(3)
        kwargs['training_game_ids_by_fold'][str(fold_for_game(games[0]))].append(games[0])
        with self.assertRaisesRegex(ValueError, 'opposite'):
            self.evaluate(labels, scores, training_game_ids_by_fold=kwargs['training_game_ids_by_fold'])
        bad_fold = kwargs['prediction_fold'].copy(); bad_fold[0] = 1-bad_fold[0]
        with self.assertRaisesRegex(ValueError, 'original whole-game'):
            self.evaluate(labels, scores, prediction_fold=bad_fold)
        with self.assertRaisesRegex(ValueError, 'target-kind'):
            self.evaluate(labels, scores, target_kinds=['unknown']*3)

    def test_group_zero_and_target_kind_cannot_be_conflated_or_filtered(self):
        labels = [0, 0, 1]
        scores = {arm: compact([1.]*3, labels) for arm in ARMS}
        result = self.evaluate(labels, scores, ['observed_no_buy_death', 'ward_only_purchase', 'other_purchase'])
        self.assertEqual(result['target_kind_counts'], {'no_buy': 1, 'ward_only': 1, 'purchase': 1})
        self.assertEqual(result['target_kind_slices']['ward_only']['status'], 'not_measured_below_30_games')
        with self.assertRaisesRegex(ValueError, 'group0'):
            self.evaluate(labels, scores, ['other_purchase']*3)

    def test_zero_baseline_loss_does_not_invent_relative_gain(self):
        labels = [1, 2]
        scores = {arm: compact([0., 0.], labels) for arm in ARMS}
        result = self.evaluate(labels, scores)
        self.assertIsNone(result['history_contrasts']['state']['relative_nll_reduction'])
        self.assertIsNone(result['history_contrasts']['state']['relative_nll_reduction_interval95'])
        self.assertFalse(result['gates']['positive_paired_relative_nll_reduction_vs_state'])

    def test_large_enough_slices_and_fold_metrics_keep_every_arm(self):
        labels = [i % 5+1 for i in range(80)]
        scores = {arm: compact([1.]*80, labels) for arm in ARMS}
        result = self.evaluate(labels, scores)
        self.assertEqual(set(result['target_kind_slices']['purchase']['arms']), set(ARMS))
        for fold in ('0', '1'):
            self.assertGreaterEqual(result['per_fold'][fold]['games'], 30)
            self.assertEqual(set(result['per_fold'][fold]['arms']), set(ARMS))


if __name__ == '__main__':
    unittest.main()
