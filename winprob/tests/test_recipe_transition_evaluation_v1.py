from copy import deepcopy
import inspect
import unittest
import numpy as np
from winprob.recipe_transition_v1 import ChoiceInputs
from winprob.recipe_transition_probe_v1 import Standardizer
from winprob.recipe_transition_evaluation_v1 import predict_correction, compact_prediction, evaluate


class EvaluationTests(unittest.TestCase):
    def predictions(self):
        n = 300; groups = np.tile([2, 1, 0], (n, 1)); result = {}
        for arm, correct in (('m2', 0), ('raw', 15), ('net', 45), ('shuffled_net', 5)):
            top = groups.copy(); top[:correct] = [1, 2, 0]
            result[arm] = {'top1': top[:, 0], 'top3': top, 'coarsened_target_logp': np.full(n, np.log(.25))}
        labels = {'main': np.ones(n, dtype=int), 'contains_wards': np.zeros(n, bool),
                  'is_no_buy': np.zeros(n, bool), 'main_size': np.ones(n, dtype=int)}
        return labels, result, groups, [str(i) for i in range(n)], np.arange(n) % 2

    def test_declared_gate_and_corrected_broken_accounting(self):
        report = evaluate(*self.predictions())
        self.assertTrue(report['metric_gate_passed'])
        self.assertEqual(report['net_minus_control']['m2']['corrected'], 45)
        self.assertEqual(report['net_minus_control']['m2']['broken'], 0)
        self.assertAlmostEqual(report['net_minus_control']['m2']['main_top1']['difference'], .15)
        self.assertEqual(report['arms']['net']['main_top3']['correct'], 300)

    def test_improvement_over_raw_control_is_required(self):
        labels, scores, groups, ids, folds = self.predictions()
        scores['raw'] = deepcopy(scores['net'])
        self.assertFalse(evaluate(labels, scores, groups, ids, folds)['metric_gate_passed'])

    def test_moving_candidates_is_not_allowed(self):
        labels, scores, groups, ids, folds = self.predictions()
        scores['net']['top3'][0, 2] = 7
        with self.assertRaises(ValueError):
            evaluate(labels, scores, groups, ids, folds)

    def test_ward_only_cannot_be_a_complete_or_typed_save_hit(self):
        labels, scores, groups, ids, folds = self.predictions()
        labels['main'][:2] = 0; labels['main_size'][:2] = 0
        labels['contains_wards'][0] = True; labels['is_no_buy'][1] = True
        for score in scores.values():
            score['top3'][:2] = [0, 1, 2]; score['top1'] = score['top3'][:, 0].copy()
        report = evaluate(labels, scores, groups, ids, folds)
        self.assertEqual(report['arms']['m2']['main_top1']['correct'], 2)
        self.assertEqual(report['arms']['m2']['complete_top1']['correct'], 1)
        self.assertEqual(report['arms']['m2']['typed_main_top1']['correct'], 1)
        self.assertEqual(report['descriptive_slices']['kind_ward_only']['status'], 'too_small_for_rates')

    def test_stable_log_probability_survives_underflow_without_target_inputs(self):
        h = np.zeros((1, 26)); h[0, 0] = 1
        f = np.zeros((1, 3, 26)); f[0, 0, 0] = 1
        inputs = ChoiceInputs(np.array([[.2, .2, .2, .4]]), h, f)
        scaler = Standardizer(np.zeros(26), np.ones(26), np.zeros(26), np.ones(26))
        theta = np.zeros((26, 26)); theta[0, 0] = 1000
        out = predict_correction(inputs, None, scaler, theta, 'net')
        self.assertEqual(out['probabilities'][0, 3], 0.)
        self.assertTrue(np.isfinite(out['log_probabilities'][0, 3]))
        compact = compact_prediction(out, np.array([[0, 1, 2]]), np.array([3]))
        self.assertLess(compact['coarsened_target_logp'][0], -900)
        self.assertNotIn('target', inspect.signature(predict_correction).parameters)


if __name__ == '__main__':
    unittest.main()
