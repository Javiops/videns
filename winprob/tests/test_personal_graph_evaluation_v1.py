from copy import deepcopy
import unittest
import numpy as np
from winprob.personal_graph_evaluation_v1 import evaluate


class GraphEvaluationTests(unittest.TestCase):
    def fixture(self):
        n = 300; groups = np.tile([2, 1, 0], (n, 1)); scores = {}
        for arm, correct in (('m2', 0), ('personal', 45), ('degree_null', 15)):
            top = groups.copy(); top[:correct] = [1, 2, 0]
            scores[arm] = {'top1': top[:, 0], 'top3': top, 'coarsened_target_logp': np.full(n, np.log(.25))}
        labels = {'main': np.ones(n, dtype=int), 'contains_wards': np.zeros(n, bool), 'is_no_buy': np.zeros(n, bool)}
        zero = {arm: np.arange(n) >= 270 for arm in ('personal', 'degree_null')}
        return labels, scores, groups, [str(i) for i in range(n)], np.arange(n) % 2, zero

    def test_gate_requires_practical_and_relational_improvement(self):
        values = self.fixture(); report = evaluate(*values)
        self.assertTrue(report['metric_gate_passed'])
        self.assertEqual(report['personal_minus_control']['m2']['corrected'], 45)
        self.assertEqual(report['personal_minus_control']['m2']['broken'], 0)
        values[1]['degree_null'] = deepcopy(values[1]['personal'])
        self.assertFalse(evaluate(*values)['metric_gate_passed'])

    def test_zero_profile_parity_is_mandatory(self):
        values = self.fixture(); values[-1]['personal'][0] = True
        with self.assertRaises(ValueError):
            evaluate(*values)

    def test_option_membership_cannot_change(self):
        values = self.fixture(); values[1]['personal']['top3'][0, 2] = 99
        with self.assertRaises(ValueError):
            evaluate(*values)

    def test_ward_only_is_not_complete_or_typed_save(self):
        values = self.fixture(); labels, scores = values[:2]
        labels['main'][:2] = 0; labels['contains_wards'][0] = True; labels['is_no_buy'][1] = True
        for s in scores.values():
            s['top3'][:2] = [0, 1, 2]; s['top1'] = s['top3'][:, 0].copy()
        report = evaluate(*values)
        self.assertEqual(report['arms']['m2']['main_top1']['correct'], 2)
        self.assertEqual(report['arms']['m2']['complete_top1']['correct'], 1)
        self.assertEqual(report['arms']['m2']['typed_main_top1']['correct'], 1)


if __name__ == '__main__':
    unittest.main()
