import unittest
import numpy as np
from winprob.route_probe_evaluation_v1 import evaluate, ARMS


class RouteEvaluationTests(unittest.TestCase):
    def fixture(self):
        n = 80; labels = {'main': np.ones(n, dtype=np.int32), 'contains_wards': np.zeros(n, bool),
                         'is_no_buy': np.zeros(n, bool), 'main_size': np.ones(n, dtype=np.int32)}
        base = np.tile([2, 1, 3], (n, 1)); real = base.copy(); real[:24] = [1, 2, 3]
        scores = {a: {'top3': v.copy(), 'top1': v[:, 0].copy(), 'catalogue_target_logp': np.full(n, -.8 if a == 'recipe' else -1.)}
                  for a, v in (('m2', base), ('recipe', real), ('permuted', base))}
        legal = {world: {arm: 0 for arm in ARMS} for world in ('16.17', '16.18')}
        return labels, scores, [str(i) for i in range(n)], np.arange(n) % 2, legal

    def test_full_catalogue_can_change_membership_and_pass_positive_gate(self):
        args = self.fixture(); args[1]['recipe']['top3'][24:] = [2, 1, 4]
        result = evaluate(*args)
        self.assertTrue(result['metric_gate_passed'])
        self.assertEqual(result['candidate_changes']['recipe']['changed_option_set'], 56)
        self.assertEqual(result['recipe_minus_control']['m2']['corrected'], 24)
        self.assertEqual(result['recipe_minus_control']['m2']['broken'], 0)

    def test_first_gain_cannot_rescue_top3_nll_or_legality(self):
        for failure in ('top3', 'nll', 'legality'):
            args = self.fixture()
            if failure == 'top3': args[1]['recipe']['top3'][24:] = [2, 3, 4]
            elif failure == 'nll': args[1]['recipe']['catalogue_target_logp'][:] = -1.1
            else: args[4]['16.17']['recipe'] = 1
            self.assertFalse(evaluate(*args)['metric_gate_passed'], failure)

    def test_ward_only_is_not_save_and_counts_preserved_in_main_target(self):
        args = self.fixture(); labels, scores = args[:2]
        labels['main'][:2] = 0; labels['main_size'][:2] = 0
        labels['contains_wards'][0] = True; labels['is_no_buy'][1] = True
        for score in scores.values(): score['top3'][:2] = [0, 2, 3]; score['top1'][:2] = 0
        report = evaluate(*args)
        self.assertEqual(report['arms']['m2']['main_top1']['correct'], 2)
        self.assertEqual(report['arms']['m2']['complete_top1']['correct'], 1)
        self.assertEqual(report['arms']['m2']['typed_main_top1']['correct'], 1)

    def test_game_or_option_duplicates_are_rejected(self):
        args = self.fixture(); args[2][1] = args[2][0]
        with self.assertRaises(ValueError): evaluate(*args)
        args = self.fixture(); args[1]['m2']['top3'][0] = [2, 2, 3]
        with self.assertRaises(ValueError): evaluate(*args)


if __name__ == '__main__':
    unittest.main()
