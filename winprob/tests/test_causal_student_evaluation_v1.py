import unittest
import numpy as np
from winprob.causal_student_evaluation_v1 import evaluate, ARMS


class StudentSelectionTests(unittest.TestCase):
    def fixture(self):
        n = 120; labels = {'main': np.ones(n, dtype=np.int32), 'contains_wards': np.zeros(n, bool),
            'is_no_buy': np.zeros(n, bool), 'main_size': np.ones(n, dtype=np.int32)}
        scores = {}
        for arm, hits in (('m2', 30), ('hard', 70), ('distilled', 65), ('shuffled_teacher', 100)):
            top = np.tile([2, 1, 3], (n, 1)); top[:hits] = [1, 2, 3]
            scores[arm] = {'top1': top[:, 0], 'top3': top, 'catalogue_target_logp': np.full(n, -1. if arm == 'm2' else -.9)}
        return [labels, scores, list(map(str, range(n))), np.arange(n)%2,
                {world: {arm: 0 for arm in ARMS} for world in ('16.17', '16.18')}]

    def test_control_is_never_promoted_and_intervals_are_adjusted(self):
        result = evaluate(*self.fixture()); self.assertEqual(result['selected_candidate'], 'hard')
        c = result['paired_contrasts']['hard__minus__m2']['main_top1']
        self.assertLessEqual(c['primary_interval97_5'][0], c['interval95'][0])
        self.assertGreaterEqual(c['primary_interval97_5'][1], c['interval95'][1])
        self.assertNotIn('shuffled_teacher', result['candidate_gates'])

    def test_neither_candidate_can_be_rescued_by_control(self):
        args = self.fixture()
        for arm in ('hard', 'distilled'): args[1][arm] = {k: v.copy() for k, v in args[1]['m2'].items()}
        result = evaluate(*args); self.assertIsNone(result['selected_candidate']); self.assertFalse(result['metric_gate_passed'])

    def test_no_harm_gates_and_tie_break(self):
        args = self.fixture(); args[1]['distilled'] = {k: v.copy() for k, v in args[1]['hard'].items()}
        self.assertEqual(evaluate(*args)['selected_candidate'], 'hard')
        args[1]['hard']['catalogue_target_logp'][:] = -1.2
        self.assertEqual(evaluate(*args)['selected_candidate'], 'distilled')
        args[4]['16.17']['shuffled_teacher'] = 1
        self.assertIsNone(evaluate(*args)['selected_candidate'])


if __name__ == '__main__': unittest.main()
