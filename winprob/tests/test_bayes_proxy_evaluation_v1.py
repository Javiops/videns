import unittest
import numpy as np
from winprob.bayes_proxy_evaluation_v1 import evaluate, ARMS


class FourArmTests(unittest.TestCase):
    def fixture(self):
        n = 80; labels = {'main': np.ones(n, dtype=np.int32), 'contains_wards': np.zeros(n, bool),
            'is_no_buy': np.zeros(n, bool), 'main_size': np.ones(n, dtype=np.int32)}
        base = np.tile([2, 1, 3], (n, 1)); real = base.copy(); real[:24] = [1, 2, 3]
        scores = {}
        for arm in ARMS:
            top = real.copy() if arm == 'history' else base.copy()
            scores[arm] = {'top1': top[:, 0], 'top3': top, 'catalogue_target_logp': np.full(n, -.8 if arm == 'history' else -1.)}
        return [labels, scores, list(map(str, range(n))), np.arange(n)%2,
                {world: {arm: 0 for arm in ARMS} for world in ('16.17', '16.18')}]

    def test_requires_both_predeclared_controls(self):
        self.assertTrue(evaluate(*self.fixture())['metric_gate_passed'])
        for arm in ('scrambled', 'static_memory'):
            args = self.fixture(); args[1][arm] = {k: v.copy() for k, v in args[1]['history'].items()}
            self.assertFalse(evaluate(*args)['metric_gate_passed'], arm)

    def test_legality_nll_and_top3_guards_cannot_be_rescued(self):
        for fail in ('legality', 'nll', 'top3'):
            args = self.fixture()
            if fail == 'legality': args[4]['16.17']['static_memory'] = 1
            elif fail == 'nll': args[1]['history']['catalogue_target_logp'][:] = -1.1
            else: args[1]['history']['top3'][24:] = [2, 3, 4]
            self.assertFalse(evaluate(*args)['metric_gate_passed'], fail)


if __name__ == '__main__': unittest.main()
