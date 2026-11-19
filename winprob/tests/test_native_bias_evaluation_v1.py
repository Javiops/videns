import unittest
import numpy as np
from winprob.native_bias_evaluation_v1 import evaluate, ARMS
from winprob.native_bias_data_v1 import opposite_fold_rows


class NativeBiasProtocolTests(unittest.TestCase):
    def fixture(self):
        n = 120
        labels = {'main': np.ones(n, dtype=np.int32), 'contains_wards': np.zeros(n, bool),
                  'is_no_buy': np.zeros(n, bool), 'main_size': np.ones(n, dtype=np.int32)}
        scores = {}
        for arm,hits in (('m2',30),('calibrated',70),('permuted',100)):
            top = np.tile([2,1,3], (n,1)); top[:hits] = [1,2,3]
            scores[arm] = {'top1': top[:,0], 'top3': top, 'catalogue_target_logp': np.full(n,-1. if arm == 'm2' else -.9)}
        return [labels,scores,list(map(str,range(n))),np.arange(n)%2,
                {world:{arm:0 for arm in ARMS} for world in ('16.17','16.18')}]

    def test_control_is_diagnostic_and_never_selected(self):
        result = evaluate(*self.fixture())
        self.assertTrue(result['metric_gate_passed']); self.assertEqual(result['selected_candidate'],'calibrated')
        self.assertFalse(result['class_identity_signal'])
        self.assertNotIn('first_positive_interval_vs_permuted',result['gate_components'])

    def test_nll_or_legality_cannot_be_ignored(self):
        args = self.fixture(); args[1]['calibrated']['catalogue_target_logp'][:] = -1.2
        self.assertIsNone(evaluate(*args)['selected_candidate'])
        args = self.fixture(); args[4]['16.17']['permuted'] = 1
        self.assertFalse(evaluate(*args)['metric_gate_passed'])

    def test_fold_gate_is_explicit_and_nonempty(self):
        f = np.array([0,1,0,1]); np.testing.assert_array_equal(opposite_fold_rows(f,0),[1,3])
        with self.assertRaises(ValueError): opposite_fold_rows(f,True)
        with self.assertRaises(ValueError): opposite_fold_rows(np.zeros(4,dtype=int),0)


if __name__ == '__main__': unittest.main()
