"""Synthetic dense-reference schema/parity tests; no model or source payloads."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import score_winprob_m2_pairs_v2 as scorer


def context():
    projection={'ts':100,'gold_est':500,'inventory':[]}
    return {'schema':'winprob-revision-source-alignment-v2','status':'aligned','match_id':'EUW1_1','participant_id':1,
            'role':'development','patch':'16.19','checks':dict.fromkeys(scorer.smoke.REQUIRED_CHECKS,True),
            'source_projection':projection,'projection_sha256':scorer.producer.value_hash(projection),
            'context_timestamp_ms':100,'undo_timestamp_ms':200,'decision_timestamp_ms':300,'certificate_b_timestamp_ms':300,
            'action_a':1036,'action_b':1052,'pair_id':'EUW1_1|1|100|200|300|1036|1052','random_focal_pair_weight':.05}


class DenseReferenceTests(unittest.TestCase):
    def test_canonical_dev19_context_and_no_role_remap(self):
        admitted={'EUW1_1':{'role':'development','patch':'16.19','historical_validation':False}}
        row=scorer.source_row(context(),admitted,set())
        self.assertEqual(row['label_counts'],{'1052':1})
        bad=context();bad['role']='fit'
        with self.assertRaisesRegex(ValueError,'role/patch'):scorer.source_row(bad,admitted,set())
        with self.assertRaises(ValueError):scorer.source_row(context(),admitted,{'EUW1_1'})
    def test_primary_dev18_and_identity_or_weight_tamper_rejected(self):
        admitted={'EUW1_1':{'role':'development','patch':'16.18','historical_validation':False}}
        bad=context();bad['patch']='16.18'
        with self.assertRaisesRegex(ValueError,'16.19 only'):scorer.source_row(bad,admitted,set())
        admitted['EUW1_1']['patch']='16.19';bad=context();bad['pair_id']='changed'
        with self.assertRaisesRegex(ValueError,'identity'):scorer.source_row(bad,admitted,set())
        bad=context();bad['random_focal_pair_weight']=None
        with self.assertRaisesRegex(ValueError,'weight'):scorer.source_row(bad,admitted,set())
    def test_native_zero_residual_override_restores_method(self):
        class Policy:
            def probabilities(self,rows):return np.asarray([[.1,.7,.2]])
            def options(self,rows):return self.probabilities(rows).argmax(axis=1).tolist()
        policy=Policy();rows=[{}];original=policy.options(rows)
        self.assertEqual(scorer.options_with_exported_probabilities(policy,rows,np.asarray([[.1,.7,.2]])),original)
        self.assertEqual(policy.options(rows),original)
        self.assertEqual(scorer.options_with_exported_probabilities(policy,rows,np.asarray([[.9,.05,.05]])),[0])
        self.assertEqual(policy.options(rows),original)
    def test_disk_backed_matrix_to_npz_preserves_float32_axis_values(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);temporary=root/'matrix.npy';destination=root/'probabilities.npz'
            matrix=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(3,5))
            expected=np.arange(15,dtype=np.float32).reshape(3,5)/15
            for start in range(3):matrix[start:start+1]=expected[start:start+1]
            matrix.flush();del matrix
            scorer.write_matrix_archive(temporary,destination)
            with np.load(destination,allow_pickle=False) as saved:
                self.assertEqual(saved.files,['probabilities']);self.assertEqual(saved['probabilities'].dtype,np.float32)
                np.testing.assert_array_equal(saved['probabilities'],expected)


if __name__=='__main__':unittest.main()
