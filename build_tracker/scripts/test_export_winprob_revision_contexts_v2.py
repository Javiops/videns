"""Synthetic generalized context-role/identity tests; no actual data reads."""
from copy import deepcopy
import unittest

import export_winprob_revision_contexts_v2 as exporter


def pair(mid,patch,role):
    return {'match_id':mid,'patch':patch,'role':role,'eligible_revision':True,'participant_id':1,
            'feature_cutoff_ms':100,'undo_ms':200,'decision_timestamp_ms':300,'action_a':1036,'action_b':1052}


class GeneralContextTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'games':[{'match_id':'EUW1_1','patch':'16.17','role':'train','historical_validation':False},
                                {'match_id':'EUW1_2','patch':'16.19','role':'development','historical_validation':False},
                                {'match_id':'EUW1_3','patch':'16.19','role':'confirmatory_reserved','historical_validation':False}]}
    def test_native_patch_and_roles_are_preserved(self):
        records=[(1,pair('EUW1_1','16.17','fit')),(2,pair('EUW1_2','16.19','development'))]
        result=exporter.select_records(records,self.manifest,{'EUW1_1','EUW1_2'},set(),'16.19',{'development'},10)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['role'],'development')
        self.assertEqual(result[0]['source_line_number'],2)
    def test_reserved_universe_rejected_before_iteration(self):
        def unreadable():raise AssertionError('must not iterate reserved source');yield
        with self.assertRaisesRegex(ValueError,'reserved identity'):
            exporter.select_records(unreadable(),self.manifest,{'EUW1_3'},{'EUW1_3'},'16.19',{'development'},10)
    def test_fold_remapping_and_duplicate_pairs_fail(self):
        with self.assertRaisesRegex(ValueError,'cannot be remapped'):
            exporter.select_records([(1,pair('EUW1_2','16.19','fit'))],self.manifest,{'EUW1_2'},set(),'16.19',{'fit'},10)
        p=pair('EUW1_1','16.17','fit')
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            exporter.select_records([(1,p),(2,p)],self.manifest,{'EUW1_1'},set(),'16.17',{'fit'},0)
    def test_context_identity_stable_across_census_indexes(self):
        p=pair('EUW1_1','16.17','fit');q={**p,'game_index':777,'source_line_number':999,'game_candidate_index':4}
        self.assertEqual(exporter.context_id(p),exporter.context_id(q))
        q['action_b']=1026
        self.assertNotEqual(exporter.context_id(p),exporter.context_id(q))


if __name__=='__main__':unittest.main()
