"""Synthetic M2 pair adapter schema checks; no policy/data loads."""
import unittest

import score_winprob_m2_pairs_v1 as scorer


class PairScoreTests(unittest.TestCase):
    def fixture(self):
        projection={'ts':100,'gold_est':500,'inventory':[]}
        return {'match_id':'EUW1_1','status':'aligned','role':'fit','patch':'16.18',
                'checks':dict.fromkeys(scorer.REQUIRED_CHECKS,True),'source_projection':projection,
                'projection_sha256':scorer.producer.value_hash(projection),'context_timestamp_ms':100,
                'certificate_b_timestamp_ms':200,'action_a':1036,'action_b':1052}
    def test_label_change_keeps_source_context(self):
        c=self.fixture();a=scorer.context_row(c,{'EUW1_1'},set(),1036);b=scorer.context_row(c,{'EUW1_1'},set())
        self.assertEqual(a['inventory'],b['inventory']);self.assertEqual(a['gold_est'],b['gold_est'])
        self.assertEqual(a['label_counts'],{'1036':1});self.assertEqual(b['label_counts'],{'1052':1})
    def test_denied_or_pre_b_or_incomplete_certificate_fails(self):
        with self.assertRaises(ValueError):scorer.context_row(self.fixture(),{'EUW1_1'},{'EUW1_1'})
        c=self.fixture();c['checks']={}
        with self.assertRaises(ValueError):scorer.context_row(c,{'EUW1_1'},set())
        c=self.fixture();c['context_timestamp_ms']=200
        with self.assertRaises(ValueError):scorer.context_row(c,{'EUW1_1'},set())
    def test_candidate_statistics_preserve_exact_mass_and_handle_zero(self):
        s=scorer.candidate_statistics(.2,.3)
        self.assertEqual(s['probability_a'],.2);self.assertAlmostEqual(s['pair_conditional_b'],.6)
        self.assertEqual(s['preference'],'B')
        self.assertIsNone(scorer.candidate_statistics(0,.1)['log_probability_b_minus_a'])
        with self.assertRaises(ValueError):scorer.candidate_statistics(float('nan'),.2)
    def test_group_mapping_matches_only_singletons(self):
        self.assertEqual(scorer.singleton_group_map([[],[(1036,1)],[(1036,2)],[(1052,1)]]),{1036:1,1052:3})
        with self.assertRaises(ValueError):scorer.singleton_group_map([[(1036,1)],[(1036,1)]])


if __name__=='__main__':unittest.main()
