import unittest
import numpy as np
from winprob.scoreboard_features_v1 import normalized_scores,permutation_for_state,shuffle_within_team


class ScoreboardFeatureTests(unittest.TestCase):
    def test_focal_channel_always_zero(self):
        raw=np.arange(60,dtype=np.int32).reshape(2,10,3);before=raw.copy();features=normalized_scores(raw)
        np.testing.assert_array_equal(features[:,0],0.)
        np.testing.assert_allclose(features[:,1:],raw[:,1:]/20.,atol=1e-7)
        np.testing.assert_array_equal(before,raw)

    def test_permutations_deterministic_and_team_local(self):
        p=permutation_for_state('synthetic-game',3,15)
        np.testing.assert_array_equal(p,permutation_for_state('synthetic-game',3,15))
        np.testing.assert_array_equal(np.sort(p[1:5]),np.arange(1,5));np.testing.assert_array_equal(np.sort(p[5:]),np.arange(5,10));self.assertEqual(p[0],0)

    def test_shuffling_preserves_paired_kda_and_team_sums(self):
        x=normalized_scores(np.arange(30,dtype=np.int32).reshape(1,10,3));p=permutation_for_state('synthetic-game',3,15)[None,:]
        shuffled=shuffle_within_team(x,p)
        for side in (slice(1,5),slice(5,10)):
            self.assertEqual(sorted(map(tuple,x[0,side])),sorted(map(tuple,shuffled[0,side])))
            np.testing.assert_allclose(x[:,side].sum(1),shuffled[:,side].sum(1),atol=1e-6)
        np.testing.assert_array_equal(shuffled[:,0],0.)

    def test_cross_team_or_focal_permutation_rejected(self):
        x=np.zeros((1,10,3),dtype=np.float32);p=np.arange(10)[None,:];p[0,[1,5]]=p[0,[5,1]]
        with self.assertRaises(ValueError):shuffle_within_team(x,p)
        p=np.arange(10)[None,:];p[0,[0,1]]=p[0,[1,0]]
        with self.assertRaises(ValueError):shuffle_within_team(x,p)

    def test_unknown_and_negative_inputs_are_rejected(self):
        with self.assertRaises(ValueError):normalized_scores(np.zeros((2,9,3),dtype=np.int32))
        with self.assertRaises(ValueError):normalized_scores(np.full((2,10,3),-1,dtype=np.int32))
        with self.assertRaises(ValueError):normalized_scores(np.zeros((2,10,3),dtype=np.float32))


if __name__=='__main__':unittest.main()
