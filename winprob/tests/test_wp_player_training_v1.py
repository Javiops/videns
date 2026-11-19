import unittest
import numpy as np
import torch
from winprob.wp_player_training_v1 import train_one,state_digest,normalized_game_weights,mean_seed_probability

torch.set_num_threads(1)


class PlayerTrainingTests(unittest.TestCase):
    def test_repeat_seed_training_is_exact_and_every_row_is_visited(self):
        rng=np.random.default_rng(7);x=torch.from_numpy(rng.normal(size=(37,173)).astype(np.float32))
        p=torch.from_numpy(rng.normal(size=(37,10,35)).astype(np.float32));y=(x[:,0]>0).float();w=torch.ones(37)
        a,ra=train_one('deepset',31,x,p,y,w,epochs=2,batch_size=16)
        b,rb=train_one('deepset',31,x,p,y,w,epochs=2,batch_size=16)
        self.assertEqual(state_digest(a),state_digest(b));self.assertEqual(ra['epoch_permutation_sha256'],rb['epoch_permutation_sha256'])
        self.assertEqual(ra['updates'],6);self.assertEqual(ra['row_visits'],74);self.assertNotEqual(ra['initial_state_sha256'],ra['final_state_sha256'])

    def test_matched_attention_control_has_identical_initialization_and_batch_order(self):
        rng=np.random.default_rng(2);x=torch.from_numpy(rng.normal(size=(16,173)).astype(np.float32))
        p=torch.from_numpy(rng.normal(size=(16,10,35)).astype(np.float32));y=(x[:,0]>0).float();w=torch.ones(16)
        _,a=train_one('attention',82,x,p,y,w,epochs=1,batch_size=8)
        _,b=train_one('attention_position_shuffled',82,x,p,y,w,epochs=1,batch_size=8)
        self.assertEqual(a['initial_state_sha256'],b['initial_state_sha256']);self.assertEqual(a['epoch_permutation_sha256'],b['epoch_permutation_sha256'])

    def test_equal_total_mass_per_game_and_all_three_seeds_required(self):
        w=normalized_game_weights(np.array([0,1,1,1]));self.assertAlmostEqual(float(w[0]),float(w[1:].sum()))
        self.assertAlmostEqual(float(w.mean()),1.)
        p=mean_seed_probability(np.array([[.1,.4],[.2,.5],[.3,.6]],dtype=np.float32))
        self.assertEqual(p.dtype,np.float64);np.testing.assert_allclose(p,[.2,.5],atol=2e-8)
        with self.assertRaisesRegex(ValueError,'three'):mean_seed_probability(np.array([[.1],[.2]]))


if __name__=='__main__':unittest.main()
