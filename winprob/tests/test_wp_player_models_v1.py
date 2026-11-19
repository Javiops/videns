import unittest
import numpy as np
import torch
from winprob.wp_player_models_v1 import PlayerWinModel,ARMS,parameter_count
from winprob.wp_player_preprocessing_v1 import position_permutations,shuffle_positions,fit_normalization,normalize

torch.set_num_threads(1)


class PlayerModelTests(unittest.TestCase):
    def test_prediction_is_invariant_to_player_record_order(self):
        torch.manual_seed(12);x=torch.randn(7,173);p=torch.randn(7,10,35);permutation=torch.randperm(10)
        for arm in ARMS:
            model=PlayerWinModel(arm).eval()
            with torch.no_grad():a=model(x,p);b=model(x,p[:,permutation])
            torch.testing.assert_close(a,b,rtol=0,atol=1e-6)

    def test_attention_and_shuffled_control_initialize_identically(self):
        torch.manual_seed(56);a=PlayerWinModel('attention')
        torch.manual_seed(56);b=PlayerWinModel('attention_position_shuffled')
        self.assertEqual(parameter_count(a),parameter_count(b))
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)

    def test_global_arm_cannot_use_player_information(self):
        model=PlayerWinModel('global');x=torch.randn(3,173)
        torch.testing.assert_close(model(x),model(x,torch.randn(3,10,35)),rtol=0,atol=0)

    def test_gradient_reaches_query_key_value_and_pool(self):
        torch.manual_seed(9);model=PlayerWinModel('attention');out=model(torch.randn(8,173),torch.randn(8,10,35))
        out.square().mean().backward()
        for name in ('blocks.0.query.weight','blocks.0.key.weight','blocks.0.value.weight','pool_query'):
            grad=dict(model.named_parameters())[name].grad;self.assertIsNotNone(grad);self.assertGreater(float(grad.abs().sum()),0)

    def test_shuffle_preserves_team_position_marginals_and_every_other_input(self):
        rng=np.random.default_rng(3);p=rng.normal(size=(5,10,35)).astype(np.float32);g=np.arange(5);t=np.full(5,180000)
        index=position_permutations(g,t);np.testing.assert_array_equal(index,position_permutations(g,t));q=shuffle_positions(p,index)
        np.testing.assert_array_equal(p[:,:,3:],q[:,:,3:]);self.assertFalse(np.array_equal(p[:,:,:3],q[:,:,:3]))
        for i in range(5):
            for sl in (slice(0,5),slice(5,10)):
                self.assertEqual(sorted(map(tuple,p[i,sl,:3])),sorted(map(tuple,q[i,sl,:3])))
        bad=index.copy();bad[0,0]=7
        with self.assertRaisesRegex(ValueError,'same five-player'):shuffle_positions(p,bad)

    def test_normalization_weights_games_equally_and_handles_constant_columns(self):
        x=np.ones((4,173),dtype=np.float32);p=np.ones((4,10,35),dtype=np.float32);g=np.array([0,1,1,1])
        x[0,0]=2;x[1:,0]=10;p[0,:,0]=4;p[1:,:,0]=8
        state=fit_normalization(x,p,g);self.assertAlmostEqual(state['global_mean'][0],6)
        self.assertAlmostEqual(state['token_mean'][0],6);self.assertEqual(state['global_scale'][1],1)
        self.assertEqual(state['token_scale'][1],1);self.assertAlmostEqual(state['global_scale'][0],4)
        self.assertAlmostEqual(state['token_scale'][0],2)
        a,b=normalize(x,p,state);self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())


if __name__=='__main__':unittest.main()
