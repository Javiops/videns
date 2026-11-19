from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wp_player_audit_helpers_v1 import forward,independent_normalization,independent_position_permutations
from winprob.wp_player_models_v1 import PlayerWinModel,ARMS
from winprob.wp_player_preprocessing_v1 import fit_normalization,position_permutations

torch.set_num_threads(1)


class IndependentPlayerAuditTests(unittest.TestCase):
    def test_functional_replay_matches_all_architectures_exactly(self):
        torch.manual_seed(43);x=torch.randn(9,173);p=torch.randn(9,10,35)
        for arm in ARMS:
            model=PlayerWinModel(arm).eval()
            weights={k:v.detach().requires_grad_(True) for k,v in model.state_dict().items()}
            with torch.no_grad():actual=forward(weights,arm,x,p);expected=model(x,p)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_independent_group_average_normalization_and_control(self):
        rng=np.random.default_rng(56);x=rng.normal(size=(17,173)).astype(np.float32);p=rng.normal(size=(17,10,35)).astype(np.float32)
        x[:,7]=1;p[:,:,5]=1;g=np.repeat([2,9,14],[2,6,9]);t=np.arange(17)*60000+180000
        a=fit_normalization(x,p,g);b=independent_normalization(x,p,g)
        for key in b:np.testing.assert_allclose(a[key],b[key],rtol=1e-12,atol=1e-10)
        np.testing.assert_array_equal(position_permutations(g,t),independent_position_permutations(g,t))


if __name__=='__main__':unittest.main()
