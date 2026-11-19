import unittest
import numpy as np
from winprob.features_wp_players_v1 import FIELDS as PLAYER_FIELDS
from winprob.features_wp_kernel_v1 import FIELDS,CHANNELS,kernel_features

IX={name:i for i,name in enumerate(PLAYER_FIELDS)}


def fixture(n=3):
    p=np.zeros((n,10,35),dtype=np.float32);p[:,:5,-1]=1;p[:,5:,-1]=-1
    p[:,:,:2]=.5;p[:,:,IX['position_observed']]=1;p[:,:,IX['total_gold_over10000']]=1
    p[:,:,IX['wallet_over2000']]=1;p[:,:,IX['wallet_observed']]=1;p[:,:,IX['net_gold_outflow_over10000']]=.8
    p[:,:,IX['health_positive']]=1;p[:,:,IX['health_ratio']]=1;p[:,:,IX['health_ratio_observed']]=1
    return p


class KernelFeatureTests(unittest.TestCase):
    def test_identical_team_distributions_have_zero_mmd_and_balance(self):
        x=kernel_features(fixture())
        for i,name in enumerate(FIELDS):
            if 'balance_' in name or name.endswith('squared_mmd'):np.testing.assert_array_equal(x[:,i],0)
            if name.endswith('cross_mean_kernel'):np.testing.assert_allclose(x[:,i],1,atol=1e-6)

    def test_player_permutation_and_side_orientation(self):
        rng=np.random.default_rng(4);p=fixture(9);p[:,:,:2]=rng.uniform(.1,.9,(9,10,2));p[:,:,IX['health_ratio']]=rng.uniform(.1,1,(9,10))
        x=kernel_features(p);permutation=rng.permutation(10);np.testing.assert_allclose(x,kernel_features(p[:,permutation]),rtol=0,atol=1e-6)
        q=p.copy();q[:,:,-1]*=-1;y=kernel_features(q)
        for i,name in enumerate(FIELDS):np.testing.assert_allclose(y[:,i],-x[:,i] if name.endswith('_diff') else x[:,i],rtol=0,atol=1e-6)

    def test_missing_positions_do_not_create_imputed_contact(self):
        p=fixture();p[:,:,:3]=0;np.testing.assert_array_equal(kernel_features(p),0)

    def test_health_location_swap_changes_the_field(self):
        p=fixture(1);p[0,0,:2]=[.5,.5];p[0,1,:2]=[.1,.1];p[0,2:5,:2]=[.2,.2];p[0,5:,:2]=[.53,.5]
        p[0,0,IX['health_ratio']]=1;p[0,1,IX['health_ratio']]=.1;q=p.copy()
        q[0,0,IX['health_ratio']]=.1;q[0,1,IX['health_ratio']]=1
        a=kernel_features(p);b=kernel_features(q)
        self.assertGreater(float(np.max(np.abs(a-b))),.05)
        for i,name in enumerate(FIELDS):
            if name.startswith('kernel_positive_health_'):np.testing.assert_array_equal(a[:,i],b[:,i])

    def test_squared_mmd_bounds_and_zero_mass_are_explicit(self):
        rng=np.random.default_rng(8);p=fixture(100);p[:,:,:2]=rng.uniform(.01,.99,(100,10,2));p[:,:,IX['health_ratio']]=rng.uniform(0,1,(100,10))
        x=kernel_features(p)
        for channel in CHANNELS:
            value=x[:,FIELDS.index('kernel_'+channel+'_squared_mmd')];self.assertTrue(np.all((value>=0)&(value<=2)))
        p[:,5:,IX['health_ratio']]=0;y=kernel_features(p)
        for channel in CHANNELS[1:]:
            np.testing.assert_array_equal(y[:,FIELDS.index('kernel_'+channel+'_both_masses_positive')],0)
            np.testing.assert_array_equal(y[:,FIELDS.index('kernel_'+channel+'_squared_mmd')],0)


if __name__=='__main__':unittest.main()
