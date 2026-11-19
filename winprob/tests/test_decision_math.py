import unittest
import numpy as np
from winprob.decision_math import (tilted_policy, episodic_wp_rewards,
    expectile_loss, advantage_weights, preference_margin, ipo_loss, dpo_loss)


class DecisionMathTests(unittest.TestCase):
    def test_zero_correction_recovers_same_conditioned_reference(self):
        np.testing.assert_allclose(tilted_policy([.6,.3,.1],[0,0,0]), [.6,.3,.1])
        np.testing.assert_allclose(tilted_policy([.6,.3,.1],[0,0,0],legal=[1,1,0]), [2/3,1/3,0])

    def test_additive_state_baseline_cancels(self):
        a = tilted_policy([.6,.4],[-.02,.03],.1)
        b = tilted_policy([.6,.4],[.78,.83],.1)
        np.testing.assert_allclose(a,b)

    def test_reference_support_and_legality_are_preserved(self):
        result = tilted_policy([.6,.4,0],[0,100,10000],legal=[1,0,1])
        np.testing.assert_array_equal(result,[1,0,0])
        with self.assertRaises(ValueError):
            tilted_policy([1,0],[0,0],legal=[0,1])

    def test_tilt_solves_variational_objective(self):
        reference=np.array([.6,.4]); score=np.array([-.02,.03]); temperature=.1
        optimum=tilted_policy(reference,score,temperature)
        def objective(p):
            return p@score-temperature*np.sum(p*np.log(p/reference))
        for q in np.linspace(.0001,.9999,1000):
            self.assertGreaterEqual(objective(optimum)+1e-12,objective(np.array([q,1-q])))

    def test_pair_normalizer_cancels(self):
        reference=np.array([.6,.3,.1]); residual=np.array([.1,-.2,.4])
        policy=tilted_policy(reference,residual)
        h=preference_margin(np.log(policy[0]),np.log(policy[1]),np.log(reference[0]),np.log(reference[1]))
        self.assertAlmostEqual(float(h), residual[0]-residual[1])

    def test_return_conservation_win_and_loss(self):
        rng=np.random.default_rng(20260930)
        for outcome in (0,1):
            for n in (1,2,30):
                values=rng.random(n)
                self.assertAlmostEqual(episodic_wp_rewards(values,outcome).sum(),outcome-values[0])
        np.testing.assert_allclose(episodic_wp_rewards([.4,.43,.7],1),[.03,.27,.30])

    def test_expectile_and_cloning_weights_are_not_probabilities(self):
        np.testing.assert_allclose(expectile_loss([-1,1],.7),[.3,.7])
        np.testing.assert_allclose(advantage_weights([0,1000],1,20),[1,20])

    def test_ipo_has_finite_target_dpo_keeps_rewarding_margin(self):
        self.assertEqual(float(ipo_loss(.5,tau=1)),0)
        self.assertGreater(float(ipo_loss(10,tau=1)),float(ipo_loss(.5,tau=1)))
        self.assertLess(float(dpo_loss(10)),float(dpo_loss(.5)))


if __name__=='__main__':
    unittest.main()
