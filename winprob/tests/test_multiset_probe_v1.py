from copy import deepcopy
import unittest
import numpy as np
import torch
from winprob.multiset_probe_v1 import (purchase_tables,canonical_log_probability,initial_models,epoch_order,compact_raw,summarize)
from winprob.multiset_likelihood_v1 import log_bag_probability


class MultisetProbeTests(unittest.TestCase):
    def test_canonical_probability_and_marginal_order_sum(self):
        tables=purchase_tables([[],[(10,2),(20,1)],[(10,6)]])
        p=torch.tensor([.2,.5,.3],dtype=torch.float64).log().expand(3,tables.local.sizes.shape[1],3).clone().requires_grad_()
        canonical=canonical_log_probability(p,tables.local,tables.path_nodes,tables.path_tokens,tables.path_mask)
        expected=torch.tensor([.2,.5**2*.3*.2,.5**6],dtype=torch.float64).log()
        torch.testing.assert_close(canonical,expected,atol=1e-12,rtol=0)
        marginal=log_bag_probability(p,tables.local)
        torch.testing.assert_close(marginal-canonical,torch.tensor([0.,np.log(3),0.],dtype=torch.float64),atol=1e-12,rtol=0)
        (-canonical.mean()).backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_models_share_initial_values_without_aliasing(self):
        models=initial_models(12,20,3,7,0)
        for key,value in models['categorical'].state.state_dict().items():
            torch.testing.assert_close(value,models['marginal'].state.state_dict()[key],atol=0,rtol=0)
            torch.testing.assert_close(value,models['canonical'].state.state_dict()[key],atol=0,rtol=0)
        for key,value in models['marginal'].head.state_dict().items():torch.testing.assert_close(value,models['canonical'].head.state_dict()[key],atol=0,rtol=0)
        before=deepcopy(models['canonical'].state.state_dict())
        with torch.no_grad():next(models['marginal'].state.parameters()).add_(1.)
        for key,value in before.items():torch.testing.assert_close(value,models['canonical'].state.state_dict()[key],atol=0,rtol=0)

    def test_epoch_permutation_is_complete_and_deterministic(self):
        indexes=np.arange(100)
        a=epoch_order(indexes,0,1);np.testing.assert_array_equal(a,epoch_order(indexes,0,1))
        np.testing.assert_array_equal(np.sort(a),indexes);self.assertFalse(np.array_equal(a,epoch_order(indexes,0,2)))

    def test_raw_catalogue_mass_is_not_silently_normalized(self):
        score=compact_raw(np.log([[.1,.2,.3],[.1,.2,.1]]),np.array([2,1]))
        np.testing.assert_allclose(np.exp(score['catalogue_log_mass']),[.6,.4])
        np.testing.assert_allclose(np.exp(score['target_log_probability']),[.5,.5])
        np.testing.assert_allclose(np.exp(score['raw_target_log_probability']),[.3,.2])
        with self.assertRaises(ValueError):compact_raw(np.log([[.5,.5,.5]]),np.array([0]))

    def test_weaker_control_gain_cannot_rescue_memory_failure(self):
        y=np.array([0,1,2,0]);good=compact_raw(np.log([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.8,.1,.1]]),y)
        bad=compact_raw(np.log([[.1,.8,.1]]*4),y)
        scores={'marginal':good,'memory':good,'categorical':bad,'canonical':bad,'old_state':bad}
        result=summarize(y,scores,np.ones(4),np.zeros(4,dtype=bool));self.assertFalse(result['metric_gate_passed'])


if __name__=='__main__':unittest.main()
