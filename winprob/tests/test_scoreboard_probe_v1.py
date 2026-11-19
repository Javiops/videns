from copy import deepcopy
from dataclasses import replace
import unittest
import numpy as np
import torch
from winprob.iql_models import PublicBoardBatch
from winprob.scoreboard_probe_v1 import initial_models,ScoreBatch,epoch_order,evaluate


def batch(n=2):
    public=PublicBoardBatch(torch.arange(1,11).expand(n,-1),torch.ones(n,10,1,dtype=torch.long),torch.ones(n,10,1),
        torch.zeros(n,10,22),torch.ones(n,10,1)/2,torch.tensor([False]+[True]*4+[False]*5).expand(n,-1),
        torch.tensor([False]*5+[True]*5).expand(n,-1),torch.zeros(n,37))
    real=torch.arange(n*30,dtype=torch.float32).reshape(n,10,3)/20.;real[:,0]=0
    p=[0,4,3,2,1,9,8,7,6,5]
    return ScoreBatch(public,real,real[:,p])


class ScoreboardProbeTests(unittest.TestCase):
    def test_shared_initial_values_and_zero_base_parity(self):
        models=initial_models(12,20,7,0);b=batch()
        expected=models['zero'].head(models['zero'].state.base(b.public))
        for arm,model in models.items():
            torch.testing.assert_close(model.predict(b),expected,atol=0,rtol=0)
            self.assertEqual(len(model.state.base.player[0]._forward_hooks),0)
            for key,value in model.state.state_dict().items():torch.testing.assert_close(value,models['zero'].state.state_dict()[key],atol=0,rtol=0)
        self.assertEqual(sum(p.numel() for p in models['zero'].state.score_projection.parameters()),192)

    def test_models_do_not_share_storage(self):
        m=initial_models(12,20,7,0);before=deepcopy(m['shuffled'].state.state_dict())
        with torch.no_grad():next(m['scoreboard'].state.parameters()).add_(1.)
        for key,value in before.items():torch.testing.assert_close(value,m['shuffled'].state.state_dict()[key],atol=0,rtol=0)

    def test_zero_arm_ignores_extra_scores_even_with_nonzero_projection(self):
        model=initial_models(12,20,7,0)['zero'];b=batch()
        with torch.no_grad():model.state.score_projection.weight.fill_(.1)
        first=model.predict(b);b.real[:,1:]*=100;b.shuffled[:,1:]*=50
        torch.testing.assert_close(first,model.predict(b),atol=0,rtol=0)

    def test_projection_gets_gradient_and_hook_is_removed(self):
        model=initial_models(12,20,7,0)['scoreboard'];b=batch()
        model.loss(b,torch.tensor([0,1])).backward()
        self.assertTrue(torch.isfinite(model.state.score_projection.weight.grad).all())
        self.assertGreater(float(model.state.score_projection.weight.grad.abs().sum()),0.)
        self.assertEqual(len(model.state.base.player[0]._forward_hooks),0)
        b.public=replace(b.public,champion_indices=torch.zeros(2,9,dtype=torch.long))
        with self.assertRaises(ValueError):model.predict(b)
        self.assertEqual(len(model.state.base.player[0]._forward_hooks),0)

    def test_focal_and_nonfinite_extra_channels_rejected(self):
        model=initial_models(12,20,7,0)['scoreboard'];b=batch();b.real[0,0,0]=1
        with self.assertRaises(ValueError):model.predict(b)
        b=batch();b.shuffled[0,1,0]=float('nan')
        with self.assertRaises(ValueError):model.predict(b)

    def test_epoch_order_and_reference_failure_gate(self):
        ids=np.arange(100);np.testing.assert_array_equal(np.sort(epoch_order(ids,0,1)),ids)
        np.testing.assert_array_equal(epoch_order(ids,0,1),epoch_order(ids,0,1))
        y=np.ones(40,dtype=np.int64);p={'top1':y.copy(),'top3':np.tile([1,0,2],(40,1)),'target_log_probability':np.full(40,-.1)}
        result=evaluate(y,dict.fromkeys(('zero','scoreboard','shuffled','memory','old_state'),p),np.zeros(40),np.ones(40),['purchase']*40)
        self.assertFalse(result['metric_gate_passed']);self.assertEqual(result['nonempty_purchase_guard']['difference'],0.)


if __name__=='__main__':unittest.main()
