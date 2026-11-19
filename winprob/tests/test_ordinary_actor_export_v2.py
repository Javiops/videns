"""Numerical regressions for the full 13,166-class frozen-actor export.

Every tensor/state below is synthetic. No real checkpoint or targets are read.
"""
from copy import deepcopy
import math
import unittest
import numpy as np
import torch

from winprob.ordinary_actor_export_v2 import GroupProjection,score_contexts
from winprob.iql_features import PublicFeatureCompiler
from test_ordinary_actor_export import static,row

torch.set_num_threads(1)
K,G=13166,9333
MAPPING=np.r_[np.arange(G),[0,0],np.arange(1,3832)].astype(np.int64)


def logits():
    rng=np.random.default_rng(7721)
    z=np.stack([rng.uniform(-25.5,23.2,K),np.full(K,-13.),np.zeros(K),
                np.linspace(-1000.,-2000.,K)]).astype(np.float32)
    z[1,0]=0.;z[3,0]=0.
    return torch.from_numpy(z)


class ConstantActor(torch.nn.Module):
    def __init__(self):
        super().__init__();self.register_buffer('base',logits()[1])
    def forward(self,state):return self.base[None,:].expand(state.batch_size,-1)


class OrdinaryActorFloat64Tests(unittest.TestCase):
    def test_full_catalogue_high_dynamic_range_normalizes_without_relaxation(self):
        self.assertEqual(len(MAPPING),K)
        z=logits();out=GroupProjection(MAPPING,G)(z)
        for value in out.values():
            self.assertEqual(value.dtype,np.float32);self.assertTrue(np.isfinite(value).all())
        for name in ('full_probabilities','group_probabilities'):
            mass=out[name].sum(1,dtype=np.float64)
            self.assertLess(float(np.max(abs(mass-1))),1e-7)
            np.testing.assert_allclose(out[name].sum(1,dtype=np.float32),1.,atol=2e-6,rtol=0)
        # Independent float64 softmax oracle over the unchanged float32 logits.
        values=z.numpy().astype(np.float64);exp=np.exp(values-values.max(1,keepdims=True))
        expected=(exp/exp.sum(1,keepdims=True)).astype(np.float32)
        np.testing.assert_allclose(out['full_probabilities'],expected,atol=0,rtol=2e-7)

    def test_group_export_is_once_rounded_sum_of_exported_full_masses(self):
        out=GroupProjection(MAPPING,G)(logits())
        expected=np.stack([np.bincount(MAPPING,weights=r.astype(np.float64),minlength=G)
                           for r in out['full_probabilities']]).astype(np.float32)
        np.testing.assert_array_equal(out['group_probabilities'],expected)
        # Group zero has SAVE + two distinct variants; none is discarded.
        np.testing.assert_array_equal(out['group_probabilities'][:,0],
            out['full_probabilities'][:,MAPPING==0].sum(1,dtype=np.float64).astype(np.float32))

    def test_finite_logmasses_keep_underflowed_group_order_and_match_logits(self):
        z=logits();out=GroupProjection(MAPPING,G)(z)
        raw=z.numpy().astype(np.float64);expected=np.empty((len(z),G))
        for group in range(G):
            values=raw[:,MAPPING==group];maximum=values.max(1)
            expected[:,group]=maximum+np.log(np.exp(values-maximum[:,None]).sum(1))
        np.testing.assert_array_equal(out['group_logits'],expected.astype(np.float32))
        normalizer=raw.max(1)+np.log(np.exp(raw-raw.max(1,keepdims=True)).sum(1))
        np.testing.assert_array_equal(out['group_log_probabilities'],(expected-normalizer[:,None]).astype(np.float32))
        self.assertEqual(out['group_probabilities'][3,9000],0.)
        self.assertEqual(out['group_probabilities'][3,9001],0.)
        self.assertGreater(out['group_logits'][3,9000],out['group_logits'][3,9001])
        np.testing.assert_array_equal(out['group_logits'].argmax(1),out['group_probabilities'].argmax(1))

    def test_all_859_rows_keep_exact_mutation_coverage_and_group_axis(self):
        rows=[row(f'synthetic-{i}') for i in range(859)]
        for i,item in enumerate(rows):
            item['tensor_row']=i
            item['prestate']['gold_est']=i
            item['source_projection']=deepcopy(item['prestate'])
        ledger,arrays,counts=score_contexts(ConstantActor(),PublicFeatureCompiler(static()),rows,MAPPING,G,batch_size=32)
        self.assertEqual(counts['scored_rows'],859);self.assertEqual(counts['mutation_checked_rows'],859)
        self.assertEqual(counts['mutation_failures'],0)
        self.assertEqual([r['bc_tensor_row'] for r in ledger],list(range(859)))
        self.assertTrue(all(r['ignored_field_mutation_equal'] for r in ledger))
        self.assertEqual(arrays['full_probabilities'].shape,(859,K))
        self.assertEqual(arrays['group_probabilities'].shape,(859,G))
        self.assertLess(float(abs(arrays['group_probabilities'].sum(1,dtype=np.float64)-1).max()),1e-7)


if __name__=='__main__':unittest.main()
