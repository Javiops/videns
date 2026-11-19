import itertools
import unittest
import torch
from winprob.multiset_likelihood_v1 import (build_lattice,log_bag_probability,MultisetTransitionHead,submultisets,
    build_catalogue_lattice,catalogue_log_probabilities)


def enumerated(logp,bag,vocabulary,maximum):
    tokens={item:i+1 for i,item in enumerate(vocabulary)}
    nodes=submultisets(tuple(sorted(bag)));lookup={n:i for i,n in enumerate(nodes)}
    ordered=sorted(set(itertools.permutations([i for i,n in bag for _ in range(n)])))
    values=[]
    for sequence in ordered:
        prefix={};value=logp.new_zeros(())
        for item in sequence:
            value=value+logp[lookup[tuple(sorted(prefix.items()))],tokens[item]]
            prefix[item]=prefix.get(item,0)+1
        if len(sequence)<maximum:value=value+logp[lookup[tuple(sorted(prefix.items()))],0]
        values.append(value)
    return torch.logsumexp(torch.stack(values),dim=0)


class MultisetLikelihoodTests(unittest.TestCase):
    def test_exact_sum_and_gradients_with_repeated_items(self):
        bag=((10,2),(20,1));lattice=build_lattice([bag],[10,20],maximum=4)
        torch.manual_seed(5);logits=torch.randn(1,lattice.sizes.shape[1],3,dtype=torch.float64,requires_grad=True)
        logp=torch.log_softmax(logits,dim=-1)
        exact=log_bag_probability(logp,lattice)[0];brute=enumerated(logp[0],bag,[10,20],4)
        torch.testing.assert_close(exact,brute,atol=1e-12,rtol=0)
        a=torch.autograd.grad(exact,logits,retain_graph=True)[0];b=torch.autograd.grad(brute,logits)[0]
        torch.testing.assert_close(a,b,atol=1e-12,rtol=0);self.assertTrue(torch.isfinite(a).all())

    def test_constant_transitions_count_three_not_six_orders(self):
        lattice=build_lattice([[(10,2),(20,1)]],[10,20],4)
        p=torch.tensor([.2,.5,.3],dtype=torch.float64).log().expand(1,lattice.sizes.shape[1],3)
        got=log_bag_probability(p,lattice).exp().item()
        self.assertAlmostEqual(got,3*.5**2*.3*.2,places=13)

    def test_normalized_over_all_bounded_multisets(self):
        bags=[[(10,a),(20,b)] for a in range(4) for b in range(4-a)]
        bags=[[(i,n) for i,n in bag if n] for bag in bags]
        lattice=build_lattice(bags,[10,20],3)
        head=MultisetTransitionHead(2,context_dim=4,item_dim=3,hidden=7,maximum=3).double()
        context=torch.tensor([[.1,.2,.3,.4]],dtype=torch.float64).expand(len(bags),-1)
        probability=head.log_probability(context,lattice).exp().sum()
        torch.testing.assert_close(probability,torch.tensor(1.,dtype=torch.float64),atol=1e-12,rtol=0)

    def test_empty_target_and_forced_final_stop(self):
        lattice=build_lattice([[],[(10,2)]],[10],2)
        logp=torch.tensor([.25,.75],dtype=torch.float64).log().expand(2,lattice.sizes.shape[1],2)
        result=log_bag_probability(logp,lattice).exp()
        torch.testing.assert_close(result,torch.tensor([.25,.75**2],dtype=torch.float64),atol=1e-12,rtol=0)

    def test_entry_permutation_and_padding_do_not_change_mass(self):
        bags=[[(10,1),(20,2)],[(20,2),(10,1)],[]]
        lattice=build_lattice(bags,[10,20],4);head=MultisetTransitionHead(2,context_dim=4,item_dim=3,hidden=7,maximum=4).double()
        context=torch.ones(3,4,dtype=torch.float64)
        values=head.log_probability(context,lattice);torch.testing.assert_close(values[0],values[1],atol=0,rtol=0)
        loss=-values.mean();loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_duplicate_pairs_unknown_items_and_overflow_rejected(self):
        for bag in ([[(10,1),(10,1)]],[[(11,1)]],[[(10,7)]]):
            with self.assertRaises(ValueError):build_lattice(bag,[10,20],6)

    def test_missing_catalogue_bags_leave_probability_mass(self):
        lattice=build_lattice([[],[(10,1)]],[10,20],2)
        logp=torch.tensor([.2,.5,.3],dtype=torch.float64).log().expand(2,lattice.sizes.shape[1],3)
        total=log_bag_probability(logp,lattice).exp().sum().item()
        self.assertAlmostEqual(total,.3,places=13)

    def test_shared_catalogue_graph_matches_local_exact_likelihoods(self):
        bags=[[(20,1),(10,2)],[],[(10,1)],[(20,2)]];vocabulary=[10,20]
        local=build_lattice(bags,vocabulary,3);shared=build_catalogue_lattice(bags,vocabulary,3)
        torch.manual_seed(19)
        head=MultisetTransitionHead(2,context_dim=4,item_dim=3,hidden=7,maximum=3).double()
        context=torch.randn(2,4,dtype=torch.float64)
        mass=catalogue_log_probabilities(head,context,shared,chunk_nodes=2)
        for row in range(2):
            expected=head.log_probability(context[row:row+1].expand(len(bags),-1),local)
            torch.testing.assert_close(mass[row],expected,rtol=0,atol=1e-12)
        self.assertTrue(torch.all(mass.exp().sum(1)<=1.))


if __name__=='__main__':unittest.main()
