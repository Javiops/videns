"""Small CPU-only synthetic checks; no GPU or real payload/training run."""
from dataclasses import replace
import unittest
import numpy as np
import torch

from winprob.iql_kernels import expectile_value_loss, observed_critic_loss
from winprob.iql_models import (PublicBoardBatch, FullActionBatch, MultisetSum, PublicStateEncoder,
    TwinFullBasketCritic, StateValue, FullCategoricalActor, make_target, polyak_update,
    expectile_loss, critic_loss, bellman_target, clipped_awr_weights, categorical_actor_loss,
    critic_value_step, collate_public_boards, collate_full_actions, GLOBAL_NAMES, INVENTORY_NAMES)

torch.set_num_threads(1)


def board(batch=2, length=3):
    champions = torch.arange(1, 11).repeat(batch, 1)
    items = torch.zeros((batch, 10, length), dtype=torch.long)
    counts = torch.zeros_like(items, dtype=torch.float32)
    if length:
        items[:, :, 0] = 1; counts[:, :, 0] = 1
    allies = torch.zeros((batch, 10), dtype=torch.bool); allies[:, 1:5] = True
    enemies = torch.zeros((batch, 10), dtype=torch.bool); enemies[:, 5:] = True
    return PublicBoardBatch(champions, items, counts, torch.zeros(batch, 10, 22),
                            torch.full((batch, 10, 1), .5), allies, enemies, torch.zeros(batch, 37))


def observed_actions():
    ids = torch.tensor([[2, 0], [2, 3]], dtype=torch.long)
    counts = torch.tensor([[1., 0.], [1., 2.]])
    return FullActionBatch(ids, counts, torch.zeros(2, 23))


class TorchIQLTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_inventory_multiset_preserves_multiplicity_without_truncation(self):
        encoder = MultisetSum(20)
        self.assertTrue(bool((encoder.embedding.weight == 0).all()))
        with torch.no_grad():
            encoder.embedding.weight[2].fill_(1.)  # emulate a learned known-item row
        once = encoder(torch.tensor([[2, 0]]), torch.tensor([[1., 0.]]))
        twice = encoder(torch.tensor([[2, 0]]), torch.tensor([[2., 0.]]))
        repeated = encoder(torch.tensor([[2, 2]]), torch.tensor([[1., 1.]]))
        torch.testing.assert_close(twice, 2*once)
        torch.testing.assert_close(twice, repeated)
        nine = encoder(torch.full((1, 9), 2), torch.ones(1, 9))
        torch.testing.assert_close(nine, 9*once)
        with self.assertRaises(ValueError):
            encoder(torch.tensor([[0]]), torch.tensor([[1.]]))

    def test_board_pooling_is_symmetric_within_teams(self):
        encoder = PublicStateEncoder(20, 20).eval()
        sample = board()
        permutation = torch.tensor([0, 4, 2, 1, 3, 9, 7, 6, 8, 5])
        shuffled = PublicBoardBatch(**{name: (getattr(sample, name)[:, permutation] if name != 'global_public' else sample.global_public)
                                      for name in sample.__dataclass_fields__})
        torch.testing.assert_close(encoder(sample), encoder(shuffled))
        self.assertEqual(encoder(sample).shape, (2, 128))

    def test_architecture_shapes_and_bounded_return_heads(self):
        state = board(); action = observed_actions()
        critic = TwinFullBasketCritic(20, 20)
        value = StateValue(20, 20)
        actor = FullCategoricalActor(20, 20, action_count=7)  # synthetic catalogue only
        q1, q2 = critic(state, action)
        self.assertEqual(q1.shape, (2,))
        self.assertTrue(bool(((q1 > 0) & (q1 < 1)).all()))
        self.assertTrue(bool(((q2 > 0) & (q2 < 1)).all()))
        self.assertEqual(value(state).shape, (2,))
        self.assertEqual(actor(state).shape, (2, 7))
        torch.testing.assert_close(actor.log_probabilities(state).exp().sum(dim=1), torch.ones(2))
        self.assertIsNot(critic.q1.state, critic.q2.state)

    def test_target_copy_frozen_and_polyak_uses_updated_online(self):
        online = TwinFullBasketCritic(20, 20); target = make_target(online)
        self.assertTrue(all(not p.requires_grad for p in target.parameters()))
        before = {name: p.clone() for name, p in target.named_parameters()}
        with torch.no_grad():
            for p in online.parameters(): p.add_(.2)
        polyak_update(online, target, .005)
        for name, p in target.named_parameters():
            torch.testing.assert_close(p, before[name]+.001)
            self.assertIsNone(p.grad)

    def test_torch_losses_match_numpy_and_detach_targets(self):
        qtarget = torch.tensor([.2, .8], requires_grad=True)
        value = torch.tensor([.4, .3], requires_grad=True)
        weights = torch.tensor([1., 2.])
        loss = expectile_loss(qtarget, value, weights=weights)
        expected = expectile_value_loss(qtarget.detach().numpy(), value.detach().numpy(), sampling_weights=[1, 2])
        self.assertAlmostEqual(float(loss.detach()), expected['loss'], places=7)
        loss.backward(); self.assertIsNone(qtarget.grad)
        np.testing.assert_allclose(value.grad.numpy(), expected['gradient_wrt_online_v'], rtol=1e-6)
        q1 = torch.tensor([.2, .8], requires_grad=True); q2 = torch.tensor([.4, .3], requires_grad=True)
        target = torch.tensor([.3, .9], requires_grad=True)
        loss = critic_loss(q1, q2, target, weights)
        expected_q = observed_critic_loss(q1.detach().numpy(), q2.detach().numpy(), target.detach().numpy(), [1, 2])
        loss.backward(); self.assertIsNone(target.grad)
        np.testing.assert_allclose(q1.grad.numpy(), expected_q['gradient_wrt_online_q1'], rtol=1e-6)

    def test_absorbing_terminal_has_no_fabricated_network_query(self):
        class NeverCalled:
            def __call__(self, state): raise AssertionError('terminal queried')
        result = bellman_target(NeverCalled(), None, torch.tensor([1., 0.]), torch.tensor([True, True]))
        torch.testing.assert_close(result, torch.tensor([1., 0.]))
        self.assertFalse(result.requires_grad)
        with self.assertRaises(ValueError):
            bellman_target(NeverCalled(), board(1), torch.tensor([1.]), torch.tensor([True]))

    def test_actor_uses_observed_full_label_and_frozen_awr(self):
        q = torch.tensor([1., .2], requires_grad=True)
        v = torch.tensor([0., .2], requires_grad=True)
        weights = clipped_awr_weights(q, q, v)
        torch.testing.assert_close(weights, torch.tensor([20., 1.]))
        self.assertFalse(weights.requires_grad)
        logits = torch.zeros(2, 3, requires_grad=True)
        loss = categorical_actor_loss(logits, torch.tensor([0, 2]), weights)
        self.assertAlmostEqual(float(loss.detach()), 10.5*np.log(3), places=5)
        loss.backward(); self.assertIsNone(q.grad); self.assertIsNone(v.grad)
        self.assertLess(logits.grad[0, 0], 0); self.assertLess(logits.grad[1, 2], 0)

    def test_one_cpu_critic_value_update_keeps_target_gradient_free(self):
        critic = TwinFullBasketCritic(20, 20); target = make_target(critic); value = StateValue(20, 20)
        qopt = torch.optim.Adam(critic.parameters(), lr=3e-4)
        vopt = torch.optim.Adam(value.parameters(), lr=3e-4)
        report = critic_value_step(critic, target, value, vopt, qopt, board(), observed_actions(),
                                  board(1), torch.tensor([0., 1.]), torch.tensor([False, True]))
        self.assertTrue(np.isfinite(report['critic_loss']))
        self.assertEqual(report['terminal_transitions'], 1)
        self.assertTrue(all(p.grad is None for p in target.parameters()))
        self.assertTrue(all(p.device.type == 'cpu' for p in critic.parameters()))

    def test_ragged_compiler_collation_has_no_fixed_inventory_cap(self):
        raw = board(1)
        compiled = {name: getattr(raw, name)[0].numpy() for name in ('champion_indices', 'inventory_static', 'visible_level',
                                                                 'ally_mask', 'enemy_mask', 'global_public')}
        compiled['inventory_bags'] = [tuple((i, 2) for i in range(1, 12))]+[()] * 9
        batch = collate_public_boards([compiled])
        self.assertEqual(batch.inventory_indices.shape, (1, 10, 11))
        self.assertEqual(int(batch.inventory_counts.sum()), 22)
        self.assertEqual(len(GLOBAL_NAMES), 37); self.assertEqual(len(INVENTORY_NAMES), 22)
        empty = collate_full_actions([((), np.r_[np.zeros(22), 1.].astype(np.float32))])
        self.assertEqual(empty.item_indices.shape, (1, 0))

    def test_unseen_public_vocab_rows_remain_zero_without_training_exposure(self):
        encoder = PublicStateEncoder(30, 30)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-4)
        encoder(board()).square().mean().backward(); optimizer.step()
        torch.testing.assert_close(encoder.champions.weight[30], torch.zeros(16))
        torch.testing.assert_close(encoder.inventory.embedding.weight[30], torch.zeros(32))
        torch.testing.assert_close(encoder.champions.weight[0], torch.zeros(16))


if __name__ == '__main__':
    unittest.main()
