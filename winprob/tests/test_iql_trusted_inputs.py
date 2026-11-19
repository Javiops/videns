"""Independent exact CPU parity checks for the validated-cache execution path.

Only synthetic tensors are used. No cache, source record, or GPU is opened.
"""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from winprob.iql_models import (PublicBoardBatch, FullActionBatch, MultisetSum,
    PublicStateEncoder, FullBasketEncoder, FullCategoricalActor, StateValue,
    TwinFullBasketCritic, categorical_actor_loss, critic_value_step, make_target,
    set_input_content_validation)
from winprob.iql_training import adam

torch.set_num_threads(1)
_spec = importlib.util.spec_from_file_location('iql_trusted_input_cli',
    Path(__file__).resolve().parents[1] / 'scripts' / 'train_iql.py')
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def states():
    champion = torch.arange(1, 11).repeat(2, 1)
    items = torch.tensor([1, 2, 0]).view(1, 1, 3).repeat(2, 10, 1)
    counts = torch.tensor([2., 1., 0.]).view(1, 1, 3).repeat(2, 10, 1)
    ally = torch.tensor([False] + [True]*4 + [False]*5).repeat(2, 1)
    enemy = ~ally; enemy[:, 0] = False
    return PublicBoardBatch(champion, items, counts, torch.randn(2, 10, 22),
        torch.rand(2, 10, 1), ally, enemy, torch.randn(2, 37))


def actions():
    return FullActionBatch(torch.tensor([[2, 0], [3, 4]]),
        torch.tensor([[2., 0.], [1., 3.]]), torch.randn(2, 23))


def initialized(model):
    # Nonzero learned embedding coordinates exercise count/pooling paths too.
    with torch.no_grad():
        for child in model.modules():
            if isinstance(child, torch.nn.Embedding):
                child.weight[1:].normal_(0., .05)
    return model


class TrustedInputParityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(907)

    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertEqual(first.dtype, second.dtype)
            self.assertEqual(first.shape, second.shape)
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first: self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second): self.assert_tree_equal(a, b)
        else:
            self.assertEqual(first, second)

    def assert_model_and_grad_equal(self, first, second):
        self.assert_tree_equal(first.state_dict(), second.state_dict())
        for (na, a), (nb, b) in zip(first.named_parameters(), second.named_parameters()):
            self.assertEqual(na, nb)
            if a.grad is None: self.assertIsNone(b.grad)
            else: self.assert_tree_equal(a.grad, b.grad)

    def test_full_catalogue_actor_forward_loss_gradient_and_adam_are_exact(self):
        strict = initialized(FullCategoricalActor(20, 20))
        fast = deepcopy(strict); set_input_content_validation(fast, False)
        first_opt, second_opt = adam(strict.parameters()), adam(fast.parameters())
        batch = states(); labels = torch.tensor([0, 13165]); weight = torch.tensor([.2, 1.8])
        first, second = strict(batch), fast(batch)
        self.assert_tree_equal(first, second)
        first_loss = categorical_actor_loss(first, labels, weight)
        second_loss = categorical_actor_loss(second, labels, weight)
        self.assert_tree_equal(first_loss, second_loss)
        first_loss.backward(); second_loss.backward()
        self.assert_model_and_grad_equal(strict, fast)
        torch.nn.utils.clip_grad_norm_(strict.parameters(), 10., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(fast.parameters(), 10., error_if_nonfinite=True)
        first_opt.step(); second_opt.step()
        self.assert_model_and_grad_equal(strict, fast)
        self.assert_tree_equal(first_opt.state_dict(), second_opt.state_dict())

    def test_entire_mixed_terminal_critic_value_polyak_step_is_exact(self):
        strict_q = initialized(TwinFullBasketCritic(20, 20))
        strict_v = initialized(StateValue(20, 20)); strict_t = make_target(strict_q)
        fast_q, fast_v, fast_t = deepcopy((strict_q, strict_v, strict_t))
        for model in (fast_q, fast_v, fast_t): set_input_content_validation(model, False)
        qopts = [adam(model.parameters()) for model in (strict_q, fast_q)]
        vopts = [adam(model.parameters()) for model in (strict_v, fast_v)]
        batch, action = states(), actions()
        self.assert_tree_equal(strict_q(batch, action), fast_q(batch, action))
        self.assert_tree_equal(strict_v(batch), fast_v(batch))
        following = batch.select(torch.tensor([0]))
        results = [critic_value_step(q, t, v, vo, qo, batch, action, following,
            torch.tensor([0., 1.]), torch.tensor([False, True]))
            for q, t, v, vo, qo in zip((strict_q, fast_q), (strict_t, fast_t),
                                     (strict_v, fast_v), vopts, qopts)]
        self.assert_tree_equal(*results)
        for first, second in ((strict_q, fast_q), (strict_v, fast_v), (strict_t, fast_t)):
            self.assert_model_and_grad_equal(first, second)
        self.assert_tree_equal(qopts[0].state_dict(), qopts[1].state_dict())
        self.assert_tree_equal(vopts[0].state_dict(), vopts[1].state_dict())
        self.assertTrue(all(p.grad is None for p in fast_t.parameters()))

    def test_default_rejects_bad_contents_and_fast_retains_schema_checks(self):
        board = states(); encoder = PublicStateEncoder(20, 20)
        self.assertTrue(all(m.check_input_contents for m in encoder.modules()
                            if hasattr(m, 'check_input_contents')))
        bad_globals = board.global_public.clone(); bad_globals[0, 0] = torch.nan
        bad_champs = board.champion_indices.clone(); bad_champs[0, 0] = 0
        bad_mask = board.ally_mask.clone(); bad_mask[0, 0] = True
        for bad in (replace(board, global_public=bad_globals),
                    replace(board, champion_indices=bad_champs), replace(board, ally_mask=bad_mask)):
            with self.assertRaises(ValueError): encoder(bad)
        with self.assertRaises(ValueError):
            MultisetSum(20)(torch.tensor([[0]]), torch.tensor([[1.]]))
        bad_action = actions(); bad_action.static_features[0, 0] = torch.inf
        with self.assertRaises(ValueError): FullBasketEncoder(20)(bad_action)
        set_input_content_validation(encoder, False)
        with self.assertRaises(ValueError): encoder(replace(board, champion_indices=board.champion_indices.float()))
        with self.assertRaises(ValueError): encoder(replace(board, global_public=board.global_public[:, :3]))
        with self.assertRaises(ValueError): set_input_content_validation(encoder, 0)

    def test_cache_gate_rejects_missing_authorization_and_graph_proof(self):
        manifest = {'training_authorized': True, 'smoke': False, 'gates': {'passed': True}}
        audit = {'terminal_return_once': True, 'states': 2}
        mutations = [('manifest', 'training_authorized', False), ('manifest', 'smoke', True),
                     ('manifest', 'gates', {'passed': False}), ('audit', 'terminal_return_once', False),
                     ('audit', 'states', 1)]
        for which, name, value in mutations:
            dataset = SimpleNamespace(manifest=deepcopy(manifest), audit=deepcopy(audit), n=2)
            getattr(dataset, which)[name] = value
            actor = FullCategoricalActor(20, 20, action_count=3)
            with self.assertRaises(ValueError):
                cli.enable_validated_cache_fastpath((actor,), SimpleNamespace(dataset=dataset))
            self.assertTrue(actor.state.check_input_contents)
        dataset = SimpleNamespace(manifest=manifest, audit=audit, n=2)
        cli.enable_validated_cache_fastpath((actor,), SimpleNamespace(dataset=dataset))
        self.assertFalse(actor.state.check_input_contents)
        self.assertFalse(actor.state.inventory.check_input_contents)

    def test_challenge_reenables_validation_before_new_tensor_forward(self):
        actor = FullCategoricalActor(20, 20, action_count=3)
        set_input_content_validation(actor, False)
        bad = states(); bad.global_public[0, 0] = torch.nan
        challenge = {'rows': [{}, {}], 'compiled': [None, None]}
        with patch.object(cli, 'collate_public_boards', return_value=bad):
            with self.assertRaises(ValueError):
                cli.evaluate_actor(actor, challenge, np.arange(3), 'cpu', lambda: None)
        self.assertTrue(actor.state.check_input_contents)
        self.assertTrue(actor.state.inventory.check_input_contents)

    def test_fast_actor_still_rejects_nonfinite_outputs_at_loss_boundary(self):
        actor = FullCategoricalActor(20, 20, action_count=3)
        set_input_content_validation(actor, False)
        with torch.no_grad(): actor.head.bias[0] = torch.nan
        with self.assertRaises(ValueError):
            categorical_actor_loss(actor(states()), torch.tensor([0, 1]), torch.ones(2))


if __name__ == '__main__':
    unittest.main()
