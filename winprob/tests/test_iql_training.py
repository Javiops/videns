"""Brief CPU-only helper tests; no real training or held-out data access."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

from winprob.iql_training import (adam, clone_actor_optimizer, epoch_order, fold_for_game,
    main_group_log_odds, padded_bags, tree_hash, validate_transition_graph, Dataset, TensorStore, HOST_FEATURE_ARRAYS)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_iql import fit_group_frequency

torch.set_num_threads(1)


def graph():
    ids = {}
    for i in range(100):
        mid = 'SYN_'+str(i); ids.setdefault(fold_for_game(mid), mid)
        if len(ids) == 2: break
    games = [{'match_id': ids[0], 'patch': '16.17', 'fold': 0}, {'match_id': ids[1], 'patch': '16.18', 'fold': 1}]
    arrays = {'game_index': np.array([0, 0, 0, 1, 1, 1]), 'episode_index': np.array([0, 0, 1, 2, 2, 3]),
              'next_state_index': np.array([1, -1, -1, 4, -1, -1]), 'terminated': np.array([False, True, True, False, True, True]),
              'reward': np.array([0, 1, 0, 0, 1, 0], dtype=np.float32), 'timestamp_ms': np.array([100, 200, 150, 100, 200, 150]),
              'elapsed_ms': np.array([100, 50, 100, 100, 50, 100]), 'fold': np.array([0, 0, 0, 1, 1, 1]),
              'patch_index': np.array([0, 0, 0, 1, 1, 1]), 'full_action_id': np.array([0, 2, 20, 0, 2, 20]),
              'main_group_id': np.zeros(6, dtype=int)}
    metadata = []
    for episode, (first, count) in enumerate(((0, 2), (2, 1), (3, 2), (5, 1))):
        gi = int(arrays['game_index'][first]); g = games[gi]
        metadata.append({'first_state': first, 'states': count, 'game_index': gi, 'fold': g['fold'],
                         'match_id': g['match_id'], 'patch': g['patch']})
    return arrays, games, metadata


class TrainingHelperTests(unittest.TestCase):
    def test_episode_metadata_and_terminal_chain_proof(self):
        arrays, games, metadata = graph()
        audit = validate_transition_graph(arrays, games, 4, metadata)
        self.assertEqual(audit['terminal_transitions'], 4)
        self.assertEqual(audit['folds']['0']['training_terminal_classes'], [0, 1])
        self.assertEqual(audit['folds']['0']['observed_full_action_support_fraction'], 1.)
        bad = deepcopy(metadata); bad[0]['first_state'] = 1
        with self.assertRaisesRegex(ValueError, 'metadata range'):
            validate_transition_graph(arrays, games, 4, bad)

    def test_next_state_cannot_cross_episode_or_repay_terminal(self):
        arrays, games, metadata = graph(); arrays['next_state_index'][0] = 4
        with self.assertRaises(ValueError): validate_transition_graph(arrays, games, 4, metadata)
        arrays, games, metadata = graph(); arrays['reward'][0] = 1
        with self.assertRaises(ValueError): validate_transition_graph(arrays, games, 4, metadata)

    def test_fold_is_whole_game_and_fixed_hash(self):
        arrays, games, metadata = graph(); arrays['fold'][1] = 1
        with self.assertRaisesRegex(ValueError, 'fold/patch'):
            validate_transition_graph(arrays, games, 4, metadata)

    def test_ragged_gather_preserves_reordering_counts_empty_bags_and_width(self):
        offsets = np.array([0, 2, 2, 5]); ids = np.array([1, 5, 3, 6, 7]); counts = np.array([2, 1, 1, 3, 2])
        padded, multiplicities = padded_bags(offsets, ids, counts, np.array([2, 0, 1]), width=4)
        np.testing.assert_array_equal(padded, [[3, 6, 7, 0], [1, 5, 0, 0], [0, 0, 0, 0]])
        np.testing.assert_array_equal(multiplicities, [[1, 3, 2, 0], [2, 1, 0, 0], [0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'truncation'):
            padded_bags(offsets, ids, counts, np.array([2]), width=2)

    def test_group_log_odds_aggregates_every_full_variant(self):
        probabilities = torch.tensor([[.3, .3, .4]], dtype=torch.float64)
        result = main_group_log_odds(probabilities.log(), [1], [0], torch.tensor([0, 0, 1]))
        self.assertAlmostEqual(float(result), np.log(.6/.4))
        # A single variant would reverse this ranking, so singleton substitution fails.
        self.assertGreater(float(result), 0)
        self.assertLess(float(probabilities[0, 0]-probabilities[0, 2]), 0)

    def test_actor_and_adam_clone_are_exact_and_independent(self):
        torch.manual_seed(9)
        actor = torch.nn.Linear(3, 5); optimizer = adam(actor.parameters())
        x = torch.tensor([[1., 2., 3.]]); target = torch.tensor([2])
        optimizer.zero_grad(); torch.nn.functional.cross_entropy(actor(x), target).backward(); optimizer.step()
        left, lo = clone_actor_optimizer(actor, optimizer); right, ro = clone_actor_optimizer(actor, optimizer)
        self.assertEqual(tree_hash(left.state_dict()), tree_hash(right.state_dict()))
        self.assertEqual(tree_hash(lo.state_dict()), tree_hash(ro.state_dict()))
        # Identical uniform continuation updates preserve exact equality, including Adam.
        for model, opt in ((left, lo), (right, ro)):
            opt.zero_grad(); torch.nn.functional.cross_entropy(model(x), target).backward(); opt.step()
        self.assertEqual(tree_hash(left.state_dict()), tree_hash(right.state_dict()))
        self.assertEqual(tree_hash(lo.state_dict()), tree_hash(ro.state_dict()))
        original = tree_hash(right.state_dict())
        with torch.no_grad(): next(left.parameters()).add_(1.)
        self.assertEqual(tree_hash(right.state_dict()), original)

    def test_matched_actor_order_has_no_arm_specific_seed(self):
        rows = np.arange(200)
        first = epoch_order(rows, 'actor_continuation', 2)
        np.testing.assert_array_equal(first, epoch_order(rows, 'actor_continuation', 2))
        self.assertFalse(np.array_equal(first, epoch_order(rows, 'actor_continuation', 3)))
        np.testing.assert_array_equal(np.sort(first), rows)

    def mapped_fixture(self, directory):
        dataset = object.__new__(Dataset)
        dataset.arrays = {}
        values = {
            'champion_indices': np.arange(1, 31, dtype=np.int32).reshape(3, 10),
            'inventory_static': np.arange(3*10*22, dtype=np.float32).reshape(3, 10, 22),
            'visible_level': np.ones((3, 10, 1), dtype=np.float32),
            'global_public': np.arange(3*37, dtype=np.float32).reshape(3, 37),
            'inventory_offsets': np.arange(31, dtype=np.int64),
            'inventory_item_indices': np.ones(30, dtype=np.int32),
            'inventory_item_counts': np.ones(30, dtype=np.int32),
            'full_action_id': np.array([1, 2, 3], dtype=np.int32),
            'patch_index': np.array([0, 1, 1], dtype=np.uint8),
            'main_group_id': np.array([8, 8, 9], dtype=np.int32),
        }
        for name, data in values.items():
            path = Path(directory)/(name+'.npy')
            np.save(path, data)
            dataset.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        dataset.focal_champion_indices = np.array(dataset.arrays['champion_indices'][:, 0], dtype=np.int32, copy=True)
        dataset.host_features_released = False
        return dataset, values

    def close_remaining(self, dataset):
        for value in dataset.arrays.values():
            mapping = getattr(value, '_mmap', None)
            if mapping is not None and not mapping.closed: mapping.close()

    def test_host_feature_release_closes_only_large_maps_and_keeps_owned_focal_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, expected = self.mapped_fixture(directory)
            owners = {name: dataset.arrays[name]._mmap for name in HOST_FEATURE_ARRAYS}
            try:
                with self.assertRaisesRegex(ValueError, 'synchronized'):
                    dataset.release_host_features(synchronized_cuda_copy=False)
                self.assertTrue(all(not mapping.closed for mapping in owners.values()))
                self.assertIsNone(dataset.focal_champion_indices.base)
                report = dataset.release_host_features(synchronized_cuda_copy=True)
                self.assertEqual(set(report['removed_host_feature_arrays']), set(HOST_FEATURE_ARRAYS))
                self.assertTrue(all(mapping.closed for mapping in owners.values()))
                self.assertFalse(HOST_FEATURE_ARRAYS & set(dataset.arrays))
                self.assertFalse(dataset.arrays['full_action_id']._mmap.closed)
                np.testing.assert_array_equal(dataset.focal_champion_indices, expected['champion_indices'][:, 0])
                model = fit_group_frequency(dataset)
                self.assertEqual(model['by_champion_patch']['21']['1']['9'], 1)
            finally:
                self.close_remaining(dataset)

    def test_resident_state_and_frequency_paths_never_touch_closed_feature_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, expected = self.mapped_fixture(directory)
            store = object.__new__(TensorStore)
            store.dataset, store.device, store.preloaded = dataset, torch.device('cpu'), True
            # Independent CPU tensors emulate the owned resident-copy path;
            # the real constructor requires CUDA and synchronizes before closing.
            store.resident = {name: torch.tensor(expected[name], dtype=torch.long if name == 'champion_indices' else torch.float32)
                              for name in ('champion_indices', 'inventory_static', 'visible_level', 'global_public')}
            store.resident['inventory_indices'] = torch.ones(3, 10, 1, dtype=torch.long)
            store.resident['inventory_counts'] = torch.ones(3, 10, 1)
            store.ally_mask = torch.tensor([False]+[True]*4+[False]*5)
            store.enemy_mask = torch.tensor([False]*5+[True]*5)
            try:
                dataset.release_host_features(synchronized_cuda_copy=True)
                batch = store.states(np.array([2, 0]))
                np.testing.assert_array_equal(batch.global_public.numpy(), expected['global_public'][[2, 0]])
                with self.assertRaisesRegex(RuntimeError, 'resident CUDA'):
                    store._cpu_state(np.array([0]))
                self.assertEqual(fit_group_frequency(dataset)['global_counts'], {'8': 2, '9': 1})
            finally:
                self.close_remaining(dataset)

    def test_fallback_keeps_maps_open_and_can_gather_host_features(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, expected = self.mapped_fixture(directory)
            store = object.__new__(TensorStore)
            store.dataset, store.device, store.preloaded = dataset, torch.device('cpu'), False
            store.ally_mask = torch.tensor([False]+[True]*4+[False]*5)
            store.enemy_mask = torch.tensor([False]*5+[True]*5)
            try:
                batch = store.states(np.array([1]))
                np.testing.assert_array_equal(batch.global_public.numpy(), expected['global_public'][[1]])
                self.assertFalse(dataset.host_features_released)
                self.assertTrue(all(not dataset.arrays[name]._mmap.closed for name in HOST_FEATURE_ARRAYS))
            finally:
                self.close_remaining(dataset)


if __name__ == '__main__': unittest.main()
