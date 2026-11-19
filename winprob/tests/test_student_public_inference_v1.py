from types import SimpleNamespace
import unittest
import numpy as np
import torch
from winprob.student_public_inference_v1 import focal_store


class FocalInputsTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(51)
        indices = rng.permutation(400)[:260]
        focals = np.arange(0, 260, 2)
        offsets = [0]; item_ids = []; item_counts = []
        bags = []
        for bag in range(4000):
            values = [(bag % 17 + k + 1, k + 2) for k in range(bag % 3)]
            bags.append(values)
            item_ids.extend(v[0] for v in values); item_counts.extend(v[1] for v in values)
            offsets.append(len(item_ids))
        features = {
            'champion_indices': np.arange(4000).reshape(400, 10),
            'inventory_static': rng.normal(size=(400, 10, 22)).astype(np.float32),
            'visible_level': rng.normal(size=(400, 10, 1)).astype(np.float32),
            'global_public': rng.normal(size=(400, 37)).astype(np.float32),
            'inventory_offsets': np.array(offsets), 'inventory_item_indices': np.array(item_ids),
            'inventory_item_counts': np.array(item_counts, dtype=np.float32)}
        arrays = {'rune_indices': np.arange(520).reshape(260, 2),
            'action_item_indices': np.array([[0, 0], [2, 3]]),
            'action_item_counts': np.array([[0., 0.], [2., 1.]], dtype=np.float32),
            'action_attributes': np.zeros((2, 23), dtype=np.float32),
            'empty_main_mask': np.array([True, False])}
        return SimpleNamespace(indices=indices, focals=focals, inventory_width=5,
            features=features, arrays=arrays, labels_gpu=None, labels_cpu=None, teacher_gpu=None), bags

    def test_focal_gather_preserves_order_counts_padding_and_last_batch(self):
        data, bags = self.fixture(); calls = []
        store, actions = focal_store(data, lambda: calls.append(True), device='cpu')
        self.assertEqual(len(calls), 2)  # 130 focals, bounded 128-row transfers.
        selected = data.indices[data.focals]
        expected_ids = np.zeros((130, 10, 5), dtype=np.int64)
        expected_counts = np.zeros((130, 10, 5), dtype=np.float32)
        for row, original in enumerate(selected):
            for player in range(10):
                for slot, (item, count) in enumerate(bags[10*original+player]):
                    expected_ids[row, player, slot] = item
                    expected_counts[row, player, slot] = count
        np.testing.assert_array_equal(store.tensors['inventory_indices'].numpy(), expected_ids)
        np.testing.assert_array_equal(store.tensors['inventory_counts'].numpy(), expected_counts)
        positions = np.array([129, 0, 128])
        board, runes = store.batch(positions)
        for name in ('champion_indices', 'inventory_static', 'visible_level', 'global_public'):
            np.testing.assert_array_equal(getattr(board, name).numpy(), data.features[name][selected[positions]])
        np.testing.assert_array_equal(runes.numpy(), data.arrays['rune_indices'][data.focals[positions]])
        np.testing.assert_array_equal(board.inventory_counts.numpy(), expected_counts[positions])
        self.assertEqual(board.ally_mask.sum(1).tolist(), [4, 4, 4])
        self.assertEqual(board.enemy_mask.sum(1).tolist(), [5, 5, 5])
        np.testing.assert_array_equal(actions[1].numpy(), [[0, 0], [2, 1]])
        self.assertFalse(hasattr(store, 'teacher'))
        self.assertFalse(hasattr(store, 'labels'))

        # Changing nonfocal dense inputs cannot enter a focal prediction batch.
        others = np.setdiff1d(np.arange(400), selected)
        for name in ('inventory_static', 'visible_level', 'global_public'):
            data.features[name][others] = np.nan
        repeat, _ = focal_store(data, device='cpu')
        for name in store.tensors:
            torch.testing.assert_close(store.tensors[name], repeat.tensors[name], rtol=0, atol=0)

    def test_supervision_rejected_before_any_feature_or_device_access(self):
        for field in ('labels_gpu', 'labels_cpu', 'teacher_gpu'):
            data = SimpleNamespace(labels_gpu=None, labels_cpu=None, teacher_gpu=None)
            setattr(data, field, object())
            with self.assertRaisesRegex(ValueError, 'supervision'):
                focal_store(data, device='cpu')


if __name__ == '__main__': unittest.main()
