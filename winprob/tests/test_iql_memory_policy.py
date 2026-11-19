"""Independent checks that closing resident feature maps preserves artifacts."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from train_iql import fit_group_frequency
from winprob.iql_training import Dataset, HOST_FEATURE_ARRAYS


class MemoryPolicyTests(unittest.TestCase):
    def fixture(self, root):
        dataset = Dataset.__new__(Dataset)
        dataset.arrays = {}
        for name in HOST_FEATURE_ARRAYS:
            value = np.lib.format.open_memmap(root/(name+'.npy'), mode='w+', dtype='int32', shape=(4, 10))
            value[:] = np.asarray([2, 2, 3, 3])[:, None]; value.flush()
            dataset.arrays[name] = value
        dataset.focal_champion_indices = np.array(dataset.arrays['champion_indices'][:, 0], copy=True)
        dataset.arrays['patch_index'] = np.asarray([0, 1, 0, 1], dtype=np.uint8)
        dataset.arrays['main_group_id'] = np.asarray([5, 6, 5, 7], dtype=np.int32)
        dataset.host_features_released = False
        return dataset

    def test_release_preserves_source_bytes_and_frequency_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); dataset = self.fixture(root)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            expected = fit_group_frequency(dataset)
            old_maps = [dataset.arrays[name]._mmap for name in HOST_FEATURE_ARRAYS]
            dataset.release_host_features(synchronized_cuda_copy=True)
            self.assertTrue(all(mapping.closed for mapping in old_maps))
            self.assertFalse(set(HOST_FEATURE_ARRAYS) & set(dataset.arrays))
            self.assertEqual(fit_group_frequency(dataset), expected)
            self.assertEqual(expected['global_counts'], {'5': 2, '6': 1, '7': 1})
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_unsynchronized_release_does_not_close_maps(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'synchronized'):
                dataset.release_host_features(synchronized_cuda_copy=False)
            self.assertTrue(all(not dataset.arrays[name]._mmap.closed for name in HOST_FEATURE_ARRAYS))
            dataset.release_host_features(synchronized_cuda_copy=True)


if __name__ == '__main__':
    unittest.main()
