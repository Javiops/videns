from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from winprob.student_checkpoint_v1 import load_checkpoint
from winprob.iql_dataset_contract import file_hash


class CheckpointMetadataTests(unittest.TestCase):
    def test_scoped_metadata_load_preserves_tensors_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'model.pt'
            state = {'w': torch.arange(8, dtype=torch.float32)}
            torch.save({'schema': 'causal-student-checkpoint-v1', 'training_game_ids': list(np.array(['g1', 'g2'])), 'state_dict': state}, p)
            sha = file_hash(p); before = list(torch.serialization.get_safe_globals())
            value = load_checkpoint(p, sha)
            self.assertEqual(value['training_game_ids'], ['g1', 'g2'])
            self.assertTrue(all(type(x) is str for x in value['training_game_ids']))
            self.assertTrue(torch.equal(value['state_dict']['w'], state['w']))
            self.assertEqual(file_hash(p), sha); self.assertEqual(torch.serialization.get_safe_globals(), before)
            with self.assertRaises(ValueError): load_checkpoint(p, '0'*64)


if __name__ == '__main__': unittest.main()
