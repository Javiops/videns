from pathlib import Path
import tempfile
import unittest
import numpy as np
from winprob.bayes_proxy_decision_v1 import decisions, swap_evidence, ARMS
from winprob.npy_row_writer_v1 import NpyRowWriter


class DecisionTests(unittest.TestCase):
    def test_no_history_all_arms_exactly_neutral(self):
        native = np.array([.4, .3, .3], dtype=np.float32)
        results = decisions(native, [.2, .6, .2], [.6, .2, .2], np.empty((0, 2)), np.zeros(0, bool))
        p = native.astype(float); p /= p.sum()
        for arm in ARMS:
            self.assertTrue(results[arm]['neutral']); np.testing.assert_array_equal(results[arm]['probabilities'], p)

    def test_scrambling_only_changes_evidence_direction(self):
        evidence = np.log([[.1, .8], [.2, .7]])
        a = decisions(np.array([.4, .3, .3]), [.6, .2, .2], [.2, .6, .2], evidence, np.ones(2, bool))
        np.testing.assert_allclose(a['history']['weights'], a['scrambled']['weights'][::-1], atol=1e-15)
        self.assertGreater(a['history']['probabilities'][1], a['scrambled']['probabilities'][1])
        self.assertEqual(swap_evidence('game', 3, 1234), swap_evidence('game', 3, 1234))

    def test_equal_proxies_remain_neutral_after_past_evidence(self):
        for value in decisions(np.array([.4, .3, .3]), [.3, .4, .3], [.3, .4, .3],
                               np.log([[.01, .9]]), np.ones(1, bool)).values(): self.assertTrue(value['neutral'])


class RowWriterTests(unittest.TestCase):
    def test_exact_irregular_roundtrip_and_duplicate_rejection(self):
        values = np.arange(30, dtype=np.float64).reshape(5, 6)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'data.npy'; writer = NpyRowWriter(p, values.shape, values.dtype)
            writer.write(np.array([3, 1]), values[[3, 1]])
            with self.assertRaises(ValueError): writer.write(np.array([1]), values[[1]])
            writer.write(np.array([0, 2, 4]), values[[0, 2, 4]]); writer.close()
            np.testing.assert_array_equal(np.load(p, allow_pickle=False), values)

    def test_incomplete_file_never_qualifies(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = NpyRowWriter(Path(directory)/'data.npy', (3, 4), np.float32)
            writer.write(np.array([0]), np.ones((1, 4), dtype=np.float32))
            with self.assertRaises(ValueError): writer.close()


if __name__ == '__main__': unittest.main()
