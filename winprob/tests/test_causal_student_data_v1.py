import unittest
import numpy as np
from winprob.causal_student_data_v1 import permitted_supervision_indices, teacher_permutation, epoch_order


class StudentPermissionsTests(unittest.TestCase):
    def test_teacher_and_hard_labels_both_reject_heldout(self):
        folds = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError): permitted_supervision_indices(folds, np.array([0]), 0)
        with self.assertRaises(ValueError): permitted_supervision_indices(folds, np.array([2]), 0, np.array([0, 1, 0, 3]))
        rows, donors = permitted_supervision_indices(folds, np.array([2, 3]), 0)
        np.testing.assert_array_equal(rows, donors)

    def test_permutation_is_train_only_and_preserves_strata(self):
        folds = np.repeat([0, 1], 16); champions = np.tile(np.repeat([1, 2], 8), 2)
        time = np.tile([0, 1, 180000, 180001]*4, 2)
        p, facts = teacher_permutation(folds, champions, time, 0); rows = np.flatnonzero(folds != 0)
        np.testing.assert_array_equal(np.sort(p[rows]), rows)
        np.testing.assert_array_equal(champions[p[rows]], champions[rows])
        np.testing.assert_array_equal(time[p[rows]]//180000, time[rows]//180000)
        np.testing.assert_array_equal(p[folds == 0], -1)
        self.assertEqual(facts['training_rows'], 16)

    def test_epoch_order_is_fixed_permutation_not_global_rng(self):
        train = np.arange(31); np.random.seed(8); before = np.random.get_state()
        a = epoch_order(train, 0, 1); after = np.random.get_state()
        np.testing.assert_array_equal(a, epoch_order(train, 0, 1))
        np.testing.assert_array_equal(np.sort(a), train)
        np.testing.assert_array_equal(before[1], after[1]); self.assertEqual(before[2:], after[2:])
        self.assertFalse(np.array_equal(a, epoch_order(train, 0, 2)))


if __name__ == '__main__': unittest.main()
