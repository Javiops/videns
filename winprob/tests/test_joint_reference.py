import unittest
import numpy as np
from winprob.joint_reference import group_marginal, lift_group_policy


class JointReferenceTests(unittest.TestCase):
    def test_group_marginal_and_normalization_preserved(self):
        base = np.array([[.3, .7], [.8, .2]])
        joint, conditional = lift_group_policy(base, np.log([[1, 3, 1], [2, 2, 1]]), [0, 0, 1])
        np.testing.assert_allclose(group_marginal(joint, [0, 0, 1], 2), base, atol=1e-15)
        np.testing.assert_allclose(joint.sum(axis=1), 1)
        np.testing.assert_allclose(conditional[0], [.25, .75, 1])

    def test_rare_group_conditional_does_not_underflow_globally(self):
        joint, _ = lift_group_policy([[.5, .5]], [[0, -10000, -10000]], [0, 1, 1])
        np.testing.assert_allclose(joint, [[.5, .25, .25]])

    def test_zeros_not_smoothed_and_unsupported_positive_group_rejected(self):
        joint, _ = lift_group_policy([[1., 0.]], [[0, -np.inf]], [0, 1])
        np.testing.assert_array_equal(joint, [[1., 0.]])
        with self.assertRaisesRegex(ValueError, 'no supported'):
            lift_group_policy([[.5, .5]], [[0, -np.inf]], [0, 1])

    def test_full_argmax_projection_not_group_policy_identity(self):
        joint, _ = lift_group_policy([[.6, .4]], [[0, 0, 0]], [0, 0, 1])
        self.assertEqual(int(np.argmax(joint[0])), 2)  # full argmax selects group1
        self.assertEqual(int(np.argmax(group_marginal(joint, [0, 0, 1], 2)[0])), 0)


if __name__ == '__main__':
    unittest.main()
