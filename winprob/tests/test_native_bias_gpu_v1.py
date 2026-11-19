import unittest
import numpy as np
from winprob.native_bias_gpu_v1 import configure, CudaBiasObjective
from winprob.native_bias_calibration_v1 import BiasObjective


class CudaBiasParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure()

    def test_cpu_gpu_parity_across_partial_batches_and_zero_support(self):
        rng = np.random.default_rng(54)
        p = rng.dirichlet(np.ones(17), size=520)
        p[:, 0] = 0; p /= p.sum(1, keepdims=True)
        rows = np.arange(513); counts = np.zeros(17, dtype=np.int64); counts[1] = len(rows)
        cpu = BiasObjective(p[rows], counts)
        gpu = CudaBiasObjective(p, rows, counts)
        try:
            for b in (np.zeros(17), rng.normal(0, .2, 17)):
                expected_value, expected_grad = cpu(b)
                value, grad = gpu(b)
                self.assertLessEqual(abs(value-expected_value), 2e-10)
                np.testing.assert_allclose(grad, expected_grad, atol=2e-10, rtol=0)
            with self.assertRaises(ValueError): gpu.set_counts(counts+1)
        finally:
            gpu.close()


if __name__ == '__main__': unittest.main()
