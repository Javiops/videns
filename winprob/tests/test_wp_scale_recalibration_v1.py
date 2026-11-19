from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from winprob.experiment import CalibratedModel

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scale_runner',ROOT/'scripts/train_wp_scale_v1.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class ScaleRecalibrationTests(unittest.TestCase):
    def test_recalibration_preserves_raw_trees_and_original_predictions(self):
        rng=np.random.default_rng(73);x=rng.normal(size=(400,3));y=(x[:,0]+.2*x[:,1]>0).astype(np.int8)
        games=np.repeat(np.arange(100),4)
        original=CalibratedModel('trajectory',[0,1,2]).fit(x,y,games).calibrate(x,y,games)
        before=original.predict(x);raw=original.raw(x);identity=runner.raw_identity(original)
        clone=deepcopy(original)
        other=x.copy();other[:,0]*=.5;other_y=y.copy();other_y[::5]=1-other_y[::5]
        clone.calibrate(other,other_y,games)
        self.assertEqual(runner.raw_identity(clone),identity)
        self.assertEqual(runner.raw_identity(original),identity)
        np.testing.assert_array_equal(clone.raw(x),raw)
        np.testing.assert_array_equal(original.predict(x),before)
        self.assertFalse(np.array_equal(clone.predict(x),before))

if __name__=='__main__':unittest.main()
