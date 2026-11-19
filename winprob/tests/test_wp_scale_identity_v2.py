from copy import deepcopy
import io
import unittest
import joblib
import numpy as np
from winprob.experiment import CalibratedModel
from winprob.wp_scale_identity_v2 import array_identity,raw_identity


class TreeIdentityTests(unittest.TestCase):
    def test_empty_arrays_preserve_shape_but_ignore_storage_stride(self):
        a=np.empty((0,8),dtype=np.uint32)
        b=np.lib.stride_tricks.as_strided(a,shape=(0,8),strides=(32,4))
        self.assertNotEqual(a.strides,b.strides)
        self.assertEqual(array_identity(a),array_identity(b))
        self.assertNotEqual(array_identity(a),array_identity(np.empty((0,9),dtype=np.uint32)))

    def test_serialized_model_copy_and_recalibration_preserve_identity_but_tree_changes_do_not(self):
        rng=np.random.default_rng(34);x=rng.normal(size=(400,3));y=(x[:,0]>.1).astype(np.int8);g=np.repeat(np.arange(100),4)
        fitted=CalibratedModel('trajectory',[0,1,2]).fit(x,y,g).calibrate(x,y,g)
        stream=io.BytesIO();joblib.dump(fitted,stream);stream.seek(0);original=joblib.load(stream)
        clone=deepcopy(original);identity=raw_identity(original);before=original.predict(x)
        self.assertEqual(raw_identity(clone),identity)
        target=y.copy();target[::7]=1-target[::7];clone.calibrate(x,target,g)
        self.assertEqual(raw_identity(clone),identity)
        np.testing.assert_array_equal(original.raw(x),clone.raw(x))
        np.testing.assert_array_equal(original.predict(x),before)
        clone.model._predictors[0][0].nodes['value'][1]+=.01
        self.assertNotEqual(raw_identity(clone),identity)


if __name__=='__main__':unittest.main()
