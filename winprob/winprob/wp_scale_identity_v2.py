"""Raw-tree identity invariant to storage strides of equivalent NumPy arrays."""
import joblib
import numpy as np


def array_identity(value):
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise TypeError('Object arrays are not supported in raw-tree identity')
        dtype=value.dtype.descr if value.dtype.names else value.dtype.str
        return ('numpy_array_values',value.shape,dtype,value.tobytes(order='C'))
    return value


def raw_identity(model):
    state=dict(vars(model.model));predictors=state.pop('_predictors')
    trees=[[{key:array_identity(value) for key,value in vars(tree).items()} for tree in iteration] for iteration in predictors]
    return joblib.hash((type(model.model).__module__,type(model.model).__qualname__,state,trees,model.indices,model.scaler))
