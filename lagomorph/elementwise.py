from pycuda import gpuarray
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import numpy as np

def dtype2ctype(dt):
    if dt == np.float32:
        return 'float'
    elif dt == np.float64:
        return 'double'
    elif dt == np.int32:
        return 'int'
    else:
        raise Exception(f"Can't convert dtype {dt} to c type name")


def sum_squares(x):
    """Just compute the sum of squares of a gpuarray"""
    krnl = ReductionKernel(np.float64, neutral="0",
                reduce_expr="a+b", map_expr="x[i]*x[i]",
                arguments="float *x")
    return krnl(x).get()

def multiply_add(x, alpha, out=None):
    if out is None:
        out = gpuarray.zeros(shape=v.shape, dtype=v.dtype, order='C')
    xtype = dtype2ctype(x.dtype)
    ma = ElementwiseKernel(
            f"{xtype} *out, {xtype} *x, {xtype} alpha",
            "out[i] += alpha*x[i]",
            "multiple_add")
    ma(out, x, x.dtype.type(alpha))
    return out
