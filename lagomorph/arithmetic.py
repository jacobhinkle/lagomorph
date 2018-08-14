from pycuda import gpuarray
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import numpy as np
import math

from .dtypes import dtype2precision, dtype2ctype
from .arithmetic_cuda import multiply_imvec_2d

def sum_squares(x):
    """Just compute the sum of squares of a gpuarray"""
    krnl = ReductionKernel(np.float64, neutral="0",
                reduce_expr="a+b", map_expr="x[i]*x[i]",
                arguments="const float *x")
    return krnl(x).get()

def multiply_add(x, alpha, out=None):
    if out is None:
        out = gpuarray.empty(shape=v.shape, dtype=v.dtype, order='C')
        out.fill(0)
    xtype = dtype2ctype(x.dtype)
    ma = ElementwiseKernel(
            f"{xtype} *out, const {xtype} *x, {xtype} alpha",
            "out[i] += alpha*x[i]",
            "multiply_add")
    ma(out, x, x.dtype.type(alpha))
    return out

def multiply(x, y, out=None):
    """Multiply images and vector fields whose shapes are identical"""
    assert x.shape == y.shape, "shapes must be equal"
    assert x.dtype == y.dtype, "dtypes must be equal"
    xtype = dtype2ctype(x.dtype)
    m = ElementwiseKernel(
            f"{xtype} *out, const {xtype} *x, const {xtype} *y",
            "out[i] = x[i]*y[i]",
            "multiply")
    if out is None:
        out = gpuarray.empty(shape=x.shape, dtype=x.dtype, order='C')
    m(out, x, y)
    return out

def multiply_imvec(im, v, out=None):
    """Multiply images and vector fields"""
    prec = dtype2precision(out.dtype)
    d = im.ndim - 1
    assert v.ndim == d+2, "v.ndim must equal im.ndim+1"
    assert im.shape[0] == v.shape[0], "im and v must have same number of fields"
    if out is None:
        out = gpuarray.empty(shape=v.shape, dtype=v.dtype, order='C')
    if d == 2:
        block = (32,32,1)
        grid = (math.ceil(im.shape[1]/block[0]), math.ceil(im.shape[2]/block[1]), 1)
        multiply_imvec_2d(out, im, v, im.shape[0], im.shape[1], im.shape[2],
                precision=prec, block=block, grid=grid)
    elif d == 3:
        raise NotImplementedError("not implemented yet")
    else:
        raise Exception(f"Unsupported dimension: {d}")
