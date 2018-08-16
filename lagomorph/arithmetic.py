from pycuda import gpuarray
import pycuda.autoinit
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import numpy as np
import math

from .memory import alloc
from .dtypes import dtype2precision, dtype2ctype
from .arithmetic_cuda import multiply_imvec_2d

_L2kernels = {xtype: ReductionKernel(np.float64, neutral="0.0",
                reduce_expr="a+b", map_expr="x[i]*y[i]",
                arguments=f"const {xtype} *x, const {xtype} *y") for xtype in
                ['float','double']}
def L2(x, y):
    """Just compute the sum of products of a gpuarray"""
    assert x.dtype == y.dtype, "dtypes must match"
    assert x.flags.c_contiguous, "x must be c contiguous"
    assert y.flags.c_contiguous, "y must be c contiguous"
    xtype = dtype2ctype(x.dtype)
    return _L2kernels[xtype](x, y).get()

_makernels = {xtype: ElementwiseKernel(
            f"{xtype} *out, const {xtype} *x, {xtype} alpha",
            "out[i] += alpha*x[i]",
            "multiply_add") for xtype in ['float', 'double']}
def multiply_add(x, alpha, out=None):
    if out is None:
        out = gpuarray.zeros_like(x, order='C')
    xtype = dtype2ctype(x.dtype)
    _makernels[xtype](out, x, x.dtype.type(alpha))
    return out

_mulkernels = {xtype: ElementwiseKernel(
            f"{xtype} *out, const {xtype} *x, const {xtype} *y",
            "out[i] = x[i]*y[i]",
            "multiply") for xtype in ['float', 'double']}
def multiply(x, y, out=None):
    """Multiply images and vector fields whose shapes are identical"""
    assert x.shape == y.shape, "shapes must be equal"
    assert x.dtype == y.dtype, "dtypes must be equal"
    xtype = dtype2ctype(x.dtype)
    if out is None:
        out = gpuarray.empty_like(x)
    _mulkernels[xtype](out, x, y)
    return out

def multiply_imvec(im, v, out=None):
    """Multiply images and vector fields"""
    d = im.ndim - 1
    assert v.ndim == d+2, "v.ndim must equal im.ndim+1"
    assert im.shape[0] == v.shape[0], "im and v must have same number of fields"
    if out is None:
        out = gpuarray.empty_like(v)
    prec = dtype2precision(out.dtype)
    if d == 2:
        block = (32,32,1)
        grid = (math.ceil(im.shape[1]/block[0]), math.ceil(im.shape[2]/block[1]), 1)
        multiply_imvec_2d(out, im, v, im.shape[0], im.shape[1], im.shape[2],
                precision=prec, block=block, grid=grid)
    elif d == 3:
        raise NotImplementedError("not implemented yet")
    else:
        raise Exception(f"Unsupported dimension: {d}")
    return out
