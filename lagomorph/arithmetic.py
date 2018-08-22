from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import numpy as np
import math

from .checks import ContextCheck, count_nans
from .dtypes import dtype2precision, dtype2ctype
from . import arithmetic_cuda as ac


_L2kernels = {xtype: ContextCheck(ReductionKernel,
                np.float64, neutral="0.0",
                reduce_expr="a+b", map_expr="x[i]*y[i]",
                arguments=f"{xtype} *x, {xtype} *y") for xtype in
                ['float','double']}
def L2(x, y):
    """Just compute the sum of products of a gpuarray"""
    assert x.dtype == y.dtype, "dtypes must match"
    assert x.flags.c_contiguous, "x must be c contiguous"
    assert y.flags.c_contiguous, "y must be c contiguous"
    xtype = dtype2ctype(x.dtype)
    return _L2kernels[xtype](x, y).get()

_makernels = {xtype: ContextCheck(ElementwiseKernel,
            f"{xtype} *out, {xtype} *x, {xtype} alpha",
            "out[i] += alpha*x[i]",
            "multiply_add") for xtype in ['float', 'double']}
def multiply_add(x, alpha, out=None):
    if out is None:
        out = gpuarray.zeros_like(x, order='C')
    xtype = dtype2ctype(x.dtype)
    _makernels[xtype](out, x, x.dtype.type(alpha))
    return out

_mulkernels = {xtype: ContextCheck(ElementwiseKernel,
            f"{xtype} *out, {xtype} *x, {xtype} *y",
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
        ac.multiply_imvec_2d(out, im, v, im.shape[0], im.shape[1], im.shape[2],
                precision=prec, block=block, grid=grid)
    elif d == 3:
        raise NotImplementedError("not implemented yet")
    else:
        raise Exception(f"Unsupported dimension: {d}")
    return out

_clipbelowkernels = {xtype: ContextCheck(ElementwiseKernel,
            f"{xtype} *out, {xtype} *x, {xtype} minval",
            "out[i] = x[i] < minval ? minval : x[i]",
            "clip_below") for xtype in ['float', 'double']}
def clip_below(x, val=0, out=None):
    """Multiply images and vector fields whose shapes are identical"""
    xtype = dtype2ctype(x.dtype)
    if out is None:
        out = gpuarray.empty_like(x)
    _clipbelowkernels[xtype](out, x, x.dtype.type(val))
    return out

def sum_along_axis(x, axis=0, out=None):
    """Compute the sum along a single axis"""
    outsh = list(x.shape)
    outsh[axis] = 1
    outsh = tuple(outsh)
    if out is None:
        out = gpuarray.empty(outsh, dtype=x.dtype, allocator=x.allocator)
    assert out.shape == outsh, "Output is of wrong shape"
    prec = dtype2precision(out.dtype)
    nn = np.prod(x.shape[:axis+1], dtype=np.int32)
    nxyz = np.prod(x.shape[axis+1:], dtype=np.int32)
    block = (1024,1,1)
    grid = (math.ceil(nxyz/block[0]),1,1)
    ac.sum_along_axis(out, x, nn, nxyz, precision=prec, block=block, grid=grid)
    return out

def multiply_add_bcast(x, y, alpha=1, out=None):
    """Multiply y by alpha and add to x, putting output in out and broadcasting
    y along the first axis"""
    assert y.shape[0] == 1, "y.shape[0] must be 1"
    if out is None:
        out = gpuarray.empty_like(x)
    prec = dtype2precision(out.dtype)
    nn = x.shape[0]
    nxyz = np.prod(x.shape[1:], dtype=np.int32)
    block = (1024,1,1)
    grid = (math.ceil(nxyz/block[0]),1,1)
    ac.multiply_add_bcast(out, x, y, x.dtype.type(alpha), nn, nxyz, precision=prec, block=block, grid=grid)
    return out
