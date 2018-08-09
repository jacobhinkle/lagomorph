from pycuda import gpuarray
import pycuda.autoinit
import numpy as np

import math

from .deform import imshape2defshape
from .dtypes import dtype2precision
from . import diff_cuda

def gradient(im, out=None):
    """
    Compute the finite-difference gradient of an image set
    """
    if out is None:
        out = gpuarray.empty(shape=imshape2defshape(im.shape), dtype=im.dtype, order='C')
    prec = dtype2precision(out.dtype)
    dim = im.ndim - 1
    if not isinstance(im, gpuarray.GPUArray):
        im = gpuarray.to_gpu(np.asarray(im))
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(im.shape[1]/block[0]), math.ceil(im.shape[2]/block[1]), 1)
        diff_cuda.gradient_2d(out, im,
                np.int32(im.shape[0]),
                np.int32(im.shape[1]),
                np.int32(im.shape[2]),
                precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotimmplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")
    return out

def divergence(v, out=None):
    """
    Compute the finite-difference divergence of a vector field set
    """
    if out is None:
        out = gpuarray.empty(shape=defshape2imshape(v.shape), dtype=v.dtype, order='C')
    prec = dtype2precision(out.dtype)
    dim = v.ndim - 2
    if not isinstance(v, gpuarray.GPUArray):
        v = gpuarray.to_gpu(np.asarray(v))
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(v.shape[2]/block[0]), math.ceil(v.shape[3]/block[1]), 1)
        diff_cuda.divergence_2d(out, v,
                np.int32(v.shape[0]),
                np.int32(v.shape[2]),
                np.int32(v.shape[3]),
                precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotimmplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")
    return out

def jacobian_times_vectorfield(v, w, transpose=False, out=None):
    """
    Compute the finite-difference Jacobian matrix of v and contract that with w
    at each point.
    """
    if out is None:
        out = gpuarray.empty(shape=v.shape, dtype=v.dtype, order='C')
    prec = dtype2precision(out.dtype)
    dim = v.ndim - 2
    if not isinstance(v, gpuarray.GPUArray):
        v = gpuarray.to_gpu(np.asarray(v))
    if not isinstance(w, gpuarray.GPUArray):
        w = gpuarray.to_gpu(np.asarray(w))
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(v.shape[2]/block[0]), math.ceil(v.shape[3]/block[1]), 1)
        if transpose:
            f = diff_cuda.jacobian_transpose_times_vectorfield_2d
        else:
            f = diff_cuda.jacobian_times_vectorfield_2d
        f(out, v, w,
            np.int32(v.shape[0]),
            np.int32(v.shape[2]),
            np.int32(v.shape[3]),
            precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotimmplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")
    return out
