from pycuda import gpuarray
import pycuda.autoinit
import numpy as np

import math

from .deform import imshape2defshape
from .dtypes import dtype2precision
from . import diff_cuda

def grad(im, out=None):
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
