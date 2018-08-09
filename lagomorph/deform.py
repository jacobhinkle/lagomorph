import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

import numpy as np
import math

from . import interp_cuda as ic
from .dtypes import dtype2precision
from .elementwise import multiply_add

def interp_image_kernel_bcastI(out, I, h):
    assert I.shape[0] == 1, "First dimension of I must be 1"
    assert I.ndim+1 == h.ndim, "Dimensions of I and h must match"
    assert out.dtype == I.dtype and I.dtype == h.dtype, "dtypes of all args must match"
    prec = dtype2precision(out.dtype)
    dim = I.ndim - 1
    if not isinstance(I, gpuarray.GPUArray):
        I = gpuarray.to_gpu(np.asarray(I))
    if not isinstance(h, gpuarray.GPUArray):
        h = gpuarray.to_gpu(np.asarray(h))
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        ic.interp_image_bcastI_2d(out, I, h,
                np.int32(h.shape[0]),
                np.int32(h.shape[2]),
                np.int32(h.shape[3]),
                precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotImplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")

def interp_image(I, h, out=None):
    """
    Interpolate an image along a vector field to compute I(h(x))

    Args:
        I: a batch of images in 2 or 3 dimensions. Rank is 3 or 4. In NWH(D)
            order.
        h: a batch of deformations of same dimension as I in NCWH(D) order where
            C indicates the coordinate. If N=1 for h but not for I, then
            broadcasting in N will be performed.
    Returns:
        Ih: warped image as same time as I
    """
    dim = I.ndim - 1
    assert I.ndim == h.ndim-1, "Dimension of I must be one less that of h"
    assert h.shape[1] == dim, "h must have same number channels as dim"
    if out is None:
        out = gpuarray.empty(shape=defshape2imshape(h.shape), dtype=I.dtype, order='C')
    assert out.shape == defshape2imshape(h.shape), "Output must have same domain as h"
    if I.shape[0] == 1 and h.shape[0] != 0:
        interp_image_kernel_bcastI(out, I, h)
    elif h.shape[0] == 1 and I.shape[0] != 0:
        interp_image_kernel_bcasth()
    else:
        assert I.shape[0] == h.shape[0], "Number non-broadcast image and deformation dimensions must match"
        if dim == 2:
            interp_image_kernel2(I, h)
        elif dim == 3:
            interp_image_kernel3(I, h)
        else:
            raise Exception("Unimplemented dimension: "+str(dim))
    return out

def interp_def(g, h, out=None):
    """Given g and h, compute $g \circ h$ by interpolating each dimension separately"""
    if out is None:
        out = gpuarray.zeros(shape=h.shape, dtype=h.dtype, order='C')
    for d in range(h.shape[1]):
        out[:,d,...] = interp_image(g[:,d,...], h)
    return out

def imshape2defshape(sh):
    """Convert the shape of an image set to corresponding deformation shape"""
    dim = len(sh)-1
    return tuple([sh[0], dim]+list(sh[1:]))

def defshape2imshape(sh):
    """Convert the shape of a deformation set to corresponding image shape"""
    return tuple([sh[0]]+list(sh[2:]))

def identity(defshape, dtype=np.float32):
    """
    Given a deformation shape in NCWH(D) order, produce an identity matrix (numpy array)
    """
    dim = len(defshape)-2
    ix = np.empty(defshape, dtype=dtype)
    for d in range(dim):
        ld = defshape[d+2]
        shd = [1]*len(defshape)
        shd[d+2] = ld
        ix[:,d,...] = np.arange(ld, dtype=dtype).reshape(shd)
    return ix


def identitylikeim(im):
    """
    Given a deformation
    """
    ix = identity(imshape2defshape(im.shape), dtype=im.dtype)
    if isinstance(im, gpuarray.GPUArray):
        ix = gpuarray.to_gpu(ix)
    return ix

def identitylikedef(h):
    """
    Given a deformation
    """
    ix = identity(h.shape, dtype=h.dtype)
    if isinstance(h, gpuarray.GPUArray):
        ix = gpuarray.to_gpu(ix)
    return ix

def composeHV(h, v, dt=1.0, out=None):
    """
    Given a deformation h, a velocity v, and a time step dt, compute h(x+dt*v(x))
    """
    if out is None:
        out = gpuarray.zeros(shape=h.shape, dtype=h.dtype, order='C')
    multiply_add(v, dt, out=out)
    # in place interp
    interp_def(h, out, out=out)
    return out