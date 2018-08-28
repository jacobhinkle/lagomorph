from pycuda.compiler import SourceModule
from pycuda import gpuarray
import pycuda.driver as drv

import numpy as np
import math

from . import interp_cuda as ic
from .dtypes import dtype2precision
from .arithmetic import multiply_add

def interp_image_kernel(out, I, h, displacement=True):
    assert I.ndim+1 == h.ndim, "Dimensions of I and h must match"
    assert out.dtype == I.dtype and I.dtype == h.dtype, "dtypes of all args must match"
    prec = dtype2precision(out.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        if displacement:
            f = ic.interp_displacement_image_2d
        else:
            f = ic.interp_image_2d
        f(out, I, h,
                np.int32(h.shape[0]),
                np.int32(h.shape[2]),
                np.int32(h.shape[3]),
                precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotImplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")

def interp_grad_kernel(out, grad, I, h, displacement=True):
    assert I.ndim+1 == h.ndim, "Dimensions of I and h must match"
    assert out.dtype == I.dtype and out.dtype == grad.dtype and \
            I.dtype == h.dtype, "dtypes of all args must match"
    prec = dtype2precision(out.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        if displacement:
            f = ic.interp_displacement_grad_2d
        else:
            f = ic.interp_grad_2d
        f(out, grad, I, h,
                np.int32(h.shape[0]),
                np.int32(h.shape[2]),
                np.int32(h.shape[3]),
                precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotImplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")

def splat_image_kernel(out, outweights, I, h, displacement=True):
    assert I.ndim+1 == h.ndim, "Dimensions of I and h must match"
    assert out.dtype == I.dtype and I.dtype == h.dtype, "dtypes of all args must match"
    prec = dtype2precision(out.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        if displacement:
            f = ic.splat_displacement_image_2d
        else:
            f = ic.splat_image_2d
        f(out, outweights, I, h,
                np.int32(h.shape[0]),
                np.int32(h.shape[2]),
                np.int32(h.shape[3]),
                precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotImplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")


def interp_vectorfield_kernel(out, g, h, displacement=True, clamp=True):
    assert g.ndim == h.ndim, "Dimensions of g and h must match"
    assert out.dtype == g.dtype and g.dtype == h.dtype, "dtypes of all args must match"
    assert out.flags.c_contiguous, "Output should be C contiguous"
    prec = dtype2precision(out.dtype)
    dim = g.ndim - 2
    assert g.shape[1] == dim, "vectorfields must have appropriate number of channels"
    assert h.shape[1] == dim, "vectorfields must have appropriate number of channels"
    assert out.shape[1] == dim, "vectorfields must have appropriate number of channels"
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(g.shape[2]/block[0]), math.ceil(g.shape[3]/block[1]), 1)
        if displacement:
            if clamp:
                f = ic.interp_displacement_vectorfield_2d
            else:
                f = ic.interp_displacement_zerobg_vectorfield_2d
        else:
            if clamp:
                f = ic.interp_vectorfield_2d
            else:
                f = ic.interp_zerobg_vectorfield_2d
        f(out, g, h,
                np.int32(h.shape[0]),
                np.int32(h.shape[2]),
                np.int32(h.shape[3]),
                precision=prec, block=block, grid=grid)
    elif dim == 3:
        raise NotImplementedError("not implemented")
    else:
        raise Exception(f"Unsupported dimension: {dim}")

def interp_image_kernel_bcastI(out, I, h):
    assert I.shape[0] == 1, "First dimension of I must be 1"
    assert I.ndim+1 == h.ndim, "Dimensions of I and h must match"
    assert out.dtype == I.dtype and I.dtype == h.dtype, "dtypes of all args must match"
    prec = dtype2precision(out.dtype)
    dim = I.ndim - 1
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

def interp_image(I, h, displacement=True, out=None):
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
        out = gpuarray.empty(shape=defshape2imshape(h.shape),
                allocator=I.allocator, dtype=I.dtype, order='C')
    assert out.shape == defshape2imshape(h.shape), "Output must have same domain as h"
    if I.shape[0] == 1 and h.shape[0] != 1:
        interp_image_kernel_bcastI(out, I, h)
    elif h.shape[0] == 1 and I.shape[0] != 1:
        interp_image_kernel_bcasth()
    else:
        assert I.shape[0] == h.shape[0], "Number non-broadcast image and deformation dimensions must match"
        interp_image_kernel(out, I, h, displacement=displacement)
    return out

def interp_grad(I, h, displacement=True, out=None, outgrad=None):
    """
    Interpolate an image along a vector field to compute I(h(x)) and also
    compute the gradient.

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
        out = gpuarray.empty(shape=I.shape, allocator=h.allocator,
                dtype=h.dtype, order='C')
    assert out.shape == I.shape, "Output must have same domain as I"
    if outgrad is None:
        outgrad = gpuarray.empty(shape=h.shape, allocator=h.allocator,
                dtype=h.dtype, order='C')
    assert outgrad.shape == h.shape, "Output gradient must have same domain as h"
    if I.shape[0] == 1 and h.shape[0] != 1:
        #interp_image_kernel_bcastI(out, I, h)
        raise NotImplementedError("not implemented")
    elif h.shape[0] == 1 and I.shape[0] != 1:
        #interp_image_kernel_bcasth()
        raise NotImplementedError("not implemented")
    else:
        assert I.shape[0] == h.shape[0], "Number non-broadcast image and deformation dimensions must match"
        interp_grad_kernel(out, outgrad, I, h, displacement=displacement)
    return out, outgrad

def splat_image(I, h, displacement=True, out=None, outweights=None):
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
        out = gpuarray.zeros_like(I)
    if outweights is None:
        outweights = gpuarray.zeros_like(I)
    assert out.shape == defshape2imshape(h.shape), "Output must have same domain as h"
    assert I.shape[0] == h.shape[0], "Number non-broadcast image and deformation dimensions must match"
    splat_image_kernel(out, outweights, I, h, displacement=displacement)
    return out, outweights

def interp_vec(g, h, displacement=True, clamp=True, out=None):
    """Given g and h, compute $g \circ h$ by interpolating each dimension separately"""
    dim = g.ndim - 2
    assert g.ndim == h.ndim, "Dimension of I must be one less that of h"
    assert g.shape[1] == dim, "g must have same number channels as dim"
    assert h.shape[1] == dim, "h must have same number channels as dim"
    if g.shape[0] == 1 and h.shape[0] != 1:
        interp_image_kernel_bcastg(out, I, h)
    if g.shape[0] == 1 and h.shape[0] != 1:
        interp_image_kernel_bcastg(out, I, h)
    if out is None:
        out = gpuarray.empty_like(g, order='C')
    assert out.shape == g.shape, "Output must have same domain as g"
    interp_vectorfield_kernel(out, g, h, displacement=displacement, clamp=clamp)
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
    return np.ascontiguousarray(ix)

def composeHV(h, v, dt=1.0, displacement=True, out=None):
    """
    Given a deformation h, a velocity v, and a time step dt, compute h(x+dt*v(x))
    """
    if out is None:
        out = gpuarray.zeros_like(h)
    dtv = multiply_add(v, dt)
    interp_vec(h, dtv, out=out, displacement=True)
    if displacement: # add the point where we interpolated
        multiply_add(v, dt, out=out)
    return out

def composeVH(v, h, dt=1.0, displacement=True, out=None):
    """
    Given a velocity v, a deformation h, and a time step dt, compute
        h(x) + dt*v(h(x))
    """
    if out is None:
        out = gpuarray.empty_like(h)
    drv.memcpy_dtod(out.gpudata, h.gpudata, size=h.nbytes)
    vh = interp_vec(v, h, displacement=displacement)
    multiply_add(vh, dt, out=out)
    return out
