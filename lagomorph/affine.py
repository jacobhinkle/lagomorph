from pycuda import gpuarray
import numpy as np

import math

from .deform import imshape2defshape, defshape2imshape
from .dtypes import dtype2precision
from . import affine_cuda

def affine_gradient(I, J, A, T, outdA=None, outdT=None):
    if outdA is None:
        outdA = gpuarray.empty(shape=A.shape,
                allocator=A.allocator, dtype=A.dtype, order='C')
    if outdT is None:
        outdT = gpuarray.empty(shape=T.shape,
                allocator=T.allocator, dtype=T.dtype, order='C')
    assert I.shape[1:] == J.shape[1:]
    assert A.shape == outdA.shape
    assert T.shape == outdT.shape
    assert A.shape[0] == T.shape[0]
    prec = dtype2precision(I.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        loss = gpuarray.zeros(shape=(1,), allocator=T.allocator, dtype=T.dtype, order='C')
        nn = I.shape[0]
        if I.shape[0] == J.shape[0]:
            f = affine_cuda.affine_grad_2d
        elif I.shape[0] == 1 and J.shape[0] > 1:
            f = affine_cuda.affine_grad_bcastI_2d
            nn = J.shape[0]
        else:
            raise NotImplementedError("Only image broadcasting is supported")
        f(outdA, outdT, loss,
            I, J, A, T,
            np.int32(nn),
            np.int32(I.shape[1]),
            np.int32(I.shape[2]),
            precision=prec, block=block, grid=grid)
        loss = np.sum(loss.get()[0])
    return outdA, outdT, loss

def interp_image_affine(I, A, T, out=None):
    if out is None:
        out = gpuarray.empty(shape=I.shape,
                allocator=I.allocator, dtype=I.dtype, order='C')
    assert I.shape == out.shape
    # no broadcasting yet
    assert I.shape[0] == A.shape[0]
    assert A.shape[0] == T.shape[0]
    prec = dtype2precision(I.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        affine_cuda.interp_image_affine_2d(
                out,
                I, A, T,
                np.int32(I.shape[0]),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
    return out

def splat_image_affine(I, A, T, out=None, outw=None):
    if out is None:
        out = gpuarray.empty(shape=I.shape,
                allocator=I.allocator, dtype=I.dtype, order='C')
    if outw is None:
        outw = gpuarray.empty(shape=I.shape,
                allocator=I.allocator, dtype=I.dtype, order='C')
    assert I.shape == out.shape
    # no broadcasting yet
    assert I.shape[0] == A.shape[0]
    assert A.shape[0] == T.shape[0]
    assert out.shape == outw.shape
    assert out.dtype == outw.dtype
    prec = dtype2precision(I.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        affine_cuda.splat_image_affine_2d(
                out, outw,
                I, A, T,
                np.int32(I.shape[0]),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
    return out, outw
