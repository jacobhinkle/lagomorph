from pycuda import gpuarray
import numpy as np


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


def composeHV(h, v, dt=1.0):
    """
    Given a deformation h, a velocity v, and a time step dt, compute h(x+dt*v(x))
    """
    raise NotImplementedError("not implemented")
