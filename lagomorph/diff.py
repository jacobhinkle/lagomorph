from pycuda import gpuarray
from .deform import imshape2defshape

def grad(im, out=None):
    """
    Compute the finite-difference gradient of an image set
    """
    if out is None:
        out = gpuarray.empty(shape=imshape2defshape(im.shape), dtype=im.dtype, order='C')
    print("WARNING: grad not implemented")
    return out
