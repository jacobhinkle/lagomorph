import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


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
    dim = I.ndims - 1
    assert I.ndims == h.ndims-1, "Dimension of I must be one less that of h"
    assert h.shape[1] == dim, "h must have same number channels as dim"
    if out is None:
        out = empty(shape=h.shape[2:], dtype=I.dtype)
    assert out.shape == h.shape[2:], "Output must have same domain as h"
    if I.shape[0] == 1 and h.shape[0] != 0:
        interp_image_kernel_bcastI(dim)
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
