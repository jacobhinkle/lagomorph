import torch
import numpy as np
import lagomorph_cuda
import math, ctypes

class AffineInterpFunction(torch.autograd.Function):
    """Interpolate an image using an affine transformation, parametrized by a
    separate matrix and translation vector.
    """
    @staticmethod
    def forward(ctx, I, A, T):
        ctx.save_for_backward(I, A, T)
        return lagomorph_cuda.affine_interp_forward(
            I.contiguous(),
            A.contiguous(),
            T.contiguous())
    @staticmethod
    def backward(ctx, grad_out):
        I, A, T = ctx.saved_tensors
        d_I, d_A, d_T = lagomorph_cuda.affine_interp_backward(
                grad_out.contiguous(),
                I.contiguous(),
                A.contiguous(),
                T.contiguous(),
                *ctx.needs_input_grad)
        return d_I, d_A, d_T
affine_interp = AffineInterpFunction.apply

class AffineInterp(torch.nn.Module):
    """Module wrapper for AffineInterpFunction"""
    def __init__(self):
        super(AffineInterp, self).__init__()
    def forward(self, I, A, T):
        return AffineInterpFunction.apply(I, A, T)

def det_2x2(A):
    return A[:,0,0]*A[:,1,1] - A[:,0,1]*A[:,1,0]

def invert_2x2(A):
    """Invert 2x2 matrix using simple formula in batch mode.
    This assumes the matrix is invertible and provides no further checks."""
    det = det_2x2(A)
    Ainv = torch.stack((A[:,1,1],-A[:,0,1],-A[:,1,0],A[:,0,0]), dim=1).view(-1,2,2)/det.view(-1,1,1)
    return Ainv

def minor(A, i, j):
    assert A.shape[1] == A.shape[2]
    n = A.shape[1]
    M = torch.cat((A.narrow(1, 0, i), A.narrow(1,i+1,n-i-1)), dim=1)
    M = torch.cat((M.narrow(2, 0, j), M.narrow(2,j+1,n-j-1)), dim=2)
    return M

def invert_3x3(A):
    """Invert a 3x3 matrix in batch mode. We use the formula involving
    determinants of minors here
    http://mathworld.wolfram.com/MatrixInverse.html
    """
    cofactors = torch.stack((
        det_2x2(minor(A,0,0)),
       -det_2x2(minor(A,0,1)),
        det_2x2(minor(A,0,2)),
       -det_2x2(minor(A,1,0)),
        det_2x2(minor(A,1,1)),
       -det_2x2(minor(A,1,2)),
        det_2x2(minor(A,2,0)),
       -det_2x2(minor(A,2,1)),
        det_2x2(minor(A,2,2)),
        ), dim=1).view(-1,3,3).transpose(1,2)
    # write determinant using minors matrix
    det =   cofactors[:,0,0]*A[:,0,0] \
          + cofactors[:,1,0]*A[:,0,1] \
          + cofactors[:,2,0]*A[:,0,2]
    return cofactors/det.view(-1,1,1)

def affine_inverse(A, T):
    """Invert an affine transformation.
    
    A transformation (A,T) is inverted by computing (A^{-1}, -A^{-1} T)
    """
    assert A.shape[1] == A.shape[2]
    assert A.shape[1] == T.shape[1]
    dim = A.shape[1]
    assert dim == 2 or dim == 3
    if dim == 2:
        Ainv = invert_2x2(A)
    elif dim == 3:
        Ainv = invert_3x3(A)
    Tinv = -torch.matmul(Ainv, T.unsqueeze(2)).squeeze(2)
    return (Ainv, Tinv)

def rotation_exp_map(v):
    """Convert a collection of tangent vectors to rotation matrices. This allows
    for rigid registration using unconstrained optimization by composing this
    function with the affine interpolation methods and a loss function.

    For 2D rotations, v should be a vector of angles in radians. For 3D
    rotations v should be an n-by-3 array of 3-vectors indicating the requested
    rotation in axis-angle format, in which case the conversion is done using
    the Rodrigues' rotation formula.
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    if v.dim() == 1: # 2D case
        c = torch.cos(v).view(-1,1)
        s = torch.sin(v).view(-1,1)
        return torch.stack((c, -s, s, c), dim=1).view(-1,2,2)
    elif v.dim() == 2 and v.size(1) == 3:
        raise NotImplementedError()
    else:
        raise Exception(f"Cannot infer dimension from v shape {v.shape}")

def rigid_inverse(v, T):
    """Invert a rigid transformation using the formula
        (R(v),T)^{-1} = (R(-v), -R(-v) T)
    """
    negv = -v
    Rinv = rotation_exp_map(negv)
    Tinv = -torch.matmul(Rinv, T.unsqueeze(2)).squeeze(2)
    return (negv, Tinv)

class RegridFunction(torch.autograd.Function):
    """Interpolate an image from one grid to another."""
    @staticmethod
    def forward(ctx, I, outshape, origin, spacing, displacement):
        outshape = [int(s) for s in outshape]
        origin = [float(o) for o in origin]
        spacing = [float(s) for s in spacing]
        ctx.inshape = I.shape[2:]
        ctx.outshape = outshape
        ctx.outorigin = origin
        ctx.outspacing = spacing
        ctx.displacement = displacement
        reg = lagomorph_cuda.regrid_forward(
            I.contiguous(),
            outshape,
            origin,
            spacing)
        if displacement:
            dim = len(I.shape) - 2
            if I.shape[1] != dim:
                raise ValueError("Incorrect num channels for regridding displacement")
            ctx.spacing_tensor = 1./torch.Tensor(spacing).type(reg.type()).to(reg.device).view(1,dim,*[1]*dim)
            reg = reg * ctx.spacing_tensor
        return reg
    @staticmethod
    def backward(ctx, grad_out):
        d_I = lagomorph_cuda.regrid_backward(
            grad_out.contiguous(),
            ctx.inshape,
            ctx.outshape,
            ctx.outorigin,
            ctx.outspacing)
        if ctx.displacement:
            d_I = d_I * ctx.spacing_tensor
        return d_I, None, None, None, None
def regrid(I, shape=None, origin=None, spacing=None, displacement=False):
    """Interpolate from one regular grid to another.

    The input grid is assumed to have the origin at (N-1)/2 where N is the size
    of a given dimension, and a spacing of 1.

    The output grid is determined by providing at least one of the optional
    arguments shape, origin, and spacing. If any of these are scalar, that value
    will be used in every dimension. The following are the rules used, with the
    given parameters in parentheses:

        () An exception is raised

        (spacing) Origin is assumed at the center of the image, and shape is
        determined in order to cover the original image domain, placing voxels
        slightly outside the domain if necessary.

        (origin) We simply translate the image by the difference in origins,
        which is equivalent to assuming the same output shape and spacing as the
        input.

        (origin, spacing) An exception is raised.

        (shape) Origin is assumed to be (I.shape-1)/2, and spacing is determined
        such that corner voxels are placed in the same place as in the input
        image: spacing = (outshape-1)/(inshape-1)

        (shape, spacing) Origin is assumed to be the middle of the image.

        (shape, origin) Spacing is assumed to be 1, as this can be used to
        easily extract a small ROI.

        (shape, origin, spacing) Specified values are used with no modification.


    The expected common use case will be upscaling an image or vector field by
    only providing the new shape.

    Note that _downscaling_ using this method is not wise. You can downscale by
    integer factors in a simple way using PyTorch's built in mean pooling.
    Alternatively, you could Gaussian filter the image then apply this function.

    If the 'displacement' argument is True, then in addition to interpolating to
    the new grid, the values will be scaled by the spacing in each dimension.
    This is only valid if the number of channels in the input is equal to the
    spatial dimension.
    """
    if shape is None:
        if origin is None:
            if spacing is None:
                raise ValueError("At least one of shape, origin, or spacing required")
            else:
                raise NotImplementedError
        else:
            if spacing is None:
                raise NotImplementedError
            else:
                raise ValueError("Shape is required if specifying origin and spacing")
    else:
        if origin is None:
            origin = tuple([(s-1)*.5 for s in I.shape[2:]])
            if spacing is None:
                spacing = tuple([(sI-1)/(s-1)
                            for sI, s in zip(I.shape[2:],shape)])
        else:
            if spacing is None:
                raise NotImplementedError
            else:
                raise NotImplementedError

    d = len(I.shape)-2
    if not isinstance(shape, (list,tuple)):
        shape = tuple([shape]*d)
    if not isinstance(origin, (list,tuple)):
        origin = tuple([origin]*d)
    if not isinstance(spacing, (list,tuple)):
        spacing = tuple([spacing]*d)
    assert len(shape)==d
    assert len(origin)==d
    assert len(spacing)==d

    return RegridFunction.apply(I, shape, origin, spacing, displacement)


class RegridModule(torch.nn.Module):
    """Module wrapper for RegridFunction"""
    def __init__(self, shape, origin, spacing):
        super(RegridModule, self).__init__()
        self.shape = shape
        self.origin = origin
        self.spacing = spacing
    def forward(self, I):
        return regrid(I, self.shape, self.origin, self.spacing)
