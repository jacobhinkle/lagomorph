import torch
import lagomorph_cuda
import numpy as np
import math

class JacobianTimesVectorFieldFunction(torch.autograd.Function):
    """
    Compute the finite-difference Jacobian matrix of v and contract that with w
    at each point.

    If the argument 'displacement' is True (False by default) then the argument
    v is treated as the displacement of a deformation whose Jacobian matrix
    should be computed (by adding ones to the diagonal) instead of v itself.

    If the argument 'transpose' is True (False by default) then the Jacobian is
    transposed before contracting with w.
    """
    @staticmethod
    def forward(ctx, v, w, displacement, transpose):
        ctx.displacement = displacement
        ctx.transpose = transpose
        ctx.save_for_backward(v, w)
        return lagomorph_cuda.jacobian_times_vectorfield_forward(v, w, displacement, transpose)
    @staticmethod
    def backward(ctx, gradout):
        v, w = ctx.saved_tensors
        d_v, d_w = lagomorph_cuda.jacobian_times_vectorfield_backward(gradout, v, w, ctx.displacement, ctx.transpose, *ctx.needs_input_grad[:2])
        return d_v, d_w, None, None
def jacobian_times_vectorfield(v, w, displacement=True, transpose=False):
    return JacobianTimesVectorFieldFunction.apply(v, w, displacement, transpose)

class JacobianTimesVectorFieldAdjointFunction(torch.autograd.Function):
    """
    The adjoint T(w)^\dagger v, of the linear operation T(w)v = (Dv)w
    """
    @staticmethod
    def forward(ctx, v, w):
        ctx.save_for_backward(v, w)
        return lagomorph_cuda.jacobian_times_vectorfield_adjoint_forward(v, w)
    @staticmethod
    def backward(ctx, gradout):
        v, w = ctx.saved_tensors
        d_v, d_w = lagomorph_cuda.jacobian_times_vectorfield_adjoint_backward(gradout, v, w, *ctx.needs_input_grad[:2])
        return d_v, d_w, None
jacobian_times_vectorfield_adjoint = JacobianTimesVectorFieldAdjointFunction.apply
