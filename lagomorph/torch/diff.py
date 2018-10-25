import torch
import lagomorph_torch_cuda
import numpy as np
import math

class JacobianTimesVectorFieldFunction(torch.autograd.Function):
    """
    Compute the finite-difference Jacobian matrix of v and contract that with w
    at each point.

    If the argument 'displacement' is True (False by default) then the argument
    v is treated as the displacement of a deformation whose Jacobian matrix
    should be computed (by adding ones to the diagonal) instead of v itself.
    """
    @staticmethod
    def forward(ctx, v, w, displacement):
        ctx.displacement = displacement
        ctx.save_for_backward(v, w)
        return lagomorph_torch_cuda.jacobian_times_vectorfield_forward(v, w, displacement)
    @staticmethod
    def backward(ctx, gradout):
        v, w = ctx.saved_tensors
        d_v, d_w = lagomorph_torch_cuda.jacobian_times_vectorfield_backward(gradout, v, w, ctx.displacement, *ctx.needs_input_grad[:2])
        return d_v, d_w, None
def jacobian_times_vectorfield(v, w, displacement=True):
    return JacobianTimesVectorFieldFunction.apply(v, w, displacement)

class JacobianTimesVectorFieldAdjointFunction(torch.autograd.Function):
    """
    The adjoint T(v)^\dagger w, of the linear operation T(w)v = (Dv)w

    If the argument 'displacement' is True (False by default) then the argument
    v is treated as the displacement of a deformation whose Jacobian matrix
    should be computed (by adding ones to the diagonal) instead of v itself.
    """
    @staticmethod
    def forward(ctx, v, w, displacement):
        ctx.displacement = displacement
        ctx.save_for_backward(v, w)
        return lagomorph_torch_cuda.jacobian_times_vectorfield_forward(v, w, displacement)
    @staticmethod
    def backward(ctx, gradout):
        v, w = ctx.saved_tensors
        d_v, d_w = lagomorph_torch_cuda.jacobian_times_vectorfield_backward(gradout, v, w, ctx.displacement, *ctx.needs_input_grad[:2])
        return d_v, d_w, None
def jacobian_times_vectorfield_adjoint(v, w, displacement=True):
    return JacobianTimesVectorFieldFunction.apply(v, w, displacement)
