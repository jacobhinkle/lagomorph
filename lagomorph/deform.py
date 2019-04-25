"""
Methods for dealing with deformation fields (displacement fields)
"""
import torch
import lagomorph_ext
import numpy as np
import math

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

class InterpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, u, dt):
        ctx.dt = dt
        ctx.save_for_backward(I, u)
        return lagomorph_ext.interp_forward(
                I.contiguous(),
                u.contiguous(),
                dt)
    @staticmethod
    def backward(ctx, gradout):
        I, u = ctx.saved_tensors
        d_I, d_u = lagomorph_ext.interp_backward(
                gradout.contiguous(),
                I.contiguous(),
                u.contiguous(),
                ctx.dt,
                *ctx.needs_input_grad[:2])
        return d_I, d_u, None
def interp(I, u, dt=1.0):
    return InterpFunction.apply(I, u, dt)

def interp_hessian_diagonal_image(I, u, dt=1.0):
    """Return the Hessian diagonal with respect to I of interp(I,u,dt)"""
    return lagomorph_ext.interp_hessian_diagonal_image(I, u, dt)

def compose(u, v, ds=1.0, dt=1.0):
    """Return ds*u(x) + dt*v(x + ds*u(x))"""
    return ds*u + dt*interp(v, u, dt=ds)

def compose_disp_vel(u, v, dt=1.0):
    """Given a displacement u, a velocity v, and a time step dt, compute
        dt*v(x) + u(x+dt*v(x))
    """
    return compose(v, u, ds=dt, dt=1.0)

def compose_vel_disp(v, u, dt=1.0):
    """
    Given a velocity v, a displacement u, and a time step dt, compute
        u(x) + dt*v(x + u(x))
    """
    return compose(u, v, ds=1.0, dt=dt)
