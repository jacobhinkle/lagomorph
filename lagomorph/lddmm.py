"""
Large Deformation Diffeomorphic Metric Mapping (LDDMM) vector and scalar
momentum shooting algorithms
"""
import torch
import numpy as np
from . import deform
from . import adjrep
import math

def expmap_advect(metric, m, T=1.0, num_steps=10, phiinv=None):
    """Compute EPDiff with vector momenta without using the integrated form.
    
    This is Euler integration of the following ODE:
        d/dt m = - ad_v^* m
    """
    d = len(m.shape)-2
    if phiinv is None:
        phiinv = torch.zeros_like(m)
    dt = T/num_steps
    v = metric.sharp(m)
    phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
    for i in range(num_steps-1):
        m = m - dt*adjrep.ad_star(v, m)
        v = metric.sharp(m)
        phiinv = deform.compose_disp_vel(phiinv, v, dt=-dt)
    return phiinv

def EPDiff_step(metric, m0, dt, phiinv, mommask=None):
    m = adjrep.Ad_star(phiinv, m0)
    if mommask is not None:
        m = m * mommask
    v = metric.sharp(m)
    return deform.compose_disp_vel(phiinv, v, dt=-dt)

class EPDiffStepsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metric, m0, dt, N, phiinv):
        ctx.metric = metric
        ctx.dt = dt
        ctx.N = N
        ctx.save_for_backward(m0, phiinv)
        with torch.no_grad():
            for n in range(N):
                phiinv = EPDiff_step(metric, m0, phiinv, dt)
        return phiinv
    @staticmethod
    def backward(ctx, gradout):
        m0, phiinv = ctx.saved_tensors
        # replay this checkpointed block
        for n in range(ctx.N):
            phiinv = EPDiff_step(ctx.metric, m0, phiinv, ctx.dt)
        phiinv.grad = gradout
        phiinv.backward()
        return None, m0.grad, None, None, phiinv.grad
EPDiff_steps = EPDiffStepsFunction.apply

def expmap(metric, m0, T=1.0, num_steps=10, phiinv=None, mommask=None, checkpoints=False):
    """
    Given an initial momentum (Lie algebra element), compute the exponential
    map.

    What we return is actually only the inverse transformation phi^{-1}
    """
    d = len(m0.shape)-2

    if phiinv is None:
        phiinv = torch.zeros_like(m0)

    if checkpoints is None or not checkpoints:
        # skip checkpointing
        dt = T/num_steps
        for i in range(num_steps):
            phiinv = EPDiff_step(metric, m0, dt, phiinv, mommask=mommask)
    else:
        if isinstance(checkpoints, int):
            cps = checkpoints
            Ncp = (num_steps+checkpoints-1)//checkpoints
        else: # automatically determine number of checkpoints to minimize memory use
            cps = int(math.sqrt(num_steps))
            Ncp = (num_steps+checkpoints-1)//checkpoints
            # adjust actual number of steps so that it's divisible by checkpoint steps
            num_steps = cps*Ncp
            dt = T/num_steps
            for i in range(num_steps):
                phiinv = EPDiff_steps(metric, m0, dt, phiinv)

    return phiinv
