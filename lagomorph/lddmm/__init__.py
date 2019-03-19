"""
Large Deformation Diffeomorphic Metric Mapping (LDDMM) vector and scalar
momentum shooting algorithms
"""
import torch
from torch.nn.functional import mse_loss
import numpy as np
from .. import deform
from .. import adjrep
from ..affine import regrid
from ..metric import FluidMetric
from ..utils import tqdm
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

def lddmm_atlas(dataset,
        I0=None,
        num_epochs=500,
        batch_size=10,
        loader_workers=8,
        lddmm_steps=1,
        lddmm_integration_steps=5,
        reg_weight=1e2,
        learning_rate_pose = 2e2,
        learning_rate_image = 1e4,
        fluid_params=[0.1,0.,.01],
        device='cuda',
        momentum_shape=None,
        momentum_preconditioning=True,
        momentum_pattern='atlas_momenta/momentum_{}.pth',
        gpu=None,
        world_size=1,
        rank=0):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    if world_size > 1:
        sampler = DistributedSampler(dataset, 
                num_replicas=world_size,
                rank=rank)
    else:
        sampler = None
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
            num_workers=loader_workers, pin_memory=True, shuffle=False)
    if I0 is None: # initialize base image to mean
        from ..affine import batch_average
        I0 = batch_average(dataloader, dim=0, returns_indices=True)
    else:
        I0 = I0.clone()
    I = I0.to(device).view(1,1,*I0.squeeze().shape)
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_image,
                                      weight_decay=0)
    metric = FluidMetric(fluid_params)
    losses = []
    reg_terms = []
    iter_losses = []
    epbar = range(num_epochs)
    if rank == 0:
        epbar = tqdm(epbar, desc='epoch')
    dim = len(I.shape)-2
    if momentum_shape is None:
        momentum_shape = I0.shape[-dim:]
    regrid_momenta = momentum_shape != I0.shape[-dim:]
    ms = torch.zeros(len(dataset),dim,*momentum_shape, dtype=I0.dtype).pin_memory()
    for epoch in epbar:
        epoch_loss = 0.0
        epoch_reg_term = 0.0
        itbar = dataloader
        if rank == 0:
            itbar = tqdm(itbar, desc='iter')
        I.requires_grad_(True)
        image_optimizer.zero_grad()
        for it, (ix, img) in enumerate(itbar):
            m = ms[ix,...].detach()
            m = m.to(device)
            img = img.to(device)
            for lit in range(lddmm_steps):
                # compute image gradient in last step
                I.requires_grad_(lit == lddmm_steps - 1)
                # enables taking multiple LDDMM step per image update
                m.requires_grad_(True)
                if m.grad is not None:
                    m.grad.detach_()
                    m.grad.zero_()
                h = expmap(metric, m, num_steps=lddmm_integration_steps)
                if regrid_momenta:
                    h = regrid(h, shape=I.shape[2:])
                Idef = deform.interp(I, h)
                v = metric.sharp(m)
                regterm = reg_weight*(v*m).sum()
                if regrid_momenta: # account for downscaling in averaging
                    regterm = regterm * (I0.numel()/v[0,0,...].numel())
                loss = (mse_loss(Idef, img, reduction='sum') + regterm) \
                        / (img.numel())
                loss.backward()
                # this makes it so that we can reduce the loss and eventually get
                # an accurate MSE for the entire dataset
                with torch.no_grad():
                    li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
                    p = m.grad
                    if momentum_preconditioning:
                        p = metric.flat(p)
                    m.add_(-learning_rate_pose, p)
                    if world_size > 1:
                        all_reduce(li)
                    iter_losses.append(li.item())
                    m = m.detach()
                    del p
            with torch.no_grad():
                epoch_loss += li
                ri = (regterm*(img.shape[0]/(img.numel()*len(dataloader.dataset)))).detach()
                epoch_reg_term += ri
                ms[ix,...] = m.detach().cpu()
            del m, h, Idef, v, loss, regterm, img
        with torch.no_grad():
            if world_size > 1:
                all_reduce(epoch_loss)
                all_reduce(epoch_reg_term)
                all_reduce(I.grad)
                I.grad = I.grad/world_size
            # average over iterations
            I.grad = I.grad / len(dataloader)
        image_optimizer.step()
        losses.append(epoch_loss.item())
        reg_terms.append(epoch_reg_term.item())
        if rank == 0:
            epbar.set_postfix(epoch_loss=epoch_loss.item(),
                    epoch_reg=epoch_reg_term.item())
    return I.detach(), ms.detach(), losses, iter_losses


