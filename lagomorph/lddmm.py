"""
Large Deformation Diffeomorphic Metric Mapping (LDDMM) vector and scalar
momentum shooting algorithms
"""
import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import all_reduce
import numpy as np
from . import deform
from . import adjrep
from .affine import regrid
from .metric import FluidMetric, Metric
from .utils import tqdm, Tool
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

class LDDMMAtlasBuilder:
    _initialized = False
    def __init__(self,
            dataset,
            I0=None,
            ms=None,
            num_epochs=500,
            batch_size=10,
            loader_workers=8,
            lddmm_steps=1,
            lddmm_integration_steps=5,
            image_update_freq=0,
            reg_weight=1e2,
            learning_rate_pose = 2e2,
            learning_rate_image = 1e4,
            metric=None,
            momentum_shape=None,
            momentum_preconditioning=False,
            device='cuda',
            world_size=1,
            rank=0):
        # just record all arguments to constructor as members
        args = vars()
        self._initvars = []
        for k,v in args.items():
            if k != 'self' and k not in vars(self): # k is an arg to this method
                setattr(self, k, v)
                self._initvars.append(k)
    def __setattr__(self, k, v):
        if self._initialized and k in self._initvars:
            raise Exception(f'Member {k} was set in constructor and cannot be '
                    'overwritten after initialization')
        self.__dict__[k] = v
    def initialize(self):
        if not self._initialized:
            self._init_dataloader()
            self._init_atlas_image()
            self._init_metric()
            self._init_losses()
            self._init_momenta()
            self._initialized = True
    def _init_dataloader(self):
        if self.world_size > 1:
            sampler = DistributedSampler(self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank)
        else:
            sampler = None
        self.dataloader = DataLoader(self.dataset, sampler=sampler,
                batch_size=self.batch_size, num_workers=self.loader_workers,
                pin_memory=True, shuffle=False)
    def _init_atlas_image(self):
        if self.I0 is None: # initialize base image to mean
            from .affine import batch_average
            with torch.no_grad():
                # initialize base image to mean
                self.I0 = batch_average(self.dataloader, dim=0,
                        returns_indices=True,
                        progress_bar=rank==0).to(self.device)
                if world_size > 1:
                    all_reduce(self.I0)
                    self.I0 /= world_size
        else:
            self.I0 = self.I0.clone().to(self.device)
        self.I = self.I0.view(1,1,*self.I0.squeeze().shape)
        self.image_optimizer = torch.optim.SGD([self.I],
                                          lr=self.learning_rate_image,
                                          weight_decay=0)
    def _init_metric(self):
        if self.metric is None:
            self.metric = FluidMetric([.1,0,.01])
    def _init_losses(self):
        # initialize losses but only if they are not preloaded
        if 'epoch_losses' not in self.__dict__:
            self.epoch_losses = []
        if 'epoch_reg_terms' not in self.__dict__:
            self.epoch_reg_terms = []
        if 'iter_losses' not in self.__dict__:
            self.iter_losses = []
        if 'iter_reg_terms' not in self.__dict__:
            self.iter_reg_terms = []
    def _init_momenta(self):
        dim = len(self.I.shape)-2
        if self.momentum_shape is None:
            self.momentum_shape = self.I.shape[-dim:]
        self.regrid_momenta = self.momentum_shape != self.I.shape[-dim:]
        if self.ms is None:
            self.ms = torch.zeros(len(self.dataset),dim,*self.momentum_shape)
        self.ms = self.ms.type(self.I.dtype).pin_memory()
    def save(self, filename):
        import h5py
        with h5py.File(filename, 'w') as f:
            atds = f.create_dataset('atlas', data=self.I.detach().cpu().numpy())
            # TODO: streaming copy of momenta
            f.create_dataset('momenta', data=self.ms.detach().cpu().numpy())
            f.create_dataset('epoch_losses', data=np.asarray(self.epoch_losses))
            f.create_dataset('epoch_reg_terms', data=np.asarray(self.epoch_reg_terms))
            f.create_dataset('iter_losses', data=np.asarray(self.iter_losses))
            f.create_dataset('iter_reg_terms', data=np.asarray(self.iter_reg_terms))
    def load(self, filename, load_image=True, load_momenta=True, load_losses=True):
        import h5py
        with h5py.File(filename, 'r') as f:
            if load_image:
                self.I0 = torch.Tensor(f['atlas'])
            if load_momenta:
                self.ms = torch.Tensor(f['momenta'])
            if load_losses:
                self.epoch_losses = list(f['epoch_losses'])
                self.epoch_reg_terms = list(f['epoch_reg_terms'])
                self.iter_losses = list(f['iter_losses'])
                self.iter_reg_terms = list(f['iter_reg_terms'])
    def update_base_image(self, force=False):
        if (self.image_iters < self.image_update_freq and not force) or \
            self.image_iters == 0:
            return
        with torch.no_grad():
            if self.world_size > 1:
                all_reduce(self.I.grad)
            self.I.grad = self.I.grad/(self.image_iters*self.world_size)
            self.image_optimizer.step()
            self.image_optimizer.zero_grad()
        self.image_iters = 0 # reset counter
    def lddmm_step(self, m, img):
        m.requires_grad_(True)
        if m.grad is not None:
            m.grad.detach_()
            m.grad.zero_()
        h = expmap(self.metric, m, num_steps=self.lddmm_integration_steps)
        if self.regrid_momenta:
            h = regrid(h, shape=self.I.shape[2:])
        Idef = deform.interp(self.I, h)
        v = self.metric.sharp(m)
        reg_term = self.reg_weight*(v*m).sum()/img.numel()
        if self.regrid_momenta: # account for downscaling in averaging
            reg_term = reg_term * (self.I.numel()/v[0,0,...].numel())
        loss = mse_loss(Idef, img, reduction='sum')/img.numel() + reg_term
        loss.backward()
        # this makes it so that we can reduce the loss and eventually get
        # an accurate MSE for the entire dataset
        with torch.no_grad():
            norm_factor = (img.shape[0]/len(self.dataloader.dataset))
            loss = (loss*norm_factor).detach()
            reg_term = (reg_term*norm_factor).detach()
            p = m.grad
            if self.momentum_preconditioning:
                p = self.metric.flat(p)
            m.add_(-self.learning_rate_pose, p)
            if self.world_size > 1:
                all_reduce(loss)
                all_reduce(reg_term)
            m = m.detach()
        return m, loss.item(), reg_term.item()
    def iteration(self, ix, img):
        m = self.ms[ix,...].detach()
        m = m.to(self.device)
        img = img.to(self.device)
        for lit in range(self.lddmm_steps):
            self.I.requires_grad_(lit == self.lddmm_steps - 1)
            m, loss, reg_term = self.lddmm_step(m, img)
        self.ms[ix,...] = m.to(self.ms.device)
        self.image_iters += 1
        self.update_base_image()
        return loss, reg_term
    def epoch(self):
        epoch_loss = 0.0
        epoch_reg_term = 0.0
        itbar = self.dataloader
        if self.rank == 0:
            itbar = tqdm(itbar, desc='iter')
        if self.image_update_freq == 0:
            self.image_optimizer.zero_grad()
        self.image_iters = 0 # how many iters accumulated
        for it, (ix, img) in enumerate(itbar):
            iter_loss, iter_reg_term = self.iteration(ix, img)
            self.iter_losses.append(iter_loss)
            self.iter_reg_terms.append(iter_reg_term)
            epoch_loss += iter_loss
            epoch_reg_term += iter_reg_term
        self.update_base_image(force=True)
        return epoch_loss, epoch_reg_term
    def run(self):
        self.initialize()
        epbar = range(self.num_epochs)
        if self.rank == 0:
            epbar = tqdm(epbar)
        self.image_optimizer.zero_grad()
        for epoch in epbar:
            epoch_loss, epoch_reg_term = self.epoch()
            self.epoch_losses.append(epoch_loss)
            self.epoch_reg_terms.append(epoch_reg_term)
            if self.rank == 0:
                epbar.set_postfix(epoch_loss=epoch_loss,
                        epoch_reg=epoch_reg_term)


class _Tool(Tool):
    """Diffeomorphic registration methods using LDDMM"""
    module_name = 'lagomorph.lddmm'
    subcommands = ['atlas']
    def atlas(self):
        """
        Build LDDMM atlas from HDF5 image dataset.

        This command will result in a new HDF5 file containing the following datasets:
            atlas: the atlas image
            momenta: a momentum vector field for each input image
            epoch_losses: mean squared error + regularization terms averaged across epochs (this is just an average of the iteration losses per epoch)
            iter_losses: loss at each iteration

        Note that metadata like the lagomorph version and parameters this
        command was invoked with are attached to the 'atlas' dataset as
        attributes.
        """
        import argparse, sys
        parser = self.new_parser('atlas')
        # prefixing the argument with -- means it's optional
        dg = parser.add_argument_group('data parameters')
        dg.add_argument('input', type=str, help='Path to input image HDF5 file')
        dg.add_argument('--force_dim', default=None, type=int, help='Force dimension of images instead of determining based on dataset shape')
        dg.add_argument('--h5key', '-k', default='images', help='Name of dataset in input HDF5 file')
        dg.add_argument('--loader_workers', default=8, help='Number of concurrent workers for dataloader')
        dg.add_argument('output', type=str, help='Path to output HDF5 file')

        ag = parser.add_argument_group('algorithm parameters')
        ag.add_argument('--initial_atlas', default=None, type=str, help="Path to h5 file with which to initialize image and momenta")
        ag.add_argument('--num_epochs', default=1000, type=int, help='Number of epochs')
        ag.add_argument('--batch_size', default=50, type=int, help='Batch size')
        ag.add_argument('--precondition_momentum', action='store_true', help='Whether to precondition momentum before gradient descent by applying metric operator')
        ag.add_argument('--image_update_freq', default=0, type=int, help='Update base image every N iterations. 0 for once per epoch')
        ag.add_argument('--lddmm_steps', default=1, type=int, help='LDDMM gradient steps to take each iteration')
        ag.add_argument('--deformation_scale', default=1, type=int, help='Amount to downscale grid for LDDMM momenta/deformation')
        ag.add_argument('--reg_weight', default=1e-1, type=float, help='Amount of regularization for deformations')
        ag.add_argument('--learning_rate_m', default=1e-3, type=float, help='Learning rate for momenta')
        ag.add_argument('--learning_rate_I', default=1e5, type=float, help='Learning rate for atlas image')

        mg = parser.add_argument_group('metric parameters')
        Metric.add_args(mg)

        self._compute_args(parser)
        args = parser.parse_args(sys.argv[2:])
        self._initialize_compute(args)

        from .data import H5Dataset
        dataset = H5Dataset(args.input, key=args.h5key, return_indices=True,
                force_dim=args.force_dim)

        if args.deformation_scale != 1:
            im0 = dataset[0][1]
            momentum_shape = [s//args.deformation_scale for s in im0.shape[1:]]
            del im0
        else:
            momentum_shape = None

        metric = Metric.from_args(args)

        builder = LDDMMAtlasBuilder(dataset,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            lddmm_steps=args.lddmm_steps,
            image_update_freq=args.image_update_freq,
            momentum_shape=momentum_shape,
            reg_weight=args.reg_weight,
            momentum_preconditioning=args.precondition_momentum,
            metric=metric,
            learning_rate_pose=args.learning_rate_m,
            learning_rate_image=args.learning_rate_I,
            loader_workers=args.loader_workers,
            world_size=self.world_size,
            rank=self.rank,
            device=f"cuda:{self.gpu}")

        if args.initial_atlas is not None:
            builder.load(args.initial_atlas)

        builder.run()

        builder.save(args.output)

        import h5py
        with h5py.File(args.output, 'a') as f:
            self._stamp_dataset(f['atlas'], args)
