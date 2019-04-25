import torch
from torch.nn.functional import mse_loss
from torch.distributed import all_reduce
import numpy as np
from .utils import tqdm, Tool
from .data import batch_average
import lagomorph_ext
import math, ctypes


class AffineInterpFunction(torch.autograd.Function):
    """Interpolate an image using an affine transformation, parametrized by a
    separate matrix and translation vector.
    """
    @staticmethod
    def forward(ctx, I, A, T):
        ctx.save_for_backward(I, A, T)
        return lagomorph_ext.affine_interp_forward(
            I.contiguous(),
            A.contiguous(),
            T.contiguous())
    @staticmethod
    def backward(ctx, grad_out):
        I, A, T = ctx.saved_tensors
        d_I, d_A, d_T = lagomorph_ext.affine_interp_backward(
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
        reg = lagomorph_ext.regrid_forward(
            I.contiguous(),
            outshape,
            origin,
            spacing)
        if displacement:
            dim = len(I.shape) - 2
            if I.shape[1] != dim:
                raise ValueError("Incorrect num channels for regridding displacement")
            ctx.spacing_tensor = 1./torch.Tensor(spacing).type(reg.type()).to(reg.device).view(1,dim,*[1]*dim)
            #reg = reg * ctx.spacing_tensor
            reg.mul_(ctx.spacing_tensor)
        return reg
    @staticmethod
    def backward(ctx, grad_out):
        d_I = lagomorph_ext.regrid_backward(
            grad_out.contiguous(),
            ctx.inshape,
            ctx.outshape,
            ctx.outorigin,
            ctx.outspacing)
        if ctx.displacement:
            d_I.mul_(ctx.spacing_tensor)
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

def affine_atlas(dataset,
                 As,
                 Ts,
                I=None,
                num_epochs=1000,
                batch_size=50,
                image_update_freq=0,
                affine_steps=1,
                reg_weightA=0e1,
                reg_weightT=0e1,
                learning_rate_A=1e-3,
                learning_rate_T=1e-2,
                learning_rate_I=1e5,
                loader_workers=8,
                gpu=None,
                world_size=1,
                rank=0):
    L2 = lambda a,b: torch.dot(a.view(-1), b.view(-1))
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    if world_size > 1:
        sampler = DistributedSampler(dataset, 
                num_replicas=world_size,
                rank=rank)
    else:
        sampler = None
    if gpu is None:
        device = 'cpu'
    else:
        device = f'cuda:{gpu}'
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            shuffle=False, num_workers=loader_workers,
                            pin_memory=True, drop_last=False)
    if I is None:
        with torch.no_grad():
            # initialize base image to mean
            I = batch_average(dataloader, dim=0, returns_indices=True,
                    progress_bar=rank==0).to(device)
            if world_size > 1:
                all_reduce(I)
                I /= world_size
    else:
        I = I.clone().to(device)
    I = I.type(As.dtype).view(1,1,*I.squeeze().shape)
    image_optimizer = torch.optim.SGD([I],
                                      lr=learning_rate_I,
                                      weight_decay=0.)
    dim = I.dim() - 2
    eye = torch.eye(dim).view(1,dim,dim).type(I.dtype).to(I.device)
    epoch_losses = []
    iter_losses = []
    epbar = range(num_epochs)
    if rank == 0:
        epbar = tqdm(epbar, desc='epoch')
    for epoch in epbar:
        epoch_loss = torch.Tensor([0.0])[0].to(device)
        itbar = dataloader
        if rank == 0:
            itbar = tqdm(dataloader, desc='iter')
        if image_update_freq == 0 or epoch == 0:
            image_optimizer.zero_grad()
        image_iters = 0 # how many iters accumulated
        for it, (ix, img) in enumerate(itbar):
            A = As[ix,...].detach().to(device).contiguous()
            T = Ts[ix,...].detach().to(device).contiguous()
            img = img.type(As.dtype).to(device)
            img.requires_grad_(False)
            for affit in range(affine_steps):
                A.requires_grad_(True)
                T.requires_grad_(True)
                if A.grad is not None:
                    A.grad.detach_()
                    A.grad.zero_()
                if T.grad is not None:
                    T.grad.detach_()
                    T.grad.zero_()
                # only accumulate image gradient at last affine step
                I.requires_grad_(affit == affine_steps-1)
                Idef = affine_interp(I, A+eye, T)
                regloss = 0.0
                if reg_weightA > 0:
                    regtermA = L2(A,A)
                    regloss = regloss + .5*reg_weightA*regtermA
                if reg_weightT > 0:
                    regtermT = L2(T,T)
                    regloss = regloss + .5*reg_weightT*regtermT
                loss = (mse_loss(Idef, img, reduction='sum')*(1./np.prod(I.shape[2:])) + regloss) \
                        / (img.shape[0])
                loss.backward()
                loss.detach_()
                with torch.no_grad():
                    li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
                    iter_losses.append(li.item())
                    A.add_(-learning_rate_A, A.grad)
                    T.add_(-learning_rate_T, T.grad)
            image_iters += 1
            if image_iters == image_update_freq:
                with torch.no_grad():
                    if world_size > 1:
                        all_reduce(epoch_loss)
                        all_reduce(I.grad)
                    I.grad = I.grad/(image_iters*world_size)
                image_optimizer.step()
                image_optimizer.zero_grad()
                image_iters = 0
            with torch.no_grad():
                li = (loss*(img.shape[0]/len(dataloader.dataset))).detach()
                epoch_loss = epoch_loss + li
            #itbar.set_postfix(minibatch_loss=itloss)
            As[ix,...] = A.detach().to(As.device)
            Ts[ix,...] = T.detach().to(Ts.device)
        if image_iters > 0:
            with torch.no_grad():
                if world_size > 1:
                    all_reduce(epoch_loss)
                    all_reduce(I.grad)
                I.grad = I.grad/(image_iters*world_size)
                image_optimizer.step()
        epoch_losses.append(epoch_loss.item())
        if rank == 0: epbar.set_postfix(epoch_loss=epoch_loss.item())
    return I.detach(), As.detach(), Ts.detach(), epoch_losses, iter_losses


class StandardizedDataset():
    def __init__(self, dataset, As, Ts, device='cuda'):
        self.dataset = dataset
        self.As = As
        self.Ts = Ts
        self.device = device
        dim = Ts.shape[1]
        self.eye = torch.eye(dim).view(1,dim,dim).type(As.dtype).to(device)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        J = self.dataset[idx]
        J = J.to(self.device).unsqueeze(0)
        A = self.As[[idx],...].to(self.device)
        T = self.Ts[[idx],...].to(self.device)
        Ainv, Tinv = affine_inverse(A+self.eye, T)
        if J.dtype not in [torch.float32, torch.float64]:
            J = J.type(torch.float32)
        return affine_interp(J, Ainv, Tinv).squeeze(0)

class _Tool(Tool):
    """Affine registration methods"""
    module_name = 'lagomorph affine'
    subcommands = ['atlas','standardize']
    def atlas(self):
        """
        Build affine atlas from HDF5 image dataset.

        This command will result in a new HDF5 file containing the following datasets:
            atlas: the atlas image
            A: d-by-d transformation matrices for each input image
            T: translation vectors for each input image
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
        dg.add_argument('--loader_workers', default=8, type=int, help='Number of concurrent workers for dataloader')
        dg.add_argument('--data_inmemory', action='store_true', help='Load entire dataset into memory first')
        dg.add_argument('output', type=str, help='Path to output HDF5 file')

        ag = parser.add_argument_group('algorithm parameters')
        ag.add_argument('--num_epochs', default=1000, type=int, help='Number of epochs')
        ag.add_argument('--batch_size', default=50, type=int, help='Batch size')
        ag.add_argument('--image_update_freq', default=0, type=int, help='Update base image every N iterations. 0 for once per epoch')
        ag.add_argument('--affine_steps', default=1, type=int, help='Affine gradient steps to take each iteration')
        ag.add_argument('--reg_weight_A', default=1e-1, type=float, help='Amount of regularization for matrix A')
        ag.add_argument('--reg_weight_T', default=1e-1, type=float, help='Amount of regularization for vector T')
        ag.add_argument('--learning_rate_A', default=1e-3, type=float, help='Learning rate for matrix A')
        ag.add_argument('--learning_rate_T', default=1e-2, type=float, help='Learning rate for vector T')
        ag.add_argument('--learning_rate_I', default=1e4, type=float, help='Learning rate for atlas image')

        self._compute_args(parser)
        args = parser.parse_args(sys.argv[2:])
        self._initialize_compute(args)

        from .data import H5Dataset, load_dataset
        dataset = load_dataset(args.input, key=args.h5key, return_indices=True,
                force_dim=args.force_dim)

        if args.data_inmemory:
            pinned = []
            for i in tqdm(range(len(dataset)), 'pre-loading data'):
                ix, x = dataset[i]
                pinned.append((ix, torch.as_tensor(x).pin_memory()))
            dataset = pinned
        # initialize affine transforms on CPU for entire dataset
        n = len(dataset)
        ds0 = dataset[0][1]
        dim = ds0.dim()-1
        del ds0
        As = torch.zeros((n, dim, dim), dtype=torch.float32)
        Ts = torch.zeros((n, dim), dtype=torch.float32)

        I, As, Ts, epoch_losses, iter_losses = affine_atlas(dataset,
                As=As,
                Ts=Ts,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                affine_steps=args.affine_steps,
                image_update_freq=args.image_update_freq,
                reg_weightA=args.reg_weight_A,
                reg_weightT=args.reg_weight_T,
                learning_rate_A=args.learning_rate_A,
                learning_rate_T=args.learning_rate_T,
                learning_rate_I=args.learning_rate_I,
                loader_workers=args.loader_workers,
                world_size=self.world_size,
                rank=self.rank,
                gpu=self.gpu)

        import h5py
        with h5py.File(args.output, 'w') as f:
            atds = f.create_dataset('atlas', data=I.cpu().numpy())
            self._stamp_dataset(atds, args)
            f.create_dataset('A', data=As.cpu().numpy())
            f.create_dataset('T', data=Ts.cpu().numpy())
            f.create_dataset('epoch_losses', data=np.asarray(epoch_losses))
            f.create_dataset('iter_losses', data=np.asarray(iter_losses))
    def standardize(self):
        """
        Standardize a dataset using transforms found during atlas building.

        Note that metadata like the lagomorph version and parameters this
        command was invoked with are attached to the output dataset
        (corresponding to the '--h5key') as attributes.
        """
        import argparse, sys
        parser = self.new_parser('standardize')
        # prefixing the argument with -- means it's optional
        parser.add_argument('inputimages', type=str, help='Path to input image HDF5 file')
        parser.add_argument('atlasoutput', type=str, help='Path to HDF5 output from affine atlas building')
        parser.add_argument('standardizedoutput', type=str, help='Path to output HDF5 file')
        parser.add_argument('--h5key', '-k', default='images', help='Name of dataset in input and HDF5 files')
        parser.add_argument('--copy_other_keys', action='store_true', help='Copy all other keys from input file into output verbatim')
        parser.add_argument('--rescale', default=None, type=float, help='Amount by which to rescale translations. Default: automatic')
        args = parser.parse_args(sys.argv[2:])

        from .data import H5Dataset, write_dataset_h5
        dataset = H5Dataset(args.inputimages, key=args.h5key)

        import h5py
        with h5py.File(args.atlasoutput, 'r') as f:
            As = torch.Tensor(f['A'])
            Ts = torch.Tensor(f['T'])
            if args.rescale is None:
                # compare size of atlas to size of dataset to be standardized
                # this determines the degree to which we scale T to compensate for
                # differences in resolution
                d = Ts.shape[1]
                shnew = dataset[0].shape[-d:]
                shatlas = f['atlas'].shape[-d:]
                if shnew != shatlas:
                    args.rescale = shnew[0] / shatlas[0]
                    for sn,sa in zip(shnew,shatlas):
                        if sn != args.rescale * sa:
                            raise Exception("Unclear how to rescale translations. You must pass the --rescale argument directly.")
                else:
                    args.rescale = 1.0
        Ts *= args.rescale

        std_ds = StandardizedDataset(dataset, As, Ts)
        write_dataset_h5(std_ds, args.standardizedoutput, key=args.h5key)
        with h5py.File(args.standardizedoutput, 'a') as fw:
            self._stamp_dataset(fw[args.h5key], args)
        if args.copy_other_keys:
            with h5py.File(args.inputimages, 'r') as fi, h5py.File(args.standardizedoutput, 'a') as fo:
                for k in tqdm(fi.keys(), desc='other keys'):
                    if k != args.h5key:
                        fi.copy(k, fo)
