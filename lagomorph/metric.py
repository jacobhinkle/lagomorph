"""
Fluid and other LDDMM metrics
"""
import torch
import lagomorph_ext
import numpy as np

class FluidMetricOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, luts, inverse, mv):
        ctx.params = params
        ctx.luts = luts
        ctx.inverse = inverse
        sh = mv.shape
        spatial_dim = len(sh)-2
        Fmv = torch.rfft(mv, spatial_dim, normalized=True)
        lagomorph_ext.fluid_operator(Fmv, inverse,
                luts['cos'], luts['sin'], *params)
        return torch.irfft(Fmv, spatial_dim, normalized=True,
                signal_sizes=sh[2:])
    @staticmethod
    def backward(ctx, outgrad):
        sh = outgrad.shape
        spatial_dim = len(sh)-2
        Fmv = torch.rfft(outgrad, spatial_dim, normalized=True)
        lagomorph_ext.fluid_operator(Fmv, ctx.inverse,
                ctx.luts['cos'], ctx.luts['sin'], *ctx.params)
        return None, None, None, torch.irfft(Fmv, spatial_dim, normalized=True,
                signal_sizes=sh[2:])


class FluidMetric(object):
    def __init__(self, params=[.1, .0, .001]):
        """
        This kernel is the Green's function for:

            L'L = - alpha nabla^2 - beta grad div + gamma

        the most commonly used kernel in LDDMM (cf. Christensen et al 1994 or
        Jacob Hinkle's PhD Thesis 2013)
        """
        self.shape = None
        self.complexshape = None
        assert len(params)==3
        self.params = params
        self.luts = None
    def initialize_luts(self, shape, dtype, device='cuda'):
        """
        Fill out lookup tables used in Fourier kernel. This amounts to just
        precomputing the sine and cosine (period N_d) for each dimension.
        """
        if self.shape != shape:
            self.shape = shape
            self.complexshape = list(shape)
            self.complexshape[-1] = self.complexshape[-1]//2+1
            self.complexshape = tuple(self.complexshape)
        if self.luts is None:
            self.luts = {'cos': [], 'sin': []}
            for (Nf,N) in zip(self.complexshape[2:], self.shape[2:]):
                self.luts['cos'].append(torch.Tensor(
                    2.*(1.-np.cos(2*np.pi*np.arange(Nf)/N))).type(dtype).to(device))
                self.luts['sin'].append(torch.Tensor(
                    np.sin(2.*np.pi*np.arange(Nf)/N)).type(dtype).to(device))
    def operator(self, mv, inverse):
        self.initialize_luts(shape=mv.shape, dtype=mv.dtype, device=mv.device)
        return FluidMetricOperator.apply(self.params, self.luts, inverse, mv)
    def sharp(self, m):
        """
        Raise indices, meaning convert a momentum (covector field) to a velocity
        (vector field) by applying the Green's function which smooths the
        momentum.
        https://en.wikipedia.org/wiki/Musical_isomorphism
        """
        return self.operator(m, inverse=True)
    def flat(self, m, out=None):
        """
        Lower indices, meaning convert a vector field to a covector field
        (a momentum) by applying the differential operator (in the Fourier
        domain).
        https://en.wikipedia.org/wiki/Musical_isomorphism
        """
        return self.operator(m, inverse=False)


class Metric:
    """Serialization and command line interface to a metric factory"""
    @staticmethod
    def add_args(parser):
        parser.add_argument("--metric_type", default="fluid", type=str, help="Type of parser. Currently only 'fluid' is supported.")
        parser.add_argument("--fluid_alpha", default=.1, type=float, help="Fluid parameter for vector Laplacian term")
        parser.add_argument("--fluid_beta", default=.0, type=float, help="Fluid parameter for gradient divergence term")
        parser.add_argument("--fluid_gamma", default=.01, type=float, help="Fluid parameter for L2 term")
    @classmethod
    def from_args(cls, args):
        if args.metric_type.lower() == 'fluid':
            return FluidMetric(params=[args.fluid_alpha, args.fluid_beta, args.fluid_gamma])
