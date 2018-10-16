"""
Fluid and other LDDMM metrics
"""
import torch
import lagomorph_torch_cuda
import numpy as np

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
        self.alpha = float(params[0])
        self.beta = float(params[1])
        self.gamma = float(params[2])
        self.luts = None
    def initialize_luts(self, shape, device='cuda'):
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
                    np.require(2.*(1.-np.cos(2*np.pi*np.arange(Nf,
                        dtype=np.float64)/N)),
                        requirements='C')).to(device))
                self.luts['sin'].append(torch.Tensor(
                    np.require(np.sin(2.*np.pi*np.arange(Nf, dtype=np.float64)/N),
                        requirements='C')).to(device))
    def operator(self, mv, inverse):
        # call the appropriate cuda kernel here
        self.initialize_luts(shape=mv.shape, device=mv.device)
        spatial_dim = mv.dim()-2
        Fmv = torch.rfft(mv, spatial_dim, normalized=True)
        lagomorph_torch_cuda.fluid_operator(Fmv, inverse,
                self.luts['cos'], self.luts['sin'],
                self.alpha, self.beta, self.gamma)
        return torch.irfft(Fmv, spatial_dim, normalized=True)
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
