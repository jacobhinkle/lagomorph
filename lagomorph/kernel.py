"""
Fluid and other LDDMM kernels
"""
import .kernel_cuda as kc
import skcuda.fft
from pycuda import gpuarray, cumath
import numpy as np

class FluidKernel(object):
    def __init__(self, shape, alpha, beta, gamma, precision='single'):
        """
        This kernel is the Green's function for:

            L'L = - alpha nabla^2 - beta grad div + gamma

        the most commonly used kernel in LDDMM (cf. Christensen et al 1994 or
        Jacob Hinkle's PhD Thesis 2013)

        The shape argument is the size of a velocity or momentum, in NCWHD order
        """
        self.shape = shape
        self.dim = len(shape)-2
        if self.dim != 2 and self.dim != 3:
            raise Exception(f"Invalid dimension: {self.dim}")
        self.precision = precision
        # initialize memory for out of place fft
        self.Fv = None
        # initialize lookup tables and send them up to the gpu
        self.initialize_luts()
        # fft plan
        self.fftplan = skcuda.fft.Plan(shape[2:], float, float, batch=shape[0])
    def initialize_luts(self):
        """
        Fill out lookup tables used in Fourier kernel. This amounts to just
        precomputing the sine and cosine (period N_d) for each dimension.
        """
        self.luts = {'cos': [], 'sin': []}
        for N in self.shape[2:]:
            self.luts['cos'].append(cumath.cos(2.*np.pi*gpuarray.arange(N)/N))
            self.luts['sin'].append(cumath.sin(2.*np.pi*gpuarray.arange(N)/N))
    def inverse_operator_fourier(self, Fm):
        # call the appropriate cuda kernel here
        if self.dim == 2:
            kc.inv2(Fm, self.luts, precision=self.precision)
        elif self.dim == 3:
            kc.inv3(Fm, self.luts, precision=self.precision)
    def inverse(self, m):
        skcuda.fft.fft(m, self.Fv, self.fftplan)
        self.inverse_operator_fourier(self.Fv)
        skcuda.fft.ifft(self.Fv, m, self.fftplan)
    def forward(self, m):
        skcuda.fft.fft(m, self.Fv, self.fftplan)
        self.forward_operator_fourier(self.Fv)
        skcuda.fft.ifft(self.Fv, m, self.fftplan)
