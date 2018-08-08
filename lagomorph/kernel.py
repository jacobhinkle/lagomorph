"""
Fluid and other LDDMM kernels
"""
from . import kernel_cuda as kc
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
        if precision == 'single':
            self.dtype = np.float32
            self.complex_dtype = np.complex64
        elif precision == 'double':
            self.dtype = np.float64
            self.complex_dtype = np.complex128
        else:
            raise Exception(f"Unsupported precision: {precision}")
        # initialize memory for out of place fft
        self.Fv = gpuarray.empty(shape=shape, dtype=self.complex_dtype, order='C')
        # initialize lookup tables and send them up to the gpu
        self.initialize_luts()
        # fft plan
        self.fftplan = skcuda.fft.Plan(shape[2:], self.dtype, self.complex_dtype, batch=shape[0])
        self.ifftplan = skcuda.fft.Plan(shape[2:], self.complex_dtype, self.dtype, batch=shape[0])
    def initialize_luts(self):
        """
        Fill out lookup tables used in Fourier kernel. This amounts to just
        precomputing the sine and cosine (period N_d) for each dimension.
        """
        self.luts = {'cos': [], 'sin': []}
        for N in self.shape[2:]:
            self.luts['cos'].append(cumath.cos(2.*np.pi*gpuarray.arange(N,
                dtype=self.dtype)/N))
            self.luts['sin'].append(cumath.sin(2.*np.pi*gpuarray.arange(N,
                dtype=self.dtype)/N))
    def inverse_operator_fourier(self, Fm):
        # call the appropriate cuda kernel here
        if self.dim == 2:
            kc.inv2(Fm, self.luts, precision=self.precision)
        elif self.dim == 3:
            kc.inv3(Fm, self.luts, precision=self.precision)
    def inverse(self, m, out=None):
        if out is None:
            out = gpuarray.zeros_like(m)
        skcuda.fft.fft(m, self.Fv, self.fftplan)
        #self.inverse_operator_fourier(self.Fv)
        print("WARNING: using dummy inverse kernel")
        skcuda.fft.ifft(self.Fv, out, self.ifftplan)
        return out
    def forward(self, m, out=None):
        if out is None:
            out = gpuarray.zeros_like(m)
        skcuda.fft.fft(m, self.Fv, self.fftplan)
        self.forward_operator_fourier(self.Fv)
        skcuda.fft.ifft(self.Fv, out, self.ifftplan)
        return out
