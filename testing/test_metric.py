from pycuda import gpuarray
import pycuda.autoinit
import skcuda.fft
from pycuda.tools import mark_cuda_test
import numpy as np
import math

import lagomorph as lm

precs = [('single',np.float32)]#, ('double', np.float64)]

np.random.seed(1)

res = 16 # which resolutions to test
dims = [2] # which dimensions to test
alpha = 0.1
beta = 0.001
gamma = 0.01
batch_sizes = [1,10] # which batch sizes to test

def test_fft_random():
    for bs in batch_sizes:
        for dim in dims:
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                x = gpuarray.to_gpu(np.asarray(np.random.rand(*defsh), dtype=dtype))
                if prec == 'single': complexdtype=np.complex64
                else: complexdtype=np.complex128
                complexsh = list(defsh)
                complexsh[-1] = complexsh[-1]//2+1
                complexsh = tuple(complexsh)
                fwdplan = skcuda.fft.Plan(defsh[2:], dtype, complexdtype, dim*bs)
                xf = gpuarray.empty(complexsh, dtype=complexdtype)
                skcuda.fft.fft(x, xf, fwdplan)
                y = gpuarray.empty_like(x)
                revplan = skcuda.fft.Plan(defsh[2:], complexdtype, dtype, dim*bs)
                skcuda.fft.ifft(xf, y, revplan, scale=True)
                xh = x.get()
                yh = y.get()
                assert np.allclose(xh, yh, atol=1e-6), (f"ifft(fft(x) != x "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"|xh-yh|^2={((xh-yh)**2).sum()}")

def test_sharp_zero():
    for bs in batch_sizes:
        for dim in dims:
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                metric = lm.FluidMetric(alpha=alpha, beta=beta, gamma=gamma, shape=defsh)
                m = gpuarray.zeros(defsh, dtype=dtype)
                v = metric.sharp(m)
                vh = v.get()
                assert np.allclose(vh, 0, atol=0.001), (f"Sharp zero is not zero image "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"|vh|^2={(vh**2).sum()}")

def test_flat_zero():
    for bs in batch_sizes:
        for dim in dims:
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                metric = lm.FluidMetric(alpha=alpha, beta=beta, gamma=gamma, shape=defsh)
                m = gpuarray.zeros(defsh, dtype=dtype, order='C')
                v = metric.flat(m)
                vh = v.get()
                assert np.allclose(vh, 0, atol=0.001), (f"Flat zero is not zero image "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"|vh|^2={(vh**2).sum()}")

def test_sharp_flat_inverse():
    for bs in batch_sizes:
        for dim in dims:
            defsh = tuple([bs,dim]+[res]*dim)
            mh0 = np.asarray(np.random.rand(*defsh), dtype=np.float64)
            for prec, dtype in precs:
                params = f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]"
                mh = mh0.astype(dtype)
                assert np.isinf(mh).sum() == 0, f"m contains infs {params}"
                assert np.isnan(mh).sum() == 0, f"m contains nans {params}"
                #mh = np.asarray(np.zeros(defsh), dtype=dtype)
                m = gpuarray.to_gpu(mh)
                metric = lm.FluidMetric(alpha=alpha, beta=beta, gamma=gamma, shape=defsh)
                v = metric.sharp(m)
                vh = v.get()
                assert np.isinf(vh).sum() == 0, f"v contains infs {params}"
                assert np.isnan(vh).sum() == 0, f"v contains nans {params}"
                vflat = metric.flat(v)
                vflath = vflat.get()
                assert np.isinf(vflath).sum() == 0, f"vflat contains infs {params}"
                assert np.isnan(vflath).sum() == 0, f"vflat contains nans {params}"
                assert np.allclose(mh, vflath, atol=1e-3), (f"flat(sharp(m)) != m {params}"
                        f"|vflath-mh|^2={((vflath-mh)**2).sum()}")
