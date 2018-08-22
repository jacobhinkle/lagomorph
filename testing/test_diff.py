from pycuda import gpuarray
import pycuda.autoinit
from pycuda.tools import mark_cuda_test
import numpy as np
import math

import lagomorph as lm

precs = [('single',np.float32), ('double', np.float64)]

np.random.seed(1)

res = 16 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,10] # which batch sizes to test

def test_grad_zero():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs]+[res]*dim)
            for prec, dtype in precs:
                I = gpuarray.zeros(imsh, dtype=dtype, order='C')
                g = lm.gradient(I)
                normsq = lm.L2(g, g)
                assert np.allclose(normsq, 0), (f"Gradient of zero is not zero vector field "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"|grad I|^2={normsq}")

def test_grad_pair():
    for dim in dims:
        imsh = tuple([1]+[res]*dim)
        for prec, dtype in precs:
            tiles = tuple([2]+[1]*dim)
            Ihost = np.tile(np.random.randn(*imsh), tiles)
            I = gpuarray.to_gpu(Ihost.astype(dtype))
            g = lm.gradient(I)
            gh = g.get()
            assert np.allclose(gh[0,...], gh[1,...]), (f"Gradients of identical images differ"
                    f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: ")

def test_divergence_zero():
    for bs in batch_sizes:
        for dim in dims:
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                v = gpuarray.zeros(defsh, dtype=dtype, order='C')
                dv = lm.divergence(v)
                normsq = lm.L2(dv, dv)
                assert np.allclose(normsq, 0), (f"Divergence of zero is not zero image "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"|div v|^2={normsq}")

def test_divergence_pair():
    for dim in dims:
        defsh = tuple([1,dim]+[res]*dim)
        for prec, dtype in precs:
            tiles = tuple([2,1]+[1]*dim)
            vhost = np.tile(np.random.randn(*defsh), tiles)
            v = gpuarray.to_gpu(vhost.astype(dtype))
            dv = lm.divergence(v)
            dvh = dv.get()
            assert np.allclose(dvh[0,...], dvh[1,...]), (f"Divergences of identical fields differ"
                    f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: ")
