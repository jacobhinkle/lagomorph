import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
import math

import lagomorph as lm

precs = [('single',np.float32), ('double', np.float64)]

np.random.seed(1)

res = 16 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,10] # which batch sizes to test

def test_interp_zero():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                I = gpuarray.zeros(imsh, dtype=dtype)
                h = lm.to_gpu(np.random.randn(*defsh).astype(dtype))
                Ih = lm.interp_image(I, h, displacement=True)
                normsq = lm.L2(Ih, Ih)
                assert np.allclose(normsq, 0), (f"Interp zero is not zero image "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"|Ih|^2={normsq}")

def test_interp_identity():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                I = lm.to_gpu(np.random.randn(*imsh).astype(dtype))
                h = gpuarray.zeros(defsh, dtype=dtype)
                Ih = lm.interp_image(I, h, displacement=True)
                assert np.allclose(I.get(), Ih.get()), (f"Interp along identity is not original image "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: ")

def test_splat_zero():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                I = gpuarray.zeros(imsh, dtype=dtype)
                h = lm.to_gpu(np.random.randn(*defsh).astype(dtype))
                Ih, _ = lm.splat_image(I, h, displacement=True)
                normsq = lm.L2(Ih, Ih)
                assert np.allclose(normsq, 0), (f"Splat along identity is not original image "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"|Ih|^2={normsq}")

def test_splat_identity():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                I = lm.to_gpu(np.random.randn(*imsh).astype(dtype))
                h = gpuarray.zeros(defsh, dtype=dtype)
                Ih, _ = lm.splat_image(I, h, displacement=True)
                assert np.allclose(I.get(), Ih.get()), (f"Interp zero is not zero image "
                        f"[batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: ")

def test_interp_splat():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            for prec, dtype in precs:
                I = lm.to_gpu(np.random.randn(*imsh).astype(dtype))
                J = lm.to_gpu(np.random.randn(*imsh).astype(dtype))
                h = lm.to_gpu(np.random.randn(*defsh).astype(dtype))
                Ih = lm.interp_image(I, h, displacement=True)
                IhJ = lm.L2(Ih, J)
                Jh, _ = lm.splat_image(J, h, displacement=True)
                JhI = lm.L2(Jh, I)
                assert np.allclose(IhJ, JhI), (f"Splat image does not pass "
                        f"adjoint check [batch_size={bs} dim={dim} precision={prec} dtype={dtype}]: "
                        f"<I\circ h, J>={IhJ} <I, splat h J>={JhI}")
