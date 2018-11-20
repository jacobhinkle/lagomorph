import torch
import numpy as np
import math

import lagomorph as lm

# This enables cuda error checking in lagomorph which causes kernels to
# synchronize, which is why it's disabled by default
lm.set_debug_mode(True)

from testing.utils import catch_gradcheck

np.random.seed(1)

res = 3 # which resolutions to test
dims = [2] # which dimensions to test
channels = [1,2,4] # numbers of channels to test
batch_sizes = [1,2] # which batch sizes to test

def test_affine_interp_gradcheck_I():
    for bs in batch_sizes:
        for dim in dims:
            for c in channels:
                imsh = tuple([bs,c]+[res]*dim)
                I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
                A = torch.randn((bs,dim,dim), dtype=I.dtype, requires_grad=False).to(I.device)
                T = torch.randn((bs,dim), dtype=I.dtype, requires_grad=False).to(I.device)
                foo = lambda Ix: lm.affine_interp(Ix, A, T)
                catch_gradcheck(f"Failed affine interp gradcheck with batch size {bs} dim {dim} channels {c}", foo, (I,))
def test_affine_interp_gradcheck_A():
    for bs in batch_sizes:
        for dim in dims:
            for c in channels:
                imsh = tuple([bs,c]+[res]*dim)
                I = torch.randn(imsh, dtype=torch.float64, requires_grad=False).cuda()
                A = torch.randn((bs,dim,dim), dtype=I.dtype, requires_grad=True).to(I.device)
                T = torch.randn((bs,dim), dtype=I.dtype, requires_grad=False).to(I.device)
                foo = lambda Ax: lm.affine_interp(I, Ax, T)
                catch_gradcheck(f"Failed affine interp gradcheck with batch size {bs} dim {dim} channels {c}", foo, (A,))
def test_affine_interp_gradcheck_T():
    for bs in batch_sizes:
        for dim in dims:
            for c in channels:
                imsh = tuple([bs,c]+[res]*dim)
                I = torch.randn(imsh, dtype=torch.float64, requires_grad=False).cuda()
                A = torch.randn((bs,dim,dim), dtype=I.dtype, requires_grad=False).to(I.device)
                T = torch.randn((bs,dim), dtype=I.dtype, requires_grad=True).to(I.device)
                foo = lambda Tx: lm.affine_interp(I, A, Tx)
                catch_gradcheck(f"Failed affine interp gradcheck with batch size {bs} dim {dim} channels {c}", foo, (T,))
def test_affine_interp_gradcheck_all():
    for bs in batch_sizes:
        for dim in dims:
            for c in channels:
                imsh = tuple([bs,c]+[res]*dim)
                I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
                A = torch.randn((bs,dim,dim), dtype=I.dtype, requires_grad=True).to(I.device)
                T = torch.randn((bs,dim), dtype=I.dtype, requires_grad=True).to(I.device)
                catch_gradcheck(f"Failed affine interp gradcheck with batch size {bs} dim {dim} channels {c}", lm.affine_interp, (I,A,T))

def test_affine_inverse():
    for bs in batch_sizes:
        for dim in dims:
            A = torch.randn((bs,dim,dim), dtype=torch.float64)
            T = torch.randn((bs,dim), dtype=A.dtype)
            x = torch.randn((bs,dim,1), dtype=T.dtype)
            Ainv, Tinv = lm.affine_inverse(A, T)
            y = torch.matmul(A, x) + T.unsqueeze(2)
            xhat = torch.matmul(Ainv, y) + Tinv.unsqueeze(2)
            assert torch.allclose(x, xhat), f"Failed affine inverse with batch size {bs} dim {dim}"
