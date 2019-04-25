import torch
import numpy as np
import math

import lagomorph as lm

# This enables cuda error checking in lagomorph which causes kernels to
# synchronize, which is why it's disabled by default
lm.set_debug_mode(True)

import pytest

from testing.utils import catch_gradcheck

np.random.seed(1)
torch.manual_seed(1)

res = 2 # which resolution to test
dims = [2,3] # which dimensions to test
channels = [1,2,4] # numbers of channels to test
batch_sizes = [1,2] # which batch sizes to test
TF = [True,False]
devices = ['cuda','cpu']

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("c", channels)
@pytest.mark.parametrize("d", devices)
def test_affine_interp_identity(d, bs, dim, c):
    imsh = tuple([bs,c]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=False).to(d)
    A = torch.zeros((bs,dim,dim), dtype=I.dtype, requires_grad=False).to(I.device)
    T = torch.zeros((bs,dim), dtype=I.dtype, requires_grad=False).to(I.device)
    for i in range(dim):
        A[:,i,i] = 1
    IAT = lm.affine_interp(I, A, T)
    assert torch.allclose(IAT, I), \
            f"Affine interp by identity is non-trivial with batch size {bs} dim {dim} channels {c}"

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("c", channels)
@pytest.mark.parametrize("testI", TF)
@pytest.mark.parametrize("testA", TF)
@pytest.mark.parametrize("testT", TF)
def test_affine_interp_gradcheck(bs, dim, c, testI, testA, testT):
    if not (testI or testA or testT): return # nothing to test
    imsh = tuple([bs,c]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=testI).cuda()
    A = torch.randn((bs,dim,dim), dtype=I.dtype, requires_grad=testA).to(I.device)
    T = torch.randn((bs,dim), dtype=I.dtype, requires_grad=testT).to(I.device)
    catch_gradcheck(f"Failed affine interp gradcheck with batch size {bs} dim {dim} channels {c}", lm.affine_interp, (I,A,T))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("c", channels)
def test_affine_interp_gpucpu_match(bs, dim, c):
    imsh = tuple([bs,c]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64).cuda()
    A = torch.randn((bs,dim,dim), dtype=I.dtype).to(I.device)
    T = torch.randn((bs,dim), dtype=I.dtype).to(I.device)
    Icpu = I.to('cpu')
    Acpu = A.to('cpu')
    Tcpu = T.to('cpu')
    Jcuda = lm.affine_interp(I, A, T)
    Jcpu = lm.affine_interp(Icpu, Acpu, Tcpu)
    assert torch.allclose(Jcuda.cpu(), Jcpu), \
            f"Affine interp GPU-CPU mismatch with batch size {bs} dim {dim} channels {c}"

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("c", channels)
@pytest.mark.parametrize("d", devices)
def test_affine_2d_match_3d(d, bs, c):
    """Test that 2D matches 3D affine interpolation"""
    with torch.no_grad():
        imsh = (bs,c,res,res)
        I2 = torch.randn(imsh, dtype=torch.float64).to(d)
        A2 = torch.randn((bs,2,2), dtype=torch.float64, device=I2.device)
        T2 = torch.randn((bs,2), dtype=A2.dtype, device=I2.device)
        I3 = I2.view(bs,c,res,res,1)
        A3 = torch.cat((
            A2[:,0,0].unsqueeze(1),
            A2[:,0,1].unsqueeze(1),
            torch.zeros((bs,1), dtype=T2.dtype, device=I2.device),
            A2[:,1,0].unsqueeze(1),
            A2[:,1,1].unsqueeze(1),
            torch.zeros((bs,3), dtype=T2.dtype, device=I2.device),
            torch.ones((bs,1), dtype=T2.dtype, device=I2.device),
            ),
                dim=1).view(bs,3,3)
        T3 = torch.cat((T2, torch.zeros((bs,1), dtype=T2.dtype,
            device=I2.device)), dim=1)
        J2 = lm.affine_interp(I2, A2, T2).view(bs,c,res,res,1)
        J3 = lm.affine_interp(I3, A3, T3)
        assert torch.allclose(J2, J3), f"Failed affine 2D==3D check with batch size {bs} channels {c}"

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_affine_inverse(bs, dim):
    A = torch.randn((bs,dim,dim), dtype=torch.float64)
    T = torch.randn((bs,dim), dtype=A.dtype)
    x = torch.randn((bs,dim,1), dtype=T.dtype)
    Ainv, Tinv = lm.affine_inverse(A, T)
    y = torch.matmul(A, x) + T.unsqueeze(2)
    xhat = torch.matmul(Ainv, y) + Tinv.unsqueeze(2)
    assert torch.allclose(x, xhat), f"Failed affine inverse with batch size {bs} dim {dim}"

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("disp", TF)
def test_regrid_identity(bs, dim, disp):
    imsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    outshape = imsh[2:]
    Ir = lm.regrid(I, shape=outshape, displacement=disp)
    assert torch.allclose(I, Ir), "Failed regrid identity check"

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("c", channels)
def test_regrid_gradcheck(bs, dim, c):
    imsh = tuple([bs,c]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    outshape = [res+1]*dim
    foo = lambda J: lm.regrid(J, shape=outshape, displacement=False)
    catch_gradcheck("Failed regrid gradcheck", foo, (I,))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_regrid_displacement_gradcheck(bs, dim):
    imsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    outshape = [res+1]*dim
    foo = lambda J: lm.regrid(J, shape=outshape, displacement=True)
    catch_gradcheck("Failed regrid displacement gradcheck", foo, (I,))
