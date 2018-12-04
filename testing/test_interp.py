import torch
import numpy as np
import math

import lagomorph as lm
import pytest

# This enables cuda error checking in lagomorph which causes kernels to
# synchronize, which is why it's disabled by default
lm.set_debug_mode(True)

from testing.utils import catch_gradcheck

np.random.seed(1)
torch.manual_seed(1)

res = 3 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_interp_gradcheck_I(bs, dim):
    imsh = tuple([bs,1]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=False).to(I.device)
    foo = lambda Ix: lm.interp(Ix, u)
    catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim}", foo, (I,))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_interp_gradcheck_u(bs, dim):
    imsh = tuple([bs,1]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=False).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
    foo = lambda ux: lm.interp(I, ux)
    catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim}", foo, (u,))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_interp_gradcheck_both(bs, dim):
    imsh = tuple([bs,1]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
    catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim}", lm.interp, (I,u))

@pytest.mark.parametrize("nc", [2,4])
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_interp_multichannel_gradcheck(nc, bs, dim):
    imsh = tuple([bs,nc]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
    catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim} num channels={nc}", lm.interp, (I,u))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_interp_broadcastI_gradcheck_u(bs, dim):
    imsh = tuple([1,1]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=False).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
    foo = lambda ux: lm.interp(I, ux)
    catch_gradcheck(f"Failed broadcastI interp gradcheck (u only) with batch size {bs} dim {dim}", foo, (u,))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_interp_broadcastI_gradcheck_I(bs, dim):
    imsh = tuple([1,1]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=False).to(I.device)
    foo = lambda Ix: lm.interp(Ix, u)
    Iu = lm.interp(I,u)
    catch_gradcheck(f"Failed broadcastI interp gradcheck (I only) with batch size {bs} dim {dim} Iu={Iu}", foo, (I,))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_interp_broadcastI_gradcheck_both(bs, dim):
    imsh = tuple([1,1]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
    catch_gradcheck(f"Failed broadcastI interp gradcheck (both) with batch size {bs} dim {dim}", lm.interp, (I,u))

@pytest.mark.parametrize("bs", batch_sizes)
def test_interp_2d_match_3d(bs):
    imsh = tuple([bs,1]+[res]*2)
    defsh = tuple([bs,2]+[res]*2)
    defsh3 = tuple([bs,3]+[res]*2+[1])
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=False).to(I.device)
    I3 = I.unsqueeze(4)
    u3 = torch.zeros(defsh3, dtype=u.dtype, device=u.device)
    u3[:,:2,...] = u.unsqueeze(4)
    Iu = lm.interp(I, u)
    Iu3 = lm.interp(I3, u3)
    assert torch.allclose(Iu.unsqueeze(4), Iu3), \
            f"Failed interp 2d match 3d with batch size {bs}"
