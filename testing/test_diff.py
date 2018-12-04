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

res = 2 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test
TF = [True,False]

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("disp", TF)
@pytest.mark.parametrize("trans", TF)
def test_jacobian_times_vectorfield_gradcheck(bs, dim, disp, trans):
    defsh = tuple([bs,dim]+[res]*dim)
    phiinv = torch.randn(defsh, dtype=torch.float64,
            requires_grad=True).cuda()
    m = torch.randn_like(phiinv)
    m.requires_grad=True
    foo = lambda v,w: lm.jacobian_times_vectorfield(v, w,
            displacement=disp, transpose=trans)
    catch_gradcheck(f"Failed jacobian_times_vectorfield gradcheck with "
    f"batch size={bs} dim={dim} displacement={disp} transpose={trans}",
    foo,
    (phiinv,m))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("disp", TF)
def test_jacobian_times_vectorfield_transpose(bs, dim, disp):
    """Test that the transpose argument gives the adjoint of point-wise
    multiplication"""
    defsh = tuple([bs,dim]+[res]*dim)
    g = torch.randn(defsh, dtype=torch.float64).cuda()
    u = torch.randn_like(g)
    v = torch.randn_like(g)
    Dgu = lm.jacobian_times_vectorfield(g, u, displacement=disp, transpose=False)
    Dguv = (Dgu*v).sum()
    DgTv = lm.jacobian_times_vectorfield(g, v, displacement=disp, transpose=True)
    uDgTv = (u*DgTv).sum()
    assert torch.allclose(Dguv, uDgTv), (f"Failed jacobian_times_vectorfield_transpose check with "
    f"batch size={bs} dim={dim} displacement={disp}")

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_jacobian_times_vectorfield_adjoint(bs, dim):
    """Test that the adjoint jacobian times vectorfield operation is an actual
    adjoint with respect to the action on the differentiated vector field"""
    defsh = tuple([bs,dim]+[res]*dim)
    u = torch.randn(defsh, dtype=torch.float64).cuda()
    v = torch.randn_like(u)
    m = torch.randn_like(u)
    Duv = lm.jacobian_times_vectorfield(u, v, displacement=False, transpose=False)
    Duvm = (Duv*m).sum()
    adjvm = lm.jacobian_times_vectorfield_adjoint(m, v)
    uadjvm = (u*adjvm).sum()
    assert torch.allclose(Duvm, uadjvm), (f"Failed jacobian_times_vectorfield_adjoint check with "
    f"batch size={bs} dim={dim}")

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_jacobian_times_vectorfield_adjoint_gradcheck(bs, dim):
    defsh = tuple([bs,dim]+[res]*dim)
    v = torch.randn(defsh, dtype=torch.float64, requires_grad=True).cuda()
    m = torch.randn_like(v)
    m.requires_grad=True
    catch_gradcheck(f"Failed jacobian_times_vectorfield_adjoint gradcheck with "
        f"batch size={bs} dim={dim}",
        lm.jacobian_times_vectorfield_adjoint,
        (v,m))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("disp", TF)
@pytest.mark.parametrize("trans", TF)
def test_jacobian_times_vectorfield_2d_match_3d(bs, disp, trans):
    defsh2 = tuple([bs,2]+[res]*2)
    defsh3 = tuple([bs,3]+[res]*2+[1])
    v2 = torch.randn(defsh2, dtype=torch.float64, requires_grad=False).cuda()
    v3 = torch.zeros(defsh3, dtype=torch.float64, requires_grad=False).cuda()
    v3[:,:2,:,:,0] = v2
    m2 = torch.randn_like(v2)
    m3 = torch.zeros_like(v3)
    m3[:,:2,:,:,0] = m2
    dvm2 = lm.jacobian_times_vectorfield(v2, m2, displacement=disp, transpose=trans)
    dvm3 = lm.jacobian_times_vectorfield(v3, m3, displacement=disp, transpose=trans)
    print(dvm2)
    print(dvm3)
    assert torch.allclose(dvm3[:,:2,:,:,0], dvm2), \
            ("Failed jacobian_times_vectorfield 2D match 3D check with "
            f"batch size={bs} disp={disp} trans={trans}")
