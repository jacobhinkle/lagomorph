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

res = 2 # which resolutions to test
dims = [2,3] # which dimensions to test
channels = [1,2,4]
batch_sizes = [1,2] # which batch sizes to test
TF = [True,False]

@pytest.mark.parametrize("nc", channels)
@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
@pytest.mark.parametrize("testI", TF)
@pytest.mark.parametrize("testu", TF)
@pytest.mark.parametrize("broadcastI", TF)
def test_interp_gradcheck(bs, nc, dim, testI, testu, broadcastI):
    if not (testI or testu): return # nothing to test
    if broadcastI:
        imsh = tuple([1,nc]+[res]*dim)
    else:
        imsh = tuple([bs,nc]+[res]*dim)
    defsh = tuple([bs,dim]+[res]*dim)
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=testI).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=testu).to(I.device)
    catch_gradcheck("Failed interp gradcheck", lm.interp, (I,u))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("nc", channels)
@pytest.mark.parametrize("broadcastI", TF)
def test_interp_2d_match_3d(bs, nc, broadcastI):
    if broadcastI:
        imsh = tuple([1,nc]+[res]*2)
    else:
        imsh = tuple([bs,nc]+[res]*2)
    defsh = tuple([bs,2]+[res]*2)
    defsh3 = tuple([bs,3]+[res]*2+[1])
    I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
    u = torch.randn(defsh, dtype=I.dtype, requires_grad=False).to(I.device)
    I3 = I.unsqueeze(4)
    u3 = torch.zeros(defsh3, dtype=u.dtype, device=u.device)
    u3[:,:2,...] = u.unsqueeze(4)
    Iu = lm.interp(I, u)
    Iu3 = lm.interp(I3, u3)
    assert torch.allclose(Iu.unsqueeze(4), Iu3), "Failed interp 2d match 3d"
