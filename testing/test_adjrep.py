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
dims = [2,3] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_Ad_star_gradcheck(bs, dim):
    defsh = tuple([bs,dim]+[res]*dim)
    phiinv = torch.randn(defsh, dtype=torch.float64, requires_grad=True).cuda()
    m = torch.randn_like(phiinv)
    catch_gradcheck(f"Failed Ad_star gradcheck with batch size {bs} dim {dim}", lm.Ad_star, (phiinv,m))
