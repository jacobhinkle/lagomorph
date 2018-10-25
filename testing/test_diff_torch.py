import torch
import numpy as np
import math

import lagomorph.torch as lt

from testing.utils import catch_gradcheck

np.random.seed(1)
torch.manual_seed(1)

res = 2 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

def test_jacobian_times_vectorfield_gradcheck():
    for bs in batch_sizes:
        for dim in dims:
            defsh = tuple([bs,dim]+[res]*dim)
            phiinv = torch.randn(defsh, dtype=torch.float64, requires_grad=True).cuda()
            m = torch.randn_like(phiinv)
            catch_gradcheck(f"Failed jacobian_times_vectorfield gradcheck with batch size {bs} dim {dim}", lt.jacobian_times_vectorfield, (phiinv,m))
