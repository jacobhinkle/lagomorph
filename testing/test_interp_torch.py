import torch
import numpy as np
import math

import lagomorph.torch as lt

from testing.utils import catch_gradcheck

np.random.seed(1)

res = 2 # which resolutions to test
dims = [5] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

def test_interp_gradcheck():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs,1]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
            u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
            catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim}", lt.interp, (I,u))
