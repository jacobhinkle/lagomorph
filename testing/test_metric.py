import torch
import numpy as np
import math

import lagomorph as lm

from testing.utils import catch_gradcheck

np.random.seed(1)

res = 2 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

def test_sharp_gradcheck():
    fluid_params = [.1,.01,.001]
    for bs in batch_sizes:
        for dim in dims:
            defsh = tuple([bs,dim]+[res]*dim)
            m = torch.randn(defsh, dtype=torch.float64, requires_grad=True).cuda()
            metric = lm.FluidMetric(fluid_params)
            catch_gradcheck(f"Failed sharp gradcheck with batch size {bs} dim {dim}", metric.sharp, (m,))
