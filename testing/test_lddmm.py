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

res = 128 # which resolutions to test
dims = [2,3] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test
TF = [True,False]
steps = [1,5]
fluid_params = [1.0,0.1,0.01]

@pytest.fixture(params=batch_sizes, ids=['bs'+str(b) for b in batch_sizes])
def bs(request): return request.param
@pytest.fixture(params=dims, ids=['dim'+str(d) for d in dims])
def dim(request): return request.param
@pytest.fixture(params=steps, ids=['step'+str(s) for s in steps])
def step(request): return request.param
@pytest.fixture(params=[fluid_params], ids=['fluid'])
def params(request): return request.param

def test_expmap_zero(bs, dim, step, params):
    defsh = tuple([bs,dim]+[res]*dim)
    m = torch.zeros(defsh, dtype=torch.float64,
            requires_grad=False).cuda()
    metric = lm.FluidMetric(params)
    h = lm.expmap(metric, m, num_steps=step)
    assert torch.allclose(m, h), "Failed expmap of zero is identity check"
