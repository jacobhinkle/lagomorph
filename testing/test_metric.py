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

res = 3 # which resolutions to test
dims = [2,3] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_fluid_sharp_gradcheck(bs, dim):
    fluid_params = [.1,.01,.001]
    defsh = tuple([bs,dim]+[res]*dim)
    m = torch.randn(defsh, dtype=torch.float64, requires_grad=True).cuda()
    metric = lm.FluidMetric(fluid_params)
    catch_gradcheck(f"Failed fluid sharp gradcheck with batch size {bs} dim {dim}", metric.sharp, (m,), 1e-4)

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_fluid_flat_gradcheck(bs, dim):
    fluid_params = [.1,.01,.001]
    defsh = tuple([bs,dim]+[res]*dim)
    v = torch.randn(defsh, dtype=torch.float64, requires_grad=True).cuda()
    metric = lm.FluidMetric(fluid_params)
    catch_gradcheck(f"Failed fluid flat gradcheck with batch size {bs} dim {dim}", metric.flat, (v,))

@pytest.mark.parametrize("bs", batch_sizes)
@pytest.mark.parametrize("dim", dims)
def test_fluid_inverse(bs, dim):
    fluid_params = [.1,.01,.001]
    defsh = tuple([bs,dim]+[res]*dim)
    m = torch.randn(defsh, dtype=torch.float64, requires_grad=False).cuda()
    metric = lm.FluidMetric(fluid_params)
    v = metric.sharp(m)
    vm = metric.flat(v)
    assert torch.allclose(vm, m, atol=1e-3), f"Failed fluid inverse check with batch size {bs} dim {dim}"
