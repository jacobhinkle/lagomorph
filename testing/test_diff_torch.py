import torch
import numpy as np
import math

import lagomorph.torch as lt

from testing.utils import catch_gradcheck

np.random.seed(1)
torch.manual_seed(1)

res = 3 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

def test_jacobian_times_vectorfield_gradcheck():
    for bs in batch_sizes:
        for dim in dims:
            for disp in [False, True]:
                for trans in [False, True]:
                    defsh = tuple([bs,dim]+[res]*dim)
                    phiinv = torch.randn(defsh, dtype=torch.float64,
                            requires_grad=True).cuda()
                    m = torch.randn_like(phiinv)
                    foo = lambda v,w: lt.jacobian_times_vectorfield(v, w,
                            displacement=disp, transpose=trans)
                    catch_gradcheck(f"Failed jacobian_times_vectorfield gradcheck with "
                    f"batch size={bs} dim={dim} displacement={disp} transpose={trans}",
                    foo,
                    (phiinv,m))

def test_jacobian_times_vectorfield_transpose():
    """Test that the transpose argument gives the adjoint"""
    for bs in batch_sizes:
        for dim in dims:
            for disp in [False, True]:
                defsh = tuple([bs,dim]+[res]*dim)
                g = torch.randn(defsh, dtype=torch.float64).cuda()
                u = torch.randn_like(g)
                v = torch.randn_like(g)
                Dgu = lt.jacobian_times_vectorfield(g, u, displacement=disp, transpose=False)
                Dguv = (Dgu*v).sum()
                DgTv = lt.jacobian_times_vectorfield(g, v, displacement=disp, transpose=True)
                uDgTv = (u*DgTv).sum()
                assert torch.allclose(Dguv, uDgTv), (f"Failed jacobian_times_vectorfield adjoint check with "
                f"batch size={bs} dim={dim} displacement={disp}")
