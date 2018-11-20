import torch
import numpy as np
import math

import lagomorph as lm

# This enables cuda error checking in lagomorph which causes kernels to
# synchronize, which is why it's disabled by default
lm.set_debug_mode(True)

from testing.utils import catch_gradcheck

np.random.seed(1)

res = 3 # which resolutions to test
dims = [2] # which dimensions to test
batch_sizes = [1,2] # which batch sizes to test

def test_interp_gradcheck_I():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs,1]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
            u = torch.randn(defsh, dtype=I.dtype, requires_grad=False).to(I.device)
            foo = lambda Ix: lm.interp(Ix, u)
            catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim}", foo, (I,))
def test_interp_gradcheck_u():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs,1]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            I = torch.randn(imsh, dtype=torch.float64, requires_grad=False).cuda()
            u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
            foo = lambda ux: lm.interp(I, ux)
            catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim}", foo, (u,))
def test_interp_gradcheck_both():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([bs,1]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
            u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
            catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim}", lm.interp, (I,u))

def test_interp_multichannel_gradcheck():
    for nc in [2,3]:
        for bs in batch_sizes:
            for dim in dims:
                imsh = tuple([bs,nc]+[res]*dim)
                defsh = tuple([bs,dim]+[res]*dim)
                I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
                u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
                catch_gradcheck(f"Failed interp gradcheck with batch size {bs} dim {dim} num channels={nc}", lm.interp, (I,u))

def test_interp_broadcastI_gradcheck_u():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([1,1]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            I = torch.randn(imsh, dtype=torch.float64, requires_grad=False).cuda()
            u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
            foo = lambda ux: lm.interp(I, ux)
            catch_gradcheck(f"Failed broadcastI interp gradcheck (u only) with batch size {bs} dim {dim}", foo, (u,))
def test_interp_broadcastI_gradcheck_I():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([1,1]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
            u = torch.randn(defsh, dtype=I.dtype, requires_grad=False).to(I.device)
            foo = lambda Ix: lm.interp(Ix, u)
            Iu = lm.interp(I,u)
            catch_gradcheck(f"Failed broadcastI interp gradcheck (I only) with batch size {bs} dim {dim} Iu={Iu}", foo, (I,))
            Iu = lm.interp(I,u)
def test_interp_broadcastI_gradcheck_both():
    for bs in batch_sizes:
        for dim in dims:
            imsh = tuple([1,1]+[res]*dim)
            defsh = tuple([bs,dim]+[res]*dim)
            I = torch.randn(imsh, dtype=torch.float64, requires_grad=True).cuda()
            u = torch.randn(defsh, dtype=I.dtype, requires_grad=True).to(I.device)
            catch_gradcheck(f"Failed broadcastI interp gradcheck (both) with batch size {bs} dim {dim}", lm.interp, (I,u))
