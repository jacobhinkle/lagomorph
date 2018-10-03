from pycuda import gpuarray
import pycuda.driver as drv
import numpy as np

import math

from .arithmetic import multiply_add, sum_along_axis, clip_below, L2
from .deform import imshape2defshape, defshape2imshape
from .dtypes import dtype2precision
from . import affine_cuda

def affine_gradient(I, J, A, T, B=None, C=None, outdA=None, outdT=None):
    if outdA is None:
        outdA = gpuarray.empty(shape=A.shape,
                allocator=A.allocator, dtype=A.dtype, order='C')
    if outdT is None:
        outdT = gpuarray.empty(shape=T.shape,
                allocator=T.allocator, dtype=T.dtype, order='C')
    use_contrast = B is not None
    if use_contrast:
        assert C is not None
        assert B.shape == C.shape
        assert B.ndim == 1
    assert I.shape[1:] == J.shape[1:]
    assert A.shape == outdA.shape
    assert T.shape == outdT.shape
    assert A.shape[0] == T.shape[0]
    prec = dtype2precision(I.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (16,32,1)
        nn = max(I.shape[0], J.shape[0])
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        loss = gpuarray.zeros(shape=(1,), allocator=I.allocator, dtype=I.dtype, order='C')
        if use_contrast:
            if I.shape[0] == J.shape[0]:
                f = affine_cuda.affine_grad_contrast_2d
            elif I.shape[0] == 1 and J.shape[0] > 1:
                f = affine_cuda.affine_grad_bcastI_contrast_2d
            else:
                raise NotImplementedError("Only base image broadcasting is supported")
            f(outdA, outdT, loss,
                I, J, A, T, B, C,
                np.int32(nn),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
        else:
            if I.shape[0] == J.shape[0]:
                f = affine_cuda.affine_grad_2d
            elif I.shape[0] == 1 and J.shape[0] > 1:
                f = affine_cuda.affine_grad_bcastI_2d
            else:
                raise NotImplementedError("Only base image broadcasting is supported")
            f(outdA, outdT, loss,
                I, J, A, T,
                np.int32(nn),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
        loss = 0.5*loss.get()[0]
    return outdA, outdT, loss

def affine_gradient_hessian(I, J, A, T, B=None, C=None, outdA=None, outdT=None, outH=None):
    """
    Compute the loss, gradient, and Hessian of the L2 loss function for affine
    image matching (with or without image broadcasting).

    The gradients with respect to A and T are provided separately, and the
    Hessian is provided as an n-by-21 matrix in row-contiguous order with A
    derivatives before T derivatives.

    **NOTE** that we do not compute derivatives with the contrast parameters B
    and C.
    """
    if outdA is None:
        outdA = gpuarray.empty(shape=A.shape,
                allocator=A.allocator, dtype=A.dtype, order='C')
    if outdT is None:
        outdT = gpuarray.empty(shape=T.shape,
                allocator=T.allocator, dtype=T.dtype, order='C')
    if outH is None:
        outH = gpuarray.empty(shape=(A.shape[0], 21),
                allocator=T.allocator, dtype=A.dtype, order='C')
    use_contrast = B is not None
    if use_contrast:
        assert C is not None
        assert B.shape == C.shape
        assert B.ndim == 1
    assert I.shape[1:] == J.shape[1:]
    assert A.shape == outdA.shape
    assert T.shape == outdT.shape
    assert A.shape[0] == T.shape[0]
    assert A.shape[0] == outH.shape[0]
    prec = dtype2precision(I.dtype)
    dim = I.ndim - 1
    outdA.fill(0)
    outdT.fill(0)
    outH.fill(0)
    if dim == 2:
        block = (12,32,1)
        nn = max(I.shape[0], J.shape[0])
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        loss = gpuarray.zeros(shape=(1,), allocator=I.allocator, dtype=I.dtype, order='C')
        if use_contrast:
            if I.shape[0] == J.shape[0]:
                f = affine_cuda.affine_grad_hessian_contrast_2d
            elif I.shape[0] == 1 and J.shape[0] > 1:
                f = affine_cuda.affine_grad_hessian_bcastI_contrast_2d
            else:
                raise NotImplementedError("Only base image broadcasting is supported")
            f(outdA, outdT, outH, loss,
                I, J, A, T, B, C,
                np.int32(nn),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
        else:
            if I.shape[0] == J.shape[0]:
                f = affine_cuda.affine_grad_hessian_2d
            elif I.shape[0] == 1 and J.shape[0] > 1:
                f = affine_cuda.affine_grad_hessian_bcastI_2d
            else:
                raise NotImplementedError("Only base image broadcasting is supported")
            f(outdA, outdT, outH, loss,
                I, J, A, T,
                np.int32(nn),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
        loss = 0.5*loss.get()[0]
    return outdA, outdT, outH, loss

def interp_image_affine(I, A, T, out=None, B=None, C=None):
    nn = max(I.shape[0], A.shape[0])
    use_contrast = B is not None
    if use_contrast:
        assert C is not None
        assert A.shape[0] == B.shape[0]
        assert B.shape == C.shape
        assert B.ndim == 1
    if out is None:
        out = gpuarray.empty(shape=[nn]+list(I.shape[1:]),
                allocator=I.allocator, dtype=I.dtype, order='C')
    assert A.shape[0] == T.shape[0]
    prec = dtype2precision(I.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        if use_contrast:
            if I.shape[0] == A.shape[0]:
                f = affine_cuda.interp_image_affine_contrast_2d
            elif I.shape[0] == 1:
                f = affine_cuda.interp_image_affine_bcastI_contrast_2d
            f(out,
                I, A, T, B, C,
                np.int32(nn),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
        else:
            if I.shape[0] == A.shape[0]:
                f = affine_cuda.interp_image_affine_2d
            elif I.shape[0]:
                f = affine_cuda.interp_image_affine_bcastI_2d
            f(out,
                I, A, T,
                np.int32(nn),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
    return out

def splat_image_affine(I, A, T, out=None, outw=None, compute_weights=True):
    if out is None:
        out = gpuarray.empty(shape=I.shape,
                allocator=I.allocator, dtype=I.dtype, order='C')
    if compute_weights:
        if outw is None:
            outw = gpuarray.empty(shape=I.shape,
                allocator=I.allocator, dtype=I.dtype, order='C')
        assert out.shape == outw.shape
        assert out.dtype == outw.dtype
    assert I.shape == out.shape
    # no broadcasting yet
    assert I.shape[0] == A.shape[0]
    assert A.shape[0] == T.shape[0]
    prec = dtype2precision(I.dtype)
    dim = I.ndim - 1
    if dim == 2:
        block = (32,32,1)
        grid = (math.ceil(I.shape[1]/block[0]), math.ceil(I.shape[2]/block[1]), 1)
        if compute_weights:
            affine_cuda.splat_image_affine_2d(
                out, outw,
                I, A, T,
                np.int32(I.shape[0]),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
            return out, outw
        else:
            affine_cuda.splat_image_affine_noweights_2d(
                out,
                I, A, T,
                np.int32(I.shape[0]),
                np.int32(I.shape[1]),
                np.int32(I.shape[2]),
                precision=prec, block=block, grid=grid)
            return out

def invert_affine(A, T):
    alloc = A.allocator
    A = A.get()
    T = T.get()
    denom = A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]
    Ainv = np.zeros_like(A)
    Ainv[:,0,0] = A[:,1,1]/denom
    Ainv[:,1,1] = A[:,0,0]/denom
    Ainv[:,1,0] = -A[:,0,1]/denom
    Ainv[:,0,1] = -A[:,1,0]/denom
    Tinv = np.zeros_like(T)
    Tinv[:,0] = -(Ainv[:,0,0]*T[:,0] + Ainv[:,0,1]*T[:,1])
    Tinv[:,1] = -(Ainv[:,1,0]*T[:,0] + Ainv[:,1,1]*T[:,1])
    return gpuarray.to_gpu(Ainv, allocator=alloc), \
            gpuarray.to_gpu(Tinv, allocator=alloc)

def match_affine(I, J, num_iters=1000, step_size_A=1e-10, step_size_T=1e-7, diagonal=False, A=None, T=None, B=None, C=None):
    # initialize to identity transform
    if A is None:
        A = np.zeros((1,2,2), dtype=J.dtype)
        A[:,0,0] = 1
        A[:,1,1] = 1
        A = gpuarray.to_gpu(A, allocator=J.allocator)
    if T is None:
        T = gpuarray.zeros((1,2), dtype=J.dtype)
    dA = gpuarray.empty_like(A)
    dT = gpuarray.empty_like(T)
    losses = []
    for it in range(num_iters):
        dA.fill(0)
        dT.fill(0)
        dA, dT, loss = affine_gradient(I, J, A, T, B=B, C=C, outdA=dA, outdT=dT)
        if diagonal:
            dAh = dA.get()
            dAh[:,0,1] = 0
            dAh[:,1,0] = 0
            dA.set(dAh)
        multiply_add(dA, -step_size_A, out=A)
        multiply_add(dT, -step_size_T, out=T)
        losses.append(loss)
    return A, T, losses

def triangular2square_numpy(U):
    """
    Given a batch of upper triangular matrices 6x6 given in row contiguous order,
    convert to the dense square symmetric matrices.
    """
    Usq = np.zeros((U.shape[0],6,6), dtype=U.dtype)
    n = 0
    for i in range(6):
        for j in range(i, 6):
            Usq[:,i,j] = U[:, n]
            if j > i:
                Usq[:,j,i] = Usq[:,i,j]
            n += 1
    return Usq

def square2triangular_numpy(S):
    assert S.shape[1] == S.shape[2]
    d = S.shape[1]
    U = np.zeros((S.shape[0],d*(d+1)//2), dtype=S.dtype)
    n = 0
    for i in range(d):
        for j in range(i, d):
            U[:,n] = S[:,i,j]
            n += 1
    return U

def batch_cholesky_triangular_numpy(H):
    """
    Given an n-by-21 gpuarray representing n 6-by-6 symmetric positive definite
    matrices, compute Cholesky factors L_n, returning another n-by-21 gpuarray.
    """
    # initialize L
    Hsq = triangular2square_numpy(H)
    Lsq = np.linalg.cholesky(Hsq)
    return square2triangular_numpy(np.swapaxes(Lsq,1,2))

def batch_cholesky_solve_numpy(L, pA, pT):
    """
    Given a batch of 6x6 cholesky factors (as serialized upper triangles), a
    batch of 2x2 matrix directions, and a batch of 2-vector translation
    directions, backsolve the directions.
    """
    x = np.zeros((pA.shape[0],6), dtype=pA.dtype)
    # backsolve the lower triangle
    x[:,0] =  pA[:,0,0] / L[:,0]
    x[:,1] = (pA[:,0,1] - x[:,0]*L[:,1]) / L[:,6]
    x[:,2] = (pA[:,1,0] - x[:,0]*L[:,2]
                        - x[:,1]*L[:,7]) / L[:,11]
    x[:,3] = (pA[:,1,1] - x[:,0]*L[:,3]
                        - x[:,1]*L[:,8]
                        - x[:,2]*L[:,12]) / L[:,15]
    x[:,4] = (  pT[:,0] - x[:,0]*L[:,4]
                        - x[:,1]*L[:,9]
                        - x[:,2]*L[:,13]
                        - x[:,3]*L[:,16]) / L[:,18]
    x[:,5] = (  pT[:,1] - x[:,0]*L[:,5]
                        - x[:,1]*L[:,10]
                        - x[:,2]*L[:,14]
                        - x[:,3]*L[:,17]
                        - x[:,4]*L[:,19]) / L[:,20]
    # now backsolve the upper triangle
    yA = np.zeros_like(pA)
    yT = np.zeros_like(pT)
    yT[:,1] =  x[:,5] / L[:,20]
    yT[:,0] = (x[:,4] - yT[:,1]*L[:,19]) / L[:,18]
    yA[:,1,1] = (x[:,3] - yT[:,0]*L[:,16]
                        - yT[:,1]*L[:,17]) / L[:,15]
    yA[:,1,0] = (x[:,2] - yA[:,1,1]*L[:,12]
                        - yT[:,0]*L[:,13]
                        - yT[:,1]*L[:,14]) / L[:,11]
    yA[:,0,1] = (x[:,1] - yA[:,1,0]*L[:,7]
                        - yA[:,1,1]*L[:,8]
                        - yT[:,0]*L[:,9]
                        - yT[:,1]*L[:,10]) / L[:,6]
    yA[:,0,0] = (x[:,0] - yA[:,0,1]*L[:,1]
                        - yA[:,1,0]*L[:,2]
                        - yA[:,1,1]*L[:,3]
                        - yT[:,0]*L[:,4]
                        - yT[:,1]*L[:,5]) / L[:,0]
    return yA, yT

def match_affine_newton(I, J, num_iters=1000,
        step_size_A=1., step_size_T=1.,
        reg_weight_A=1.0, reg_weight_T=1.0,
        diagonal=False, A=None, T=None, B=None, C=None):
    # initialize to identity transform
    if A is None:
        A = np.zeros((1,2,2), dtype=J.dtype)
        A[:,0,0] = 1
        A[:,1,1] = 1
        A = gpuarray.to_gpu(A, allocator=J.allocator)
    if T is None:
        T = gpuarray.zeros((1,2), dtype=J.dtype)
    dA = gpuarray.empty_like(A)
    dT = gpuarray.empty_like(T)
    losses = []
    for it in range(num_iters):
        dA.fill(0)
        dT.fill(0)
        dA, dT, H, loss = affine_gradient_hessian(I, J, A, T, B=B, C=C, outdA=dA, outdT=dT)
        # add prior terms
        centeredA = A.get()
        centeredA[:,0,0] -= 1.
        centeredA[:,1,1] -= 1.
        loss += reg_weight_A*np.sum(centeredA**2) + reg_weight_T*L2(T,T)
        Hh = H.get()
        dAh = dA.get()
        dAh += reg_weight_A*A.get()
        dAh[:,0,0] -= reg_weight_A
        dAh[:,1,1] -= reg_weight_A
        dTh = dT.get()
        dTh += reg_weight_T*T.get()
        Hh[:,[0,6,11,15]] += reg_weight_A
        Hh[:,[18,20]] += reg_weight_T
        # cholesky and backsolve to get update direction
        L = batch_cholesky_triangular_numpy(Hh)
        pA, pT = batch_cholesky_solve_numpy(L, dAh, dTh)
        if diagonal:
            pA[:,0,1] = 0
            pA[:,1,0] = 0
        pA = gpuarray.to_gpu(pA, allocator=I.allocator)
        pT = gpuarray.to_gpu(pT, allocator=I.allocator)
        multiply_add(pA, -step_size_A, out=A)
        multiply_add(pT, -step_size_T, out=T)
        losses.append(loss)
    return A, T, losses

def atlas_update_base_image_affine(J, A, T, B=None, C=None, I=None,
        l2_weight=0.01, num_iters=10, step_size=.5, use_mpi=False):
    if use_mpi:
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        except ImportError:
            print("use_mpi option passed but mpi4py not available")
            raise
    def allreduce_gpu(xd):
        # divide by a biggish constant to prevent overflows
        scale = np.prod(J.shape[1:])
        xh = np.ascontiguousarray(xd.get().ravel())/scale
        xr = comm.allreduce(sendobj=xh, op=MPI.SUM)
        xr *= scale
        xd.set(xr.reshape(xd.shape))
    if I is None:
        Ish = tuple([1] + list(J.shape[1:]))
        I = gpuarray.zeros(shape=Ish, dtype=J.dtype, allocator=J.allocator)
    num = gpuarray.zeros_like(I)
    denom = gpuarray.zeros_like(I)
    prec = dtype2precision(J.dtype)
    block = (16,32,1)
    grid = (math.ceil(J.shape[1]/block[0]), math.ceil(J.shape[2]/block[1]), 1)
    ssed = gpuarray.zeros((1,), dtype=J.dtype, allocator=J.allocator)
    lastloss = np.inf
    losses = []
    for it in range(num_iters):
        num.fill(0.0)
        denom.fill(0.0)
        ssed.fill(0.0)
        if B is None:
            affine_cuda.affine_atlas_jacobi_2d(ssed, num, denom, I, J, A, T,
                    np.int32(J.shape[0]), np.int32(J.shape[1]), np.int32(J.shape[2]),
                    block=block, grid=grid, precision=prec)
        else:
            affine_cuda.affine_atlas_jacobi_contrast_2d(ssed, num, denom, I, J, A, T,
                    B, C,
                    np.int32(J.shape[0]), np.int32(J.shape[1]), np.int32(J.shape[2]),
                    block=block, grid=grid, precision=prec)
        if use_mpi:
            # allreduce g and sumw
            allreduce_gpu(num)
            allreduce_gpu(denom)
            allreduce_gpu(ssed)
        # Add the l2 regularization term
        multiply_add(I, l2_weight, out=num)
        denom += l2_weight
        loss = .5*(ssed.get()[0] + l2_weight*L2(I,I))
        losses.append(loss)
        lastloss = loss
        #print(f"it={it} loss={loss}")
        num /= denom    
        multiply_add(num, -step_size, out=I)
    return I, losses

def atlas_update_contrast_affine(I, J, A, T, B=None, C=None, use_mpi=False):
    nxy = J.shape[1]*J.shape[2]
    Jh = J.get()
    sumJ = Jh.sum(axis=(1,2)) # precompute sum of each image
    phiI = interp_image_affine(I, A, T).get()
    sumphiI = (phiI).sum(axis=(1,2))
    phiIJ = (phiI*Jh).sum(axis=(1,2))
    L2phiI = (phiI**2).sum(axis=(1,2))
    denom = np.clip(L2phiI*nxy - sumphiI**2, .01, None)
    Bh = (nxy*phiIJ - sumJ*sumphiI)/denom
    Ch = (L2phiI*sumJ - sumphiI*phiIJ)/denom
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        sumBh = comm.allreduce(sendobj=Bh.sum(), op=MPI.SUM)
        sumCh = comm.allreduce(sendobj=Ch.sum(), op=MPI.SUM)
        nsubjects = comm.allreduce(sendobj=np.ones_like(Ch).sum(), op=MPI.SUM)
        meanBh = sumBh/nsubjects
        meanCh = sumCh/nsubjects
    else:
        meanBh = np.mean(Bh)
        meanCh = np.mean(Ch)
    #Bh -= meanBh - 1.0
    #Ch -= meanCh
    if B is None:
        B = gpuarray.to_gpu(Bh.astype(I.dtype), allocator=I.allocator)
    else:
        B.set(Bh)
    if C is None:
        C = gpuarray.to_gpu(Ch.astype(I.dtype), allocator=I.allocator)
    else:
        C.set(Ch)
    return B, C

def atlas_affine(J, num_iters=5, N_affine=100, step_size_A=1e-10,
        step_size_T=1e-7, image_l2=0.01, diagonal=True, I=None, contrast=True,
        use_mpi=False):
    try:
        from mpi4py import MPI
    except ImportError:
        if use_mpi:
            print("use_mpi==True but mpi4py cannot be imported")
            raise
    losses = []
    A = np.zeros((J.shape[0],2,2), dtype=J.dtype)
    A[:,0,0] = 1
    A[:,1,1] = 1
    A = gpuarray.to_gpu(A, allocator=J.allocator)
    T = gpuarray.zeros((J.shape[0],2), dtype=J.dtype, allocator=J.allocator)
    B = C = None
    if I is None:  # initialize to average image
        I = sum_along_axis(J, axis=0)
        I /= float(J.shape[0])
    else:
        I, lossesI = atlas_update_base_image_affine(J, A, T, B=B, C=C, l2_weight=image_l2, I=I)
        losses.extend(lossesI)
    from datetime import datetime
    tstart = datetime.now()
    for it in range(num_iters):
        tstart_it = datetime.now()
        for _ in range(1):
            if contrast:
                B, C = atlas_update_contrast_affine(I, J, A, T, B=B, C=C,
                        use_mpi=use_mpi)
            I, lossesI = atlas_update_base_image_affine(J, A, T, I=I, B=B, C=C,
                    use_mpi=use_mpi, l2_weight=image_l2)
            losses.extend(lossesI)
        Il2 = L2(I,I)
        A, T, losses_pose = match_affine(I, J, N_affine, step_size_A, step_size_T, diagonal=diagonal, A=A, T=T, B=B, C=C)
        # add base image regularization term
        losses_pose = [l + .5*image_l2*Il2 for l in losses_pose]
        if use_mpi:
            comm = MPI.COMM_WORLD
            # reduce the losses too
            losses_pose = list(comm.allreduce(np.asarray(losses_pose), op=MPI.SUM))
        losses.extend(losses_pose)
        if len(losses_pose) > 1 and (not use_mpi or MPI.COMM_WORLD.Get_rank() == 0):
            tnow = datetime.now()
            print(f"Iter {it+1} of {num_iters}. Loss went from {losses_pose[0]:15.2f} to {losses_pose[-1]:15.2f} iter: {tnow-tstart_it} elapsed: {tnow-tstart}")
            import sys
            sys.stdout.flush()
    return I, A, T, B, C, losses