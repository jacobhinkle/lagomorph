import torch
import h5py
import numpy as np

import atexit

import lagomorph.torch as lt

def standardize(J, A, T, B, C, out=None):
    """Apply inverse contrast and inverse affine to J to get standardized J"""
    Ainv, Tinv = lm.invert_affine(A, T)
    if B is None:
        return lm.interp_image_affine(J, Ainv, Tinv, out=out)
    else:
        # Inverse contrast is simple
        Binv = 1./B.get()
        Cinv = -C.get()*Binv
        Binv = gpuarray.to_gpu(Binv, allocator=B.allocator)
        Cinv = gpuarray.to_gpu(Cinv, allocator=T.allocator)
        return lm.interp_image_affine(J, Ainv, Tinv, out=out, B=Binv, C=Cinv)

if __name__ == '__main__':
    from mpi4py import MPI

    use_mpi = True

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()
    atexit.register(MPI.Finalize)

    border=2

    w = 16
    with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}.h5', 'r') as f:
	# I crop the boundary pixels out since they are often introduce discontinuities
        #Jhost = np.asarray(f['/images/train'][rank::world_size,border:-border,border:-border],
        Jhost = np.asarray(f['/images/train'][0:1,border:-border,border:-border],
                dtype=np.float32)
    J = torch.from_numpy(Jhost).pin_memory().cuda(non_blocking=True)
    del Jhost

    # atlas params
    use_contrast = False
    diagonal = True
    regA = regT = 1e-3
    interp = lt.AffineInterpImage(dim=2)
    A = J.new_zeros((J.size(0), 2, 2))
    A[:,0,0] = 1.
    A[:,1,1] = 1.
    A = torch.nn.Parameter(A)
    T = torch.nn.Parameter(J.new_zeros((J.size(0), 2)))
    # initialize image is just arithmetic mean of inputs
    I = J.mean(dim=0, keepdim=True)
    atlas_iters = 10
    base_image_iters = 10
    match_iters = 100
    base_image_stepsize = .1
    criterion = torch.nn.MSELoss()
    losses = []
    for ait in range(atlas_iters):
        optimizer = torch.optim.Adadelta([A,T])
        for mit in range(match_iters):
            optimizer.zero_grad()
            Itx = interp(I, A, T)
            loss = criterion(Itx, J)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        # update base image iteratively
        for bit in range(base_image_iters):
            gI = torch.autograd.grad(loss, I)
            # TODO: do an allreduce here
            # TODO: use custom Jacobi method function here instead of fixed GD
            I -= base_image_stepsize * gI
            Itx = interp(I, A, T)
            loss = criterion(Itx, J)
            losses.append(loss.item())
        print(loss.item())

    # only write if rank 0
    if rank == 0:
        if contrast:
            np.savez(f'cxr14_{w}_result.npz', I=I.numpy(), A=A.numpy(), T=T.numpy(),  B=B.numpy(), C=C.numpy(), losses=losses_atlas)
        else:
            np.savez(f'cxr14_{w}_result.npz', I=I.numpy(), A=A.numpy(), T=T.numpy(), losses=losses_atlas)

    # now do standardization of each split and save result
    with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}.h5', 'r') as f:
        #with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}_affine_standardized.h5', 'w',
                #driver='mpio', comm=MPI.COMM_WORLD) as fw:
        with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}_affine_standardized_{rank}.h5', 'w') as fw:
            # train
            Jstd = standardize(J, A, T, B, C)
            ds = fw.create_dataset('/images/train', shape=f['/images/train'].shape, dtype=f['/images/train'].dtype)
            ds[rank::world_size,border:-border,border:-border] = Jstd.numpy()
            # val
            ds = fw.create_dataset('/images/val', shape=f['/images/val'].shape, dtype=f['/images/val'].dtype)
            J = gpuarray.to_gpu(np.ascontiguousarray(f['/images/val'][rank::world_size,border:-border,border:-border],
                dtype=np.float32))
            A,T,B,C = lm.match_affine(I, J)
            Jstd = standardize(J, A, T, B, C)
            ds[rank::world_size,border:-border,border:-border] = Jstd.get()
            # test
            ds = fw.create_dataset('/images/test', shape=f['/images/test'].shape, dtype=f['/images/test'].dtype)
            J = gpuarray.to_gpu(np.ascontiguousarray(f['/images/test'][rank::world_size,border:-border,border:-border],
                dtype=np.float32))
            A,T,B,C = lm.match_affine(I, J)
            Jstd = standardize(J, A, T, B, C)
            ds[rank::world_size,border:-border,border:-border] = Jstd.get()
