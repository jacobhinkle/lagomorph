raise Exception("DEPRECATED: This example has not been updated to use the new pytorch interface")

import numpy as np
import h5py

import atexit

from pycuda import gpuarray
import lagomorph as lm

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

    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    rank = comm.Get_rank()

    import pycuda.driver as drv
    drv.init()
    dev = drv.Device(rank)
    ctx = dev.make_context()
    # make sure we clean up
    atexit.register(ctx.pop)
    atexit.register(MPI.Finalize)

    border=2

    w = 128
    with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}.h5', 'r') as f:
	# I crop the boundary pixels out since they are often introduce discontinuities
        Jhost = np.ascontiguousarray(f['/images/train'][rank::world_size,border:-border,border:-border],
                dtype=np.float32)
    J = gpuarray.to_gpu(Jhost)
    del Jhost

    use_mpi = True

    contrast = False
    I, A, T, B, C, losses_atlas = lm.atlas_affine(J, num_iters=1,
	N_affine=1, contrast=contrast, diagonal=True, use_mpi=use_mpi)

    # only write if rank 0
    if rank == 0:
        if contrast:
            np.savez(f'cxr14_{w}_result.npz', I=I.get(), A=A.get(), T=T.get(),  B=B.get(), C=C.get(), losses=losses_atlas)
        else:
            np.savez(f'cxr14_{w}_result.npz', I=I.get(), A=A.get(), T=T.get(), losses=losses_atlas)

    # now do standardization of each split and save result
    with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}.h5', 'r') as f:
        #with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}_affine_standardized.h5', 'w',
                #driver='mpio', comm=MPI.COMM_WORLD) as fw:
        with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}_affine_standardized_{rank}.h5', 'w') as fw:
            # train
            Jstd = standardize(J, A, T, B, C)
            ds = fw.create_dataset('/images/train', shape=f['/images/train'].shape, dtype=f['/images/train'].dtype)
            ds[rank::world_size,border:-border,border:-border] = Jstd.get()
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
