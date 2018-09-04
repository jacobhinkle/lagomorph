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

    w = 128
    with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}.h5', 'r') as f:
	# I crop the boundary pixels out since they are often introduce discontinuities
        Jhost = np.ascontiguousarray(f['/images/train'][rank:10000:world_size,2:-2,2:-2],
                dtype=np.float32)
    J = gpuarray.to_gpu(Jhost)
    del Jhost

    I, A, T, B, C, losses_atlas = lm.atlas_affine(J, num_iters=50,
	N_affine=100, contrast=True, diagonal=True, use_mpi=True)

    np.savez(f'cxr14_{w}_result.npz', I=I.get(), A=A.get(), T=T.get(),  B=B.get(), C=C.get())

