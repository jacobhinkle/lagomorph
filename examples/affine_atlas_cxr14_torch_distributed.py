import torch
import torch.distributed as dist
import h5py
import numpy as np
import lagomorph.torch as lt
import math
import atexit


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

    num_local_gpus = torch.cuda.device_count()

    dist.init_process_group(backend="nccl",
                            init_method="file:///tmp/distributed_test",
                            world_size=world_size,
                            rank=rank)

    border=0

    w = 128
    Js = []
    with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}.h5', 'r') as f:
        train_imgs = f['/images/train']
        num_samples = train_imgs.shape[0]
        # determine how many (max) examples per device
        total_gpus = num_local_gpus * world_size
        samples_per_gpu = (num_samples + total_gpus - 1)//total_gpus
	# I crop the boundary pixels out since they are often introduce discontinuities
        for d in range(num_local_gpus):
            start = (rank*num_local_gpus + d)*samples_per_gpu
            end = start + samples_per_gpu
            if border > 0:
                sl = train_imgs[start:end, border:-border, border:-border]
            else:
                sl = train_imgs[start:end, ...]
            Jhost = np.asarray(sl, dtype=np.float32)/255.0
            Js.append(torch.from_numpy(Jhost).cuda(d, non_blocking=True))
    print("Done loading data")

    # atlas params
    use_contrast = False
    diagonal = True
    regA = regT = 1e-3
    #interp = lt.AffineInterpImage()
    interp = lt.AffineInterpImageFunction.apply
    As = [J.new_zeros((J.size(0), 2, 2)) for J in Js]
    for A in As:
        A[:,0,0] = 1.
        A[:,1,1] = 1.
    Ts = [J.new_zeros((J.size(0), 2)) for J in Js]
    # initialize image is just arithmetic mean of inputs
    Is = [J.sum(dim=0, keepdim=True)/num_samples for J in Js]
    print("reducing base image initial averages")
    dist.all_reduce_multigpu(Is)
    print("done reducing base image initial averages")
    atlas_iters = 100
    base_image_iters = 10
    match_iters = 100
    base_image_stepsize = 1e2
    stepsize_A = 1e2
    stepsize_T = 1e5
    criterion = torch.nn.MSELoss()
    it_losses = []
    for ait in range(atlas_iters):
        Is = [torch.autograd.Variable(I, requires_grad=False) for I in Is]
        As = [torch.autograd.Variable(A, requires_grad=True) for A in As]
        Ts = [torch.autograd.Variable(T, requires_grad=True) for T in Ts]
        optimizers = [torch.optim.SGD([{'params':[A], 'lr':stepsize_A},
                        {'params':[T], 'lr':stepsize_T}], momentum=0.0) for
                        (I,A,T) in zip(Is, As, Ts)]
        for mit in range(match_iters):
            losses = []
            for (I,A,T,J,optimizer) in zip(Is, As, Ts, Js, optimizers):
                optimizer.zero_grad()
                optimizers.append(optimizers)
                Itx = interp(I, A, T)
                loss = criterion(Itx, J)
                loss.backward()
                losses.append(loss)
                optimizer.step()
            #dist.all_reduce_multigpu(losses)
            it_losses.append(losses[0].item())
            print(f"rank={rank} mit={mit} loss={loss[0].item()}")
        if ait == 0 and len(it_losses) > 0:
            print(f"Initial MSE=", it_losses[0])
        print(f"ait={ait:4d} AT MSE=", it_losses[-1])
        # update base image iteratively
        Is = [torch.autograd.Variable(I, requires_grad=True) for I in Is]
        As = [torch.autograd.Variable(A, requires_grad=False) for A in As]
        Ts = [torch.autograd.Variable(T, requires_grad=False) for T in Ts]
        # TODO: use custom Jacobi method function here instead of fixed GD
        for bit in range(base_image_iters):
            losses = []
            for (i, (I,A,T,J)) in enumerate(zip(Is, As, Ts, Js)):
                Itx = interp(I, A, T)
                loss = criterion(Itx, J)
                losses.append(loss)
                gI, = torch.autograd.grad(loss, I)
                Is[i] = I/world_size - base_image_stepsize * gI
            dist.all_reduce_multigpu([l for l in losses])
            dist.all_reduce_multigpu(Is)
            it_losses.append(losses[0].item())
        print(f"ait={ait:4d}  I MSE=", it_losses[-1])
