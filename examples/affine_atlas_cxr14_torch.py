raise Exception("DEPRECATED: This example has not been updated to use the new pytorch interface")

import torch
import h5py
import numpy as np
import lagomorph.torch as lt
import math
from tqdm import tqdm
import atexit


if __name__ == '__main__':
    border=0
    w = 128
    with h5py.File(f'/raid/ChestXRay14/chestxray14_{w}.h5', 'r') as f:
        train_imgs = f['/images/train'][:65000,...]
        num_samples = train_imgs.shape[0]
	# I crop the boundary pixels out since they are often introduce discontinuities
        if border > 0:
            sl = train_imgs[:, border:-border, border:-border]
        else:
            sl = train_imgs
        Jhost = np.asarray(sl, dtype=np.float32)/255.0
        J = torch.from_numpy(Jhost).cuda(non_blocking=True)

    # atlas params
    use_contrast = False
    diagonal = True
    regA = regT = 1e-3
    #interp = lt.AffineInterpImage()
    interp = lt.AffineInterpImageFunction.apply
    A = J.new_zeros((J.size(0), 2, 2))
    A[:,0,0] = 1.
    A[:,1,1] = 1.
    T = J.new_zeros((J.size(0), 2))
    # initialize image is just arithmetic mean of inputs
    I = J.mean(dim=0, keepdim=True)
    atlas_iters = 100
    base_image_iters = 10
    match_iters = 100
    base_image_stepsize = 1e2
    stepsize_A = 1e2
    stepsize_T = 1e5
    criterion = torch.nn.MSELoss()
    it_losses = []
    with tqdm(total=atlas_iters,desc=f'Atlas', position=0) as tatlas, \
         tqdm(total=match_iters,desc=f'Match', position=1) as tmatch, \
         tqdm(total=base_image_iters,desc=f'Image', position=2) as timage:
        for ait in range(atlas_iters):
            # reset progress bars
            tmatch.n = timage.n = 0
            tmatch.refresh(), timage.refresh()
            I = torch.autograd.Variable(I, requires_grad=False)
            A = torch.autograd.Variable(A, requires_grad=True)
            T = torch.autograd.Variable(T, requires_grad=True)
            optimizer = torch.optim.SGD([{'params':[A], 'lr':stepsize_A},
                            {'params':[T], 'lr':stepsize_T}], momentum=0.0)
            for mit in range(match_iters):
                optimizer.zero_grad()
                Itx = interp(I, A, T)
                loss = criterion(Itx, J)
                loss.backward()
                optimizer.step()
                lossi = loss.item()
                it_losses.append(lossi)
                tmatch.set_postfix({'loss':lossi})
                tmatch.update()
            # update base image iteratively
            I = torch.autograd.Variable(I, requires_grad=True)
            A = torch.autograd.Variable(A, requires_grad=False)
            T = torch.autograd.Variable(T, requires_grad=False)
            # TODO: use custom Jacobi method function here instead of fixed GD
            for bit in range(base_image_iters):
                Itx = interp(I, A, T)
                loss = criterion(Itx, J)
                gI, = torch.autograd.grad(loss, I)
                I = I - base_image_stepsize * gI
                lossi = loss.item()
                it_losses.append(lossi)
                timage.set_postfix({'loss':lossi})
                timage.update()
            tatlas.set_postfix({'rel_loss': lossi/it_losses[0], 'loss':lossi})
            tatlas.update()
