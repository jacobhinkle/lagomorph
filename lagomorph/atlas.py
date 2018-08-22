import skcuda.misc
from pycuda import gpuarray
import numpy as np

from .arithmetic import multiply_add, L2, clip_below, sum_along_axis, multiply_add_bcast
from .deform import interp_image, imshape2defshape, splat_image
from .metric import FluidMetric
from .shooting import expmap, jacobi_field_backward


class Atlas:
    def __init__(self, J, fluid_params, reg_weight=1.0):
        self.J = J
        self.m = gpuarray.zeros(imshape2defshape(J.shape), dtype=J.dtype,
            allocator=J.allocator, order='C')
        self.metric = FluidMetric(self.m.shape, params=fluid_params,
                allocator=J.allocator)
        baseimsh = tuple([1]+list(self.J.shape[1:]))
        self.base_image = gpuarray.zeros(baseimsh, dtype=J.dtype,
            allocator=J.allocator, order='C')
        self.reg_weight = reg_weight
        self.loss = []
        self.sse = []
        self.defnormsq = []
    def update_base_image(self):
        # assume self.h is up to date
        splatJ, w = splat_image(self.J, self.h)
        # skcuda.misc.sum is broken:
        # https://github.com/lebedov/scikit-cuda/issues/173
        sumJ = sum_along_axis(splatJ, axis=0)
        sumw = sum_along_axis(w, axis=0)
        clip_below(sumw, .001, out=sumw) # avoid divide by 0
        self.base_image = sumJ/sumw
    def build(self, num_iters=10, step_size=.1,
            constrain_mean_momentum_zero=False, callbacks=None):
        if callbacks is None:
            callbacks = [self.print_last_loss]
        # adjust step size to account for image size
        step_size /= np.prod(self.J.shape[1:])
        for it in range(num_iters+1):
            if it > 0: # don't take step on first iter
                # integrate adjoint equation
                mu0 = jacobi_field_backward(self.m, self.metric, self.h, diff,
                        self.base_image)
                # complete gradient by adding regularization term to mu(0)
                multiply_add(self.m, self.reg_weight, out=mu0)
                if constrain_mean_momentum_zero:
                    meanmu0 = sum_along_axis(mu0, axis=0)
                    multiply_add_bcast(mu0, meanmu0, mu0.dtype.type(-1./float(mu0.shape[0])), out=mu0)
                # take gradient step for v0
                multiply_add(mu0, -step_size, out=self.m)
            # exponentiate and recompute optimal image
            self.h = expmap(self.m, self.metric)
            self.update_base_image()
            # apply to image
            diff = interp_image(self.base_image, self.h)
            # compute L2 difference
            diff -= self.J
            sse = L2(diff, diff)
            v = self.metric.sharp(self.m)
            msq = L2(self.m, v)
            loss = sse + self.reg_weight*msq
            self.defnormsq.append(msq)
            self.sse.append(sse)
            self.loss.append(loss)
            for cb in callbacks:
                cb(it, num_iters)
    def print_last_loss(self, it, num_iters):
        sse = self.sse[-1]
        msq = self.defnormsq[-1]
        loss = self.loss[-1]
        print(f"Iteration {it:3d} of {num_iters} SSE={sse:10.2f} |m|^2={msq:10.2f} E={loss:10.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", "--num_iters", type=int, default=10, help="Number of iterations of gradient descent")
    parser.add_argument("-a", "--alpha", type=float, default=0.1, help="Fluid kernel parameter")
    parser.add_argument("-b", "--beta", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("-c", "--gamma", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("--step_size", type=float, default=0.1, help="Gradient descent step size")
    parser.add_argument("--reg_weight", type=float, default=0.1, help="Relative weight of regularization term in loss")
    parser.add_argument("images", metavar='imfile', nargs="+", help='List of filenames of images') 

    args = parser.parse_args()

    import pycuda.autoinit # fire up pycuda
    atlas = Atlas(J, args.alpha, args.beta, args.gamma, reg_weight=args.reg_weight)
    atlas.build(num_iters=args.num_iters, step_size=args.step_size)
