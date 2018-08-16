from pycuda import gpuarray
import pycuda.autoinit

import lagomorph as lm
import numpy as np

def circle_image(W, H, X, Y, R):
    XY = lm.identity((1,2,W,H))
    Rs = (XY[:,0,...] - X)**2 + (XY[:,1,...] - Y)**2
    im = (Rs <= R**2).astype(np.float32)
    return np.ascontiguousarray(im.reshape((1,W,H)))

def shooting_example(alpha, beta, gamma, width = 256, num_iters=10, step_size=.1, reg_weight=1.0):
    source = gpuarray.to_gpu(circle_image(width, width, width//2, width//2,
        width//4), allocator=lm.alloc)
    target = gpuarray.to_gpu(circle_image(width, width, width*5//8, width//2,
        width//4), allocator=lm.alloc)

    # initialize velocity field
    m0 = gpuarray.zeros(lm.imshape2defshape(source.shape), dtype=np.float32,
            allocator=lm.alloc, order='C')
    metric = lm.FluidMetric(alpha=alpha, beta=beta, gamma=gamma, shape=m0.shape)

    for it in range(num_iters):
        print(f"Iteration {it+1:3d} of {num_iters} ", end='')
        h = lm.expmap(m0, metric)
        # apply to image
        diff = lm.interp_image(source, h)
        # we'll need the gradient of the deformed image
        lamT = lm.gradient(diff)
        # compute L2 difference
        diff -= target
        sse = lm.L2(diff, diff)
        print(f" SSE={sse:10.2f} ", end='')
        v0 = metric.sharp(m0)
        m0sq = lm.L2(m0, v0)
        print(f"|m0|^2={m0sq:10.2f} ", end='')
        loss = sse + reg_weight*m0sq
        print(f"E={loss:10.2f}")
        # initialize lambda to difference times gradient of deformed image
        #lamT = lm.multiply_imvec(diff, lamT, out=lamT)
        # integrate adjoint equation
        mu0 = lm.jacobi_field_backward(m0, metric, h, diff, source)
        # complete gradient by adding regularization term to mu(0)
        lm.multiply_add(m0, -reg_weight, out=mu0)
        # take gradient step for v0
        lm.multiply_add(mu0, step_size, out=m0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", "--num_iters", type=int, default=10, help="Number of iterations of gradient descent")
    parser.add_argument("-w", "--width", type=int, default=256, help="Image width in pixels")
    parser.add_argument("-a", "--alpha", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("-b", "--beta", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("-c", "--gamma", type=float, default=0.1, help="Fluid kernel parameter")
    parser.add_argument("--step_size", type=float, default=0.1, help="Gradient descent step size")
    parser.add_argument("--reg_weight", type=float, default=0.1, help="Relative weight of regularization term in loss")


    args = parser.parse_args()
    shooting_example(**vars(args))
