from pycuda import gpuarray
import pycuda.autoinit

import lagomorph as lm
import numpy as np

def circle_image(W, H, X, Y, R):
    XY = lm.identity((1,2,W,H))
    Rs = (XY[:,0,...] - X)**2 + (XY[:,1,...] - Y)**2
    im = (Rs <= R**2).astype(np.float32)
    return im.reshape((1,W,H))

def shooting_example(alpha, beta, gamma, width = 256, num_iters=10):
    source = gpuarray.to_gpu(circle_image(width, width, width//2, width//2, width//4))
    target = gpuarray.to_gpu(circle_image(width, width, width*5//8, width//2, width//4))

    # initialize velocity field
    m0 = np.zeros(lm.imshape2defshape(source.shape), dtype=np.float32)
    metric = lm.FluidMetric(alpha=alpha, beta=beta, gamma=gamma, shape=m0.shape)

    for it in range(num_iters):
        print(f"Iteration {it+1:2d} of {num_iters}")
        h1 = lm.expmap(m0, metric)

        # TODO apply to image
        I1 = lm.interp_image(source, h1)
        # TODO compute L2 difference
        I1 -= target
        L2 = lm.sum_squares(I1)
        print(f"L2 difference = {L2}")
        # TODO integrate adjoint equation
        # TODO take gradient step for v0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", "--num_iters", type=int, default=10, help="Number of iterations of gradient descent")
    parser.add_argument("-w", "--width", type=int, default=256, help="Image width in pixels")
    parser.add_argument("-a", "--alpha", type=float, default=0.1, help="Fluid kernel parameter")
    parser.add_argument("-b", "--beta", type=float, default=0.1, help="Fluid kernel parameter")
    parser.add_argument("-c", "--gamma", type=float, default=0.1, help="Fluid kernel parameter")

    args = parser.parse_args()
    shooting_example(**vars(args))
