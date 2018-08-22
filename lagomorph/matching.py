from pycuda import gpuarray
import numpy as np

from .arithmetic import multiply_add, L2
from .deform import interp_image, imshape2defshape
from .metric import FluidMetric
from .shooting import expmap, jacobi_field_backward


def match(source, target, fluid_params, num_iters=10, step_size=.1, reg_weight=1.0):
    step_size /= float(np.prod(source.shape[1:]))

    # initialize velocity field
    m0 = gpuarray.zeros(imshape2defshape(source.shape), dtype=source.dtype,
            allocator=source.allocator, order='C')
    # set up metric
    metric = FluidMetric(shape=m0.shape, params=fluid_params,
            allocator=source.allocator)

    for it in range(num_iters):
        print(f"Iteration {it+1:3d} of {num_iters} ", end='')
        h = expmap(m0, metric)
        # apply to image
        diff = interp_image(source, h)
        # compute L2 difference
        diff -= target
        sse = L2(diff, diff)
        print(f" SSE={sse:10.2f} ", end='')
        v0 = metric.sharp(m0)
        m0sq = L2(m0, v0)
        print(f"|m0|^2={m0sq:10.2f} ", end='')
        loss = sse + reg_weight*m0sq
        print(f"E={loss:10.2f}")
        # integrate adjoint equation
        mu0 = jacobi_field_backward(m0, metric, h, diff, source)
        # complete gradient by adding regularization term to mu(0)
        multiply_add(m0, reg_weight, out=mu0)
        # take gradient step for v0
        multiply_add(mu0, -step_size, out=m0)

    return m0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", "--num_iters", type=int, default=10, help="Number of iterations of gradient descent")
    parser.add_argument("-a", "--alpha", type=float, default=0.1, help="Fluid kernel parameter")
    parser.add_argument("-b", "--beta", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("-c", "--gamma", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("--step_size", type=float, default=0.1, help="Gradient descent step size")
    parser.add_argument("--reg_weight", type=float, default=0.1, help="Relative weight of regularization term in loss")

    args = parser.parse_args()

    import pycuda.autoinit # fire up pycuda
    match(**vars(args))
