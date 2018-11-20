raise Exception("DEPRECATED: This example has not been updated to use the new pytorch interface")

from pycuda import gpuarray
from pycuda.tools import DeviceMemoryPool
from pycuda.driver import mem_alloc
import numpy as np

from lagomorph.atlas import Atlas

def circle_image(W, H, X, Y, R, dtype=np.float32):
    from lagomorph.deform import identity
    XY = identity((1,2,W,H))
    Rs = (XY[:,0,...] - X)**2 + (XY[:,1,...] - Y)**2
    im = (Rs <= R**2).astype(dtype)
    return np.ascontiguousarray(im.reshape((1,W,H)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", "--num_iters", type=int, default=10, help="Number of iterations of gradient descent")
    parser.add_argument("-w", "--width", type=int, default=256, help="Image width in pixels")
    parser.add_argument("-a", "--alpha", type=float, default=0.1, help="Fluid kernel parameter")
    parser.add_argument("-b", "--beta", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("-c", "--gamma", type=float, default=0.01, help="Fluid kernel parameter")
    parser.add_argument("--step_size", type=float, default=0.1, help="Gradient descent step size")
    parser.add_argument("--reg_weight", type=float, default=0.1, help="Relative weight of regularization term in loss")
    args = parser.parse_args()

    import pycuda.autoinit # fire up pycuda
    pool = DeviceMemoryPool()
    alloc = pool.allocate
    #alloc = mem_alloc
    source = circle_image(args.width, args.width, args.width//2,
        args.width//2, args.width//4)
    target = circle_image(args.width, args.width,
        args.width*5//8, args.width//2, args.width//4)
    J = np.concatenate((source, target), axis=0)
    J = gpuarray.to_gpu(J, allocator=alloc)

    fluid_params = [args.alpha, args.beta, args.gamma]
    atlas = Atlas(J, fluid_params=fluid_params, reg_weight=args.reg_weight)
    atlas.build(num_iters=args.num_iters, step_size=args.step_size)
