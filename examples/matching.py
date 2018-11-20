raise Exception("DEPRECATED: This example has not been updated to use the new pytorch interface")

import numpy as np

import lagomorph as lm

def circle_image(W, H, X, Y, R, dtype=np.float32):
    from lagomorph.deform import identity
    XY = identity((1,2,W,H))
    Rs = (XY[:,0,...] - X)**2 + (XY[:,1,...] - Y)**2
    im = (Rs <= R**2).astype(dtype)
    return np.ascontiguousarray(im.reshape((1,W,H)))

if __name__ == '__main__':
    import pycuda.autoinit # fire up pycuda
    from pycuda import gpuarray
    from pycuda.tools import DeviceMemoryPool
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

    pool = DeviceMemoryPool()
    alloc = pool.allocate
    #alloc = mem_alloc
    source = gpuarray.to_gpu(circle_image(args.width, args.width, args.width//2,
        args.width//2, args.width//4), allocator=alloc)
    target = gpuarray.to_gpu(circle_image(args.width, args.width,
        args.width*5//8, args.width//2, args.width//4), allocator=alloc)

    lm.match(source, target, **vars(args))
