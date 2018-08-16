import pycuda.autoinit
from pycuda import gpuarray
from pycuda.tools import DeviceMemoryPool

import numpy as np

_device_memory_pool = DeviceMemoryPool()
alloc = _device_memory_pool.allocate

def to_gpu(arr):
    return gpuarray.to_gpu(np.ascontiguousarray(arr))
