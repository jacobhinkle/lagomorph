import pycuda.autoinit
from pycuda.tools import DeviceMemoryPool

_device_memory_pool = DeviceMemoryPool()
alloc = _device_memory_pool.allocate
