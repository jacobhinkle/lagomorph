from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from pycuda import gpuarray
import numpy as np

from pkg_resources import resource_filename
import os

from .checks import ContextCheck, count_nans, count_infs

_defsfile = resource_filename(__name__, "defs.cuh")
_incdir = os.path.dirname(_defsfile)

class CudaFunc:
    def __init__(self, mod, func_name):
        self.name = func_name
        self.mod = mod
    def __call__(self, *args, precision='double', **kwargs):
        f = self.mod.get_func(self.name, precision)
        # automatically convert a few python types to their c equivalent here
        args = list(args)
        for (i,a) in enumerate(args):
            if isinstance(a, int):
                args[i] = np.int32(a)
            elif isinstance(a, float):
                if precision == 'single':
                    args[i] = np.float32(a)
                else:
                    args[i] = np.float64(a)
            elif False and isinstance(a, gpuarray.GPUArray) and \
                a.dtype in [np.float32, np.float64]:
                nn = count_nans(a)
                if nn > 0:
                    raise Exception(f"Found {nn} nans")
                ni = count_infs(a)
                if ni > 0:
                    raise Exception(f"Found {ni} infs")

        ret = f(*args, **kwargs)
        return ret

class CudaModule:
    def __init__(self, cuda_source, extra_nvcc_flags=[]):
        self.source = cuda_source
        self.nvcc_flags = DEFAULT_NVCC_FLAGS + ['-std=c++11',f'-I{_incdir}'] + extra_nvcc_flags
        self.mods = {}
    def compile(self, precision='single'):
        if precision == 'single':
            nvcc_flags = self.nvcc_flags + ['-DReal=float', '-DComplex=cuFloatComplex']
        elif precision == 'double':
            nvcc_flags = self.nvcc_flags + ['-DReal=double', '-DComplex=cuDoubleComplex']
        else:
            raise Exception('Unrecognized precision: {}'.format(precision))
        return SourceModule(self.source, options=nvcc_flags, no_extern_c=True)
    def get_func(self, func_name, precision):
        try:
            mp = self.mods[precision]
        except KeyError:
            self.mods[precision] = self.compile(precision)
            mp = self.mods[precision]
        return mp.get_function(func_name)
    def func(self, func_name):
        return CudaFunc(self, func_name)
