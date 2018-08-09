# vim: set syntax=cuda:
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from pycuda import gpuarray

_cu = '''
#include <stdio.h>

__inline__ __device__ Real safe_sqrt(Real x) {
    if (x < 0.) return 0.;
    return SQRT(x);


__inline__ __device__ Real diff_central_unsafe(Real *x, int offset, int stride) {
        return .5*(x[offset+stride] - x[offset-stride]);
    }
__inline__ __device__ Real diff_fwd_unsafe(Real *x, int offset, int stride) {
        return x[offset+stride] - x[offset];
    }
__inline__ __device__ Real diff_rev_unsafe(Real *x, int offset, int stride) {
        return x[offset] - x[offset-stride];
    }
}


'''

def getmod(precision='single'):
    if precision == 'single':
        nvcc_flags = DEFAULT_NVCC_FLAGS + ['-DReal=float']
    elif precision == 'double':
        nvcc_flags = DEFAULT_NVCC_FLAGS + ['-DReal=double']
    else:
        raise Exception('Unrecognized precision: {}'.format(precision))

    return SourceModule(_cu, options=nvcc_flags)

class CudaFunc:
    def __init__(self, func_name):
        self.name = func_name
        self.mods = {}
    def __call__(self, *args, precision='double', **kwargs):
        if not precision in self.mods:
            self.mods[precision] = getmod(precision)
        mod = self.mods[precision]
        f = mod.get_function(self.name)
        return f(*args, **kwargs)

diff = CudaFunc("diff")
