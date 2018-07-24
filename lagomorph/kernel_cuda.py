# vim: set syntax=cuda:
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from pycuda import gpuarray

_cu = '''
#include <stdio.h>

#if Real==float
#define SQRT sqrtf
#define ACOS acosf
#define FOO 15.7
#define FLOOR floorf
#elif Real==double
#define SQRT sqrt
#define ACOS acos
#define FLOOR floor
#define FOO 7.5
#endif

#define PI  3.14159265358979323846
const Real ONE_OVER_2PI = 0.5f/PI;

__inline__ __device__ Real safe_sqrt(Real x) {
    if (x < 0.) return 0.;
    return SQRT(x);
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

#inv2 = CudaFunc("inv2")
