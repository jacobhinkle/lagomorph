# -*- mode: cuda -*-
# vi: set syntax=cuda:
from .cudamod import CudaModule
mod = CudaModule('''
extern "C" {
    __global__ void multiply_imvec_2d(Real* out, Real* im, Real* v,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        int nxy = nx*ny;
        int inim = i*ny + j;
        int inv = inim;
        out[inv] = v[inv]*im[inim];
        inv += nxy;
        out[inv] = v[inv]*im[inim];
    }
}
''')
multiply_imvec_2d = mod.func("multiply_imvec_2d")
