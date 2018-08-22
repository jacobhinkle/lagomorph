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
        for (int n=0; n<nn; ++n) {
            out[inv] = v[inv]*im[inim];
            inv += nxy;
            out[inv] = v[inv]*im[inim];
            inv += nxy;
            inim += nxy;
        }
    }
    __global__ void sum_along_axis(Real* out, Real* arr, int nn, int nxyz) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= nxyz) return;
        double s = 0.0f;
        int ix = i;
        for (int n=0; n<nn; ++n) {
            s += arr[ix];
            ix += nxyz;
        }
        out[i] = static_cast<Real>(s);
    }
    __global__ void multiply_add_bcast(Real* out, Real* x, Real* y, Real alpha,
        int nn, int nxyz) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= nxyz) return;
        int ix = i;
        for (int n=0; n<nn; ++n) {
            out[ix] = x[ix] + alpha*y[i];
            ix += nxyz;
        }
    }
}
''')
multiply_imvec_2d = mod.func("multiply_imvec_2d")
sum_along_axis = mod.func("sum_along_axis")
multiply_add_bcast = mod.func("multiply_add_bcast")
