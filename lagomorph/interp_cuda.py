# -*- mode: cuda -*-
# vi: set syntax=cuda:
from .cudamod import CudaModule
mod = CudaModule('''
#include "interp.cuh"
#include "defs.cuh"

template<BackgroundStrategy backgroundStrategy, int displacement>
inline __device__
void
interp_vectorfield_kernel_2d(int i, int j, Real* out, const Real* g, const Real* h,
            int nn, int nx, int ny) {
    if (i >= nx || j >= ny) return;
    int nxy = nx*ny;
    int inx = i*ny + j;
    int iny = inx + nxy;
    int ino = inx;
    Real hx, hy;
    Real fi = static_cast<Real>(i);
    Real fj = static_cast<Real>(j);
    const Real* gd = g;
    for (int n=0; n < nn; ++n) {
        if (displacement) {
            hx = h[inx] + fi;
            hy = h[iny] + fj;
        } else {
            hx = h[inx];
            hy = h[iny];
        }
        out[ino] = biLerp<DEFAULT_BACKGROUND_STRATEGY>(gd,
            hx, hy,
            nx, ny,
            0.f);
        ino += nxy;
        gd += nxy;
        out[ino] = biLerp<DEFAULT_BACKGROUND_STRATEGY>(gd,
            hx, hy,
            nx, ny,
            0.f);
        ino += nxy;
        gd += nxy;
        inx += 2*nxy;
        iny += 2*nxy;
    }
}

template<BackgroundStrategy backgroundStrategy, int displacement,
    int broadcast_image=0>
inline __device__
void
interp_image_kernel_2d(int i, int j, Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
    if (i >= nx || j >= ny) return;
    int nxy = nx*ny;
    int inx = i*ny + j;
    int iny = inx + nxy;
    int ino = inx;
    Real* In = I;
    for (int n=0; n < nn; ++n) {
        Real hx = h[inx];
        Real hy = h[iny];
        if (displacement) {
            hx += static_cast<Real>(i);
            hy += static_cast<Real>(j);
        }
        out[ino] = biLerp<DEFAULT_BACKGROUND_STRATEGY>(In,
            hx, hy,
            nx, ny,
            0.f);
        inx += 2*nxy;
        iny += 2*nxy;
        ino += nxy;
        if (!broadcast_image)
            In += nxy;
    }
}

template<BackgroundStrategy backgroundStrategy, int displacement,
    int broadcast_image=0>
inline __device__
void
interp_grad_kernel_2d(int i, int j, Real* out, Real* g, Real* I, Real* h,
            int nn, int nx, int ny) {
    if (i >= nx || j >= ny) return;
    int nxy = nx*ny;
    int inx = i*ny + j;
    int iny = inx + nxy;
    int inim = inx;
    Real gx, gy, Ix;
    Real* In = I;
    for (int n=0; n < nn; ++n) {
        Real hx = h[inx];
        Real hy = h[iny];
        if (displacement) {
            hx += static_cast<Real>(i);
            hy += static_cast<Real>(j);
        }
        biLerp_grad<DEFAULT_BACKGROUND_STRATEGY>(Ix, gx, gy,
            In,
            hx, hy,
            nx, ny,
            0.f);
        out[inim] = Ix;
        g[inx] = gx;
        g[iny] = gy;
        inx += 2*nxy;
        iny += 2*nxy;
        inim += nxy;
        if (!broadcast_image)
            In += nxy;
    }
}

template<BackgroundStrategy backgroundStrategy, int displacement, int write_weights>
inline __device__
void
splat_image_kernel_2d(int i, int j, Real* d_wd, Real* d_ww, Real* I, Real* h,
            int nn, int nx, int ny) {
    if (i >= nx || j >= ny) return;
    int nxy = nx*ny;
    int inx = i*ny + j;
    int iny = inx + nxy;
    int ino = inx;
    Real* dn = d_wd;
    Real* wn = d_ww;
    for (int n=0; n < nn; ++n) {
        Real hx = h[inx];
        Real hy = h[iny];
        if (displacement) {
            hx += static_cast<Real>(i);
            hy += static_cast<Real>(j);
        }
        atomicSplat<DEFAULT_BACKGROUND_STRATEGY, write_weights>(dn, wn,
            I[ino], hx, hy, nx, ny);
        inx += 2*nxy;
        iny += 2*nxy;
        dn += nxy;
        wn += nxy;
        ino += nxy;
    }
}

extern "C" {
    __global__ void splat_image_2d(Real* splats, Real* weights, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        splat_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 1>(i, j,
            splats, weights, I, h, nn, nx, ny);
    }
    __global__ void splat_displacement_image_2d(Real* d_wd, Real* d_ww, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        splat_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 1>(i, j,
            d_wd, d_ww, I, h, nn, nx, ny);
    }
    __global__ void splat_image_noweights_2d(Real* splats, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        splat_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0>(i, j,
            splats, NULL, I, h, nn, nx, ny);
    }
    __global__ void splat_displacement_image_noweights_2d(Real* d_wd, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        splat_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0>(i, j,
            d_wd, NULL, I, h, nn, nx, ny);
    }
    __global__ void interp_image_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0>(i, j, out, I, h, nn, nx, ny);
    }
    __global__ void interp_displacement_image_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0>(i, j, out, I, h, nn, nx, ny);
    }
    __global__ void interp_grad_2d(Real* out, Real*g, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_grad_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0>(i, j, out, g, I, h, nn, nx, ny);
    }
    __global__ void interp_displacement_grad_2d(Real* out, Real*g, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_grad_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0>(i, j, out, g, I, h, nn, nx, ny);
    }
    __global__ void interp_vectorfield_2d(Real* out, Real* g, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_vectorfield_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0>(i, j, out, g, h, nn, nx, ny);
    }
    __global__ void interp_displacement_vectorfield_2d(Real* out, const Real* g, const Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_vectorfield_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1>(i, j, out, g, h, nn, nx, ny);
    }
    __global__ void interp_zerobg_vectorfield_2d(Real* out, Real* g, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_vectorfield_kernel_2d<BACKGROUND_STRATEGY_ZERO, 0>(i, j, out, g, h, nn, nx, ny);
    }
    __global__ void interp_displacement_zerobg_vectorfield_2d(Real* out, const Real* g, const Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_vectorfield_kernel_2d<BACKGROUND_STRATEGY_ZERO, 1>(i, j, out, g, h, nn, nx, ny);
    }
    __global__ void interp_image_bcastI_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 1>(i, j, out, I, h, nn, nx, ny);
    }
    __global__ void interp_displacement_image_bcastI_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 1>(i, j, out, I, h, nn, nx, ny);
    }
    __global__ void interp_image_bcastI_3d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny, int nz) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        int nxy = nx*ny;
        int nxyz = nxy*nz;
        int inn = 0;
        for (int n=0; n < nn; ++n) {
            int inx = inn +         i*ny + j;
            int iny = inn +   nxy + i*ny + j;
            int inz = inn + 2*nxy + i*ny + j;
            int ino = inx;
            for (int k=0; k < nz; ++k) {
                Real hx = h[inx];
                Real hy = h[iny];
                Real hz = h[inz];
                out[ino] = triLerp<DEFAULT_BACKGROUND_STRATEGY>(I,
                    hx, hy, hz,
                    nx, ny, nz,
                    0.f);
                ino += nxy;
                inx += 3*nxy;
                iny += 3*nxy;
                inz += 3*nxy;
            }
            inn += nxyz;
        }
    }
}
''', extra_nvcc_flags=[
        '-DDEFAULT_BACKGROUND_STRATEGY=BACKGROUND_STRATEGY_CLAMP'])
splat_image_2d = mod.func("splat_image_2d")
splat_displacement_image_2d = mod.func("splat_displacement_image_2d")
splat_image_noweights_2d = mod.func("splat_image_noweights_2d")
splat_displacement_image_noweights_2d = mod.func("splat_displacement_image_noweights_2d")
interp_image_2d = mod.func("interp_image_2d")
interp_displacement_image_2d = mod.func("interp_displacement_image_2d")
interp_grad_2d = mod.func("interp_grad_2d")
interp_displacement_grad_2d = mod.func("interp_displacement_grad_2d")
interp_vectorfield_2d = mod.func("interp_vectorfield_2d")
interp_displacement_vectorfield_2d = mod.func("interp_displacement_vectorfield_2d")
interp_zerobg_vectorfield_2d = mod.func("interp_zerobg_vectorfield_2d")
interp_displacement_zerobg_vectorfield_2d = mod.func("interp_displacement_zerobg_vectorfield_2d")
interp_image_bcastI_2d = mod.func("interp_image_bcastI_2d")
interp_image_bcastI_2d = mod.func("interp_displacement_image_bcastI_2d")
