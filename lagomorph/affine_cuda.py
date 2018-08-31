# -*- mode: cuda -*-
# vi: set syntax=cuda:
from .cudamod import CudaModule
mod = CudaModule('''
#include "interp.cuh"
#include "diff.cuh"
#include "defs.cuh"

template<BackgroundStrategy backgroundStrategy, int broadcast_I, int broadcast_J>
inline __device__
void
old_affine_grad_kernel_2d(Real* gA, Real* gT, Real* loss, // outputs
        Real* I, Real* J, // input image sets
        Real* A, Real* T, // Affine transform matrix and translation
        int nn, int nx, int ny) {
    int blockstartx = blockDim.x * blockIdx.x;
    int blockstarty = blockDim.y * blockIdx.y;
    int i = blockstartx + threadIdx.x;
    int j = blockstarty + threadIdx.y;
    // get a linear thread index to index into the shared arrays
    int tid = threadIdx.x*blockDim.y + threadIdx.y;
    __shared__ Real dlA00[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlA01[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlA10[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlA11[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlT0[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlT1[MAX_THREADS_PER_BLOCK];
    __shared__ Real losses[MAX_THREADS_PER_BLOCK];
    int nxy = nx*ny;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    const Real* In = I; // pointer to current vector field v
    const Real* Jn = J; // pointer to current vector field v
    Real* gAn = gA; // pointer to current vector field v
    Real* gTn = gT; // pointer to current vector field v
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-ox;
    Real diff, gx, gy, hx, hy;
    Real sumloss = 0.0f;
    for (int n=0; n < nn; ++n) {
        if (i >= nx || j >= ny) {
            dlA00[tid] = 0;
            dlA01[tid] = 0;
            dlA10[tid] = 0;
            dlA11[tid] = 0;
            dlT0[tid] = 0;
            dlT1[tid] = 0;
            losses[tid] = 0;
        } else {
            // apply affine transform to map i, j to lookup point
            hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
            hy = An[2]*fi + An[3]*fj + Tn[1] + oy;
            // get interp value and gradient at lookup point
            biLerp_grad<backgroundStrategy>(diff, gx, gy,
                In,
                hx, hy,
                nx, ny,
                0.f);
            // use Jn point to compute image difference at this point
            diff -= Jn[ix];
            gx *= diff; // save a few multiplies by premultiplying
            gy *= diff;
            // compute the outer product terms that will be summed
            dlA00[tid] = gx*fi;
            dlA01[tid] = gx*fj;
            dlA10[tid] = gy*fi;
            dlA11[tid] = gy*fj;
            dlT0[tid] = gx;
            dlT1[tid] = gy;
            losses[tid] = diff*diff;
        }

        // reduce this block
        __syncthreads();
        static_assert(MAX_THREADS_PER_BLOCK <= 1024, "MAX_THREADS_PER_BLOCK > 1024 not supported");
        // ensure counterpart in second half of arrays is not outside
        // pixel domain
#define REDUCE_BLOCK(N) \
	if (MAX_THREADS_PER_BLOCK > N) { \
            if (tid < N) { \
                dlA00[tid] += dlA00[tid + N]; \
                dlA01[tid] += dlA01[tid + N]; \
                dlA10[tid] += dlA10[tid + N]; \
                dlA11[tid] += dlA11[tid + N]; \
                dlT0[tid] += dlT0[tid + N]; \
                dlT1[tid] += dlT1[tid + N]; \
                losses[tid] += losses[tid + N]; \
            } \
            __syncthreads(); \
        }
        REDUCE_BLOCK(512)
        REDUCE_BLOCK(256)
        REDUCE_BLOCK(128)
        REDUCE_BLOCK(64)
        REDUCE_BLOCK(32)
        REDUCE_BLOCK(16)
        REDUCE_BLOCK(8)
        REDUCE_BLOCK(4)
        REDUCE_BLOCK(2)
        REDUCE_BLOCK(1)
        if (tid == 0) {
            atomicAdd(&gAn[0], dlA00[0]);
            atomicAdd(&gAn[1], dlA01[0]);
            atomicAdd(&gAn[2], dlA10[0]);
            atomicAdd(&gAn[3], dlA11[0]);
            atomicAdd(&gTn[0], dlT0[0]);
            atomicAdd(&gTn[1], dlT1[0]);
            sumloss += losses[0];
        }

        if (!broadcast_I)
            In += nxy;
        if (!broadcast_J)
            Jn += nxy;
        An += 4;
        Tn += 2;
        gAn += 4;
        gTn += 2;
    }
    // record sum loss for this block
    if (tid == 0) {
        atomicAdd(loss, sumloss);
    }
}
// This function computes the gradient of the sum of squared error image match
// term after an affine transformation. gAt should be an array of Reals of
// length six times the number of blocks in the grid.
template<BackgroundStrategy backgroundStrategy, int broadcast_I, int broadcast_J, int use_contrast>
inline __device__
void
affine_grad_kernel_2d(Real* gA, Real* gT, Real* loss, // outputs
        Real* I, Real* J, // input image sets
        Real* A, Real* T, // Affine transform matrix and translation
        Real* B, Real* C, // contrast scale and offset
        int nn, int nx, int ny) {
    int blockstartx = blockDim.x * blockIdx.x;
    int blockstarty = blockDim.y * blockIdx.y;
    int i = blockstartx + threadIdx.x;
    int j = blockstarty + threadIdx.y;
    // get a linear thread index to index into the shared arrays
    int tid = threadIdx.x*blockDim.y + threadIdx.y;
    __shared__ Real dlA00[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlA01[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlA10[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlA11[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlT0[MAX_THREADS_PER_BLOCK];
    __shared__ Real dlT1[MAX_THREADS_PER_BLOCK];
    __shared__ Real losses[MAX_THREADS_PER_BLOCK];
    int nxy = nx*ny;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    const Real* In = I; // pointer to current vector field v
    const Real* Jn = J; // pointer to current vector field v
    Real* gAn = gA; // pointer to current vector field v
    Real* gTn = gT; // pointer to current vector field v
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    const Real* Bn = B;
    const Real* Cn = C;
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-ox;
    Real diff, gx, gy, hx, hy;
    Real sumloss = 0.0f;
    for (int n=0; n < nn; ++n) {
        if (i >= nx || j >= ny) {
            dlA00[tid] = 0;
            dlA01[tid] = 0;
            dlA10[tid] = 0;
            dlA11[tid] = 0;
            dlT0[tid] = 0;
            dlT1[tid] = 0;
            losses[tid] = 0;
        } else {
            // apply affine transform to map i, j to lookup point
            hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
            hy = An[2]*fi + An[3]*fj + Tn[1] + oy;
            // get interp value and gradient at lookup point
            biLerp_grad<backgroundStrategy>(diff, gx, gy,
                In,
                hx, hy,
                nx, ny,
                0.f);
            if (use_contrast) {
                diff = Bn[0]*diff + Cn[0];
                gx *= Bn[0];
                gy *= Bn[0];
            }
            // use Jn point to compute image difference at this point
            diff -= Jn[ix];
            gx *= diff; // save a few multiplies by premultiplying
            gy *= diff;
            // compute the outer product terms that will be summed
            dlA00[tid] = gx*fi;
            dlA01[tid] = gx*fj;
            dlA10[tid] = gy*fi;
            dlA11[tid] = gy*fj;
            dlT0[tid] = gx;
            dlT1[tid] = gy;
            losses[tid] = diff*diff;
        }

        // reduce this block
        __syncthreads();
        static_assert(MAX_THREADS_PER_BLOCK <= 1024, "MAX_THREADS_PER_BLOCK > 1024 not supported");
        // ensure counterpart in second half of arrays is not outside
        // pixel domain
#define REDUCE_BLOCK(N) \
	if (MAX_THREADS_PER_BLOCK > N) { \
            if (tid < N) { \
                dlA00[tid] += dlA00[tid + N]; \
                dlA01[tid] += dlA01[tid + N]; \
                dlA10[tid] += dlA10[tid + N]; \
                dlA11[tid] += dlA11[tid + N]; \
                dlT0[tid] += dlT0[tid + N]; \
                dlT1[tid] += dlT1[tid + N]; \
                losses[tid] += losses[tid + N]; \
            } \
            __syncthreads(); \
        }
        REDUCE_BLOCK(512)
        REDUCE_BLOCK(256)
        REDUCE_BLOCK(128)
        REDUCE_BLOCK(64)
        REDUCE_BLOCK(32)
        REDUCE_BLOCK(16)
        REDUCE_BLOCK(8)
        REDUCE_BLOCK(4)
        REDUCE_BLOCK(2)
        REDUCE_BLOCK(1)
        if (tid == 0) {
            atomicAdd(&gAn[0], dlA00[0]);
            atomicAdd(&gAn[1], dlA01[0]);
            atomicAdd(&gAn[2], dlA10[0]);
            atomicAdd(&gAn[3], dlA11[0]);
            atomicAdd(&gTn[0], dlT0[0]);
            atomicAdd(&gTn[1], dlT1[0]);
            sumloss += losses[0];
        }

        if (!broadcast_I)
            In += nxy;
        if (!broadcast_J)
            Jn += nxy;
        An += 4;
        Tn += 2;
        if (use_contrast) {
            Bn += 1;
            Cn += 1;
        }
        gAn += 4;
        gTn += 2;
    }
    // record sum loss for this block
    if (tid == 0) {
        atomicAdd(loss, sumloss);
    }
}

template<BackgroundStrategy backgroundStrategy, int broadcast_I, int use_contrast>
inline __device__
void
interp_image_affine_kernel_2d(Real* out,
        Real* I, Real* A, Real* T,
        Real* B, Real* C,
        int nn, int nx, int ny) {
    int blockstartx = blockDim.x * blockIdx.x;
    int blockstarty = blockDim.y * blockIdx.y;
    int i = blockstartx + threadIdx.x;
    int j = blockstarty + threadIdx.y;
    if (i >= nx || j >= ny) return;
    // get a linear thread index to index into the shared arrays
    int nxy = nx*ny;
    const Real* In = I; // pointer to current vector field v
    Real* outn = out; // pointer to current vector field v
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    const Real* Bn = B;
    const Real* Cn = C;
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-oy;
    Real hx, hy;
    Real Inx;
    for (int n=0; n < nn; ++n) {
        // apply affine transform to map i, j to lookup point
        hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
        hy = An[2]*fi + An[3]*fj + Tn[1] + oy;
        Inx = biLerp<backgroundStrategy>(In,
            hx, hy,
            nx, ny,
            0.f);
        if (use_contrast) {
            Inx = Bn[0]*Inx + Cn[0];
        }
        outn[ix] = Inx;

        if (!broadcast_I)
            In += nxy;
        outn += nxy;
        An += 4;
        Tn += 2;
        if (use_contrast) {
            Bn++;
            Cn++;
        }
    }
}

template<BackgroundStrategy backgroundStrategy, int write_weights>
inline __device__
void
splat_image_affine_kernel_2d(Real* out, Real* w,
        Real* I, Real* A, Real* T,
        int nn, int nx, int ny) {
    int blockstartx = blockDim.x * blockIdx.x;
    int blockstarty = blockDim.y * blockIdx.y;
    int i = blockstartx + threadIdx.x;
    int j = blockstarty + threadIdx.y;
    if (i >= nx || j >= ny) return;
    // get a linear thread index to index into the shared arrays
    int nxy = nx*ny;
    const Real* In = I; // pointer to current vector field v
    Real* outn = out; // pointer to current vector field v
    Real* wn = w; // pointer to current vector field v
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-oy;
    Real hx, hy;
    for (int n=0; n < nn; ++n) {
        // apply affine transform to map i, j to lookup point
        hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
        hy = An[2]*fi + An[3]*fj + Tn[1] + oy;

        atomicSplat<backgroundStrategy, write_weights>(outn, wn,
            In[ix], hx, hy, nx, ny);

        In += nxy;
        outn += nxy;
        An += 4;
        Tn += 2;
    }
}

extern "C" {
    __global__ void affine_grad_2d(Real* gA, Real* gT, Real* loss,
            Real* I, Real* J, Real* A, Real* T,
            int nn, int nx, int ny) {
        affine_grad_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0, 0>(
            gA, gT, loss, I, J, A, T, NULL, NULL, nn, nx, ny);
    }
    __global__ void affine_grad_contrast_2d(Real* gA, Real* gT, Real* loss,
            Real* I, Real* J, Real* A, Real* T, Real* B, Real* C,
            int nn, int nx, int ny) {
        affine_grad_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0, 1>(
            gA, gT, loss, I, J, A, T, B, C, nn, nx, ny);
    }
    __global__ void affine_grad_bcastI_2d(Real* gA, Real* gT, Real* loss,
            Real* I, Real* J, Real* A, Real* T,
            int nn, int nx, int ny) {
        affine_grad_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0, 0>(
            gA, gT, loss, I, J, A, T, NULL, NULL, nn, nx, ny);
    }
    __global__ void affine_grad_bcastI_contrast_2d(Real* gA, Real* gT, Real* loss,
            Real* I, Real* J, Real* A, Real* T, Real* B, Real* C,
            int nn, int nx, int ny) {
        affine_grad_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0, 1>(
            gA, gT, loss, I, J, A, T, B, C, nn, nx, ny);
    }
    __global__ void interp_image_affine_2d(Real* out,
            Real* I, Real* A, Real* T,
            int nn, int nx, int ny) {
        interp_image_affine_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0>(
            out, I, A, T, NULL, NULL, nn, nx, ny);
    }
    __global__ void interp_image_affine_contrast_2d(Real* out,
            Real* I, Real* A, Real* T, Real* B, Real* C,
            int nn, int nx, int ny) {
        interp_image_affine_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 1>(
            out, I, A, T, B, C, nn, nx, ny);
    }
    __global__ void interp_image_affine_bcastI_2d(Real* out,
            Real* I, Real* A, Real* T,
            int nn, int nx, int ny) {
        interp_image_affine_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0>(
            out, I, A, T, NULL, NULL, nn, nx, ny);
    }
    __global__ void interp_image_affine_bcastI_contrast_2d(Real* out,
            Real* I, Real* A, Real* T, Real* B, Real* C,
            int nn, int nx, int ny) {
        interp_image_affine_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 1>(
            out, I, A, T, B, C, nn, nx, ny);
    }
    __global__ void splat_image_affine_2d(Real* out, Real* w,
            Real* I, Real* A, Real* T,
            int nn, int nx, int ny) {
        splat_image_affine_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1>(
            out, w, I, A, T, nn, nx, ny);
    }
    __global__ void splat_image_affine_noweights_2d(Real* out,
            Real* I, Real* A, Real* T,
            int nn, int nx, int ny) {
        splat_image_affine_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0>(
            out, NULL, I, A, T, nn, nx, ny);
    }
}
''', extra_nvcc_flags=[
        '-DDEFAULT_BACKGROUND_STRATEGY=BACKGROUND_STRATEGY_CLAMP'])
affine_grad_2d = mod.func("affine_grad_2d")
affine_grad_contrast_2d = mod.func("affine_grad_contrast_2d")
affine_grad_bcastI_2d = mod.func("affine_grad_bcastI_2d")
affine_grad_bcastI_contrast_2d = mod.func("affine_grad_bcastI_contrast_2d")
interp_image_affine_2d = mod.func("interp_image_affine_2d")
interp_image_affine_contrast_2d = mod.func("interp_image_affine_contrast_2d")
interp_image_affine_bcastI_2d = mod.func("interp_image_affine_bcastI_2d")
interp_image_affine_bcastI_contrast_2d = mod.func("interp_image_affine_bcastI_contrast_2d")
splat_image_affine_2d = mod.func("splat_image_affine_2d")
splat_image_affine_noweights_2d = mod.func("splat_image_affine_noweights_2d")
