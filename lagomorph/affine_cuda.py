# -*- mode: cuda -*-
# vi: set syntax=cuda:
from .cudamod import CudaModule
mod = CudaModule('''
#include "interp.cuh"
#include "diff.cuh"
#include "defs.cuh"

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

// This function computes the gradient and Hessian of the sum of squared error
// image match term after an affine transformation. gAt should be an array of
// Reals of length six times the number of blocks in the grid.
//
// The output Hessian, H, is symmetric of size 6x6, but should be allocated as
// an n by 21 matrix
// The entries are as follows:
// 
// H = HA00A00 HA00A01 HA00A10 HA00A11 HA00T0 HA00T1
//             HA01A01 HA01A10 HA01A11 HA01T0 HA01T1
//                     HA10A10 HA10A11 HA10T0 HA10T1
//                             HA11A11 HA11T0 HA11T1
//                                      HT0T0  HT0T1
//                                             HT1T1
//   =      H0      H1      H2      H3     H4     H5
//                  H6      H7      H8     H9    H10
//                         H11     H12    H13    H14
//                                 H15    H16    H17
//                                        H18    H19
//                                               H20
template<BackgroundStrategy backgroundStrategy, int broadcast_I, int broadcast_J, int use_contrast>
inline __device__
void
affine_grad_hessian_kernel_2d(Real* gA, Real* gT,
        Real* H, // Hessian wrt flattened A concatenated with T (6x6 symmetric)
        Real* loss, // outputs
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
#define THREADS_PER_BLOCK_HESSIAN 12*32
    __shared__ Real dlA00[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real dlA01[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real dlA10[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real dlA11[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real dlT0[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real dlT1[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA00A00[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA00A01[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA00A10[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA00A11[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA01A01[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA01A10[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA01A11[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA10A10[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA10A11[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA11A11[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA00T0[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA01T0[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA10T0[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA11T0[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA00T1[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA01T1[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA10T1[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HA11T1[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HT0T0[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HT0T1[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real HT1T1[THREADS_PER_BLOCK_HESSIAN];
    __shared__ Real losses[THREADS_PER_BLOCK_HESSIAN];
    int nxy = nx*ny;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    const Real* In = I; // pointer to current vector field v
    const Real* Jn = J; // pointer to current vector field v
    Real* gAn = gA; // pointer to current vector field v
    Real* gTn = gT; // pointer to current vector field v
    Real* Hn = H;
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    const Real* Bn = B;
    const Real* Cn = C;
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-ox;
    Real diff, gx, gy, hx, hy, Hxy;
    Real sumloss = 0.0f;
    for (int n=0; n < nn; ++n) {
        if (i >= nx || j >= ny) {
            dlA00[tid] = 0;
            dlA01[tid] = 0;
            dlA10[tid] = 0;
            dlA11[tid] = 0;
            dlT0[tid] = 0;
            dlT1[tid] = 0;
            // Hessian
            HA00A00[tid] = 0;
            HA00A01[tid] = 0;
            HA00A10[tid] = 0;
            HA00A11[tid] = 0;
            HA01A01[tid] = 0;
            HA01A10[tid] = 0;
            HA01A11[tid] = 0;
            HA10A10[tid] = 0;
            HA10A11[tid] = 0;
            HA11A11[tid] = 0;

            HA00T0[tid] = 0;
            HA01T0[tid] = 0;
            HA10T0[tid] = 0;
            HA11T0[tid] = 0;
            HA00T1[tid] = 0;
            HA01T1[tid] = 0;
            HA10T1[tid] = 0;
            HA11T1[tid] = 0;

            HT0T0[tid] = 0;
            HT0T1[tid] = 0;
            HT1T1[tid] = 0;
            losses[tid] = 0;
        } else {
            // apply affine transform to map i, j to lookup point
            hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
            hy = An[2]*fi + An[3]*fj + Tn[1] + oy;
            // get interp value and gradient at lookup point
            biLerp_grad_hessian<backgroundStrategy>(diff, gx, gy,
                Hxy,
                In,
                hx, hy,
                nx, ny,
                0.f);
            if (use_contrast) {
                diff = Bn[0]*diff + Cn[0];
                gx *= Bn[0];
                gy *= Bn[0];
                // Use fact that Hessian of image is zero on diagonal
                //Hxx *= Bn[0];
                Hxy *= Bn[0];
                //yy *= Bn[0];
            }
            // use Jn point to compute image difference at this point
            diff -= Jn[ix];
            gx *= diff; // save a few multiplies by premultiplying
            gy *= diff;
            // compute the outer product terms that will be summed
            losses[tid] = diff*diff;
            // gradient
            dlA00[tid] = gx*fi;
            dlA01[tid] = gx*fj;
            dlA10[tid] = gy*fi;
            dlA11[tid] = gy*fj;
            dlT0[tid] = gx;
            dlT1[tid] = gy;
            // Hessian. Note that zero elements of Hessian are commented out
            HA00A00[tid] = gx*fi*gx*fi;// + diff*Hxx*fi*fi;
            HA00A01[tid] = gx*fi*gx*fj;// + diff*Hxx*fi*fj;
            HA00A10[tid] = gx*fi*gy*fi + diff*Hxy*fi*fi;
            HA00A11[tid] = gx*fi*gy*fj + diff*Hxy*fi*fj;
            HA01A01[tid] = gx*fj*gx*fj;// + diff*Hxx*fj*fj;
            HA01A10[tid] = gx*fj*gy*fi + diff*Hxy*fj*fi;
            HA01A11[tid] = gx*fj*gy*fj + diff*Hxy*fj*fj;
            HA10A10[tid] = gy*fi*gy*fi;// + diff*Hyy*fi*fi;
            HA10A11[tid] = gy*fi*gy*fj;// + diff*Hyy*fi*fj;
            HA11A11[tid] = gy*fj*gy*fj;// + diff*Hyy*fj*fj;

            HA00T0[tid] = gx*fi*gx;// + diff*Hxx*fi;
            HA01T0[tid] = gx*fj*gx;// + diff*Hxx*fj;
            HA10T0[tid] = gy*fi*gx + diff*Hxy*fi;
            HA11T0[tid] = gy*fj*gx + diff*Hxy*fj;
            HA00T1[tid] = gx*fi*gy + diff*Hxy*fi;
            HA01T1[tid] = gx*fj*gy + diff*Hxy*fj;
            HA10T1[tid] = gy*fi*gy;// + diff*Hyy*fi;
            HA11T1[tid] = gy*fj*gy;// + diff*Hyy*fj;

            HT0T0[tid] = gx*gx;// + diff*Hxx;
            HT0T1[tid] = gx*gy + diff*Hxy;
            HT1T1[tid] = gy*gy;// + diff*Hyy;
        }

        // reduce this block
        __syncthreads();
        static_assert(THREADS_PER_BLOCK_HESSIAN <= 1024, "THREADS_PER_BLOCK_HESSIAN > 1024 not supported");
        // ensure counterpart in second half of arrays is not outside
        // pixel domain
#define REDUCE_BLOCK_HESSIAN(N) \
	if (THREADS_PER_BLOCK_HESSIAN > N) { \
            if (tid < N && tid+N < THREADS_PER_BLOCK_HESSIAN) { \
                dlA00[tid] += dlA00[tid + N]; \
                dlA01[tid] += dlA01[tid + N]; \
                dlA10[tid] += dlA10[tid + N]; \
                dlA11[tid] += dlA11[tid + N]; \
                dlT0[tid] += dlT0[tid + N]; \
                dlT1[tid] += dlT1[tid + N]; \
                HA00A00[tid] += HA00A00[tid + N]; \
                HA00A01[tid] += HA00A01[tid + N]; \
                HA00A10[tid] += HA00A10[tid + N]; \
                HA00A11[tid] += HA00A11[tid + N]; \
                HA01A01[tid] += HA01A01[tid + N]; \
                HA01A10[tid] += HA01A10[tid + N]; \
                HA01A11[tid] += HA01A11[tid + N]; \
                HA10A10[tid] += HA10A10[tid + N]; \
                HA10A11[tid] += HA10A11[tid + N]; \
                HA11A11[tid] += HA11A11[tid + N]; \
                HA00T0[tid] += HA00T0[tid + N]; \
                HA01T0[tid] += HA01T0[tid + N]; \
                HA10T0[tid] += HA10T0[tid + N]; \
                HA11T0[tid] += HA11T0[tid + N]; \
                HA00T1[tid] += HA00T1[tid + N]; \
                HA01T1[tid] += HA01T1[tid + N]; \
                HA10T1[tid] += HA10T1[tid + N]; \
                HA11T1[tid] += HA11T1[tid + N]; \
                HT0T0[tid] += HT0T0[tid + N]; \
                HT0T1[tid] += HT0T1[tid + N]; \
                HT1T1[tid] += HT1T1[tid + N]; \
                losses[tid] += losses[tid + N]; \
            } \
            __syncthreads(); \
        }
        REDUCE_BLOCK_HESSIAN(512)
        REDUCE_BLOCK_HESSIAN(256)
        REDUCE_BLOCK_HESSIAN(128)
        REDUCE_BLOCK_HESSIAN(64)
        REDUCE_BLOCK_HESSIAN(32)
        REDUCE_BLOCK_HESSIAN(16)
        REDUCE_BLOCK_HESSIAN(8)
        REDUCE_BLOCK_HESSIAN(4)
        REDUCE_BLOCK_HESSIAN(2)
        REDUCE_BLOCK_HESSIAN(1)
        if (tid == 0) {
            atomicAdd(&gAn[0], dlA00[0]);
            atomicAdd(&gAn[1], dlA01[0]);
            atomicAdd(&gAn[2], dlA10[0]);
            atomicAdd(&gAn[3], dlA11[0]);
            atomicAdd(&gTn[0],  dlT0[0]);
            atomicAdd(&gTn[1],  dlT1[0]);
            atomicAdd( &Hn[0], HA00A00[0]);
            atomicAdd( &Hn[1], HA00A01[0]);
            atomicAdd( &Hn[2], HA00A10[0]);
            atomicAdd( &Hn[3], HA00A11[0]);
            atomicAdd( &Hn[4], HA00T0[0]);
            atomicAdd( &Hn[5], HA00T1[0]);
            atomicAdd( &Hn[6], HA01A01[0]);
            atomicAdd( &Hn[7], HA01A10[0]);
            atomicAdd( &Hn[8], HA01A11[0]);
            atomicAdd( &Hn[9], HA01T0[0]);
            atomicAdd(&Hn[10], HA01T1[0]);
            atomicAdd(&Hn[11], HA10A10[0]);
            atomicAdd(&Hn[12], HA10A11[0]);
            atomicAdd(&Hn[13], HA10T0[0]);
            atomicAdd(&Hn[14], HA10T1[0]);
            atomicAdd(&Hn[15], HA11A11[0]);
            atomicAdd(&Hn[16], HA11T0[0]);
            atomicAdd(&Hn[17], HA11T1[0]);
            atomicAdd(&Hn[18], HT0T0[0]);
            atomicAdd(&Hn[19], HT0T1[0]);
            atomicAdd(&Hn[20], HT1T1[0]);
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
        Hn += 21;
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
        const Real* I, const Real* A, const Real* T,
        const Real* B, const Real* C,
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

template<BackgroundStrategy backgroundStrategy, int use_contrast>
__device__
void
affine_atlas_jacobi_kernel(Real* sse, Real* num, Real* denom,
        const Real* I, const Real* J, const Real* A, const Real* T,
        const Real* B, const Real* C,
        int nn, int nx, int ny) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    // thread number within this block
    int tid = threadIdx.x*blockDim.y + threadIdx.y;
    __shared__ Real sses[MAX_THREADS_PER_BLOCK];
    int nxy = nx*ny;
    const Real* Jn = J; // pointer to current image J
    const Real* An = A; // pointer to current matrix A
    const Real* Tn = T; // pointer to current vector T
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-oy;
    Real hx, hy;
    Real sset = 0.0f;
    if (i < nx && j < ny) {
        for (int n=0; n < nn; ++n) {
            // apply affine transform to map i, j to lookup point
            hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
            hy = An[2]*fi + An[3]*fj + Tn[1] + oy;

            if (use_contrast)
                sset += atlas_jacobi_point<backgroundStrategy>(num, denom, I, Jn,
                    B[n], C[n], i, j, hx, hy, nx, ny);
            else
                sset += atlas_jacobi_point<backgroundStrategy>(num, denom, I, Jn,
                    (Real)1.0, (Real)0.0, i, j, hx, hy, nx, ny);

            Jn += nxy;
            An += 4;
            Tn += 2;
        }
    }
    sses[tid] = sset;
    // reduce this block of sses
    __syncthreads();
    static_assert(MAX_THREADS_PER_BLOCK <= 1024, "MAX_THREADS_PER_BLOCK > 1024 not supported");
    // ensure counterpart in second half of arrays is not outside
    // pixel domain
#define REDUCE_SSE(N) \
    if (MAX_THREADS_PER_BLOCK > N) { \
        if (tid < N) { \
            sses[tid] += sses[tid + N]; \
        } \
        __syncthreads(); \
    }
    REDUCE_SSE(512)
    REDUCE_SSE(256)
    REDUCE_SSE(128)
    REDUCE_SSE(64)
    REDUCE_SSE(32)
    REDUCE_SSE(16)
    REDUCE_SSE(8)
    REDUCE_SSE(4)
    REDUCE_SSE(2)
    REDUCE_SSE(1)
    if (tid == 0) {
        atomicAdd(sse, sses[0]);
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
    __global__ void affine_grad_hessian_2d(Real* gA, Real* gT,
            Real* H, Real* loss,
            Real* I, Real* J, Real* A, Real* T,
            int nn, int nx, int ny) {
        affine_grad_hessian_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0, 0>(
            gA, gT, H, loss, I, J, A, T, NULL, NULL, nn, nx, ny);
    }
    __global__ void affine_grad_hessian_contrast_2d(Real* gA, Real* gT,
            Real* H, Real* loss,
            Real* I, Real* J, Real* A, Real* T, Real* B, Real* C,
            int nn, int nx, int ny) {
        affine_grad_hessian_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0, 0, 1>(
            gA, gT, H, loss, I, J, A, T, B, C, nn, nx, ny);
    }
    __global__ void affine_grad_hessian_bcastI_2d(Real* gA, Real* gT,
            Real* H, Real* loss,
            Real* I, Real* J, Real* A, Real* T,
            int nn, int nx, int ny) {
        affine_grad_hessian_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0, 0>(
            gA, gT, H, loss, I, J, A, T, NULL, NULL, nn, nx, ny);
    }
    __global__ void affine_grad_hessian_bcastI_contrast_2d(Real* gA, Real* gT,
            Real* H, Real* loss,
            Real* I, Real* J, Real* A, Real* T, Real* B, Real* C,
            int nn, int nx, int ny) {
        affine_grad_hessian_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1, 0, 1>(
            gA, gT, H, loss, I, J, A, T, B, C, nn, nx, ny);
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
    __global__ void affine_atlas_jacobi_2d(Real* sse, Real* num, Real* denom,
        const Real* I, const Real* J, const Real* A, const Real* T,
        int nn, int nx, int ny) {
        affine_atlas_jacobi_kernel<DEFAULT_BACKGROUND_STRATEGY, 0>(
            sse, num, denom, I, J, A, T, (Real*)NULL, (Real*)NULL, nn, nx, ny);
    }
    __global__ void affine_atlas_jacobi_contrast_2d(Real* sse, Real* num, Real* denom,
        const Real* I, const Real* J, const Real* A, const Real* T,
        const Real* B, const Real* C,
        int nn, int nx, int ny) {
        affine_atlas_jacobi_kernel<DEFAULT_BACKGROUND_STRATEGY, 1>(
            sse, num, denom, I, J, A, T, B, C, nn, nx, ny);
    }
}
''', extra_nvcc_flags=[
        '-DDEFAULT_BACKGROUND_STRATEGY=BACKGROUND_STRATEGY_CLAMP'])
affine_grad_2d = mod.func("affine_grad_2d")
affine_grad_contrast_2d = mod.func("affine_grad_contrast_2d")
affine_grad_bcastI_2d = mod.func("affine_grad_bcastI_2d")
affine_grad_bcastI_contrast_2d = mod.func("affine_grad_bcastI_contrast_2d")
affine_grad_hessian_2d = mod.func("affine_grad_hessian_2d")
affine_grad_hessian_contrast_2d = mod.func("affine_grad_hessian_contrast_2d")
affine_grad_hessian_bcastI_2d = mod.func("affine_grad_hessian_bcastI_2d")
affine_grad_hessian_bcastI_contrast_2d = mod.func("affine_grad_hessian_bcastI_contrast_2d")
interp_image_affine_2d = mod.func("interp_image_affine_2d")
interp_image_affine_contrast_2d = mod.func("interp_image_affine_contrast_2d")
interp_image_affine_bcastI_2d = mod.func("interp_image_affine_bcastI_2d")
interp_image_affine_bcastI_contrast_2d = mod.func("interp_image_affine_bcastI_contrast_2d")
splat_image_affine_2d = mod.func("splat_image_affine_2d")
splat_image_affine_noweights_2d = mod.func("splat_image_affine_noweights_2d")
affine_atlas_jacobi_2d = mod.func("affine_atlas_jacobi_2d")
affine_atlas_jacobi_contrast_2d = mod.func("affine_atlas_jacobi_contrast_2d")
