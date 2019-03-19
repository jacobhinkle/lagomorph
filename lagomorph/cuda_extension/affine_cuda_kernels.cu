#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "atomic.cuh"
#include "interp.cuh"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define INTERP_BACKWARD_THREADS_X 16
#define INTERP_BACKWARD_THREADS_Y 32
const dim3 INTERP_BACKWARD_THREADS(INTERP_BACKWARD_THREADS_X,
    INTERP_BACKWARD_THREADS_Y);
const auto INTERP_BACKWARD_NUM_THREADS =
    INTERP_BACKWARD_THREADS_X*INTERP_BACKWARD_THREADS_Y;


template<typename Real, bool broadcast_I>
__global__ void affine_interp_kernel_2d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        const Real* __restrict__ A, const Real* __restrict__ T,
        const Real* __restrict__ B, const Real* __restrict__ C,
        size_t nn, size_t nc, size_t nx, size_t ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    // get a linear thread index to index into the shared arrays
    const int nxy = nx*ny;
    const Real* In = I; // pointer to current vector field v
    Real* outn = out; // pointer to current vector field v
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
    Real Inx;
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) In = I;
        // apply affine transform to map i, j to lookup point
        hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
        hy = An[2]*fi + An[3]*fj + Tn[1] + oy;
        for (int c=0; c < nc; ++c) {
            Inx = biLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                hx, hy,
                nx, ny);
            outn[ix] = Inx;
            outn += nxy;
            In += nxy;
        }
        An += 4;
        Tn += 2;
    }
}

template<typename Real, bool broadcast_I>
__global__ void affine_interp_kernel_3d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        const Real* __restrict__ A, const Real* __restrict__ T,
        const Real* __restrict__ B, const Real* __restrict__ C,
        size_t nn, size_t nc, size_t nx, size_t ny, size_t nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    // get a linear thread index to index into the shared arrays
    const int nxyz = nx*ny*nz;
    const Real* In = I; // pointer to current vector field v
    Real* outn = out; // pointer to current vector field v
    const Real* An = A; // pointer to current vector field v
    const Real* Tn = T; // pointer to current vector field v
    // index of current output point (first component. add nxy for second)
    int ix = (i*ny + j)*nz;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real oz = .5*static_cast<Real>(nz-1);
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-oy;
    Real hx, hy, hz;
    Real Inx;
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) In = I;
        else In = I + n*nc*nxyz;
        for (int c=0; c < nc; ++c) {
            for (int k=0; k < nz; ++k) {
                Real fk=static_cast<Real>(k)-oz;
                hx = An[0]*fi + An[1]*fj + An[2]*fk + Tn[0] + ox;
                hy = An[3]*fi + An[4]*fj + An[5]*fk + Tn[1] + oy;
                hz = An[6]*fi + An[7]*fj + An[8]*fk + Tn[2] + oz;
                Inx = triLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                    hx, hy, hz,
                    nx, ny, nz);
                outn[ix + k] = Inx;
            }
            In += nxyz;
            outn += nxyz;
        }
        An += 9;
        Tn += 3;
    }
}

at::Tensor affine_interp_cuda_forward(
    at::Tensor I,
    at::Tensor A,
    at::Tensor T) {
    AT_ASSERTM(A.size(0) == T.size(0), "A and T must have same first dimension")
    auto d = I.dim() - 2;
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional affine interpolation is supported")

    const dim3 threads(16, 32);
    const dim3 blocks((I.size(2) + threads.x - 1) / threads.x,
                    (I.size(3) + threads.y - 1) / threads.y);

    const bool broadcast_I = I.size(0) == 1 && A.size(0) > 1;

    at::Tensor Itx;

    if (d == 2) {
        Itx = at::zeros({A.size(0), I.size(1), I.size(2), I.size(3)}, I.type());
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_cuda_forward", ([&] {
            affine_interp_kernel_2d<scalar_t, broadcastI><<<blocks, threads>>>(
                Itx.data<scalar_t>(),
                I.data<scalar_t>(),
                A.data<scalar_t>(),
                T.data<scalar_t>(),
                NULL,
                NULL,
                A.size(0),
                I.size(1),
                I.size(2),
                I.size(3));
            }));
        }));
    } else {
        Itx = at::zeros({A.size(0), I.size(1), I.size(2), I.size(3), I.size(4)}, I.type());
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_cuda_forward", ([&] {
            affine_interp_kernel_3d<scalar_t, broadcastI><<<blocks, threads>>>(
                Itx.data<scalar_t>(),
                I.data<scalar_t>(),
                A.data<scalar_t>(),
                T.data<scalar_t>(),
                NULL,
                NULL,
                A.size(0),
                I.size(1),
                I.size(2),
                I.size(3),
                I.size(4));
            }));
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return Itx;
}

template<typename Real, bool broadcast_I,
    bool need_I, bool need_A, bool need_T>
__global__ void affine_interp_kernel_backward_2d(
        Real* __restrict__ d_I,
        Real* __restrict__ d_A,
        Real* __restrict__ d_T,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ I,
        const Real* __restrict__ A,
        const Real* __restrict__ T,
        const Real* __restrict__ B,
        const Real* __restrict__ C,
        size_t nn, size_t nc, size_t nx, size_t ny) {
    const auto ii = threadIdx.x;
    const auto jj = threadIdx.y;
    const auto n = blockIdx.y;
    const auto c = blockIdx.z;
    // get a linear thread index to index into the shared arrays
    int tid = threadIdx.x*blockDim.y + threadIdx.y;
    const auto THREADS_PER_BLOCK = INTERP_BACKWARD_NUM_THREADS;
    __shared__ Real dlA00[THREADS_PER_BLOCK];
    __shared__ Real dlA01[THREADS_PER_BLOCK];
    __shared__ Real dlA10[THREADS_PER_BLOCK];
    __shared__ Real dlA11[THREADS_PER_BLOCK];
    __shared__ Real dlT0[THREADS_PER_BLOCK];
    __shared__ Real dlT1[THREADS_PER_BLOCK];
    Real dlA00i = 0;
    Real dlA01i = 0;
    Real dlA10i = 0;
    Real dlA11i = 0;
    Real dlT0i = 0;
    Real dlT1i = 0;
    const size_t cxy = c*nx*ny;
    const size_t nncxy = n*nc*nx*ny;
    const Real* gon = grad_out + nncxy + cxy;
    Real* d_In = d_I + cxy;
    const Real* In = I +cxy;
    if (!broadcast_I) {
        In += nncxy;
        d_In += nncxy;
    }
    const Real* An = A + 4*n;
    const Real* Tn = T + 2*n;
    Real* d_An = d_A + 4*n;
    Real* d_Tn = d_T + 2*n;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real fi, fj;
    Real diff, interpval, gx, gy, hx, hy;
    for (unsigned int i=ii; i < nx; i += blockDim.x) {
        fi=static_cast<Real>(i)-ox;
        for (unsigned int j=jj, ix = i*ny+jj; j < ny; j += blockDim.y, ix += blockDim.y) {
            fj=static_cast<Real>(j)-oy;

            // apply affine transform to map i, j to lookup point
            hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
            hy = An[2]*fi + An[3]*fj + Tn[1] + oy;

            // what for L2 loss is diff now given as grad_out
            diff = gon[ix];

            // derivative with respect to input is just splat of grad_out
            if (need_I)
                atomicSplat<Real, DEFAULT_BACKGROUND_STRATEGY, false>(d_In, NULL,
                    diff, hx, hy, nx, ny);
            // get interp value and gradient at lookup point
            if (need_A || need_T) {
                biLerp_grad<Real, DEFAULT_BACKGROUND_STRATEGY>(interpval, gx, gy,
                    In,
                    hx, hy,
                    nx, ny);
                gx *= diff; // save a few multiplies by premultiplying
                gy *= diff;
                // compute the outer product terms that will be summed
                if (need_A) {
                    dlA00i += gx*fi;
                    dlA01i += gx*fj;
                    dlA10i += gy*fi;
                    dlA11i += gy*fj;
                }
                if (need_T) {
                    dlT0i += gx;
                    dlT1i += gy;
                }
            }
        }
    }
    if (need_A || need_T) {
        if (need_A) {
            dlA00[tid] = dlA00i;
            dlA01[tid] = dlA01i;
            dlA10[tid] = dlA10i;
            dlA11[tid] = dlA11i;
        }
        if (need_T) {
            dlT0[tid] = dlT0i;
            dlT1[tid] = dlT1i;
        }

        // reduce this block
        __syncthreads();
        static_assert(THREADS_PER_BLOCK <= 1024, "THREADS_PER_BLOCK > 1024 not supported");
        // ensure counterpart in second half of arrays is not outside
        // pixel domain
#define AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(N) \
        if (THREADS_PER_BLOCK > N) { \
            if (tid < N) { \
                if (need_A) { \
                    dlA00[tid] += dlA00[tid + N]; \
                    dlA01[tid] += dlA01[tid + N]; \
                    dlA10[tid] += dlA10[tid + N]; \
                    dlA11[tid] += dlA11[tid + N]; \
                } \
                if (need_T) { \
                    dlT0[tid] += dlT0[tid + N]; \
                    dlT1[tid] += dlT1[tid + N]; \
                } \
            } \
            __syncthreads(); \
        }
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(512)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(256)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(128)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(64)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(32)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(16)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(8)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(4)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(2)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_2D(1)
        if (tid == 0) {
            if (nc == 1) {
                if (need_A) {
                    d_An[0] = dlA00[0];
                    d_An[1] = dlA01[0];
                    d_An[2] = dlA10[0];
                    d_An[3] = dlA11[0];
                }
                if (need_T) {
                    d_Tn[0] = dlT0[0];
                    d_Tn[1] = dlT1[0];
                }
            } else {
                if (need_A) {
                    atomicAdd(&d_An[0], dlA00[0]);
                    atomicAdd(&d_An[1], dlA01[0]);
                    atomicAdd(&d_An[2], dlA10[0]);
                    atomicAdd(&d_An[3], dlA11[0]);
                }
                if (need_T) {
                    atomicAdd(&d_Tn[0], dlT0[0]);
                    atomicAdd(&d_Tn[1], dlT1[0]);
                }
            }
        }
    }
}

template<typename Real, bool broadcast_I,
    bool need_I, bool need_A, bool need_T>
__global__ void affine_interp_kernel_backward_3d(
        Real* __restrict__ d_I,
        Real* __restrict__ d_A,
        Real* __restrict__ d_T,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ I,
        const Real* __restrict__ A,
        const Real* __restrict__ T,
        const Real* __restrict__ B,
        const Real* __restrict__ C,
        size_t nn, size_t nc, size_t nx, size_t ny, size_t nz) {
    const auto ii = threadIdx.x;
    const auto jj = threadIdx.y;
    const auto n = blockIdx.y;
    const auto c = blockIdx.z;
    // get a linear thread index to index into the shared arrays
    int tid = threadIdx.x*blockDim.y + threadIdx.y;
    const auto THREADS_PER_BLOCK = INTERP_BACKWARD_NUM_THREADS;
    __shared__ Real dlA00[THREADS_PER_BLOCK];
    __shared__ Real dlA01[THREADS_PER_BLOCK];
    __shared__ Real dlA02[THREADS_PER_BLOCK];
    __shared__ Real dlA10[THREADS_PER_BLOCK];
    __shared__ Real dlA11[THREADS_PER_BLOCK];
    __shared__ Real dlA12[THREADS_PER_BLOCK];
    __shared__ Real dlA20[THREADS_PER_BLOCK];
    __shared__ Real dlA21[THREADS_PER_BLOCK];
    __shared__ Real dlA22[THREADS_PER_BLOCK];
    __shared__ Real dlT0[THREADS_PER_BLOCK];
    __shared__ Real dlT1[THREADS_PER_BLOCK];
    __shared__ Real dlT2[THREADS_PER_BLOCK];
    Real dlA00i = 0;
    Real dlA01i = 0;
    Real dlA02i = 0;
    Real dlA10i = 0;
    Real dlA11i = 0;
    Real dlA12i = 0;
    Real dlA20i = 0;
    Real dlA21i = 0;
    Real dlA22i = 0;
    Real dlT0i = 0;
    Real dlT1i = 0;
    Real dlT2i = 0;
    const size_t cxyz = c*nx*ny*nz;
    const size_t nncxyz = n*nc*nx*ny*nz;
    const Real* gon = grad_out + nncxyz + cxyz;
    Real* d_In = d_I + cxyz;
    const Real* In = I +cxyz;
    if (!broadcast_I) {
        In += nncxyz;
        d_In += nncxyz;
    }
    const Real* An = A + 9*n;
    const Real* Tn = T + 3*n;
    Real* d_An = d_A + 9*n;
    Real* d_Tn = d_T + 3*n;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real oz = .5*static_cast<Real>(nz-1);
    Real fi, fj, fk;
    Real diff, interpval, gx, gy, gz, hx, hy, hz;
    for (unsigned int i=ii; i < nx; i += blockDim.x) {
        fi=static_cast<Real>(i)-ox;
        for (unsigned int j=jj; j < ny; j += blockDim.y) {
            fj=static_cast<Real>(j)-oy;
            for (unsigned int k=0, ix=(i*ny+j)*nz; k < nz; ++k, ix++) {
                fk=static_cast<Real>(k)-oz;

                // apply affine transform to map i, j to lookup point
                hx = An[0]*fi + An[1]*fj + An[2]*fk + Tn[0] + ox;
                hy = An[3]*fi + An[4]*fj + An[5]*fk + Tn[1] + oy;
                hz = An[6]*fi + An[7]*fj + An[8]*fk + Tn[2] + oz;

                // what for L2 loss is diff now given as grad_out
                diff = gon[ix];

                // derivative with respect to input is just splat of grad_out
                if (need_I)
                    atomicSplat<Real, DEFAULT_BACKGROUND_STRATEGY, false>(d_In, NULL,
                        diff, hx, hy, hz, nx, ny, nz);
                // get interp value and gradient at lookup point
                if (need_A || need_T) {
                    triLerp_grad<Real, DEFAULT_BACKGROUND_STRATEGY>(interpval,
                        gx, gy, gz,
                        In,
                        hx, hy, hz,
                        nx, ny, nz);
                    gx *= diff; // save a few multiplies by premultiplying
                    gy *= diff;
                    gz *= diff;
                    // compute the outer product terms that will be summed
                    if (need_A) {
                        dlA00i += gx*fi;
                        dlA01i += gx*fj;
                        dlA02i += gx*fk;
                        dlA10i += gy*fi;
                        dlA11i += gy*fj;
                        dlA12i += gy*fk;
                        dlA20i += gz*fi;
                        dlA21i += gz*fj;
                        dlA22i += gz*fk;
                    }
                    if (need_T) {
                        dlT0i += gx;
                        dlT1i += gy;
                        dlT2i += gz;
                    }
                }
            }
        }
    }
    if (need_A || need_T) {
        if (need_A) {
            dlA00[tid] = dlA00i;
            dlA01[tid] = dlA01i;
            dlA02[tid] = dlA02i;
            dlA10[tid] = dlA10i;
            dlA11[tid] = dlA11i;
            dlA12[tid] = dlA12i;
            dlA20[tid] = dlA20i;
            dlA21[tid] = dlA21i;
            dlA22[tid] = dlA22i;
        }
        if (need_T) {
            dlT0[tid] = dlT0i;
            dlT1[tid] = dlT1i;
            dlT2[tid] = dlT2i;
        }

        // reduce this block
        __syncthreads();
        static_assert(THREADS_PER_BLOCK <= 1024, "THREADS_PER_BLOCK > 1024 not supported");
        // ensure counterpart in second half of arrays is not outside
        // pixel domain
#define AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(N) \
        if (THREADS_PER_BLOCK > N) { \
            if (tid < N) { \
                if (need_A) { \
                    dlA00[tid] += dlA00[tid + N]; \
                    dlA01[tid] += dlA01[tid + N]; \
                    dlA02[tid] += dlA02[tid + N]; \
                    dlA10[tid] += dlA10[tid + N]; \
                    dlA11[tid] += dlA11[tid + N]; \
                    dlA12[tid] += dlA12[tid + N]; \
                    dlA20[tid] += dlA20[tid + N]; \
                    dlA21[tid] += dlA21[tid + N]; \
                    dlA22[tid] += dlA22[tid + N]; \
                } \
                if (need_T) { \
                    dlT0[tid] += dlT0[tid + N]; \
                    dlT1[tid] += dlT1[tid + N]; \
                    dlT2[tid] += dlT2[tid + N]; \
                } \
            } \
            __syncthreads(); \
        }
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(512)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(256)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(128)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(64)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(32)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(16)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(8)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(4)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(2)
        AFFINE_INTERP_BACKWARD_REDUCE_BLOCK_3D(1)
        if (tid == 0) {
            if (nc == 1) {
                if (need_A) {
                    d_An[0] = dlA00[0];
                    d_An[1] = dlA01[0];
                    d_An[2] = dlA02[0];
                    d_An[3] = dlA10[0];
                    d_An[4] = dlA11[0];
                    d_An[5] = dlA12[0];
                    d_An[6] = dlA20[0];
                    d_An[7] = dlA21[0];
                    d_An[8] = dlA22[0];
                }
                if (need_T) {
                    d_Tn[0] = dlT0[0];
                    d_Tn[1] = dlT1[0];
                    d_Tn[2] = dlT2[0];
                }
            } else {
                if (need_A) {
                    atomicAdd(&d_An[0], dlA00[0]);
                    atomicAdd(&d_An[1], dlA01[0]);
                    atomicAdd(&d_An[2], dlA02[0]);
                    atomicAdd(&d_An[3], dlA10[0]);
                    atomicAdd(&d_An[4], dlA11[0]);
                    atomicAdd(&d_An[5], dlA12[0]);
                    atomicAdd(&d_An[6], dlA20[0]);
                    atomicAdd(&d_An[7], dlA21[0]);
                    atomicAdd(&d_An[8], dlA22[0]);
                }
                if (need_T) {
                    atomicAdd(&d_Tn[0], dlT0[0]);
                    atomicAdd(&d_Tn[1], dlT1[0]);
                    atomicAdd(&d_Tn[2], dlT2[0]);
                }
            }
        }
    }
}

std::vector<at::Tensor> affine_interp_cuda_backward(
    at::Tensor grad_out,
    at::Tensor I,
    at::Tensor A,
    at::Tensor T,
    bool need_I,
    bool need_A,
    bool need_T) {
    AT_ASSERTM(I.size(1) == grad_out.size(1), "I and grad_out must have same number of channels")
    AT_ASSERTM(A.size(0) == T.size(0), "A and T must have same first dimension")
    auto d = I.dim() - 2;
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional affine interpolation is supported")

    // avoid allocating memory for gradients we don't need to compute
	auto d_I = need_I ? at::zeros_like(I) : at::zeros({0}, I.type());
	auto d_A = need_A ? at::zeros_like(A) : at::zeros({0}, A.type());
	auto d_T = need_T ? at::zeros_like(T) : at::zeros({0}, T.type());

    const auto threads = INTERP_BACKWARD_THREADS;
    const dim3 blocks(1, A.size(0), I.size(1));

    const bool broadcast_I = I.size(0) == 1 && grad_out.size(0) > 1;

    if (d == 2) {
        LAGOMORPH_DISPATCH_BOOL(need_I, needI, ([&] {
        LAGOMORPH_DISPATCH_BOOL(need_A, needA, ([&] {
        LAGOMORPH_DISPATCH_BOOL(need_T, needT, ([&] {
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_cuda_backward", ([&] {
            affine_interp_kernel_backward_2d<scalar_t, broadcastI, needI, needA, needT><<<blocks, threads>>>(
                d_I.data<scalar_t>(),
                d_A.data<scalar_t>(),
                d_T.data<scalar_t>(),
                grad_out.data<scalar_t>(),
                I.data<scalar_t>(),
                A.data<scalar_t>(),
                T.data<scalar_t>(),
                NULL,
                NULL,
                grad_out.size(0),
                grad_out.size(1),
                grad_out.size(2),
                grad_out.size(3));
            }));
        })); })); })); }));
    } else {
        LAGOMORPH_DISPATCH_BOOL(need_I, needI, ([&] {
        LAGOMORPH_DISPATCH_BOOL(need_A, needA, ([&] {
        LAGOMORPH_DISPATCH_BOOL(need_T, needT, ([&] {
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_cuda_backward", ([&] {
            affine_interp_kernel_backward_3d<scalar_t, broadcastI, needI, needA, needT><<<blocks, threads>>>(
                d_I.data<scalar_t>(),
                d_A.data<scalar_t>(),
                d_T.data<scalar_t>(),
                grad_out.data<scalar_t>(),
                I.data<scalar_t>(),
                A.data<scalar_t>(),
                T.data<scalar_t>(),
                NULL,
                NULL,
                grad_out.size(0),
                grad_out.size(1),
                grad_out.size(2),
                grad_out.size(3),
                grad_out.size(4));
            }));
        })); })); })); }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return {d_I, d_A, d_T};
}

template<typename Real>
__global__ void regrid_forward_kernel_2d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        int Nx, int Ny,
        Real Ox, Real Oy,
        Real Sx, Real Sy,
        size_t nn, size_t nc, size_t nx, size_t ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    const int Nxy = Nx*Ny;
    const int nxy = nx*ny;
    const Real* In = I;
    Real* outn = out;
    int ix = i*Ny + j;
    // center in new coordinates
    Real ox = .5*static_cast<Real>(Nx-1);
    Real oy = .5*static_cast<Real>(Ny-1);
    auto hx = (i-ox)*Sx + Ox;
    auto hy = (j-oy)*Sy + Oy;
    for (int n=0; n < nn; ++n) {
        for (int c=0; c < nc; ++c) {
            auto Inx = biLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                hx, hy,
                nx, ny);
            outn[ix] = Inx;
            outn += Nxy;
            In += nxy;
        }
    }
}

template<typename Real>
__global__ void regrid_forward_kernel_3d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        int Nx, int Ny, int Nz,
        Real Ox, Real Oy, Real Oz,
        Real Sx, Real Sy, Real Sz,
        size_t nn, size_t nc, size_t nx, size_t ny, size_t nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    const int Nxyz = Nx*Ny*Nz;
    const int nxyz = nx*ny*nz;
    const Real* In = I;
    Real* outn = out;
    int ix = (i*Ny + j)*Nz;
    // center in new coordinates
    Real ox = .5*static_cast<Real>(Nx-1);
    Real oy = .5*static_cast<Real>(Ny-1);
    Real oz = .5*static_cast<Real>(Nz-1);
    Real hx = (i-ox)*Sx + Ox;
    Real hy = (j-oy)*Sy + Oy;
    for (int n=0; n < nn; ++n) {
        for (int c=0; c < nc; ++c) {
            Real hz = Oz - oz*Sz;
            for (int k=0; k < Nz; ++k) {
                //Real hz = (k-oz)*Sz + Oz;
                outn[ix+k] = triLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                    hx, hy, hz,
                    nx, ny, nz);
                hz += Sz;
            }
            outn += Nxyz;
            In += nxyz;
        }
    }
}

at::Tensor regrid_forward(
    at::Tensor I,
    std::vector<int> shape,
    std::vector<double> origin,
    std::vector<double> spacing) {
    auto d = I.dim() - 2;
    CHECK_INPUT(I)
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional regridding is supported")
    AT_ASSERTM(shape.size() == d, "Shape should be vector of size d (not 2+d)")
    AT_ASSERTM(origin.size() == d, "Origin should be vector of size d (not 2+d)")
    AT_ASSERTM(spacing.size() == d, "Spacing should be vector of size d (not 2+d)")

    const dim3 threads(32, 32);
    const dim3 blocks((shape[0] + threads.x - 1) / threads.x,
                    (shape[1] + threads.y - 1) / threads.y);

    at::Tensor Itx;

    if (d == 2) {
        Itx = at::zeros({I.size(0), I.size(1), shape[0], shape[1]}, I.type());
        AT_DISPATCH_FLOATING_TYPES(I.type(), "regrid_forward", ([&] {
        regrid_forward_kernel_2d<scalar_t><<<blocks, threads>>>(
            Itx.data<scalar_t>(),
            I.data<scalar_t>(),
            shape[0], shape[1],
            origin[0], origin[1],
            spacing[0], spacing[1],
            I.size(0),
            I.size(1),
            I.size(2),
            I.size(3));
        }));
    } else {
        Itx = at::zeros({I.size(0), I.size(1), shape[0], shape[1], shape[2]}, I.type());
        AT_DISPATCH_FLOATING_TYPES(I.type(), "regrid_forward", ([&] {
        regrid_forward_kernel_3d<scalar_t><<<blocks, threads>>>(
            Itx.data<scalar_t>(),
            I.data<scalar_t>(),
            shape[0], shape[1], shape[2],
            origin[0], origin[1], origin[2],
            spacing[0], spacing[1], spacing[2],
            I.size(0),
            I.size(1),
            I.size(2),
            I.size(3),
            I.size(4));
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return Itx;
}

template<typename Real>
__global__ void regrid_backward_kernel_2d(
        Real* __restrict__ d_I,
        const Real* __restrict__ grad_out,
        size_t Nx, size_t Ny,
        Real Ox, Real Oy,
        Real Sx, Real Sy,
        size_t nn, size_t nc, size_t nx, size_t ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    const int Nxy = Nx*Ny;
    const int nxy = nx*ny;
    Real* d_In = d_I;
    const Real* gon = grad_out;
    int ix = i*Ny + j;
    // center in new coordinates
    Real ox = .5*static_cast<Real>(Nx-1);
    Real oy = .5*static_cast<Real>(Ny-1);
    auto hx = (i-ox)*Sx + Ox;
    auto hy = (j-oy)*Sy + Oy;
    for (int n=0; n < nn; ++n) {
        for (int c=0; c < nc; ++c) {
            atomicSplat<Real, DEFAULT_BACKGROUND_STRATEGY, false>(d_In, NULL,
                gon[ix], hx, hy, nx, ny);
            gon += Nxy;
            d_In += nxy;
        }
    }
}

template<typename Real>
__global__ void regrid_backward_kernel_3d(
        Real* __restrict__ d_I,
        const Real* __restrict__ grad_out,
        size_t Nx, size_t Ny, size_t Nz,
        Real Ox, Real Oy, Real Oz,
        Real Sx, Real Sy, Real Sz,
        size_t nn, size_t nc, size_t nx, size_t ny, size_t nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= Nx || j >= Ny) return;
    const int Nxyz = Nx*Ny*Nz;
    const int nxyz = nx*ny*nz;
    Real* d_In = d_I;
    const Real* gon = grad_out;
    int ix = (i*Ny + j)*Nz;
    // center in new coordinates
    Real ox = .5*static_cast<Real>(Nx-1);
    Real oy = .5*static_cast<Real>(Ny-1);
    Real oz = .5*static_cast<Real>(Nz-1);
    auto hx = (i-ox)*Sx + Ox;
    auto hy = (j-oy)*Sy + Oy;
    for (int n=0; n < nn; ++n) {
        for (int c=0; c < nc; ++c) {
            for (int k=0; k < Nz; ++k) {
                auto hz = (k-oz)*Sz + Oz;
                atomicSplat<Real, DEFAULT_BACKGROUND_STRATEGY, false>(d_In, NULL,
                    gon[ix+k], hx, hy, hz, nx, ny, nz);
            }
            gon += Nxyz;
            d_In += nxyz;
        }
    }
}

at::Tensor regrid_backward(
    at::Tensor grad_out,
    std::vector<int> inshape,
    std::vector<int> shape,
    std::vector<double> origin,
    std::vector<double> spacing) {
    auto d = grad_out.dim() - 2;
    CHECK_INPUT(grad_out)
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional regridding is supported")
    AT_ASSERTM(inshape.size() == d, "Input shape should be vector of size d (not 2+d)")
    AT_ASSERTM(shape.size() == d, "Shape should be vector of size d (not 2+d)")
    AT_ASSERTM(origin.size() == d, "Origin should be vector of size d (not 2+d)")
    AT_ASSERTM(spacing.size() == d, "Spacing should be vector of size d (not 2+d)")

    const dim3 threads(32, 32);
    const dim3 blocks((shape[0] + threads.x - 1) / threads.x,
                    (shape[1] + threads.y - 1) / threads.y);

    at::Tensor d_I;

    if (d == 2) {
        d_I = at::zeros({grad_out.size(0), grad_out.size(1), inshape[0], inshape[1]}, grad_out.type());
        AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "regrid_backward", ([&] {
        regrid_backward_kernel_2d<scalar_t><<<blocks, threads>>>(
            d_I.data<scalar_t>(),
            grad_out.data<scalar_t>(),
            shape[0], shape[1],
            origin[0], origin[1],
            spacing[0], spacing[1],
            d_I.size(0),
            d_I.size(1),
            inshape[0],
            inshape[1]);
        }));
    } else {
        d_I = at::zeros({grad_out.size(0), grad_out.size(1), inshape[0], inshape[1], inshape[2]}, grad_out.type());
        AT_DISPATCH_FLOATING_TYPES(grad_out.type(), "regrid_backward", ([&] {
        regrid_backward_kernel_3d<scalar_t><<<blocks, threads>>>(
            d_I.data<scalar_t>(),
            grad_out.data<scalar_t>(),
            shape[0], shape[1], shape[2],
            origin[0], origin[1], origin[2],
            spacing[0], spacing[1], spacing[2],
            d_I.size(0),
            d_I.size(1),
            inshape[0],
            inshape[1],
            inshape[2]);
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return d_I;
}

