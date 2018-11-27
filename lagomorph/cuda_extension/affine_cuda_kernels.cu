#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "atomic.cuh"
#include "interp.cuh"

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
    // initialize shmem
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
#define INTERP_BACKWARD_REDUCE_BLOCK(N) \
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
        INTERP_BACKWARD_REDUCE_BLOCK(512)
        INTERP_BACKWARD_REDUCE_BLOCK(256)
        INTERP_BACKWARD_REDUCE_BLOCK(128)
        INTERP_BACKWARD_REDUCE_BLOCK(64)
        INTERP_BACKWARD_REDUCE_BLOCK(32)
        INTERP_BACKWARD_REDUCE_BLOCK(16)
        INTERP_BACKWARD_REDUCE_BLOCK(8)
        INTERP_BACKWARD_REDUCE_BLOCK(4)
        INTERP_BACKWARD_REDUCE_BLOCK(2)
        INTERP_BACKWARD_REDUCE_BLOCK(1)
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

std::vector<at::Tensor> affine_interp_cuda_backward(
    at::Tensor grad_out,
    at::Tensor I,
    at::Tensor A,
    at::Tensor T,
    bool need_I,
    bool need_A,
    bool need_T) {
    AT_ASSERTM(I.size(1) == grad_out.size(1), "I and grad_out must have same number of channels")

    // avoid allocating memory for gradients we don't need to compute
	auto d_I = need_I ? at::zeros_like(I) : at::zeros({0}, I.type());
	auto d_A = need_A ? at::zeros_like(A) : at::zeros({0}, A.type());
	auto d_T = need_T ? at::zeros_like(T) : at::zeros({0}, T.type());

    const auto threads = INTERP_BACKWARD_THREADS;
    const dim3 blocks(1, A.size(0), I.size(1));

    const bool broadcast_I = I.size(0) == 1 && grad_out.size(0) > 1;

    LAGOMORPH_DISPATCH_BOOL(need_I, needI, ([&] {
    LAGOMORPH_DISPATCH_BOOL(need_A, needA, ([&] {
    LAGOMORPH_DISPATCH_BOOL(need_T, needT, ([&] {
    LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
        AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_cuda_backward", ([&] {
        affine_interp_kernel_backward_2d<scalar_t, broadcastI, false, needI, needA, needT><<<blocks, threads>>>(
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
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return {d_I, d_A, d_T};
}
