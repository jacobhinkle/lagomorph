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


template<typename Real, bool broadcast_I, bool use_contrast>
__global__ void affine_interp_image_kernel_2d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        const Real* __restrict__ A, const Real* __restrict__ T,
        const Real* __restrict__ B, const Real* __restrict__ C,
        size_t nn, size_t nx, size_t ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    // get a linear thread index to index into the shared arrays
    const int nxy = nx*ny;
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
        Inx = biLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
            hx, hy,
            nx, ny);
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

at::Tensor affine_interp_image_cuda_forward(
    at::Tensor I,
    at::Tensor A,
    at::Tensor T) {
  auto Itx = at::empty_like(I);

  const auto batch_size = I.size(0);

  const dim3 threads(32, 32);
  const dim3 blocks((I.size(1) + threads.x - 1) / threads.x,
                    (I.size(2) + threads.y - 1) / threads.y);

  AT_DISPATCH_FLOATING_TYPES(I.type(), "affine_interp_image_cuda_forward", ([&] {
    affine_interp_image_kernel_2d<scalar_t, false, false><<<blocks, threads>>>(
        Itx.data<scalar_t>(),
        I.data<scalar_t>(),
        A.data<scalar_t>(),
        T.data<scalar_t>(),
        NULL,
        NULL,
        batch_size,
        I.size(1),
        I.size(2));
  }));

  return Itx;
}

template<typename Real, bool broadcast_I, bool use_contrast,
    bool need_I, bool need_A, bool need_T>
__global__ void affine_interp_image_kernel_backward_2d(
        Real* __restrict__ d_I,
        Real* __restrict__ d_A,
        Real* __restrict__ d_T,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ I,
        const Real* __restrict__ A, const Real* __restrict__ T,
        const Real* __restrict__ B, const Real* __restrict__ C,
        size_t nn, size_t nx, size_t ny) {
    const size_t i = blockDim.x*blockIdx.x + threadIdx.x;
    const size_t j = blockDim.y*blockIdx.y + threadIdx.y;
    // get a linear thread index to index into the shared arrays
    int tid = threadIdx.x*blockDim.y + threadIdx.y;
    const auto THREADS_PER_BLOCK = INTERP_BACKWARD_NUM_THREADS;
    __shared__ Real dlA00[THREADS_PER_BLOCK];
    __shared__ Real dlA01[THREADS_PER_BLOCK];
    __shared__ Real dlA10[THREADS_PER_BLOCK];
    __shared__ Real dlA11[THREADS_PER_BLOCK];
    __shared__ Real dlT0[THREADS_PER_BLOCK];
    __shared__ Real dlT1[THREADS_PER_BLOCK];
    const size_t nxy = nx*ny;
    const Real* gon = grad_out;
    const Real* In = I;
    const Real* An = A;
    const Real* Tn = T;
    const Real* Bn = B;
    const Real* Cn = C;
    Real* gIn = d_I;
    Real* gAn = d_A;
    Real* gTn = d_T;
    // index of current output point (first component. add nxy for second)
    size_t ix = i*ny + j;
    // center of rotation
    Real ox = .5*static_cast<Real>(nx-1);
    Real oy = .5*static_cast<Real>(ny-1);
    Real fi=static_cast<Real>(i)-ox;
    Real fj=static_cast<Real>(j)-oy;
    Real diff, gx, gy, hx, hy;
    for (int n=0; n < nn; ++n) {
        if (i >= nx || j >= ny) {
            if (need_A) {
                dlA00[tid] = 0;
                dlA01[tid] = 0;
                dlA10[tid] = 0;
                dlA11[tid] = 0;
            }
            if (need_T) {
                dlT0[tid] = 0;
                dlT1[tid] = 0;
            }
        } else {
            // what for L2 loss is diff now given as grad_out
            diff = gon[ix];

            // apply affine transform to map i, j to lookup point
            hx = An[0]*fi + An[1]*fj + Tn[0] + ox;
            hy = An[2]*fi + An[3]*fj + Tn[1] + oy;

            // derivative with respect to input is just splat of grad_out
            if (need_I) {
                atomicSplat<Real, DEFAULT_BACKGROUND_STRATEGY, false>(d_I, NULL,
                    diff, hx, hy, nx, ny);
            }

            // get interp value and gradient at lookup point
            if (need_A || need_T)
                biLerp_grad<Real, DEFAULT_BACKGROUND_STRATEGY>(diff, gx, gy,
                    In,
                    hx, hy,
                    nx, ny,
                    0.f);
            if (use_contrast) {
                diff = Bn[0]*diff + Cn[0];
                gx *= Bn[0];
                gy *= Bn[0];
            }
            gx *= diff; // save a few multiplies by premultiplying
            gy *= diff;
            // compute the outer product terms that will be summed
            if (need_A) {
                dlA00[tid] = gx*fi;
                dlA01[tid] = gx*fj;
                dlA10[tid] = gy*fi;
                dlA11[tid] = gy*fj;
            }
            if (need_T) {
                dlT0[tid] = gx;
                dlT1[tid] = gy;
            }
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
            if (need_A) {
                atomicAdd(&gAn[0], dlA00[0]);
                atomicAdd(&gAn[1], dlA01[0]);
                atomicAdd(&gAn[2], dlA10[0]);
                atomicAdd(&gAn[3], dlA11[0]);
            }
            if (need_T) {
                atomicAdd(&gTn[0], dlT0[0]);
                atomicAdd(&gTn[1], dlT1[0]);
            }
        }

        if (!broadcast_I) {
            In += nxy;
            gIn += nxy;
        }
        gon += nxy;
        An += 4;
        Tn += 2;
        if (use_contrast) {
            Bn++;
            Cn++;
        }
        gAn += 4;
        gTn += 2;
    }
}

template<bool need_I, bool need_A, bool need_T>
std::vector<at::Tensor> affine_interp_image_cuda_backward_impl(
    at::Tensor grad_out,
    at::Tensor I,
    at::Tensor A,
    at::Tensor T) {
    // avoid allocating memory for gradients we don't need to compute
    at::Tensor d_I, d_A, d_T;
    if (need_I) d_I = at::zeros_like(I);
    if (need_A) d_A = at::zeros_like(A);
    if (need_T) d_T = at::zeros_like(T);

    const auto threads = INTERP_BACKWARD_THREADS;
    const dim3 blocks((d_I.size(1) + threads.x - 1) / threads.x,
                      (d_I.size(2) + threads.y - 1) / threads.y);

    const bool broadcast_I = I.size(0) == 1 && grad_out.size(0) > 1;

    if (broadcast_I) {
        AT_DISPATCH_FLOATING_TYPES(d_I.type(), "affine_interp_image_cuda_backward", ([&] {
        affine_interp_image_kernel_backward_2d<scalar_t, true, false, need_I, need_A, need_T><<<blocks, threads>>>(
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
            d_I.size(1),
            d_I.size(2));
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(d_I.type(), "affine_interp_image_cuda_backward", ([&] {
        affine_interp_image_kernel_backward_2d<scalar_t, false, false, need_I, need_A, need_T><<<blocks, threads>>>(
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
            d_I.size(1),
            d_I.size(2));
        }));
    }

    return {d_I, d_A, d_T};
}

std::vector<at::Tensor> affine_interp_image_cuda_backward(
    at::Tensor grad_out,
    at::Tensor I,
    at::Tensor A,
    at::Tensor T,
    bool need_I,
    bool need_A,
    bool need_T) {
    if (need_I) {
        if (need_A) {
            if (need_T) {
                return affine_interp_image_cuda_backward_impl<true, true, true>(grad_out, I, A, T);
            } else {
                return affine_interp_image_cuda_backward_impl<true, true, false>(grad_out, I, A, T);
            }
        } else {
            if (need_T) {
                return affine_interp_image_cuda_backward_impl<true, false, true>(grad_out, I, A, T);
            } else {
                return affine_interp_image_cuda_backward_impl<true, false, false>(grad_out, I, A, T);
            }
        }
    } else {
        if (need_A) {
            if (need_T) {
                return affine_interp_image_cuda_backward_impl<false, true, true>(grad_out, I, A, T);
            } else {
                return affine_interp_image_cuda_backward_impl<false, true, false>(grad_out, I, A, T);
            }
        } else {
            if (need_T) {
                return affine_interp_image_cuda_backward_impl<false, false, true>(grad_out, I, A, T);
            } else {
                return affine_interp_image_cuda_backward_impl<false, false, false>(grad_out, I, A, T);
            }
        }
    }
}

