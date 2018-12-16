#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "atomic.cuh"
#include "defs.cuh"
#include "interp.cuh"

// Generic interpolation kernel. Given an input image I with nc channels (this
// could represent a vector field), interpolate the image via the displacement
// field u. dt is a time step that multiplies the displacements.
template<typename Real, bool broadcast_I>
__global__ void interp_kernel_2d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        const Real* __restrict__ u,
        double dt,
        size_t nn, size_t nc, size_t nx, size_t ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int nxy = nx*ny;
    const Real* In = I; // pointer to first channel of input
    Real* outn = out; // pointer to current vector field v
    const Real* un = u; // pointer to current vector field v
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    int iy = ix + nxy;
    Real fi=static_cast<Real>(i);
    Real fj=static_cast<Real>(j);
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) In = I;
        auto hx = fi + dt*un[ix];
        auto hy = fj + dt*un[iy];
        for (int c=0; c < nc; ++c) {
            outn[ix] = biLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                hx, hy, nx, ny);
            In += nxy;
            outn += nxy;
        }
        un += 2*nxy;
    }
}
template<typename Real, bool broadcast_I>
__global__ void interp_kernel_3d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        const Real* __restrict__ u,
        double dt,
        size_t nn, size_t nc, size_t nx, size_t ny, size_t nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int nxyz = nx*ny*nz;
    const Real* In = I;
    Real fi=static_cast<Real>(i);
    Real fj=static_cast<Real>(j);
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) In = I;
        for (int c=0; c < nc; ++c) {
            auto ix =    (((n*3     )*nx + i)*ny + j)*nz;
            auto ixout = (((n*nc + c)*nx + i)*ny + j)*nz;
            for (int k=0; k < nz; ++k, ++ix, ++ixout) {
                Real fk=static_cast<Real>(k);
                auto hx = fi + dt*u[ix       ];
                auto hy = fj + dt*u[ix+  nxyz];
                auto hz = fk + dt*u[ix+2*nxyz];
                out[ixout] = \
                    triLerp<Real, BACKGROUND_STRATEGY_CLAMP>(In,
                        hx, hy, hz, nx, ny, nz);
            }
            In += nxyz;
        }
    }
}

at::Tensor interp_cuda_forward(
    at::Tensor Iv,
    at::Tensor u,
    double dt) {
    auto d = Iv.dim() - 2;
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional interpolation is supported")
    const dim3 threads(32, 32);
    const dim3 blocks((Iv.size(2) + threads.x - 1) / threads.x,
                    (Iv.size(3) + threads.y - 1) / threads.y);

    const auto batch_size = (u.size(0) > Iv.size(0)) ? u.size(0) : Iv.size(0);

    const bool broadcast_I = Iv.size(0) < batch_size;

    at::Tensor out;
    if (d == 2) {
        out = at::zeros({batch_size, Iv.size(1), Iv.size(2), Iv.size(3)}, Iv.type());
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(Iv.type(), "interp_cuda_forward", ([&] {
            interp_kernel_2d<scalar_t, broadcastI><<<blocks, threads>>>(
                out.data<scalar_t>(),
                Iv.data<scalar_t>(),
                u.data<scalar_t>(),
                dt,
                batch_size,
                Iv.size(1),
                Iv.size(2),
                Iv.size(3));
            }));
        }));
    } else {
        out = at::zeros({batch_size, Iv.size(1), Iv.size(2), Iv.size(3), Iv.size(4)}, Iv.type());
        LAGOMORPH_DISPATCH_BOOL(broadcast_I, broadcastI, ([&] {
            AT_DISPATCH_FLOATING_TYPES(Iv.type(), "interp_cuda_forward", ([&] {
            interp_kernel_3d<scalar_t, broadcastI><<<blocks, threads>>>(
                out.data<scalar_t>(),
                Iv.data<scalar_t>(),
                u.data<scalar_t>(),
                dt,
                batch_size,
                Iv.size(1),
                Iv.size(2),
                Iv.size(3),
                Iv.size(4));
            }));
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return out;
}

template<typename Real, bool broadcast_I, bool need_I, bool need_u>
__global__ void interp_kernel_backward_2d(
        Real* __restrict__ d_I,
        Real* __restrict__ d_u,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ I,
        const Real* __restrict__ u,
        double dt,
        size_t nn, size_t nc, size_t nx, size_t ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int nxy = nx*ny;
    const Real* In = I; // pointer to first channel of input
    Real* d_In = d_I; // pointer to first channel of gradient wrt input
    const Real* un = u; // pointer to current vector field v
    Real* d_un = d_u; // pointer to gradient wrt current vector field v
    const Real* gon = grad_out;
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    int iy = ix + nxy;
    Real Ih=0, gx=0, gy=0; // gradient at interpolated point
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) {
            In = I; // reset to first channel of first image
            d_In = d_I;
        }
        // apply displacement to get interpolated point for this vector field
        Real hx = i + dt*un[ix];
        Real hy = j + dt*un[iy];
        for (int c=0; c < nc; ++c) {
            Real diff = gon[ix];
            if (need_I) {
                atomicSplat<Real, DEFAULT_BACKGROUND_STRATEGY, false>(d_In, NULL,
                    diff, hx, hy, nx, ny);
            }
            if (need_u) {
                biLerp_grad<Real, DEFAULT_BACKGROUND_STRATEGY>(Ih, gx, gy,
                    In, hx, hy, nx, ny);
                diff *= dt;
                d_un[ix] = d_un[ix] + gx*diff;
                d_un[iy] = d_un[iy] + gy*diff;
            }
            // move to next channel/image
            In += nxy;
            d_In += nxy;
            gon += nxy;
        }
        un += 2*nxy;
        d_un += 2*nxy;
    }
}

template<typename Real, bool broadcast_I, bool need_I, bool need_u>
__global__ void interp_kernel_backward_3d(
        Real* __restrict__ d_I,
        Real* __restrict__ d_u,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ I,
        const Real* __restrict__ u,
        double dt,
        size_t nn, size_t nc, size_t nx, size_t ny, size_t nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int nxy = nx*ny;
    const Real* In = I; // pointer to first channel of input
    Real* d_In = d_I; // pointer to first channel of gradient wrt input
    const Real* un = u; // pointer to current vector field v
    Real* d_un = d_u; // pointer to gradient wrt current vector field v
    const Real* gon = grad_out;
    // index of current output point (first component. add nxy for second)
    auto ix = i*ny + j;
    auto iy = ix + nxy;
    auto iz = iy + nxy;
    Real Ih=0, gx=0, gy=0, gz=0; // gradient at interpolated point
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) {
            In = I; // reset to first channel of first image
            d_In = d_I;
        }
        // apply displacement to get interpolated point for this vector field
        for (int c=0; c < nc; ++c) {
            for (int k=0; k < nz; ++k) {
                auto hx = i + dt*un[ix*nz + k];
                auto hy = j + dt*un[iy*nz + k];
                auto hz = k + dt*un[iz*nz + k];
                auto diff = gon[ix*nz + k];
                if (need_I) {
                    atomicSplat<Real, DEFAULT_BACKGROUND_STRATEGY, false>(d_In, NULL,
                        diff, hx, hy, hz, nx, ny, nz);
                }
                if (need_u) {
                    triLerp_grad<Real, DEFAULT_BACKGROUND_STRATEGY>(Ih,
                        gx, gy, gz,
                        In,
                        hx, hy, hz,
                        nx, ny, nz);
                    diff *= dt;
                    d_un[ix*nz+k] = d_un[ix*nz+k] + gx*diff;
                    d_un[iy*nz+k] = d_un[iy*nz+k] + gy*diff;
                    d_un[iz*nz+k] = d_un[iz*nz+k] + gz*diff;
                }
            }
            // move to next channel/image
            In += nxy*nz;
            d_In += nxy*nz;
            gon += nxy*nz;
        }
        un += 3*nxy*nz;
        d_un += 3*nxy*nz;
    }
}

std::vector<at::Tensor> interp_cuda_backward(
    at::Tensor grad_out,
    at::Tensor Iv,
    at::Tensor u,
    double dt,
    bool need_I,
    bool need_u) {
    auto d = Iv.dim() - 2;
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional interpolation is supported")
    const dim3 threads(16, 32);
    const dim3 blocks((Iv.size(2) + threads.x - 1) / threads.x,
                    (Iv.size(3) + threads.y - 1) / threads.y);

    const auto batch_size = (u.size(0) > Iv.size(0)) ? u.size(0) : Iv.size(0);
    const bool broadcast_I = Iv.size(0) < batch_size;

    // For now, just compute all gradients
    at::Tensor d_I = at::zeros_like(Iv);
    at::Tensor d_u = at::zeros_like(u);

    if (d == 2) {
        LAGOMORPH_DISPATCH_BOOL(need_I, needI, ([&] {
            LAGOMORPH_DISPATCH_BOOL(need_u, needu, ([&] {
                LAGOMORPH_DISPATCH_BOOL(broadcast_I, bcastI, ([&] {
                    AT_DISPATCH_FLOATING_TYPES(Iv.type(), "interp_cuda_backward", ([&] {
                    interp_kernel_backward_2d<scalar_t, bcastI, needI,
                            needu><<<blocks, threads>>>(
                        d_I.data<scalar_t>(),
                        d_u.data<scalar_t>(),
                        grad_out.data<scalar_t>(),
                        Iv.data<scalar_t>(),
                        u.data<scalar_t>(),
                        dt,
                        batch_size,
                        Iv.size(1),
                        Iv.size(2),
                        Iv.size(3));
                    }));
                }));
            }));
        }));
    } else {
        LAGOMORPH_DISPATCH_BOOL(need_I, needI, ([&] {
            LAGOMORPH_DISPATCH_BOOL(need_u, needu, ([&] {
                LAGOMORPH_DISPATCH_BOOL(broadcast_I, bcastI, ([&] {
                    AT_DISPATCH_FLOATING_TYPES(Iv.type(), "interp_cuda_backward", ([&] {
                    interp_kernel_backward_3d<scalar_t, bcastI, needI,
                            needu><<<blocks, threads>>>(
                        d_I.data<scalar_t>(),
                        d_u.data<scalar_t>(),
                        grad_out.data<scalar_t>(),
                        Iv.data<scalar_t>(),
                        u.data<scalar_t>(),
                        dt,
                        batch_size,
                        Iv.size(1),
                        Iv.size(2),
                        Iv.size(3),
                        Iv.size(4));
                    }));
                }));
            }));
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return {d_I, d_u};
}

// This kernel computes the diagonal of the Hessian with respect to the
// interpolated image I
template<typename Real, bool broadcast_I,
    BackgroundStrategy backgroundStrategy>
__global__ void interp_hessian_diagonal_image_kernel_2d(
        Real* __restrict__ out,
        const Real* __restrict__ I,
        const Real* __restrict__ u,
        double dt,
        size_t nn, size_t nc, size_t nx, size_t ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int nxy = nx*ny;
    const Real* In = I; // pointer to first channel of input
    Real* outn = out; // pointer to current vector field v
    const Real* un = u; // pointer to current vector field v
    // index of current output point (first component. add nxy for second)
    int ix = i*ny + j;
    int iy = ix + nxy;
    Real fi=static_cast<Real>(i);
    Real fj=static_cast<Real>(j);
    for (int n=0; n < nn; ++n) {
        if (broadcast_I) In = I; // reset to first channel
        auto hx = fi + dt*un[ix];
        auto hy = fj + dt*un[iy];
        for (int c=0; c < nc; ++c) {
            interp_hessian_diagonal_image_point<Real, backgroundStrategy>(out, hx, hy, nx, ny);
            // move to next channel/image
            In += nxy;
            outn += nxy;
        }
        un += 2*nxy;
    }
}

at::Tensor interp_hessian_diagonal_image(
    at::Tensor Iv,
    at::Tensor u,
    double dt) {
    const dim3 threads(32, 32);
    const dim3 blocks((Iv.size(2) + threads.x - 1) / threads.x,
                    (Iv.size(3) + threads.y - 1) / threads.y);

    const auto batch_size = (u.size(0) > Iv.size(0)) ? u.size(0) : Iv.size(0);

    const bool broadcast_I = Iv.size(0) < batch_size;

    auto out = at::zeros_like(Iv);

    LAGOMORPH_DISPATCH_BOOL(broadcast_I, bcastI, ([&] {
        AT_DISPATCH_FLOATING_TYPES(Iv.type(), "interp_hessian_diagonal_image", ([&] {
        interp_hessian_diagonal_image_kernel_2d<scalar_t, bcastI, DEFAULT_BACKGROUND_STRATEGY><<<blocks, threads>>>(
            out.data<scalar_t>(),
            Iv.data<scalar_t>(),
            u.data<scalar_t>(),
            dt,
            batch_size,
            Iv.size(1),
            Iv.size(2),
            Iv.size(3));
        }));
    }));
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return out;
}
