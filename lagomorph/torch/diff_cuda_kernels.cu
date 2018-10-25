#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "defs.cuh"
#include "diff.cuh"

// Templated function to compute the Jacobian matrix of the first vector field
// and contract it with the second vector field in a point-wise fashion. The
// Jacobian will be transposed first if the template argument 'transpose' is 1
// instead of 0. If the template argument displacement is 1 then the vector
// field v will be treated as the displacement of a deformation whose jacobian
// we compute.
template<typename Real, bool displacement, bool transpose, BackgroundStrategy backgroundStrategy>
__global__
void
jacobian_times_vectorfield_forward_kernel_2d(Real* __restrict__ out, const Real* __restrict__ v, const Real* __restrict__ w,
        int nn, int nc, int nx, int ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    Real gx, gy; // gradient of component of v
    const int nxy = nx*ny;
    Real* outn = out;
    const Real* vn = v; // pointer to current vector field v
    const Real* wn = w;
    int ix = i*ny + j; // indices of first and second component of wn
    int iy = ix + nxy;
    for (int n=0; n < nn; ++n) {
        if (transpose) {
            grad_point<Real, backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
            if (displacement) gx += 1.0;
            outn[ix] = gx*wn[ix];
            outn[iy] = gy*wn[ix];
            vn += nxy;
            grad_point<Real, backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
            if (displacement) gy += 1.0;
            outn[ix] += gx*wn[iy];
            outn[iy] += gy*wn[iy];
            outn += 2*nxy;
            vn += nxy;
        } else {
            for (int c=0; c < nc; ++c) {
                // get gradient of each component of vn
                grad_point<Real, backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
                if (displacement) {
                    if (c == 0) gx += 1.0;
                    if (c == 1) gy += 1.0;
                }
                outn[ix] = gx*wn[ix] + gy*wn[iy];
                vn += nxy; // move to next component
                outn += nxy;
            }
        }
        // increment w lookups
        wn += 2*nxy;
    }
}

at::Tensor jacobian_times_vectorfield_forward(
    at::Tensor g,
    at::Tensor v,
    bool displacement,
	bool transpose) {
    const dim3 threads(32, 32);
    const dim3 blocks((g.size(2) + threads.x - 1) / threads.x,
                    (g.size(3) + threads.y - 1) / threads.y);

    const auto dim = g.dim()-2;
    AT_ASSERTM(v.size(1) == dim, "vector field is of wrong dimension")
    AT_ASSERTM(!displacement || g.size(1) == dim, "Displacement mode only defined for vector fields")
    if (transpose)
        AT_ASSERTM(g.size(1) == dim, "Jacobian transpose only implemented for vector fields")

    const auto batch_size = (v.size(0) > g.size(0)) ? v.size(0) : g.size(0);

    auto out = at::zeros({batch_size, g.size(1), g.size(2), g.size(3)}, g.type());

    LAGOMORPH_DISPATCH_BOOL(transpose, trans, ([&] {
        LAGOMORPH_DISPATCH_BOOL(displacement, disp, ([&] {
            AT_DISPATCH_FLOATING_TYPES(g.type(), "jacobian_times_vectorfield_forward", ([&] {
            jacobian_times_vectorfield_forward_kernel_2d<scalar_t, disp, trans, DEFAULT_BACKGROUND_STRATEGY><<<blocks, threads>>>(
                out.data<scalar_t>(),
                g.data<scalar_t>(),
                v.data<scalar_t>(),
                out.size(0),
                out.size(1),
                out.size(2),
                out.size(3));
            }));
        }));
    }));
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return out;
}

template<typename Real, bool displacement, bool transpose,
    BackgroundStrategy backgroundStrategy, bool need_v, bool need_w>
__global__
void
jacobian_times_vectorfield_backward_kernel_2d(
        Real* __restrict__ d_v,
        Real* __restrict__ d_w,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ v, const Real* __restrict__ w,
        int nn, int nc, int nx, int ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    Real gx, gy; // gradient of component of v
    const int nxy = nx*ny;
    Real* d_vn = d_v;
    Real* d_wn = d_w;
    const Real* gon = grad_out;
    const Real* vn = v; // pointer to current vector field v
    const Real* wn = w;
    int ix = i*ny + j; // indices of first and second component of wn
    int iy = ix + nxy;
    for (int n=0; n < nn; ++n) {
        if (transpose) {
            if (need_w) {
                grad_point<Real, backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
                if (displacement) gx += 1.0;
                d_wn[ix] += gx*gon[ix] + gy*gon[iy];
                vn += nxy;
                d_wn += nxy;
                grad_point<Real, backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
                if (displacement) gy += 1.0;
                d_wn[ix] += gx*gon[ix] + gy*gon[iy];
                vn += nxy;
                d_wn += nxy;
            }
            if (need_v) {
                if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP) {
                    if (i == 0) {
                        d_vn[ix] += -.5 * (wn[ix]*gon[ix] + wn[ix+ny]*gon[ix+ny]);
                        d_vn[iy] += -.5 * (wn[iy]*gon[ix] + wn[iy+ny]*gon[ix+ny]);
                    } else if (i == nx-1) {
                        d_vn[ix] +=  .5 * (wn[ix]*gon[ix] + wn[ix-ny]*gon[ix-ny]);
                        d_vn[iy] +=  .5 * (wn[iy]*gon[ix] + wn[iy-ny]*gon[ix-ny]);
                    } else {
                        d_vn[ix] += -.5 * (wn[ix+ny]*gon[ix+ny] - wn[ix-ny]*gon[ix-ny]);
                        d_vn[iy] += -.5 * (wn[iy+ny]*gon[ix+ny] - wn[iy-ny]*gon[ix-ny]);
                    }
                    if (j == 0) {
                        d_vn[ix] += -.5 * (wn[ix]*gon[iy] + wn[ix+1]*gon[iy+1]);
                        d_vn[iy] += -.5 * (wn[iy]*gon[iy] + wn[iy+1]*gon[iy+1]);
                    } else if (j == ny-1) {
                        d_vn[ix] +=  .5 * (wn[ix]*gon[iy] + wn[ix-1]*gon[iy-1]);
                        d_vn[iy] +=  .5 * (wn[iy]*gon[iy] + wn[iy-1]*gon[iy-1]);
                    } else {
                        d_vn[ix] += -.5 * (wn[ix+1]*gon[iy+1] - wn[ix-1]*gon[iy-1]);
                        d_vn[iy] += -.5 * (wn[iy+1]*gon[iy+1] - wn[iy-1]*gon[iy-1]);
                    }
                }
                d_vn += 2*nxy;
            }
            gon += 2*nxy;
        } else {
            for (int c=0; c < nc; ++c) {
                if (need_w) {
                    grad_point<Real, backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
                    if (displacement) {
                        if (c == 0) gx += 1.0;
                        if (c == 1) gy += 1.0;
                    }
                    d_wn[ix] += gx*gon[ix];
                    d_wn[iy] += gy*gon[ix];
                }
                if (need_v) {
                    if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP) {
                        if (i == 0)
                            d_vn[ix] += -.5 * (wn[ix]*gon[ix] + wn[ix+ny]*gon[ix+ny]);
                        else if (i == nx-1)
                            d_vn[ix] +=  .5 * (wn[ix]*gon[ix] + wn[ix-ny]*gon[ix-ny]);
                        else
                            d_vn[ix] += -.5 * (wn[ix+ny]*gon[ix+ny] - wn[ix-ny]*gon[ix-ny]);

                        if (j == 0)
                            d_vn[ix] += -.5 * (wn[iy]*gon[ix] + wn[iy+1]*gon[ix+1]);
                        else if (j == ny-1)
                            d_vn[ix] +=  .5 * (wn[iy]*gon[ix] + wn[iy-1]*gon[ix-1]);
                        else
                            d_vn[ix] += -.5 * (wn[iy+1]*gon[ix+1] - wn[iy-1]*gon[ix-1]);
                    }
                }
                vn += nxy; // move to next component
                d_vn += nxy;
                gon += nxy;
            }
        }
        // increment w lookups
        wn += 2*nxy;
        d_wn += 2*nxy;
    }
}

std::vector<at::Tensor> jacobian_times_vectorfield_backward(
    at::Tensor grad_out,
    at::Tensor v,
    at::Tensor w,
    bool displacement,
    bool transpose,
    bool need_v,
    bool need_w) {
    need_v = true; // hardcode derivatives for both arguments for now
    need_w = true;

    const dim3 threads(16, 32);
    const dim3 blocks((v.size(2) + threads.x - 1) / threads.x,
                    (v.size(3) + threads.y - 1) / threads.y);

    const auto dim = v.dim()-2;
    AT_ASSERTM(w.size(1) == dim, "vector field is of wrong dimension")
    AT_ASSERTM(!displacement || v.size(1) == dim, "Displacement mode only defined for vector fields")

    const auto batch_size = (w.size(0) > v.size(0)) ? w.size(0) : v.size(0);

    at::Tensor d_v, d_w;
    if (need_v) d_v = at::zeros_like(v);
    if (need_w) d_w = at::zeros_like(w);

    LAGOMORPH_DISPATCH_BOOL(transpose, trans, ([&] {
        LAGOMORPH_DISPATCH_BOOL(displacement, disp, ([&] {
            AT_DISPATCH_FLOATING_TYPES(v.type(), "jacobian_times_vectorfield_backward", ([&] {
            jacobian_times_vectorfield_backward_kernel_2d<scalar_t, disp, trans, DEFAULT_BACKGROUND_STRATEGY, true, true><<<blocks, threads>>>(
                d_v.data<scalar_t>(),
                d_w.data<scalar_t>(),
                grad_out.data<scalar_t>(),
                v.data<scalar_t>(),
                w.data<scalar_t>(),
                batch_size,
                v.size(1),
                v.size(2),
                v.size(3));
            }));
        }));
    }));
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return {d_v, d_w};
}
