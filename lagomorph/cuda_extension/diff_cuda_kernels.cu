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

template<typename Real, bool displacement, bool transpose, BackgroundStrategy backgroundStrategy>
__global__
void
jacobian_times_vectorfield_forward_kernel_3d(Real* __restrict__ out,
        const Real* __restrict__ v, const Real* __restrict__ w,
        int nn, int nc, int nx, int ny, int nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    Real gx, gy, gz; // gradient of component of v
    const int nxyz = nx*ny*nz;
    Real* outn = out;
    const Real* vn = v; // pointer to current vector field v
    const Real* wn = w;
    int ix = (i*ny + j)*nz; // indices of first and second component of wn
    int iy = ix + nxyz;
    int iz = iy + nxyz;
    for (int n=0; n < nn; ++n) {
        if (transpose) {
            for (int k=0; k < nz; ++k) {
                grad_point<Real, backgroundStrategy>(gx, gy, gz, vn       , nx, ny, nz, i, j, k);
                if (displacement) gx += 1.0;
                outn[ix + k]  = gx*wn[ix + k];
                outn[iy + k]  = gy*wn[ix + k];
                outn[iz + k]  = gz*wn[ix + k];

                grad_point<Real, backgroundStrategy>(gx, gy, gz, vn+  nxyz, nx, ny, nz, i, j, k);
                if (displacement) gy += 1.0;
                outn[ix + k] += gx*wn[iy + k];
                outn[iy + k] += gy*wn[iy + k];
                outn[iz + k] += gz*wn[iy + k];

                grad_point<Real, backgroundStrategy>(gx, gy, gz, vn+2*nxyz, nx, ny, nz, i, j, k);
                if (displacement) gz += 1.0;
                outn[ix + k] += gx*wn[iz + k];
                outn[iy + k] += gy*wn[iz + k];
                outn[iz + k] += gz*wn[iz + k];
            }
            vn += 3*nxyz;
            outn += 3*nxyz;
        } else {
            for (int c=0; c < nc; ++c) {
                for (int k=0; k < nz; ++k) {
                    // get gradient of each component of vn
                    grad_point<Real, backgroundStrategy>(gx, gy, gz,
                        vn,
                        nx, ny, nz,
                        i, j, k);
                    if (displacement) {
                        if (c == 0) gx += 1.0;
                        if (c == 1) gy += 1.0;
                        if (c == 2) gz += 1.0;
                    }
                    outn[ix+k] = gx*wn[ix       +k] + \
                                 gy*wn[ix+  nxyz+k] + \
                                 gz*wn[ix+2*nxyz+k];
                }
                vn += nxyz; // move to next component
                outn += nxyz;
            }
        }
        // increment w lookups
        wn += 3*nxyz;
    }
}

at::Tensor jacobian_times_vectorfield_forward(
    at::Tensor g,
    at::Tensor v,
    bool displacement,
	bool transpose) {
    auto d = g.dim() - 2;
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional jacobian times vectorfield is supported")
    for (size_t dd=0; dd < d; ++dd)
        AT_ASSERTM(v.size(dd+2) > 1, "Jacobian times vectorfield not implemented for 'thin' dimensions")
    const dim3 threads(16, 32);
    const dim3 blocks((g.size(2) + threads.x - 1) / threads.x,
                    (g.size(3) + threads.y - 1) / threads.y);

    AT_ASSERTM(g.size(0) == v.size(0), "arguments must have same batch size dimension")
    AT_ASSERTM(v.size(1) == d, "vector field is of wrong dimension")
    AT_ASSERTM(!displacement || g.size(1) == d, "Displacement mode only defined for vector fields")
    if (transpose)
        AT_ASSERTM(g.size(1) == d, "Jacobian transpose only implemented for vector fields")

    at::Tensor out = at::zeros_like(g);

    if (d == 2) {
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
    } else {
        LAGOMORPH_DISPATCH_BOOL(transpose, trans, ([&] {
            LAGOMORPH_DISPATCH_BOOL(displacement, disp, ([&] {
                AT_DISPATCH_FLOATING_TYPES(g.type(), "jacobian_times_vectorfield_forward", ([&] {
                jacobian_times_vectorfield_forward_kernel_3d<scalar_t, disp, trans, DEFAULT_BACKGROUND_STRATEGY><<<blocks, threads>>>(
                    out.data<scalar_t>(),
                    g.data<scalar_t>(),
                    v.data<scalar_t>(),
                    out.size(0),
                    out.size(1),
                    out.size(2),
                    out.size(3),
                    out.size(4));
                }));
            }));
        }));
    }
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
                grad_point<Real, backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
                if (displacement) gy += 1.0;
                d_wn[iy] += gx*gon[ix] + gy*gon[iy];
                vn += nxy;
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

template<typename Real, bool displacement, bool transpose,
    BackgroundStrategy backgroundStrategy, bool need_v, bool need_w>
__global__
void
jacobian_times_vectorfield_backward_kernel_3d(
        Real* __restrict__ d_v,
        Real* __restrict__ d_w,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ v, const Real* __restrict__ w,
        size_t nn, size_t nc, size_t nx, size_t ny, size_t nz) {
    const auto i = blockDim.x*blockIdx.x + threadIdx.x;
    const auto j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    Real gx, gy, gz; // gradient of component of v
    const auto nyz = ny*nz;
    const auto nxyz = nx*ny*nz;
    Real* d_vn = d_v;
    Real* d_wn = d_w;
    const Real* gon = grad_out;
    const Real* vn = v; // pointer to current vector field v
    const Real* wn = w;
    int ix, iy, iz;
    for (int n=0; n < nn; ++n) {
        if (transpose) {
            for (int k=0; k < nz; ++k) {
                ix = (i*ny+j)*nz+k; // indices of each component of w
                iy = ix + nxyz;
                iz = iy + nxyz;
                if (need_w) {
                    grad_point<Real, backgroundStrategy>(gx, gy, gz,
                            vn,
                            nx, ny, nz,
                            i, j, k);
                    if (displacement) gx += 1.0;
                    d_wn[ix] += gx*gon[ix] + gy*gon[iy] + gz*gon[iz];

                    grad_point<Real, backgroundStrategy>(gx, gy, gz,
                            vn + nxyz,
                            nx, ny, nz,
                            i, j, k);
                    if (displacement) gy += 1.0;
                    d_wn[iy] += gx*gon[ix] + gy*gon[iy] + gz*gon[iz];

                    grad_point<Real, backgroundStrategy>(gx, gy, gz,
                            vn + 2*nxyz,
                            nx, ny, nz,
                            i, j, k);
                    if (displacement) gz += 1.0;
                    d_wn[iz] += gx*gon[ix] + gy*gon[iy] + gz*gon[iz];
                }
                if (need_v) {
                    if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP) {
                        if (i == 0) {
                            d_vn[ix] += -.5 * (wn[ix+nyz]*gon[ix+nyz]
                                             + wn[ix    ]*gon[ix    ]);
                            d_vn[iy] += -.5 * (wn[iy+nyz]*gon[ix+nyz]
                                             + wn[iy    ]*gon[ix    ]);
                            d_vn[iz] += -.5 * (wn[iz+nyz]*gon[ix+nyz]
                                             + wn[iz    ]*gon[ix    ]);
                        } else if (i == nx-1) {
                            d_vn[ix] +=  .5 * (wn[ix    ]*gon[ix    ]
                                             + wn[ix-nyz]*gon[ix-nyz]);
                            d_vn[iy] +=  .5 * (wn[iy    ]*gon[ix    ]
                                             + wn[iy-nyz]*gon[ix-nyz]);
                            d_vn[iz] +=  .5 * (wn[iz    ]*gon[ix    ]
                                             + wn[iz-nyz]*gon[ix-nyz]);
                        } else {
                            d_vn[ix] += -.5 * (wn[ix+nyz]*gon[ix+nyz]
                                             - wn[ix-nyz]*gon[ix-nyz]);
                            d_vn[iy] += -.5 * (wn[iy+nyz]*gon[ix+nyz]
                                             - wn[iy-nyz]*gon[ix-nyz]);
                            d_vn[iz] += -.5 * (wn[iz+nyz]*gon[ix+nyz]
                                             - wn[iz-nyz]*gon[ix-nyz]);
                        }
                        if (j == 0) {
                            d_vn[ix] += -.5 * (wn[ix+nz]*gon[iy+nz]
                                             + wn[ix   ]*gon[iy   ]);
                            d_vn[iy] += -.5 * (wn[iy+nz]*gon[iy+nz]
                                             + wn[iy   ]*gon[iy   ]);
                            d_vn[iz] += -.5 * (wn[iz+nz]*gon[iy+nz]
                                             + wn[iz   ]*gon[iy   ]);
                        } else if (j == ny-1) {
                            d_vn[ix] +=  .5 * (wn[ix   ]*gon[iy   ]
                                             + wn[ix-nz]*gon[iy-nz]);
                            d_vn[iy] +=  .5 * (wn[iy   ]*gon[iy   ]
                                             + wn[iy-nz]*gon[iy-nz]);
                            d_vn[iz] +=  .5 * (wn[iz   ]*gon[iy   ]
                                             + wn[iz-nz]*gon[iy-nz]);
                        } else {
                            d_vn[ix] += -.5 * (wn[ix+nz]*gon[iy+nz]
                                             - wn[ix-nz]*gon[iy-nz]);
                            d_vn[iy] += -.5 * (wn[iy+nz]*gon[iy+nz]
                                             - wn[iy-nz]*gon[iy-nz]);
                            d_vn[iz] += -.5 * (wn[iz+nz]*gon[iy+nz]
                                             - wn[iz-nz]*gon[iy-nz]);
                        }
                        if (k == 0) {
                            d_vn[ix] += -.5 * (wn[ix+1]*gon[iz+1]
                                             + wn[ix  ]*gon[iz  ]);
                            d_vn[iy] += -.5 * (wn[iy+1]*gon[iz+1]
                                             + wn[iy  ]*gon[iz  ]);
                            d_vn[iz] += -.5 * (wn[iz+1]*gon[iz+1]
                                             + wn[iz  ]*gon[iz  ]);
                        } else if (k == nz-1) {
                            d_vn[ix] +=  .5 * (wn[ix  ]*gon[iz  ]
                                             + wn[ix-1]*gon[iz-1]);
                            d_vn[iy] +=  .5 * (wn[iy  ]*gon[iz  ]
                                             + wn[iy-1]*gon[iz-1]);
                            d_vn[iz] +=  .5 * (wn[iz  ]*gon[iz  ]
                                             + wn[iz-1]*gon[iz-1]);
                        } else {
                            d_vn[ix] += -.5 * (wn[ix+1]*gon[iz+1]
                                             - wn[ix-1]*gon[iz-1]);
                            d_vn[iy] += -.5 * (wn[iy+1]*gon[iz+1]
                                             - wn[iy-1]*gon[iz-1]);
                            d_vn[iz] += -.5 * (wn[iz+1]*gon[iz+1]
                                             - wn[iz-1]*gon[iz-1]);
                        }
                    }
                }
            }
            d_vn += 3*nxyz;
            vn += 3*nxyz;
            gon += 3*nxyz;
        } else {
            for (int c=0; c < nc; ++c) {
                for (int k=0; k < nz; ++ k) {
                    ix = (i*ny+j)*nz+k; // indices of each component of w
                    iy = ix + nxyz;
                    iz = iy + nxyz;
                    if (need_w) {
                        grad_point<Real, backgroundStrategy>(gx, gy, gz,
                                vn,
                                nx, ny, nz,
                                i, j, k);
                        if (displacement) {
                            if (c == 0) gx += 1.0;
                            if (c == 1) gy += 1.0;
                            if (c == 2) gz += 1.0;
                        }
                        d_wn[ix] += gx*gon[ix];
                        d_wn[iy] += gy*gon[ix];
                        d_wn[iz] += gz*gon[ix];
                    }
                    if (need_v) {
                        if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP) {
                            if (i == 0)
                                d_vn[ix] += -.5 * (wn[ix    ]*gon[ix    ]
                                                 + wn[ix+nyz]*gon[ix+nyz]);
                            else if (i == nx-1)
                                d_vn[ix] +=  .5 * (wn[ix    ]*gon[ix    ]
                                                 + wn[ix-nyz]*gon[ix-nyz]);
                            else
                                d_vn[ix] += -.5 * (wn[ix+nyz]*gon[ix+nyz]
                                                 - wn[ix-nyz]*gon[ix-nyz]);

                            if (j == 0)
                                d_vn[ix] += -.5 * (wn[iy   ]*gon[ix   ]
                                                 + wn[iy+nz]*gon[ix+nz]);
                            else if (j == ny-1)
                                d_vn[ix] +=  .5 * (wn[iy   ]*gon[ix   ]
                                                 + wn[iy-nz]*gon[ix-nz]);
                            else
                                d_vn[ix] += -.5 * (wn[iy+nz]*gon[ix+nz]
                                                 - wn[iy-nz]*gon[ix-nz]);

                            if (k == 0)
                                d_vn[ix] += -.5 * (wn[iz  ]*gon[ix  ]
                                                 + wn[iz+1]*gon[ix+1]);
                            else if (k == nz-1)
                                d_vn[ix] +=  .5 * (wn[iz  ]*gon[ix  ]
                                                 + wn[iz-1]*gon[ix-1]);
                            else
                                d_vn[ix] += -.5 * (wn[iz+1]*gon[ix+1]
                                                 - wn[iz-1]*gon[ix-1]);
                        }
                    }
                }
                vn += nxyz; // move to next component
                d_vn += nxyz;
                gon += nxyz;
            }
        }
        // increment w lookups
        wn += 3*nxyz;
        d_wn += 3*nxyz;
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
    need_v = true;
    need_w = true;
    const dim3 threads(16, 32);
    const dim3 blocks((v.size(2) + threads.x - 1) / threads.x,
                    (v.size(3) + threads.y - 1) / threads.y);

    AT_ASSERTM(v.size(0) == w.size(0), "arguments must have same batch size dimension")
    const auto d = v.dim()-2;
    for (size_t dd=0; dd < d; ++dd)
        AT_ASSERTM(v.size(dd+2) > 1, "Jacobian times vectorfield not implemented for 'thin' dimensions")
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional jacobian times vectorfield is supported")
    AT_ASSERTM(w.size(1) == d, "vector field is of wrong dimension")
    AT_ASSERTM(!displacement || v.size(1) == d, "Displacement mode only defined for vector fields")

    at::Tensor d_v, d_w;
    if (need_v) d_v = at::zeros_like(v);
    if (need_w) d_w = at::zeros_like(w);

    if (d == 2) {
        LAGOMORPH_DISPATCH_BOOL(transpose, trans, ([&] {
            LAGOMORPH_DISPATCH_BOOL(displacement, disp, ([&] {
                AT_DISPATCH_FLOATING_TYPES(v.type(), "jacobian_times_vectorfield_backward", ([&] {
                jacobian_times_vectorfield_backward_kernel_2d<scalar_t, disp, trans, DEFAULT_BACKGROUND_STRATEGY, true, true><<<blocks, threads>>>(
                    d_v.data<scalar_t>(),
                    d_w.data<scalar_t>(),
                    grad_out.data<scalar_t>(),
                    v.data<scalar_t>(),
                    w.data<scalar_t>(),
                    v.size(0),
                    v.size(1),
                    v.size(2),
                    v.size(3));
                }));
            }));
        }));
    } else {
        LAGOMORPH_DISPATCH_BOOL(transpose, trans, ([&] {
            LAGOMORPH_DISPATCH_BOOL(displacement, disp, ([&] {
                AT_DISPATCH_FLOATING_TYPES(v.type(), "jacobian_times_vectorfield_backward", ([&] {
                jacobian_times_vectorfield_backward_kernel_3d<scalar_t, disp, trans, DEFAULT_BACKGROUND_STRATEGY, true, true><<<blocks, threads>>>(
                    d_v.data<scalar_t>(),
                    d_w.data<scalar_t>(),
                    grad_out.data<scalar_t>(),
                    v.data<scalar_t>(),
                    w.data<scalar_t>(),
                    v.size(0),
                    v.size(1),
                    v.size(2),
                    v.size(3),
                    v.size(4));
                }));
            }));
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return {d_v, d_w};
}

// This is the _adjoint_ version of the regular jacobian times vectorfield
// operation (Dv)w. The adjoint with respect to the action on w is trivial and
// is implemented via the 'transpose' argument to the above function. The below
// kernel is for the adjoint with respect to the linear action of w on v
template<typename Real, BackgroundStrategy backgroundStrategy>
__global__
void
jacobian_times_vectorfield_adjoint_forward_kernel_2d(Real* __restrict__ out, const Real* __restrict__ z, const Real* __restrict__ w,
        int nn, int nc, int nx, int ny) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int nxy = nx*ny;
    Real* outn = out;
    const Real* zn = z; // pointer to current vector field v
    const Real* wn = w;
    int ix = i*ny + j; // indices of first and second component of wn
    int iy = ix + nxy;
    for (int n=0; n < nn; ++n) {
        for (int c=0; c < nc; ++c) {
            if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP) {
                if (i == 0)
                    outn[ix] += -.5 * (wn[ix]*zn[ix] + wn[ix+ny]*zn[ix+ny]);
                else if (i == nx-1)
                    outn[ix] +=  .5 * (wn[ix]*zn[ix] + wn[ix-ny]*zn[ix-ny]);
                else
                    outn[ix] += -.5 * (wn[ix+ny]*zn[ix+ny] - wn[ix-ny]*zn[ix-ny]);

                if (j == 0)
                    outn[ix] += -.5 * (wn[iy]*zn[ix] + wn[iy+1]*zn[ix+1]);
                else if (j == ny-1)
                    outn[ix] +=  .5 * (wn[iy]*zn[ix] + wn[iy-1]*zn[ix-1]);
                else
                    outn[ix] += -.5 * (wn[iy+1]*zn[ix+1] - wn[iy-1]*zn[ix-1]);
            }
            outn += nxy;
            zn += nxy;
        }
        // increment w lookups
        wn += 2*nxy; // move to next component
    }
}
template<typename Real, BackgroundStrategy backgroundStrategy>
__global__
void
jacobian_times_vectorfield_adjoint_forward_kernel_3d(Real* __restrict__ out, const Real* __restrict__ z, const Real* __restrict__ w,
        int nn, int nc, int nx, int ny, int nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int nyz = ny*nz;
    const int nxyz = nx*ny*nz;
    Real* outn = out;
    const Real* zn = z; // pointer to current vector field v
    const Real* wn = w;
    for (int n=0; n < nn; ++n) {
        for (int c=0; c < nc; ++c) {
            for (int k=0; k < nz; ++k) {
                int ix = (i*ny + j)*nz + k; // indices of first and second component of wn
                int iy = ix + nxyz;
                int iz = iy + nxyz;
                if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP) {
                    if (i == 0)
                        outn[ix] += -.5 * (wn[ix+nyz]*zn[ix+nyz] + wn[ix    ]*zn[ix    ]);
                    else if (i == nx-1)
                        outn[ix] +=  .5 * (wn[ix    ]*zn[ix    ] + wn[ix-nyz]*zn[ix-nyz]);
                    else
                        outn[ix] += -.5 * (wn[ix+nyz]*zn[ix+nyz] - wn[ix-nyz]*zn[ix-nyz]);

                    if (j == 0)
                        outn[ix] += -.5 * (wn[iy+nz]*zn[ix+nz] + wn[iy   ]*zn[ix   ]);
                    else if (j == ny-1)
                        outn[ix] +=  .5 * (wn[iy   ]*zn[ix   ] + wn[iy-nz]*zn[ix-nz]);
                    else
                        outn[ix] += -.5 * (wn[iy+nz]*zn[ix+nz] - wn[iy-nz]*zn[ix-nz]);

                    if (k == 0)
                        outn[ix] += -.5 * (wn[iz+1]*zn[ix+1] + wn[iz  ]*zn[ix  ]);
                    else if (k == nz-1)
                        outn[ix] +=  .5 * (wn[iz  ]*zn[ix  ] + wn[iz-1]*zn[ix-1]);
                    else
                        outn[ix] += -.5 * (wn[iz+1]*zn[ix+1] - wn[iz-1]*zn[ix-1]);
                }
            }
            outn += nxyz;
            zn += nxyz;
        }
        // increment w lookups
        wn += 3*nxyz; // move to next component
    }
}

at::Tensor jacobian_times_vectorfield_adjoint_forward(
    at::Tensor g,
    at::Tensor v) {
    const dim3 threads(16, 32);
    const dim3 blocks((g.size(2) + threads.x - 1) / threads.x,
                    (g.size(3) + threads.y - 1) / threads.y);
    const auto d = g.dim()-2;
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional jacobian times vectorfield is supported")
    AT_ASSERTM(v.size(1) == d, "vector field is of wrong dimension")
    auto out = at::zeros_like(g);

    if (d == 2) {
        AT_DISPATCH_FLOATING_TYPES(g.type(), "jacobian_times_vectorfield_adjoint_forward", ([&] {
        jacobian_times_vectorfield_adjoint_forward_kernel_2d<scalar_t, BACKGROUND_STRATEGY_CLAMP><<<blocks, threads>>>(
            out.data<scalar_t>(),
            g.data<scalar_t>(),
            v.data<scalar_t>(),
            out.size(0),
            out.size(1),
            out.size(2),
            out.size(3));
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(g.type(), "jacobian_times_vectorfield_adjoint_forward", ([&] {
        jacobian_times_vectorfield_adjoint_forward_kernel_3d<scalar_t, BACKGROUND_STRATEGY_CLAMP><<<blocks, threads>>>(
            out.data<scalar_t>(),
            g.data<scalar_t>(),
            v.data<scalar_t>(),
            out.size(0),
            out.size(1),
            out.size(2),
            out.size(3),
            out.size(4));
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return out;
}

template<typename Real, BackgroundStrategy backgroundStrategy, bool need_v, bool need_w>
__global__
void
jacobian_times_vectorfield_adjoint_backward_kernel_2d(
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
        grad_point<Real, backgroundStrategy>(gx, gy, gon, nx, ny, i, j);
        if (need_w) {
            d_wn[ix] = gx*vn[ix];
            d_wn[iy] = gy*vn[ix];
        }
        if (need_v) {
            d_vn[ix] += gx*wn[ix] + gy*wn[iy];
            d_vn += nxy;
        }
        gon += nxy;
        grad_point<Real, backgroundStrategy>(gx, gy, gon, nx, ny, i, j);
        if (need_w) {
            d_wn[ix] += gx*vn[iy];
            d_wn[iy] += gy*vn[iy];
            vn += 2*nxy;
            d_wn += 2*nxy;
        }
        if (need_v) {
            d_vn[ix] += gx*wn[ix] + gy*wn[iy];
            d_vn += nxy;
            wn += 2*nxy;
        }
        gon += nxy;
    }
}

template<typename Real, BackgroundStrategy backgroundStrategy, bool need_v, bool need_w>
__global__
void
jacobian_times_vectorfield_adjoint_backward_kernel_3d(
        Real* __restrict__ d_v,
        Real* __restrict__ d_w,
        const Real* __restrict__ grad_out,
        const Real* __restrict__ v, const Real* __restrict__ w,
        int nn, int nc, int nx, int ny, int nz) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    Real gx, gy, gz; // gradient of component of v
    const int nxyz = nx*ny*nz;
    Real* d_vn = d_v;
    Real* d_wn = d_w;
    const Real* gon = grad_out;
    const Real* vn = v; // pointer to current vector field v
    const Real* wn = w;
    for (int n=0; n < nn; ++n) {
        int ix = (i*ny + j)*nz;
        int iy = ix + nxyz;
        int iz = iy + nxyz;
        for (int k=0; k < nz; ++k, ++ix, ++iy, ++iz) {
            grad_point<Real, backgroundStrategy>(gx, gy, gz, gon, nx, ny, nz, i, j, k);
            if (need_w) {
                d_wn[ix] = gx*vn[ix];
                d_wn[iy] = gy*vn[ix];
                d_wn[iz] = gz*vn[ix];
            }
            if (need_v) {
                d_vn[ix] += gx*wn[ix] + gy*wn[iy] + gz*wn[iz];
            }
            grad_point<Real, backgroundStrategy>(gx, gy, gz, gon+nxyz, nx, ny, nz, i, j, k);
            if (need_w) {
                d_wn[ix] += gx*vn[iy];
                d_wn[iy] += gy*vn[iy];
                d_wn[iz] += gz*vn[iy];
            }
            if (need_v) {
                d_vn[iy] += gx*wn[ix] + gy*wn[iy] + gz*wn[iz];
            }
            grad_point<Real, backgroundStrategy>(gx, gy, gz, gon+2*nxyz, nx, ny, nz, i, j, k);
            if (need_w) {
                d_wn[ix] += gx*vn[iz];
                d_wn[iy] += gy*vn[iz];
                d_wn[iz] += gz*vn[iz];
            }
            if (need_v) {
                d_vn[iz] += gx*wn[ix] + gy*wn[iy] + gz*wn[iz];
            }
        }
        vn += 3*nxyz;
        wn += 3*nxyz;
        d_vn += 3*nxyz;
        d_wn += 3*nxyz;
        gon += 3*nxyz;
    }
}


std::vector<at::Tensor> jacobian_times_vectorfield_adjoint_backward(
    at::Tensor grad_out,
    at::Tensor v,
    at::Tensor w,
    bool need_v,
    bool need_w) {
    need_v = true; // hardcode derivatives for both arguments for now
    need_w = true;

    const dim3 threads(16, 32);
    const dim3 blocks((v.size(2) + threads.x - 1) / threads.x,
                    (v.size(3) + threads.y - 1) / threads.y);

    const auto d = v.dim()-2;
    AT_ASSERTM(d == 2 || d == 3, "Only two- and three-dimensional jacobian times vectorfield is supported")
    AT_ASSERTM(w.size(1) == d, "vector field is of wrong dimension")

    at::Tensor d_v, d_w;
    if (need_v) d_v = at::zeros_like(v);
    if (need_w) d_w = at::zeros_like(w);

    if (d == 2) {
        AT_DISPATCH_FLOATING_TYPES(v.type(), "jacobian_times_vectorfield_adjoint_backward", ([&] {
        jacobian_times_vectorfield_adjoint_backward_kernel_2d<scalar_t, DEFAULT_BACKGROUND_STRATEGY, true, true><<<blocks, threads>>>(
            d_v.data<scalar_t>(),
            d_w.data<scalar_t>(),
            grad_out.data<scalar_t>(),
            v.data<scalar_t>(),
            w.data<scalar_t>(),
            v.size(0),
            v.size(1),
            v.size(2),
            v.size(3));
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(v.type(), "jacobian_times_vectorfield_adjoint_backward", ([&] {
        jacobian_times_vectorfield_adjoint_backward_kernel_3d<scalar_t, DEFAULT_BACKGROUND_STRATEGY, true, true><<<blocks, threads>>>(
            d_v.data<scalar_t>(),
            d_w.data<scalar_t>(),
            grad_out.data<scalar_t>(),
            v.data<scalar_t>(),
            w.data<scalar_t>(),
            v.size(0),
            v.size(1),
            v.size(2),
            v.size(3),
            v.size(4));
        }));
    }
	LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);

    return {d_v, d_w};
}
