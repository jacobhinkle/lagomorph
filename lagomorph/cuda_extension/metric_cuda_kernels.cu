#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "atomic.cuh"
#include "defs.cuh"
#include "interp.cuh"

#define PI  3.14159265358979323846

template <typename Real>
__inline__ __device__ Real safe_sqrt(Real x) {
    if (x < 1e-8) return 1e-4;
    return sqrt(x);
}

template <typename Real>
__device__
void
CholeskyFactor( Real& ooG00,
                Real& G10, Real& ooG11,
                Real L00,
                Real L10, Real L11) {
    //
    // Given that A is pos-def symetric matrix, solve Ax=b by finding
    // cholesky decomposition GG'=A
    // and then performing 2 back-solves, Gy=b and then G'x=y to get x.
    //

    // 1. find cholesky decomposition by finding G such that GG'=A.
    //    A must be positive definite symetric (we assume that here)
    //    G is then lower triangular, see algorithm 4.2.1 p142-3
    //    in Golub and Van Loan
    // Note: these are in matlab notation 1:3
    // [ G(1,1)   0      0    ]   [ G(1,1) G(2,1) G(3,1) ]
    // [ G(2,1) G(2,2)   0    ] * [   0    G(2,2) G(3,2) ] = Amatrix
    // [ G(3,1) G(3,2) G(3,3) ]   [   0      0    G(3,3) ]
    ooG00 = 1./safe_sqrt(L00);
    G10 = L10 * ooG00;
    ooG11 = L11 - G10 * G10;
    ooG11 = 1./safe_sqrt(ooG11);
}

template <typename Real>
__device__
void
CholeskyFactor( Real& ooG00,
                Real&   G10, Real& ooG11,
                Real&   G20, Real&   G21, Real& ooG22,
                Real L00,
                Real L10, Real L11,
                Real L20, Real L21, Real L22) {
    //
    // Given that A is pos-def symetric matrix, solve Ax=b by finding
    // cholesky decomposition GG'=A
    // and then performing 2 back-solves, Gy=b and then G'x=y to get x.
    //

    // 1. find cholesky decomposition by finding G such that GG'=A.
    //    A must be positive definite symetric (we assume that here)
    //    G is then lower triangular, see algorithm 4.2.1 p142-3
    //    in Golub and Van Loan
    // Note: these are in matlab notation 1:3
    // [ G(1,1)   0      0    ]   [ G(1,1) G(2,1) G(3,1) ]
    // [ G(2,1) G(2,2)   0    ] * [   0    G(2,2) G(3,2) ] = Amatrix
    // [ G(3,1) G(3,2) G(3,3) ]   [   0      0    G(3,3) ]
    ooG00 = 1./safe_sqrt(L00);
    G10 = L10 * ooG00;
    G20 = (L20) * ooG00;
    ooG11 = L11 - G10*G10;
    ooG11 = 1./safe_sqrt(ooG11);
    G21 = (L21 - G20*G10) * ooG11;
    ooG22 = L22 - G20*G20 - G21*G21;
    ooG22 = 1./safe_sqrt(ooG22);
}

template <typename Real>
__device__
void
CholeskySolve(  Real& bXr,
                Real& bXi,
                Real& bYr,
                Real& bYi,
                Real ooG00,
                Real G10, Real ooG11) {
    // back-solve Gy=b to get a temporary vector y
    // back-solve G'x=y to get answer in x
    //
    // Note: these are in matlab notation 1:3
    // [ G(1,1)   0      0    ]   [ y(1) ] = b(1)
    // [ G(2,1) G(2,2)   0    ] * [ y(2) ] = b(2)
    // [ G(3,1) G(3,2) G(3,3) ]   [ y(3) ] = b(3)
    //
    // [ G(1,1) G(2,1) G(3,1) ]   [ x(1) ] = y(1)
    // [   0    G(2,2) G(3,2) ] * [ x(2) ] = y(2)
    // [   0      0    G(3,3) ]   [ x(3) ] = y(3)
    auto y0 = bXr * ooG00;
    auto y1 = (bYr - G10*y0) * ooG11;
    bYr = y1 * ooG11;
    bXr = (y0 - G10*bYr) * ooG00;

    y0 = bXi * ooG00;
    y1 = (bYi - G10*y0) * ooG11;
    bYi = y1 * ooG11;
    bXi = (y0 - G10*bYi) * ooG00;
}

template <typename Real>
__device__
void
CholeskySolve(  Real& bXr,
                Real& bXi,
                Real& bYr,
                Real& bYi,
                Real& bZr,
                Real& bZi,
                Real ooG00,
                Real   G10, Real ooG11,
                Real   G20, Real   G21, Real ooG22) {
    // back-solve Gy=b to get a temporary vector y
    // back-solve G'x=y to get answer in x
    //
    // Note: these are in matlab notation 1:3
    // [ G(1,1)   0      0    ]   [ y(1) ] = b(1)
    // [ G(2,1) G(2,2)   0    ] * [ y(2) ] = b(2)
    // [ G(3,1) G(3,2) G(3,3) ]   [ y(3) ] = b(3)
    //
    // [ G(1,1) G(2,1) G(3,1) ]   [ x(1) ] = y(1)
    // [   0    G(2,2) G(3,2) ] * [ x(2) ] = y(2)
    // [   0      0    G(3,3) ]   [ x(3) ] = y(3)
    auto y0 = bXr * ooG00;
    auto y1 = (bYr - G10*y0) * ooG11;
    auto y2 = (bZr - G20*y0 - G21*y1) * ooG22;
    bZr = y2 * ooG22;
    bYr = (y1 - G21*bZr) * ooG11;
    bXr = (y0 - G10*bYr - G20*bZr) * ooG00;

    y0 = bXi * ooG00;
    y1 = (bYi - G10*y0) * ooG11;
    y2 = (bZi - G20*y0 - G21*y1) * ooG22;
    bZi = y2 * ooG22;
    bYi = (y1 - G21*bZi) * ooG11;
    bXi = (y0 - G10*bYi - G20*bZi) * ooG00;
}

template <typename Real>
__device__
void
OperatorMultiply(Real& bXr,
                 Real& bXi,
                 Real& bYr,
                 Real& bYi,
                 Real L00,
                 Real L10, Real L11) {
    // We need a temporary to avoid aliasing
    auto x = L00*bXr + L10*bYr;
    bYr = L10*bXr + L11*bYr;
    bXr = x;
    x = L00*bXi + L10*bYi;
    bYi = L10*bXi + L11*bYi;
    bXi = x;
}

template <typename Real>
__device__
void
OperatorMultiply(Real& bXr,
                 Real& bXi,
                 Real& bYr,
                 Real& bYi,
                 Real& bZr,
                 Real& bZi,
                 Real L00,
                 Real L10, Real L11,
                 Real L20, Real L21, Real L22) {
    // We need a temporary to avoid aliasing
    auto x = L00*bXr + L10*bYr + L20*bZr;
    auto y = L10*bXr + L11*bYr + L21*bZr;
    bZr = L20*bXr + L21*bYr + L22*bZr;
    bXr = x;
    bYr = y;
    x = L00*bXi + L10*bYi + L20*bZi;
    y = L10*bXi + L11*bYi + L21*bZi;
    bZi = L20*bXi + L21*bYi + L22*bZi;
    bXi = x;
    bYi = y;
}

template <typename Real, bool inverseOp>
__global__
void
fluid_kernel_2d(Real* __restrict__ Fm,
        const Real* __restrict__ cosX, const Real* __restrict__ sinX,
        const Real* __restrict__ cosY, const Real* __restrict__ sinY,
        double alpha, double beta, double gamma,
        size_t nn, size_t nx, size_t ny) {
    const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const Real wx = cosX[i];
    const Real wy = cosY[j];
    Real L00;
    Real L10, L11;

    const auto nxy = 2*nx*ny;
    // indices into the FFT
    auto ix     = 2*(j + i * ny);
    auto iy     = ix + nxy;

    // alpha and gamma parts are diagonal in Fourier
    const Real lambda = gamma + alpha * (wx + wy);

    Real l00 = lambda - beta * wx;
    Real l11 = lambda - beta * wy;
    Real l10 = beta * sinX[i] * sinY[j];
    // square this matrix
    L00 = l00*l00 + l10*l10;
    L10 = l00*l10 + l10*l11;
    L11 = l11*l11 + l10*l10;

    Real ooG00, G10, ooG11;
    if (inverseOp) { // compute Cholesky factor once, apply to all in batch
        CholeskyFactor<Real>(ooG00,
                         G10, ooG11,
                         L00,
                         L10, L11);
    }
    for (size_t n=0; n < nn; ++n, ix+=2*nxy, iy+=2*nxy) {
      //
      // compute L (it is symmetric, only need lower triangular part)
      //
      Real Fmxr=Fm[ix], Fmxi=Fm[ix+1], Fmyr=Fm[iy], Fmyi=Fm[iy+1];
      if (inverseOp)
        CholeskySolve<Real>(Fmxr, Fmxi, Fmyr, Fmyi, ooG00, G10, ooG11);
      else
        OperatorMultiply(Fmxr, Fmxi, Fmyr, Fmyi, L00, L10, L11);
      Fm[ix] = Fmxr;
      Fm[ix+1] = Fmxi;
      Fm[iy] = Fmyr;
      Fm[iy+1] = Fmyi;
    }
}

template <typename Real, bool inverseOp>
__global__
void
fluid_kernel_3d(Real* __restrict__ Fm,
        const Real* __restrict__ cosX, const Real* __restrict__ sinX,
        const Real* __restrict__ cosY, const Real* __restrict__ sinY,
        const Real* __restrict__ cosZ, const Real* __restrict__ sinZ,
        double alpha, double beta, double gamma,
        size_t nn, size_t nx, size_t ny, size_t nz) {
    const size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const Real wx = cosX[i];
    const Real wy = cosY[j];

    const auto nxyz = 2*nx*ny*nz;
    // indices into the FFT
    auto ix     = 2*(j + i * ny)*nz;
    auto iy     = ix + nxyz;
    auto iz     = iy + nxyz;

    for (size_t k=0; k < nz; ++k) {
        const Real wz = cosZ[k];
        // alpha and gamma parts are diagonal in Fourier
        const Real lambda = gamma + alpha * (wx + wy + wz);

        Real l00 = lambda - beta * wx;
        Real l11 = lambda - beta * wy;
        Real l22 = lambda - beta * wz;
        Real l10 = beta * sinX[i] * sinY[j];
        Real l20 = beta * sinX[i] * sinZ[k];
        Real l21 = beta * sinY[j] * sinZ[k];
        // square this real-valued symmetric matrix
        auto L00 = l00*l00 + l10*l10 + l20*l20;
        auto L10 = l00*l10 + l10*l11 + l20*l21;
        auto L11 = l10*l10 + l11*l11 + l21*l21;
        auto L20 = l00*l20 + l10*l21 + l20*l22;
        auto L21 = l10*l20 + l11*l21 + l21*l22;
        auto L22 = l20*l20 + l21*l21 + l22*l22;


        Real  ooG00,
                G10, ooG11,
                G20,   G21, ooG22;
        if (inverseOp) { // compute Cholesky factor once, apply to all in batch
            CholeskyFactor<Real>(ooG00,
                                   G10, ooG11,
                                   G20,   G21, ooG22,
                                   L00,
                                   L10, L11,
                                   L20, L21, L22);
        }
        for (size_t n=0; n < nn; ++n, ix+=3*nxyz, iy+=3*nxyz, iz+=3*nxyz) {
          //
          // compute L (it is symmetric, only need lower triangular part)
          //
          Real Fmxr=Fm[ix], Fmxi=Fm[ix+1];
          Real Fmyr=Fm[iy], Fmyi=Fm[iy+1];
          Real Fmzr=Fm[iz], Fmzi=Fm[iz+1];
          if (inverseOp)
            CholeskySolve<Real>(Fmxr, Fmyr, Fmzr, Fmxi, Fmyi, Fmzi,
                                 ooG00,
                                   G10, ooG11,
                                   G20,   G21, ooG22);
          else
            OperatorMultiply(Fmxr, Fmyr, Fmzr, Fmxi, Fmyi, Fmzi,
                L00,
                L10, L11,
                L20, L21, L22);
          Fm[ix+2*k] = Fmxr;
          Fm[ix+2*k+1] = Fmxi;
          Fm[iy+2*k] = Fmyr;
          Fm[iy+2*k+1] = Fmyi;
          Fm[iz+2*k] = Fmzr;
          Fm[iz+2*k+1] = Fmzi;
        }
    }
}

void fluid_operator_cuda(
    at::Tensor Fmv,
    bool inverse,
    std::vector<at::Tensor> cosluts,
    std::vector<at::Tensor> sinluts,
    double alpha,
    double beta,
    double gamma) {
    auto dim = Fmv.dim() - 3;
    AT_ASSERTM(dim == 2 || dim == 3, "Only two- and three-dimensional fluid metric is supported")
    AT_ASSERTM(Fmv.size(1) == dim, "Vector field has incorrect shape for dimension")
    AT_ASSERTM(Fmv.type() == cosluts[0].type(), "Type of LUTs must equal that of image")

    const dim3 threads(16, 32);
    const dim3 blocks((Fmv.size(2) + threads.x - 1) / threads.x,
                    (Fmv.size(3) + threads.y - 1) / threads.y);

    if (dim == 2) {
        LAGOMORPH_DISPATCH_BOOL(inverse, do_inverse, ([&] {
            AT_DISPATCH_FLOATING_TYPES(Fmv.type(), "fluid_operator_cuda", ([&] {
                fluid_kernel_2d<scalar_t, do_inverse><<<blocks, threads>>>(
                    Fmv.data<scalar_t>(),
                    cosluts[0].data<scalar_t>(),
                    sinluts[0].data<scalar_t>(),
                    cosluts[1].data<scalar_t>(),
                    sinluts[1].data<scalar_t>(),
                    alpha, beta, gamma,
                    Fmv.size(0), Fmv.size(2), Fmv.size(3));
                }));
            }));
    } else if (dim == 3) {
        LAGOMORPH_DISPATCH_BOOL(inverse, do_inverse, ([&] {
            AT_DISPATCH_FLOATING_TYPES(Fmv.type(), "fluid_operator_cuda", ([&] {
                fluid_kernel_3d<scalar_t, do_inverse><<<blocks, threads>>>(
                    Fmv.data<scalar_t>(),
                    cosluts[0].data<scalar_t>(),
                    sinluts[0].data<scalar_t>(),
                    cosluts[1].data<scalar_t>(),
                    sinluts[1].data<scalar_t>(),
                    cosluts[2].data<scalar_t>(),
                    sinluts[2].data<scalar_t>(),
                    alpha, beta, gamma,
                    Fmv.size(0), Fmv.size(2), Fmv.size(3), Fmv.size(4));
                }));
            }));
    }
    LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);
}
