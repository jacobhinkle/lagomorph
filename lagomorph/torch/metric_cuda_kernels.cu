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
    if (x < 0.) return 0.;
    return sqrt(x);
}

template <typename Real>
__device__
void
InverseOperatorMultiply(Real& bXr,
                        Real& bXi,
                        Real& bYr,
                        Real& bYi,
                        Real L00,
                        Real L10, Real L11) {
    Real ooG00;
    Real G10, ooG11;
    Real y0, y1;
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
    y0 = bXr * ooG00;
    y1 = (bYr - G10*y0) * ooG11;
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
OperatorMultiply(Real& bXr,
                 Real& bXi,
                 Real& bYr,
                 Real& bYi,
                 Real L00,
                 Real L10, Real L11) {
    bXr = L00*bXr + L10*bYr;
    bYr = L10*bXr + L11*bYr;
    bXi = L00*bXi + L10*bYi;
    bYi = L10*bXi + L11*bYi;
}

template <typename Real, bool inverseOp>
__global__
void
fluid_kernel_2d(Real* __restrict__ Fm,
        const Real* __restrict__ cosX, const Real* __restrict__ sinX,
        const Real* __restrict__ cosY, const Real* __restrict__ sinY,
        double alpha, double beta, double gamma,
        size_t nn, size_t nx, size_t ny,
        size_t cutoffX, size_t cutoffY) {
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

    for (size_t n=0; n < nn; ++n, ix+=2*nxy, iy+=2*nxy) {
      //
      // compute L (it is symmetric, only need lower triangular part)
      //
      Real Fmxr=Fm[ix], Fmxi=Fm[ix+1], Fmyr=Fm[iy], Fmyi=Fm[iy+1];
      if (inverseOp)
        InverseOperatorMultiply(Fmxr, Fmyr, Fmxi, Fmyi, L00, L10, L11);
      else
        OperatorMultiply(Fmxr, Fmyr, Fmxi, Fmyi, L00, L10, L11);
      Fm[ix] = Fmxr;
      Fm[ix+1] = Fmxi;
      Fm[iy] = Fmyr;
      Fm[iy+1] = Fmyi;

      // set to zero outside cutoff (except if cutoff is 0)
      if (cutoffX == 0 && cutoffY == 0)
        continue;

      Real weight = 1.0f;
      // change coordinates from eclipse to circle
      // use fft coordinates, and rescale to S^1
      Real xF, yF, rF = 0.0f;
      if (cutoffX > 0)
        xF = static_cast<Real>(min(i, nx-1-i))/static_cast<Real>(cutoffX);
      if (cutoffY > 0)
        yF = static_cast<Real>(min(j, ny-1-j))/static_cast<Real>(cutoffY);

      rF = safe_sqrt(xF*xF + yF*yF);
      if (rF <= .7) {
        weight = 1.0;
      } else if (rF < 1.3) {
        // soft threshold at 1, this is transition
        weight = 0.5*(1-sin(PI*(rF-1.0)/0.6));
      } else {
        weight = 0.0;
      }
      if (inverseOp && weight > 0.0) weight = 1./weight;

      Fm[ix] *= weight;
      Fm[ix+1] *= weight;
      Fm[iy] *= weight;
      Fm[iy+1] *= weight;
    }
}

template<typename Real, typename Complex, bool inverseOp>
__global__ void operator_2d(Complex* Fm,
        const Real* cosX, const Real* sinX,
        const Real* cosY, const Real* sinY,
        const double alpha, const double beta, const double gamma,
        const int nn, const int nx, const int ny,
        const int cutoffX, const int cutoffY) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    fluid_kernel_2d<inverseOp>(i, j, Fm, cosX, sinX, cosY, sinY,
        alpha, beta, gamma, nn, nx, ny, cutoffX, cutoffY);
}

void fluid_operator_cuda(
    at::Tensor Fmv,
    bool inverse,
    std::vector<at::Tensor> cosluts,
    std::vector<at::Tensor> sinluts,
    double alpha,
    double beta,
    double gamma,
    const int cutoffX,
    const int cutoffY) {
    auto dim = Fmv.dim() - 3;
    AT_ASSERTM(Fmv.size(1) == dim, "Vector field has incorrect shape for dimension")
    AT_ASSERTM(Fmv.type() == cosluts[0].type(), "Type of LUTs must equal that of image")

    const dim3 threads(16, 32);
    const dim3 blocks((Fmv.size(2) + threads.x - 1) / threads.x,
                    (Fmv.size(3) + threads.y - 1) / threads.y);

    if (dim == 2) {
        if (inverse) {
            AT_DISPATCH_FLOATING_TYPES(Fmv.type(), "fluid_operator_cuda", ([&] {
                fluid_kernel_2d<scalar_t, true><<<blocks, threads>>>(
                    Fmv.data<scalar_t>(),
                    cosluts[0].data<scalar_t>(),
                    sinluts[0].data<scalar_t>(),
                    cosluts[1].data<scalar_t>(),
                    sinluts[1].data<scalar_t>(),
                    alpha, beta, gamma,
                    Fmv.size(0), Fmv.size(2), Fmv.size(3),
                    cutoffX, cutoffY);
                }));
        } else {
            AT_DISPATCH_FLOATING_TYPES(Fmv.type(), "fluid_operator_cuda", ([&] {
                fluid_kernel_2d<scalar_t, false><<<blocks, threads>>>(
                    Fmv.data<scalar_t>(),
                    cosluts[0].data<scalar_t>(),
                    sinluts[0].data<scalar_t>(),
                    cosluts[1].data<scalar_t>(),
                    sinluts[1].data<scalar_t>(),
                    alpha, beta, gamma,
                    Fmv.size(0), Fmv.size(2), Fmv.size(3),
                    cutoffX, cutoffY);
                }));
        }
    } else if (dim == 3) {
    } else {
    }
    LAGOMORPH_CUDA_CHECK(__FILE__,__LINE__);
}
