# -*- mode: cuda -*-
# vi: set syntax=cuda:
# Much of this was borrowed from PyCA's GFluidKernelFFTKernels.cu
from .cudamod import CudaModule
mod = CudaModule('''
#include <cuComplex.h>
#include <stdio.h>

#if Real==float
#define SQRT sqrtf
#define SIN sinf
#define ACOS acosf
#define FLOOR floorf
#elif Real==double
#define SQRT sqrt
#define SIN sin
#define ACOS acos
#define FLOOR floor
#endif
#if Complex==cuFloatComplex
#define REALPART cuCrealf
#define IMAGPART cuCimagf
#define MAKECOMPLEX make_cuFloatComplex
#elif Complex==cuDoubleComplex
#define REALPART cuCreal
#define IMAGPART cuCimag
#define MAKECOMPLEX make_cuDoubleComplex
#endif

#define PI  3.14159265358979323846

__inline__ __device__ Real safe_sqrt(Real x) {
    if (x < 0.) return 0.;
    return SQRT(x);
}

__device__
void
InverseOperatorMultiply(Complex& bX,
                        Complex& bY,
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
    //    in Golub and VanLoan
    // Note: these are in matlab notation 1:3
    // [ G(1,1)   0      0    ]   [ G(1,1) G(2,1) G(3,1) ]
    // [ G(2,1) G(2,2)   0    ] * [   0    G(2,2) G(3,2) ] = Amatrix
    // [ G(3,1) G(3,2) G(3,3) ]   [   0      0    G(3,3) ]
    Real bRealX = bX.x;
    Real bRealY = bY.x;

    Real bImagX = bX.y;
    Real bImagY = bY.y;

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
    y0 = bRealX * ooG00;
    y1 = (bRealY - G10*y0) * ooG11;
    bY.x = y1 * ooG11;
    bX.x = (y0 - G10*bY.x) * ooG00;

    y0 = bImagX * ooG00;
    y1 = (bImagY - G10*y0) * ooG11;
    bY.y = y1 * ooG11;
    bX.y = (y0 - G10*bY.y) * ooG00;
}

__device__
void
OperatorMultiply(Complex& bX,
                 Complex& bY,
                 Real L00,
                 Real L10, Real L11) {
    Real bRealX = bX.x;
    Real bRealY = bY.x;
    Real bImagX = bX.y;
    Real bImagY = bY.y;

    bX.x = L00*bRealX + L10*bRealY;
    bY.x = L10*bRealX + L11*bRealY;
    bX.y = L00*bImagX + L10*bImagY;
    bY.y = L10*bImagX + L11*bImagY;
}

template<bool inverseOp>
__device__
void
fluid_kernel_2d(int i, int j,
        Complex* Fm,
        const Real* cosX, const Real* sinX,
        const Real* cosY, const Real* sinY,
        Real alpha, Real beta, Real gamma,
        int nn, int nx, int ny,
        int cutoffX, int cutoffY) {
    if (i >= nx || j >= ny) return;
    const Real wx = cosX[i];
    const Real wy = cosY[j];
    Real L00;
    Real L10, L11;

    const int nxy = nx*ny;
    int ix     = j + i * ny;
    int iy     = j + i * ny + nxy;

    // alpha and gamma parts are diagonal in Fourier
    const Real lambda = gamma + alpha * (wx + wy);

    Real l00 = lambda - beta * wx;
    Real l11 = lambda - beta * wy;
    Real l10 = beta * sinX[i] * sinY[j];
    // square this matrix
    L00 = l00*l00 + l10*l10;
    L10 = l00*l10 + l10*l11;
    L11 = l11*l11 + l10*l10;

    for (int n=0; n < nn; ++n, ix+=2*nxy, iy+=2*nxy) {
      //
      // compute L (it is symmetric, only need lower triangular part)
      //
      Complex Fmx=Fm[ix], Fmy=Fm[iy];
      if (inverseOp)
        InverseOperatorMultiply(Fmx, Fmy, L00, L10, L11);
      else
        OperatorMultiply(Fmx, Fmy, L00, L10, L11);
      Fm[ix].x = Fmx.x;
      Fm[ix].y = Fmx.y;
      Fm[iy].x = Fmy.x;
      Fm[iy].y = Fmy.y;

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
        weight = 0.5*(1-SIN(PI*(rF-1.0)/0.6));
      } else {
        weight = 0.0;
      }

      Fm[ix].x *= weight;
      Fm[ix].y *= weight;
      Fm[iy].x *= weight;
      Fm[iy].y *= weight;
    }
}

extern "C" {
    __global__ void forward_operator_2d(Complex* Fm,
            const Real* cosX, const Real* sinX,
            const Real* cosY, const Real* sinY,
            const Real alpha, const Real beta, const Real gamma,
            const int nn, const int nx, const int ny,
            const int cutoffX, const int cutoffY) {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        fluid_kernel_2d<false>(i, j, Fm, cosX, sinX, cosY, sinY,
            alpha, beta, gamma, nn, nx, ny, cutoffX, cutoffY);
    }
    __global__ void inverse_operator_2d(Complex* Fm,
            const Real* cosX, const Real* sinX,
            const Real* cosY, const Real* sinY,
            const Real alpha, const Real beta, const Real gamma,
            const int nn, const int nx, const int ny,
            const int cutoffX, const int cutoffY) {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        fluid_kernel_2d<true>(i, j, Fm, cosX, sinX, cosY, sinY,
            alpha, beta, gamma, nn, nx, ny, cutoffX, cutoffY);
    }
}

''')
forward_operator_2d = mod.func("forward_operator_2d")
inverse_operator_2d = mod.func("inverse_operator_2d")
