# vim: set syntax=cuda:
# Much of this was borrowed from PyCA's GFluidKernelFFTKernels.cu
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from pycuda import gpuarray

_cu = '''
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
    Real G00;
    Real G10, G11;
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

    Real bRealX = REALPART(bX);
    Real bRealY = REALPART(bY);

    Real bImagX = IMAGPART(bX);
    Real bImagY = IMAGPART(bY);

    Real vRealX = bRealX;
    Real vRealY = bRealY;

    Real vImagX = bImagX;
    Real vImagY = bImagY;

    G00 = safe_sqrt(L00);
    G10 = L10 / G00;

    G11 = L11 - G10 * G10;
    G11 = safe_sqrt(G11);

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
    y0 = bRealX / G00;
    y1 = (bRealY - G10*y0) / G11;

    vRealY = y1 / G11;
    vRealX = (y0 - G10*vRealY) / G00;

    y0 = bImagX / G00;
    y1 = (bImagY - G10*y0) / G11;

    vImagY = y1 / G11;
    vImagX = (y0 - G10*vImagY) / G00;

    // set outputs
    bX = MAKECOMPLEX(vRealX, vImagX);
    bY = MAKECOMPLEX(vRealY, vImagY);
}

__device__
void
OperatorMultiply(Complex& bX,
                 Complex& bY,
                 Real L00,
                 Real L10, Real L11) {
    Real bRealX = REALPART(bX);
    Real bRealY = REALPART(bY);

    Real bImagX = IMAGPART(bX);
    Real bImagY = IMAGPART(bY);

    Real vRealX = L00*bRealX + L10*bRealY;
    Real vRealY = L10*bRealX + L11*bRealY;

    Real vImagX = L00*bImagX + L10*bImagY;
    Real vImagY = L10*bImagX + L11*bImagY;

    bX = MAKECOMPLEX(vRealX, vImagX);
    bY = MAKECOMPLEX(vRealY, vImagY);
}

template<bool inverseOp>
inline __device__
void
fluid_kernel_2d(const int i, const int j,
        Complex* Fm,
        const Real* cosX, const Real* sinX,
        const Real* cosY, const Real* sinY,
        const Real alpha, const Real beta, const Real gamma,
        const int nn, const int nx, const int ny,
        const int cutoffX, const int cutoffY, const int cutoffZ) {
    Real wx = cosX[i];
    Real wy = cosY[j];
    Real lambda;
    Real L00;
    Real L10, L11;

    const int planeSize = nx * ny;
    int ix     = i + j * nx;
    int iy     = i + j * nx + planeSize;

    // alpha and gamma parts are diagonal in Fourier
    lambda = -alpha * (wx + wy) + gamma;

    L00 = lambda - beta * cosX[i];
    L11 = lambda - beta * cosY[j];
    L10 = beta * sinX[i] * sinY[j];

    for (int n=0; n < nn ; ++n, ix+=planeSize, iy+=planeSize) {
      //
      // compute L (it is symmetric, only need lower triangular part)
      //
      if (inverseOp)
        InverseOperatorMultiply(Fm[ix], Fm[iy], L00, L10, L11);
      else
        OperatorMultiply(Fm[ix], Fm[iy], L00, L10, L11);

      // set to zero outside cutoff (except if cutoff is 0)
      if (cutoffX == 0 && cutoffY == 0 && cutoffZ == 0)
        continue;

      Real weight = 1.0f;
      // change coordinates from eclipse to circle
      // use fft coordinates, and rescale to S^1
      Real xF, yF, rF = 0.0f;
      if (cutoffX > 0)
        xF = static_cast<Real>(i)/static_cast<Real>(cutoffX);
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
            const int cutoffX, const int cutoffY, const int cutoffZ) {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        fluid_kernel_2d<false>(i, j, Fm, cosX, sinX, cosY, sinY,
            alpha, beta, gamma, nn, nx, ny, cutoffX, cutoffY, cutoffZ);
    }
    __global__ void inverse_operator_2d(Complex* Fm,
            const Real* cosX, const Real* sinX,
            const Real* cosY, const Real* sinY,
            const Real alpha, const Real beta, const Real gamma,
            const int nn, const int nx, const int ny,
            const int cutoffX, const int cutoffY, const int cutoffZ) {
        const int i = blockDim.x * blockIdx.x + threadIdx.x;
        const int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        fluid_kernel_2d<true>(i, j, Fm, cosX, sinX, cosY, sinY,
            alpha, beta, gamma, nn, nx, ny, cutoffX, cutoffY, cutoffZ);
    }
}

'''

def getmod(precision='single'):
    if precision == 'single':
        nvcc_flags = DEFAULT_NVCC_FLAGS + ['-DReal=float', '-DComplex=cuFloatComplex']
    elif precision == 'double':
        nvcc_flags = DEFAULT_NVCC_FLAGS + ['-DReal=double', '-DComplex=cuDoubleComplex']
    else:
        raise Exception('Unrecognized precision: {}'.format(precision))

    return SourceModule(_cu, options=nvcc_flags, no_extern_c=True)

class CudaFunc:
    def __init__(self, func_name):
        self.name = func_name
        self.mods = {}
    def __call__(self, *args, precision='double', **kwargs):
        if not precision in self.mods:
            self.mods[precision] = getmod(precision)
        mod = self.mods[precision]
        f = mod.get_function(self.name)
        return f(*args, **kwargs)

inverse_operator_2d = CudaFunc("inverse_operator_2d")
