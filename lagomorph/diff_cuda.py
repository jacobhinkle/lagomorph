# vim: set syntax=cuda:
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
from pycuda import gpuarray

_cu = '''
#include <stdio.h>

enum BackgroundStrategy { BACKGROUND_STRATEGY_PARTIAL_ID,
                          BACKGROUND_STRATEGY_ID,
                          BACKGROUND_STRATEGY_PARTIAL_ZERO,
                          BACKGROUND_STRATEGY_ZERO,
                          BACKGROUND_STRATEGY_CLAMP,
                          BACKGROUND_STRATEGY_WRAP,
			  BACKGROUND_STRATEGY_VAL};

inline __device__ Real get_pixel_3d(int x, int y, int z,
                                    const Real* d_i,
                                    int sizeX, int sizeY, int sizeZ)
{
    int index = (z * sizeY + y) * sizeX + x;
    return d_i[index];
}

inline __device__ Real get_pixel_2d(int x, int y,
                                    const Real* d_i,
                                    int sizeX, int sizeY)
{
    int index =  y * sizeX + x;
    return d_i[index];
}

// Wrap around strategy
// Old version
// __device__ void wrap(int& r, int b){
//     if (r < 0) r += b;
//     else if (r > b) r %= b;
// }
inline __device__ int safe_mod(int r, int b){
    int m = r % b;
    return (m < 0) ? m + b : m;
}

inline __device__ void wrap(int& r, int b){
    r %= b;
    if (r < 0) {
        r += b;
    }
}

inline __device__ void wrapBackground(int& floorX,
                                      int& ceilX,
                                      int sizeX) {
    wrap(floorX, sizeX);
    wrap(ceilX, sizeX);
}
inline __device__ void wrapBackground(int& floorX,int& floorY,
                                      int& ceilX, int& ceilY,
                                      int  sizeX, int  sizeY) {
    wrapBackground(floorX, ceilX, sizeX);
    wrapBackground(floorY, ceilY, sizeY);
}
inline __device__ void wrapBackground(int& floorX,int& floorY,int& floorZ,
                                      int& ceilX, int& ceilY, int& ceilZ,
                                      int  sizeX, int  sizeY, int  sizeZ) {
    wrapBackground(floorX, ceilX, sizeX);
    wrapBackground(floorY, ceilY, sizeY);
    wrapBackground(floorZ, ceilZ, sizeZ);
}

// Clamp strategy
inline __device__ void clamp(int& r, int b){
    if (r < 0) r = 0;
    else if (r >= b) r = b - 1;
}

inline __device__ void clampBackground(int& floorX,
                                       int& ceilX,
                                       int sizeX) {
    if(floorX < 0) {
      floorX = 0;
      if(ceilX < 0) ceilX = 0;
    }
    if(ceilX >= sizeX) {
      ceilX = sizeX-1;
      if(floorX >= sizeX) floorX = sizeX-1;
    }
}
inline __device__ void clampBackground(int& floorX, int& floorY,
                                       int& ceilX, int& ceilY,
                                       int  sizeX, int  sizeY) {
    clampBackground(floorX, ceilX, sizeX);
    clampBackground(floorY, ceilY, sizeY);
}
inline __device__ void clampBackground(int& floorX, int& floorY, int& floorZ,
                                       int& ceilX, int& ceilY, int& ceilZ,
                                       int  sizeX, int  sizeY, int  sizeZ) {
    clampBackground(floorX, ceilX, sizeX);
    clampBackground(floorY, ceilY, sizeY);
    clampBackground(floorZ, ceilZ, sizeZ);
}


// Check if the point is completely inside the boundary
inline __device__ bool isInside(int floorX, int floorY,
                                int ceilX, int ceilY,
                                int  sizeX, int  sizeY){

    return (floorX >= 0 && ceilX < sizeX &&
            floorY >= 0 && ceilY < sizeY);
}
inline __device__ bool isInside(int floorX,int floorY,int floorZ,
                                int ceilX, int ceilY, int ceilZ,
                                int  sizeX, int  sizeY, int  sizeZ){

    return (floorX >= 0 && ceilX < sizeX &&
            floorY >= 0 && ceilY < sizeY &&
            floorZ >= 0 && ceilZ < sizeZ);
}

template<BackgroundStrategy backgroundStrategy>
inline __device__
Real
get_value_safe(const Real* arr, int nx, int ny, int i, int j, Real background=0.0f) {
    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(i, nx);
        wrap(j, ny);
        return arr[j*nx + i];
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(i, nx);
        clamp(j, ny);
        return arr[j*nx + i];
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){

	if(backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	    background = 0.f;
	}

        if (i >= 0 && i < nx && j >= 0 && j < ny)
            return arr[j*nx + i];
        else
            return background;
    }else{
	// unknown background strategy, don't allow compilation
	static_assert(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_VAL,
                      "Unknown background strategy");
	return 0.f;
    }
}

template<BackgroundStrategy backgroundStrategy>
inline __device__
void
grad_point(Real& gx, Real& gy,
        const Real* arr,
        int nx, int ny,
        int i, int j) {
    gx = 0.5f*(get_value_safe<backgroundStrategy>(arr, nx, ny, i+1, j)
             - get_value_safe<backgroundStrategy>(arr, nx, ny, i-1, j));
    gy = 0.5f*(get_value_safe<backgroundStrategy>(arr, nx, ny, i, j+1)
             - get_value_safe<backgroundStrategy>(arr, nx, ny, i, j-1));
}

// Just a simple image gradient
template<BackgroundStrategy backgroundStrategy>
inline __device__
void
gradient_kernel(Real* out, const Real* im,
        int nn, int nx, int ny, int i, int j) {
    Real gx, gy; // gradient of im
    int nxy = nx*ny;
    const Real* imn = im; // pointer to current image
    // index of current output point (first component. add nxy for second)
    int ino = j*nx + i;
    for (int n=0; n < nn; ++n) {
        grad_point<backgroundStrategy>(gx, gy, imn, nx, ny, i, j);
        out[ino] = gx;
        ino += nxy;
        out[ino] = gy;
        ino += nxy;
        imn += nxy;
    }
}

// Templated function to compute the Jacobian matrix of the first vector field
// and contract it with the second vector field in a point-wise fashion. The
// Jacobian will be transposed first if the template argument 'transpose' is 1
// instead of 0.
template<BackgroundStrategy backgroundStrategy, int transpose>
inline __device__
void
jacobian_times_vectorfield_kernel(Real* out, const Real* v, const Real* w,
        int nn, int nx, int ny, int i, int j) {
    Real gx, gy, gx2, gy2; // gradient of component of v
    int nxy = nx*ny;
    const Real* vn = v; // pointer to current vector field v
    // index of current output point (first component. add nxy for second)
    int ino = j*nx + i;
    int inx = ino;
    int iny = nxy + ino;
    for (int n=0; n < nn; ++n) {
        if (transpose) {
            // get gradient of each component of vn
            grad_point<backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
            vn += nxy; // move to next component
            grad_point<backgroundStrategy>(gx2, gy2, vn, nx, ny, i, j);
            out[ino] = gx*w[inx] + gx2*w[iny];
            ino += nxy;
            out[ino] = gy*w[inx] + gy2*w[iny];
            vn += nxy; // move to next image
            ino += nxy;
        } else {
            // get gradient of each component of vn
            grad_point<backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
            out[ino] = gx*w[inx] + gy*w[iny];
            vn += nxy; // move to next component
            ino += nxy;
            grad_point<backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
            out[ino] = gx*w[inx] + gy*w[iny];
            vn += nxy; // move to next image
            ino += nxy;
        }
        // increment w lookups
        inx += 2*nxy;
        iny += 2*nxy;
    }
}

extern "C" {
    __global__ void gradient_2d(Real* out,
            const Real* im,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        gradient_kernel<DEFAULT_BACKGROUND_STRATEGY>(out, im, nn, nx, ny, i, j);
    }
    __global__ void jacobian_times_vectorfield_2d(Real* out,
            const Real* v, const Real* w,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        jacobian_times_vectorfield_kernel<DEFAULT_BACKGROUND_STRATEGY, 0>(
            out, v, w, nn, nx, ny, i, j);
    }
    __global__ void jacobian_transpose_times_vectorfield_2d(Real* out,
            const Real* v, const Real* w,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        jacobian_times_vectorfield_kernel<DEFAULT_BACKGROUND_STRATEGY, 1>(
            out, v, w, nn, nx, ny, i, j);
    }
}

'''

def getmod(precision='single', background_strategy='BACKGROUND_STRATEGY_CLAMP'):
    nvcc_flags = DEFAULT_NVCC_FLAGS + ['-std=c++11',
            f'-DDEFAULT_BACKGROUND_STRATEGY={background_strategy}']
    if precision == 'single':
        nvcc_flags.append('-DReal=float')
    elif precision == 'double':
        nvcc_flags.append('-DReal=double')
    else:
        raise Exception(f'Unrecognized precision: {precision}')

    return SourceModule(_cu, options=nvcc_flags, no_extern_c=1)

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

gradient_2d = CudaFunc("gradient_2d")
jacobian_times_vectorfield_2d = CudaFunc("jacobian_times_vectorfield_2d")
jacobian_transpose_times_vectorfield_2d = \
        CudaFunc("jacobian_transpose_times_vectorfield_2d")
