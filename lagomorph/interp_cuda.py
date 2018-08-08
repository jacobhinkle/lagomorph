# vim: set syntax=cuda:
from pycuda import gpuarray
from pycuda.compiler import SourceModule, DEFAULT_NVCC_FLAGS
import pycuda.autoinit

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

// Bilerp function for single array input
template<BackgroundStrategy backgroundStrategy>
inline __device__
Real
biLerp(const Real* img,
	Real x, Real y,
	int sizeX, int sizeY,
	Real background = 0.f)
{
    int floorX = (int)(x);
    int floorY = (int)(y);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;

    // this is not truly ceiling, but floor + 1, which is usually ceiling
    int ceilX = floorX + 1;
    int ceilY = floorY + 1;

    Real t = x - floorX;
    Real u = y - floorY;

    Real oneMinusT = 1.f- t;
    Real oneMinusU = 1.f- u;

    Real v0, v1, v2, v3;
    int inside = 1;

    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY,
                       ceilX, ceilY,
                       sizeX, sizeY);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY,
                        ceilX, ceilY,
                        sizeX, sizeY);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){

	if(backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	    background = 0.f;
	}

        inside = isInside(floorX, floorY,
                          ceilX, ceilY,
                          sizeX, sizeY);
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


    if (inside){
        v0 = get_pixel_2d(floorX, floorY, img, sizeX, sizeY);
        v1 = get_pixel_2d(ceilX, floorY, img, sizeX, sizeY);
        v2 = get_pixel_2d(ceilX, ceilY, img, sizeX, sizeY);
        v3 = get_pixel_2d(floorX, ceilY, img, sizeX, sizeY);
    }else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;

        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;

        v0 = (floorXIn && floorYIn) ? get_pixel_2d(floorX, floorY, img, sizeX, sizeY): background;
        v1 = (ceilXIn && floorYIn)  ? get_pixel_2d(ceilX, floorY, img, sizeX, sizeY): background;
        v2 = (ceilXIn && ceilYIn)   ? get_pixel_2d(ceilX, ceilY, img, sizeX, sizeY): background;
        v3 = (floorXIn && ceilYIn)  ? get_pixel_2d(floorX, ceilY, img, sizeX, sizeY): background;
    }

    //
    // do bilinear interpolation
    //

    //
    // this is the basic bilerp function...
    //
    //     h =
    //       v0 * (1 - t) * (1 - u) +
    //       v1 * t       * (1 - u) +
    //       v2 * t       * u       +
    //       v3 * (1 - t) * u
    //
    // the following nested version saves 2 multiplies.
    //
    return  oneMinusT * (oneMinusU * v0  +
                         u         * v3) +
            t         * (oneMinusU * v1  +
                         u         * v2);
}

// Trilerp function for single array input
template<BackgroundStrategy backgroundStrategy>
inline __device__
Real
triLerp(const Real* img,
	Real x, Real y, Real z,
	int sizeX, int sizeY, int sizeZ,
	Real background = 0.f)
{
    int floorX = (int)(x);
    int floorY = (int)(y);
    int floorZ = (int)(z);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;
    if (z < 0 && z != (int)(z)) --floorZ;

    // this is not truly ceiling, but floor + 1, which is usually ceiling
    int ceilX = floorX + 1;
    int ceilY = floorY + 1;
    int ceilZ = floorZ + 1;

    Real t = x - floorX;
	Real u = y - floorY;
	Real v = z - floorZ;

    Real oneMinusT = 1.f- t;
	Real oneMinusU = 1.f- u;
    Real oneMinusV = 1.f- v;

    Real v0, v1, v2, v3, v4, v5, v6, v7;
    int inside = 1;

    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY, floorZ,
                       ceilX, ceilY, ceilZ,
                       sizeX, sizeY, sizeZ);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY, floorZ,
                        ceilX, ceilY, ceilZ,
                        sizeX, sizeY, sizeZ);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){

	if(backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	    background = 0.f;
	}

        inside = isInside(floorX, floorY, floorZ,
                          ceilX, ceilY, ceilZ,
                          sizeX, sizeY, sizeZ);
    }else{
	// unknown background strategy, don't allow compilation
	STATIC_ASSERT(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_VAL);
	return 0.f;
    }


    if (inside){
        v0 = get_pixel_3d(floorX, floorY, floorZ, img, sizeX, sizeY, sizeZ);
        v1 = get_pixel_3d(ceilX, floorY, floorZ,  img, sizeX, sizeY, sizeZ);
        v2 = get_pixel_3d(ceilX, ceilY, floorZ,   img, sizeX, sizeY, sizeZ);
        v3 = get_pixel_3d(floorX, ceilY, floorZ,  img, sizeX, sizeY, sizeZ);

        v4 = get_pixel_3d(floorX, ceilY, ceilZ,  img, sizeX, sizeY, sizeZ);
        v5 = get_pixel_3d(ceilX, ceilY, ceilZ,   img, sizeX, sizeY, sizeZ);
        v6 = get_pixel_3d(ceilX, floorY, ceilZ,  img, sizeX, sizeY, sizeZ);
        v7 = get_pixel_3d(floorX, floorY, ceilZ, img, sizeX, sizeY, sizeZ);
    }else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;
        bool floorZIn = floorZ >= 0 && floorZ < sizeZ;

        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
        bool ceilZIn = ceilZ >= 0 && ceilZ < sizeZ;

        v0 = (floorXIn && floorYIn && floorZIn) ? get_pixel_3d(floorX, floorY, floorZ, img, sizeX, sizeY, sizeZ): background;
        v1 = (ceilXIn && floorYIn && floorZIn)  ? get_pixel_3d(ceilX, floorY, floorZ,  img, sizeX, sizeY, sizeZ): background;
        v2 = (ceilXIn && ceilYIn && floorZIn)   ? get_pixel_3d(ceilX, ceilY, floorZ,   img, sizeX, sizeY, sizeZ): background;
        v3 = (floorXIn && ceilYIn && floorZIn)  ? get_pixel_3d(floorX, ceilY, floorZ,  img, sizeX, sizeY, sizeZ): background;

        v4 = (floorXIn && ceilYIn && ceilZIn)   ? get_pixel_3d(floorX, ceilY, ceilZ,  img, sizeX, sizeY, sizeZ): background;
        v5 = (ceilXIn && ceilYIn && ceilZIn)    ? get_pixel_3d(ceilX, ceilY, ceilZ,   img, sizeX, sizeY, sizeZ): background;
        v6 = (ceilXIn && floorYIn && ceilZIn)   ? get_pixel_3d(ceilX, floorY, ceilZ,  img, sizeX, sizeY, sizeZ): background;
        v7 = (floorXIn && floorYIn && ceilZIn)  ? get_pixel_3d(floorX, floorY, ceilZ, img, sizeX, sizeY, sizeZ): background;
    }

    //
    // do trilinear interpolation
    //

    //
    // this is the basic trilerp function...
    //
    //     h =
    //       v0 * (1 - t) * (1 - u) * (1 - v) +
    //       v1 * t       * (1 - u) * (1 - v) +
    //       v2 * t       * u       * (1 - v) +
    //       v3 * (1 - t) * u       * (1 - v) +
    //       v4 * (1 - t) * u       * v       +
    //       v5 * t       * u       * v       +
    //       v6 * t       * (1 - u) * v       +
    //       v7 * (1 - t) * (1 - u) * v;
    //
    // the following nested version saves 10 multiplies.
    //
    return  oneMinusT * (oneMinusU * (v0 * oneMinusV + v7 * v)  +
                         u         * (v3 * oneMinusV + v4 * v)) +
            t         * (oneMinusU * (v1 * oneMinusV + v6 * v)  +
                         u         * (v2 * oneMinusV + v5 * v));
}

extern "C" {
    __global__ void interp_image_bcastI_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        Real xi = biLerp<BACKGROUND_STRATEGY_CLAMP>(I,
            0.5f, 0.5f,
            nx, ny,
            0.f);
    }
    __global__ void interp_image_bcastI_3d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny, int nz) {
    }
}
'''

def getmod(precision='double'):
    nvcc_flags = DEFAULT_NVCC_FLAGS + ['-std=c++11']
    if precision == 'single':
        nvcc_flags.append('-DReal=float')
    elif precision == 'double':
        nvcc_flags.append('-DReal=double')
    else:
        raise Exception('Unrecognized precision: {}'.format(precision))

    return SourceModule(_cu, options=nvcc_flags, no_extern_c=1)

class CudaFunc:
    def __init__(self, func_name):
        self.name = func_name
        self.funcs = {}
    def __call__(self, *args, precision='double', **kwargs):
        if not precision in self.funcs:
            self.funcs[precision] = getmod(precision).get_function(self.name)
        f = self.funcs[precision]
        return f(*args, **kwargs)

interp_image_bcastI_2d = CudaFunc("interp_image_bcastI_2d")
