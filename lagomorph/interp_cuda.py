# -*- mode: cuda -*-
# vi: set syntax=cuda:
from .cudamod import CudaModule
mod = CudaModule('''
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
    int index = (x * sizeY + y) * sizeZ + z;
    return d_i[index];
}

__device__ Real get_pixel_2d(int x, int y,
                                    const Real* d_i,
                                    int sizeX, int sizeY)
{
    int index =  x * sizeY + y;
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
inline __device__ void splat_neighbor(Real* d_wd, Real* d_ww,
        Real ww, Real mass,
        int xInt, int yInt,
        int w, int h) {
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(xInt, w);
        wrap(yInt, h);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(xInt, w);
        clamp(yInt, h);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
        if (xInt < 0 || xInt >= w) return;
        if (yInt < 0 || yInt >= h) return;
    }else{
	// unknown background strategy, don't allow compilation
	static_assert(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_VAL,
                      "Unknown background strategy");
	return;
    }
    int nid = xInt * h + yInt;
    atomicAdd(&d_ww[nid], ww);
    ww *= mass;
    atomicAdd(&d_wd[nid], ww);
}

template<BackgroundStrategy backgroundStrategy>
inline __device__ void splat_neighbor(Real* d_wd, Real* d_ww,
        Real ww, Real mass,
        int xInt, int yInt, int zInt,
        int w, int h, int l) {
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(xInt, w);
        wrap(yInt, h);
        wrap(zInt, l);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(xInt, w);
        clamp(yInt, h);
        clamp(zInt, l);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){

	if(backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	    background = 0.f;
	}
        if (xInt < 0 || xInt >= w) return;
        if (yInt < 0 || yInt >= h) return;
        if (zInt < 0 || zInt >= l) return;
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
    int nid = (xInt * h + yInt) * l + zInt;
    atomicAdd(&d_ww[nid], ww);
    ww *= mass;
    atomicAdd(&d_wd[nid], ww);
}

template<BackgroundStrategy backgroundStrategy>
inline  __device__ void atomicSplat(Real* d_wd, Real* d_ww,
        Real mass, Real x, Real y,
        int w, int h)
{
    int xInt = int(x);
    int yInt = int(y);

    if (x < 0 && x != xInt) --xInt;
    if (y < 0 && y != yInt) --yInt;

    Real dx = 1.f - (x - xInt);
    Real dy = 1.f - (y - yInt);

    for (int xi=xInt;xi<xInt+2;xi++) {
        for (int yi=yInt;yi<yInt+2;yi++) {
            splat_neighbor<backgroundStrategy>(d_ww, d_wd, dx * dy,
                mass, xi, yi, w, h);
            dy = 1.-dy;
        }
        dx = 1.f-dx;
    }
}
template<BackgroundStrategy backgroundStrategy>
inline  __device__ void atomicSplat(Real* d_wd, Real* d_ww,
        Real mass, Real x, Real y, Real z,
        int w, int h, int l)
{
    int xInt = int(x);
    int yInt = int(y);
    int zInt = int(z);

    if (x < 0 && x != xInt) --xInt;
    if (y < 0 && y != yInt) --yInt;
    if (z < 0 && z != zInt) --zInt;

    Real dx = 1.f - (x - xInt);
    Real dy = 1.f - (y - yInt);
    Real dz = 1.f - (z - zInt);

    splat_neighbor<backgroundStrategy>(d_ww, d_wd, dx * dy * dz,
        mass, xInt, yInt, zInt, w, h, l);
    splat_neighbor<backgroundStrategy>(d_ww, d_wd, dx * dy * (1.f-dz),
        mass, xInt, yInt, zInt+1, w, h, l);
    splat_neighbor<backgroundStrategy>(d_ww, d_wd, dx * (1.f-dy) * dz,
        mass, xInt, yInt+1, zInt, w, h, l);
    splat_neighbor<backgroundStrategy>(d_ww, d_wd, dx * (1.f-dy) * (1.f-dz),
        mass, xInt, yInt+1, zInt+1, w, h, l);
    splat_neighbor<backgroundStrategy>(d_ww, d_wd, (1.f-dx) * dy * dz,
        mass, xInt+1, yInt, zInt, w, h, l);
    splat_neighbor<backgroundStrategy>(d_ww, d_wd, (1.f-dx) * dy * (1.f-dz),
        mass, xInt+1, yInt, zInt+1, w, h, l);
    splat_neighbor<backgroundStrategy>(d_ww, d_wd, (1.f-dx) * (1.f-dy) * dz,
        mass, xInt+1, yInt+1, zInt, w, h, l);
    splat_neighbor<backgroundStrategy>(d_ww, d_wd, (1.f-dx) * (1.f-dy) * (1.f-dz),
        mass, xInt+1, yInt+1, zInt+1, w, h, l);
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
	static_assert(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_VAL,
                      "Unknown background strategy");
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

template<BackgroundStrategy backgroundStrategy, int displacement>
inline __device__
void
interp_vectorfield_kernel_2d(int i, int j, Real* out, const Real* g, const Real* h,
            int nn, int nx, int ny) {
    if (i >= nx || j >= ny) return;
    int nxy = nx*ny;
    int inx = i*ny + j;
    int iny = inx + nxy;
    int ino = inx;
    Real hx, hy;
    Real fi = static_cast<Real>(i);
    Real fj = static_cast<Real>(j);
    const Real* gd = g;
    for (int n=0; n < nn; ++n) {
        if (displacement) {
            hx = h[inx] + fi;
            hy = h[iny] + fj;
        } else {
            hx = h[inx];
            hy = h[iny];
        }
        out[ino] = biLerp<DEFAULT_BACKGROUND_STRATEGY>(gd,
            hx, hy,
            nx, ny,
            0.f);
        ino += nxy;
        gd += nxy;
        out[ino] = biLerp<DEFAULT_BACKGROUND_STRATEGY>(gd,
            hx, hy,
            nx, ny,
            0.f);
        ino += nxy;
        gd += nxy;
        inx += 2*nxy;
        iny += 2*nxy;
    }
}

template<BackgroundStrategy backgroundStrategy, int displacement>
inline __device__
void
interp_image_kernel_2d(int i, int j, Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
    if (i >= nx || j >= ny) return;
    int nxy = nx*ny;
    int inx = i*ny + j;
    int iny = inx + nxy;
    int ino = inx;
    for (int n=0; n < nn; ++n) {
        Real hx = h[inx];
        Real hy = h[iny];
        if (displacement) {
            hx += static_cast<Real>(i);
            hy += static_cast<Real>(j);
        }
        out[ino] = biLerp<DEFAULT_BACKGROUND_STRATEGY>(I,
            hx, hy,
            nx, ny,
            0.f);
        inx += 2*nxy;
        iny += 2*nxy;
        ino += nxy;
    }
}

template<BackgroundStrategy backgroundStrategy, int displacement>
inline __device__
void
splat_image_kernel_2d(int i, int j, Real* d_wd, Real* d_ww, Real* I, Real* h,
            int nn, int nx, int ny) {
    if (i >= nx || j >= ny) return;
    int nxy = nx*ny;
    int inx = i*ny + j;
    int iny = inx + nxy;
    int ino = inx;
    Real* dn = d_wd;
    Real* wn = d_ww;
    for (int n=0; n < nn; ++n) {
        Real hx = h[inx];
        Real hy = h[iny];
        if (displacement) {
            hx += static_cast<Real>(i);
            hy += static_cast<Real>(j);
        }
        atomicSplat<DEFAULT_BACKGROUND_STRATEGY>(dn, wn,
            I[ino], hx, hy, nx, ny);
        inx += 2*nxy;
        iny += 2*nxy;
        dn += nxy;
        wn += nxy;
        ino += nxy;
    }
}

extern "C" {
    __global__ void splat_image_2d(Real* d_wd, Real* d_ww, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        splat_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0>(i, j,
            d_wd, d_ww, I, h, nn, nx, ny);
    }
    __global__ void splat_displacement_image_2d(Real* d_wd, Real* d_ww, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        splat_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1>(i, j,
            d_wd, d_ww, I, h, nn, nx, ny);
    }
    __global__ void interp_image_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0>(i, j, out, I, h, nn, nx, ny);
    }
    __global__ void interp_displacement_image_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_image_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1>(i, j, out, I, h, nn, nx, ny);
    }
    __global__ void interp_vectorfield_2d(Real* out, Real* g, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_vectorfield_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 0>(i, j, out, g, h, nn, nx, ny);
    }
    __global__ void interp_displacement_vectorfield_2d(Real* out, const Real* g, const Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        interp_vectorfield_kernel_2d<DEFAULT_BACKGROUND_STRATEGY, 1>(i, j, out, g, h, nn, nx, ny);
    }
    __global__ void interp_image_bcastI_2d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        int nxy = nx*ny;
        int inx = i*ny + j;
        int iny = nxy + i*ny + j;
        int ino = inx;
        for (int n=0; n < nn; ++n) {
            Real hx = h[inx];
            Real hy = h[iny];
            out[ino] = biLerp<DEFAULT_BACKGROUND_STRATEGY>(I,
                hx, hy,
                nx, ny,
                0.f);
            inx += 2*nxy;
            iny += 2*nxy;
            ino += nxy;
        }
    }
    __global__ void interp_image_bcastI_3d(Real* out, Real* I, Real* h,
            int nn, int nx, int ny, int nz) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        int nxy = nx*ny;
        int nxyz = nxy*nz;
        int inn = 0;
        for (int n=0; n < nn; ++n) {
            int inx = inn +         i*ny + j;
            int iny = inn +   nxy + i*ny + j;
            int inz = inn + 2*nxy + i*ny + j;
            int ino = inx;
            for (int k=0; k < nz; ++k) {
                Real hx = h[inx];
                Real hy = h[iny];
                Real hz = h[inz];
                out[ino] = triLerp<DEFAULT_BACKGROUND_STRATEGY>(I,
                    hx, hy, hz,
                    nx, ny, nz,
                    0.f);
                ino += nxy;
                inx += 3*nxy;
                iny += 3*nxy;
                inz += 3*nxy;
            }
            inn += nxyz;
        }
    }
}
''', extra_nvcc_flags=[
        '-DDEFAULT_BACKGROUND_STRATEGY=BACKGROUND_STRATEGY_CLAMP'])
splat_image_2d = mod.func("splat_image_2d")
splat_displacement_image_2d = mod.func("splat_displacement_image_2d")
interp_image_2d = mod.func("interp_image_2d")
interp_displacement_image_2d = mod.func("interp_displacement_image_2d")
interp_vectorfield_2d = mod.func("interp_vectorfield_2d")
interp_displacement_vectorfield_2d = mod.func("interp_displacement_vectorfield_2d")
interp_image_bcastI_2d = mod.func("interp_image_bcastI_2d")
