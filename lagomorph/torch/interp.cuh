/* vim: set ft=cuda: */
#pragma once

#include "atomic.cuh"
#include "extrap.cuh"
#include "defs.cuh"

// Bilerp function for single array input
template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__ Real biLerp(const Real* __restrict__ img,
	Real x, Real y,
	size_t sizeX, size_t sizeY)
{
    auto floorX = (int)(x);
    auto floorY = (int)(y);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;

    // this is not truly ceiling, but floor + 1, which is usually ceiling
    auto ceilX = floorX + 1;
    auto ceilY = floorY + 1;

    Real t = x - floorX;
    Real u = y - floorY;

    Real oneMinusT = 1.f- t;
    Real oneMinusU = 1.f- u;

    Real v0, v1, v2, v3;

    clampBackground(floorX, floorY,
                    ceilX, ceilY,
                    sizeX, sizeY);

    v0 = get_pixel(floorX, floorY, img, sizeX, sizeY);
    v1 = get_pixel(ceilX, floorY, img, sizeX, sizeY);
    v2 = get_pixel(ceilX, ceilY, img, sizeX, sizeY);
    v3 = get_pixel(floorX, ceilY, img, sizeX, sizeY);

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


// Bilerp function for single array input
template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
void
biLerp_grad(Real& Ix, Real& gx, Real& gy,
        const Real* __restrict__ img,
	Real x, Real y,
	int sizeX, int sizeY)
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
    // inside indicates whether we can safely do lookups with the mapped indices
    bool inside = map_point<backgroundStrategy>(floorX, floorY,
                       ceilX, ceilY,
                       sizeX, sizeY);
    if (inside){
        v0 = get_pixel(floorX, floorY, img, sizeX, sizeY);
        v1 = get_pixel(ceilX, floorY, img, sizeX, sizeY);
        v2 = get_pixel(ceilX, ceilY, img, sizeX, sizeY);
        v3 = get_pixel(floorX, ceilY, img, sizeX, sizeY);
    }else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;

        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;

        v0 = (floorXIn && floorYIn) ? get_pixel(floorX, floorY, img, sizeX, sizeY): 0;
        v1 = (ceilXIn && floorYIn)  ? get_pixel(ceilX, floorY, img, sizeX, sizeY): 0;
        v2 = (ceilXIn && ceilYIn)   ? get_pixel(ceilX, ceilY, img, sizeX, sizeY): 0;
        v3 = (floorXIn && ceilYIn)  ? get_pixel(floorX, ceilY, img, sizeX, sizeY): 0;
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
    // The derivative is 
    //    dh/dx = (-v0) * (1 - u) +
    //              v1  * (1 - u) +
    //              v2  * u       +
    //            (-v3) * u
    //    dh/dy = (-v0) * (1 - t) +
    //            (-v1) * t       +
    //              v2  * t       +
    //              v3  * (1 - t)
    //
    Ix =    oneMinusT * (oneMinusU * v0  +
                         u         * v3) +
            t         * (oneMinusU * v1  +
                         u         * v2);
    gx = v1 - v0 + u * (v2 - v3 - v1 + v0);
    gy = v3 - v0 + t * (v2 - v1 - v3 + v0);
}


template<typename Real, BackgroundStrategy backgroundStrategy, bool write_weights>
inline __device__ void splat_neighbor(Real* d_wd, Real* d_ww,
        Real ww, Real mass,
        int xInt, int yInt,
        int w, int h) {
    int i=xInt, j=yInt;
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(i, w);
        wrap(j, h);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(i, w);
        clamp(j, h);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
        if (i < 0 || i >= w) return;
        if (j < 0 || j >= h) return;
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
    int nid = i * h + j;
    if (write_weights)
        atomicAdd(&d_ww[nid], ww);
    atomicAdd(&d_wd[nid], ww*mass);
}

template<typename Real, BackgroundStrategy backgroundStrategy, bool write_weights>
inline __device__ void splat_neighbor(Real* d_wd, Real* d_ww,
        Real ww, Real mass,
        int xInt, int yInt, int zInt,
        int w, int h, int l) {
	Real background = 0.0f;
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
	return;
    }
    int nid = (xInt * h + yInt) * l + zInt;
    if (write_weights)
        atomicAdd(&d_ww[nid], ww);
    atomicAdd(&d_wd[nid], ww*mass);
}

template<typename Real, BackgroundStrategy backgroundStrategy, bool write_weights>
inline  __device__ void atomicSplat(Real* d_wd, Real* d_ww,
        Real mass, Real x, Real y,
        int w, int h)
{
    int xInt = int(x);
    int yInt = int(y);

    if (x < 0 && x != xInt) --xInt;
    if (y < 0 && y != yInt) --yInt;

    Real dx = 1.f - (x - static_cast<Real>(xInt));
    Real dy = 1.f - (y - static_cast<Real>(yInt));

    for (int xi=xInt;xi<xInt+2;xi++) {
        for (int yi=yInt;yi<yInt+2;yi++) {
            splat_neighbor<Real, backgroundStrategy, write_weights>(d_wd, d_ww, dx * dy,
                mass, xi, yi, w, h);
            dy = 1.f-dy;
        }
        dx = 1.f-dx;
    }
}
template<typename Real, BackgroundStrategy backgroundStrategy, bool write_weights>
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

    for (int xi=xInt;xi<xInt+2;xi++) {
        for (int yi=yInt;yi<yInt+2;yi++) {
            for (int zi=zInt;zi<zInt+2;zi++) {
                splat_neighbor<Real, backgroundStrategy, write_weights>(d_wd, d_ww, dx * dy * dz,
                    mass, xi, yi, zi, w, h, l);
                dz = 1.f-dz;
            }
            dy = 1.f-dy;
        }
        dx = 1.f-dx;
    }
}

