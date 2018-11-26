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

// Trilerp function for single array input
template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__ Real triLerp(const Real* __restrict__ img,
	Real x, Real y, Real z,
	size_t sizeX, size_t sizeY, size_t sizeZ)
{
    auto floorX = (int)(x);
    auto floorY = (int)(y);
    auto floorZ = (int)(z);

    if (x < 0 && x != (int)(x)) --floorX;
    if (y < 0 && y != (int)(y)) --floorY;
    if (z < 0 && z != (int)(z)) --floorZ;

    // this is not truly ceiling, but floor + 1, which is usually ceiling
    auto ceilX = floorX + 1;
    auto ceilY = floorY + 1;
    auto ceilZ = floorZ + 1;

    Real t = x - floorX;
    Real u = y - floorY;
    Real v = z - floorZ;

    Real oneMinusT = 1.f- t;
    Real oneMinusU = 1.f- u;
    Real oneMinusV = 1.f- v;

    Real v0, v1, v2, v3, v4, v5, v6, v7;

    clampBackground(floorX, floorY, floorZ,
                    ceilX, ceilY, ceilZ,
                    sizeX, sizeY, sizeZ);

    v0 = get_pixel(floorX, floorY, floorZ, img, sizeX, sizeY, sizeZ);
    v1 = get_pixel(ceilX, floorY, floorZ, img, sizeX, sizeY, sizeZ);
    v2 = get_pixel(ceilX, ceilY, floorZ, img, sizeX, sizeY, sizeZ);
    v3 = get_pixel(floorX, ceilY, floorZ, img, sizeX, sizeY, sizeZ);
    v4 = get_pixel(floorX, floorY, ceilZ, img, sizeX, sizeY, sizeZ);
    v5 = get_pixel(ceilX, floorY, ceilZ, img, sizeX, sizeY, sizeZ);
    v6 = get_pixel(ceilX, ceilY, ceilZ, img, sizeX, sizeY, sizeZ);
    v7 = get_pixel(floorX, ceilY, ceilZ, img, sizeX, sizeY, sizeZ);

    //
    // this is the basic trilerp function...
    //
    //     h =
    //       v0 * (1 - t) * (1 - u) * (1 - v) +
    //       v1 * t       * (1 - u) * (1 - v) +
    //       v2 * t       * u       * (1 - v) +
    //       v3 * (1 - t) * u       * (1 - v) +
    //       v4 * (1 - t) * (1 - u) * v       +
    //       v5 * t       * (1 - u) * v       +
    //       v6 * t       * u       * v       +
    //       v7 * (1 - t) * u       * v
    //
    // the following nested version saves 10 multiplies.
    //
    return oneMinusV * (oneMinusU * (oneMinusT * v0  +
                                     t         * v1) +
                        u         * (oneMinusT * v3  +
                                     t         * v2) ) +
           v         * (oneMinusU * (oneMinusT * v4  +
                                     t         * v5) +
                        u         * (oneMinusT * v7  +
                                     t         * v6) );
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

// This computes the diagonal elements of the Hessian of a sum of squared
// differences loss with respect to the interpolated image.
// Note that this implementation uses global atomic adds.
template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__ void interp_hessian_diagonal_image_point(
        Real* __restrict__ H,
        Real x, Real y,
        size_t sizeX, size_t sizeY) {
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

    Real w0, w1, w2, w3; // interp weights
    int ix0, ix1, ix2, ix3; // indices of corners
    bool inside = 1;

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
        }

        inside = isInside(floorX, floorY,
                          ceilX, ceilY,
                          sizeX, sizeY);
    } else {
        // unknown background strategy, don't allow compilation
        static_assert(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
                  backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
                  backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
                  backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
                  backgroundStrategy== BACKGROUND_STRATEGY_VAL,
                          "Unknown background strategy");
    }

    ix0 = floorX*sizeY + floorY;
    ix1 = ceilX *sizeY + floorY;
    ix2 = ceilX *sizeY + ceilY;
    ix3 = floorX*sizeY + ceilY;

    // interpolation weights
    w0 = (oneMinusT * oneMinusU);
    w1 = (t         * oneMinusU);
    w2 = (t         * u        );
    w3 = (oneMinusT * u        );

    if (inside) {
        // set the diagonal of the Hessian directly
        atomicAdd(&H[ix0], w0*w0);
        atomicAdd(&H[ix1], w1*w1);
        atomicAdd(&H[ix2], w2*w2);
        atomicAdd(&H[ix3], w3*w3);
    } else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;
        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;

        if (floorXIn && floorYIn) atomicAdd(&H[ix0], w0*w0);
        if (ceilXIn && floorYIn) atomicAdd(&H[ix1], w1*w1);
        if (ceilXIn && ceilYIn) atomicAdd(&H[ix2], w2*w2);
        if (floorXIn && ceilYIn) atomicAdd(&H[ix3], w3*w3);
    }
}
