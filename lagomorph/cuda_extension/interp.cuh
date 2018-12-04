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

    v0 = get_value(img, sizeX, sizeY, floorX, floorY);
    v1 = get_value(img, sizeX, sizeY, ceilX, floorY);
    v2 = get_value(img, sizeX, sizeY, ceilX, ceilY);
    v3 = get_value(img, sizeX, sizeY, floorX, ceilY);

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

    v0 = get_value(img, sizeX, sizeY, sizeZ, floorX, floorY, floorZ);
    v1 = get_value(img, sizeX, sizeY, sizeZ, ceilX, floorY, floorZ);
    v2 = get_value(img, sizeX, sizeY, sizeZ, ceilX, ceilY, floorZ);
    v3 = get_value(img, sizeX, sizeY, sizeZ, floorX, ceilY, floorZ);
    v4 = get_value(img, sizeX, sizeY, sizeZ, floorX, floorY, ceilZ);
    v5 = get_value(img, sizeX, sizeY, sizeZ, ceilX, floorY, ceilZ);
    v6 = get_value(img, sizeX, sizeY, sizeZ, ceilX, ceilY, ceilZ);
    v7 = get_value(img, sizeX, sizeY, sizeZ, floorX, ceilY, ceilZ);

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
        v0 = get_value(img, sizeX, sizeY, floorX, floorY);
        v1 = get_value(img, sizeX, sizeY, ceilX, floorY);
        v2 = get_value(img, sizeX, sizeY, ceilX, ceilY);
        v3 = get_value(img, sizeX, sizeY, floorX, ceilY);
    }else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;

        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;

        v0 = (floorXIn && floorYIn) ? get_value(img, sizeX, sizeY, floorX, floorY): 0;
        v1 = (ceilXIn && floorYIn)  ? get_value(img, sizeX, sizeY, ceilX, floorY): 0;
        v2 = (ceilXIn && ceilYIn)   ? get_value(img, sizeX, sizeY, ceilX, ceilY): 0;
        v3 = (floorXIn && ceilYIn)  ? get_value(img, sizeX, sizeY, floorX, ceilY): 0;
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

template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
void
triLerp_grad(Real& Ix, Real& gx, Real& gy, Real& gz,
        const Real* __restrict__ img,
	Real x, Real y, Real z,
	int sizeX, int sizeY, int sizeZ)
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

    //
    // this is the basic trilerp function...
    // inside indicates whether we can safely do lookups with the mapped indices
    bool inside = map_point<backgroundStrategy>(floorX, floorY, floorZ,
                       ceilX, ceilY, ceilZ,
                       sizeX, sizeY, sizeZ);
    if (inside){
        v0 = get_value(img, sizeX, sizeY, sizeZ, floorX, floorY, floorZ);
        v1 = get_value(img, sizeX, sizeY, sizeZ, ceilX, floorY, floorZ);
        v2 = get_value(img, sizeX, sizeY, sizeZ, ceilX, ceilY, floorZ);
        v3 = get_value(img, sizeX, sizeY, sizeZ, floorX, ceilY, floorZ);
        v4 = get_value(img, sizeX, sizeY, sizeZ, floorX, floorY, ceilZ);
        v5 = get_value(img, sizeX, sizeY, sizeZ, ceilX, floorY, ceilZ);
        v6 = get_value(img, sizeX, sizeY, sizeZ, ceilX, ceilY, ceilZ);
        v7 = get_value(img, sizeX, sizeY, sizeZ, floorX, ceilY, ceilZ);
    }else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;
        bool floorZIn = floorZ >= 0 && floorZ < sizeZ;

        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;
        bool ceilZIn = ceilZ >= 0 && ceilZ < sizeZ;

        v0 = (floorXIn && floorYIn && floorZIn) ? get_value(img, sizeX, sizeY, sizeZ, floorX, floorY, floorZ): 0;
        v1 = (ceilXIn && floorYIn && floorZIn)  ? get_value(img, sizeX, sizeY, sizeZ, ceilX, floorY, floorZ): 0;
        v2 = (ceilXIn && ceilYIn && floorZIn)   ? get_value(img, sizeX, sizeY, sizeZ, ceilX, ceilY, floorZ): 0;
        v3 = (floorXIn && ceilYIn && floorZIn)  ? get_value(img, sizeX, sizeY, sizeZ, floorX, ceilY, floorZ): 0;
        v4 = (floorXIn && floorYIn && ceilZIn) ? get_value(img, sizeX, sizeY, sizeZ, floorX, floorY, ceilZ): 0;
        v5 = (ceilXIn && floorYIn && ceilZIn)  ? get_value(img, sizeX, sizeY, sizeZ, ceilX, floorY, ceilZ): 0;
        v6 = (ceilXIn && ceilYIn && ceilZIn)   ? get_value(img, sizeX, sizeY, sizeZ, ceilX, ceilY, ceilZ): 0;
        v7 = (floorXIn && ceilYIn && ceilZIn)  ? get_value(img, sizeX, sizeY, sizeZ, floorX, ceilY, ceilZ): 0;
    }

    // trilinear interpolation:
    //     h =
    //       v0 * (1 - t) * (1 - u) * (1 - v) +
    //       v1 * t       * (1 - u) * (1 - v) +
    //       v2 * t       * u       * (1 - v) +
    //       v3 * (1 - t) * u       * (1 - v) +
    //       v4 * (1 - t) * (1 - u) * v       +
    //       v5 * t       * (1 - u) * v       +
    //       v6 * t       * u       * v       +
    //       v7 * (1 - t) * u       * v
    // The derivative is 
    //    dh/dx = (-v0) * (1 - u) * (1 - v) +
    //              v1  * (1 - u) * (1 - v) +
    //              v2  * u       * (1 - v) +
    //            (-v3) * u       * (1 - v) +
    //            (-v4) * (1 - u) * v       +
    //              v5  * (1 - u) * v       +
    //              v6  * u       * v       +
    //            (-v7) * u       * v
    //    dh/dy = (-v0) * (1 - t) * (1 - v) +
    //            (-v1) * t       * (1 - v) +
    //              v2  * t       * (1 - v) +
    //              v3  * (1 - t) * (1 - v) +
    //            (-v4) * (1 - t) * v       +
    //            (-v5) * t       * v       +
    //              v6  * t       * v       +
    //              v7  * (1 - t) * v
    //    dh/dz = (-v0) * (1 - t) * (1 - u) +
    //            (-v1) * t       * (1 - u) +
    //            (-v2) * t       * u       +
    //            (-v3) * (1 - t) * u       +
    //              v4  * (1 - t) * (1 - u) +
    //              v5  * t       * (1 - u) +
    //              v6  * t       * u       +
    //              v7  * (1 - t) * u       +
    //
    Ix =   oneMinusV * (oneMinusU * (oneMinusT * v0  +
                                     t         * v1) +
                        u         * (oneMinusT * v3  +
                                     t         * v2) ) +
           v         * (oneMinusU * (oneMinusT * v4  +
                                     t         * v5) +
                        u         * (oneMinusT * v7  +
                                     t         * v6) );
    gx = oneMinusV * (oneMinusU * (v1 - v0)  +
                      u         * (v2 - v3)) +
         v         * (oneMinusU * (v5 - v4)  +
                      u         * (v6 - v7));
    gy = oneMinusV * (oneMinusT * (v3 - v0)  +
                      t         * (v2 - v1)) +
         v         * (oneMinusT * (v7 - v4)  +
                      t         * (v6 - v5));
    gz = oneMinusU * (oneMinusT * (v4 - v0)  +
                      t         * (v5 - v1)) +
         u         * (oneMinusT * (v7 - v3)  +
                      t         * (v6 - v2));
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
    int i=xInt, j=yInt, k=zInt;
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(i, w);
        wrap(j, h);
        wrap(k, l);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(i, w);
        clamp(j, h);
        clamp(k, l);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
        if (i < 0 || i >= w) return;
        if (j < 0 || j >= h) return;
        if (k < 0 || k >= l) return;
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
    int nid = (i * h + j) * l + k;
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
