#ifndef INTERP_CUH
#define INTERP_CUH

#include "extrap.cuh"
#include "defs.cuh"

template<BackgroundStrategy backgroundStrategy, int write_weights>
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

template<BackgroundStrategy backgroundStrategy, int write_weights>
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
    if (write_weights)
        atomicAdd(&d_ww[nid], ww);
    atomicAdd(&d_wd[nid], ww*mass);
}

template<BackgroundStrategy backgroundStrategy, int write_weights>
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
            splat_neighbor<backgroundStrategy, write_weights>(d_wd, d_ww, dx * dy,
                mass, xi, yi, w, h);
            dy = 1.f-dy;
        }
        dx = 1.f-dx;
    }
}
template<BackgroundStrategy backgroundStrategy, int write_weights>
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
                splat_neighbor<backgroundStrategy, write_weights>(d_wd, d_ww, dx * dy * dz,
                    mass, xi, yi, zi, w, h, l);
                dz = 1.f-dz;
            }
            dy = 1.f-dy;
        }
        dx = 1.f-dx;
    }
}

// This computes the loss and gradient of the image match at a point with
// respect to the base image I. The gradient is splatted back into two images:
// num and denom. Denom holds the Jacobi preconditioner, and num holds the
// unconditioned gradient. To do a Jacobi preconditioned gradient step or a
// Jacobi iteration step, all that is needed is to divide num by denom after
// this kernel has completed. The squared difference between I(x,y) *B + C and
// J(i,j) is returned.
template<BackgroundStrategy backgroundStrategy>
inline __device__
Real
atlas_jacobi_point(Real* num, Real* denom,
        const Real* I,
        const Real* J,
        Real B, Real C,
        int i, int j,
        Real x, Real y,
        int sizeX, int sizeY,
        Real background = 0.f) {
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
    Real w0, w1, w2, w3; // interp weights
    int ix0, ix1, ix2, ix3; // indices of corners
    Real diff;
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
    } else {
        // unknown background strategy, don't allow compilation
        static_assert(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
                  backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
                  backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
                  backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
                  backgroundStrategy== BACKGROUND_STRATEGY_VAL,
                          "Unknown background strategy");
        return;
    }

    if (inside) {
        ix0 = floorX*sizeY + floorY;
        ix1 = ceilX *sizeY + floorY;
        ix2 = ceilX *sizeY + ceilY;
        ix3 = floorX*sizeY + ceilY;

        v0 = I[ix0];
        v1 = I[ix1];
        v2 = I[ix2];
        v3 = I[ix3];

        w0 = (oneMinusT * oneMinusU)*B;
        w1 = (t         * oneMinusU)*B;
        w2 = (t         * u        )*B;
        w3 = (oneMinusT * u        )*B;

        // point image difference
        diff = (w0*v0 + w1*v1 + w2*v2 + w3*v3) + C - J[i*sizeY + j];

        // set the denominator which is the diagonal of the Hessian
        atomicAdd(&denom[ix0], w0*w0);
        atomicAdd(&denom[ix1], w1*w1);
        atomicAdd(&denom[ix2], w2*w2);
        atomicAdd(&denom[ix3], w3*w3);
        // splat B*Ix+C - Ji into num
        atomicAdd(&num[ix0], w0*diff);
        atomicAdd(&num[ix1], w1*diff);
        atomicAdd(&num[ix2], w2*diff);
        atomicAdd(&num[ix3], w3*diff);
    } else {
        bool floorXIn = floorX >= 0 && floorX < sizeX;
        bool floorYIn = floorY >= 0 && floorY < sizeY;

        bool ceilXIn = ceilX >= 0 && ceilX < sizeX;
        bool ceilYIn = ceilY >= 0 && ceilY < sizeY;

        v0 = (floorXIn && floorYIn) ? get_pixel_2d(floorX, floorY, I, sizeX, sizeY): background;
        v1 = (ceilXIn && floorYIn)  ? get_pixel_2d(ceilX, floorY, I, sizeX, sizeY): background;
        v2 = (ceilXIn && ceilYIn)   ? get_pixel_2d(ceilX, ceilY, I, sizeX, sizeY): background;
        v3 = (floorXIn && ceilYIn)  ? get_pixel_2d(floorX, ceilY, I, sizeX, sizeY): background;
    }
    return diff*diff;
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

// Bilerp function for single array input
template<BackgroundStrategy backgroundStrategy>
inline __device__
void
biLerp_grad(Real& Ix, Real& gx, Real& gy,
        const Real* img,
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
        Ix = 0.f;
        gx = 0.f;
        gy = 0.f;
	return;
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
    gx = oneMinusU * (v1 - v0) + u * (v2 - v3);
    gy = oneMinusT * (v3 - v0) + t * (v2 - v1);
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


#endif /* INTERP_CUH */
