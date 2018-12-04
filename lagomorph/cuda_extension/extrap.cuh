/* vim: set ft=cuda: */
#pragma once

#include "defs.cuh"

template<typename Real>
__device__ Real get_value(const Real* d_i,
                          size_t sizeX, size_t sizeY,
                          int x, int y)
{
    auto index =  x * sizeY + y;
    return d_i[index];
}
template<typename Real>
inline __device__ Real get_value(const Real* d_i,
                                 size_t sizeX, size_t sizeY, size_t sizeZ,
                                 int x, int y, int z)
{
    auto index = (x * sizeY + y) * sizeZ + z;
    return d_i[index];
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

// Clamp strategy
inline __device__ void clamp(int& r, size_t b){
    if (r < 0) r = 0;
    else if (r >= b) r = b - 1;
}

inline __device__ void clampBackground(int& floorX,
                                       int& ceilX,
                                       size_t sizeX) {
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
                                       size_t  sizeX, size_t  sizeY) {
    clampBackground(floorX, ceilX, sizeX);
    clampBackground(floorY, ceilY, sizeY);
}
inline __device__ void clampBackground(int& floorX, int& floorY, int& floorZ,
                                       int& ceilX, int& ceilY, int& ceilZ,
                                       size_t  sizeX, size_t  sizeY, size_t  sizeZ) {
    clampBackground(floorX, ceilX, sizeX);
    clampBackground(floorY, ceilY, sizeY);
    clampBackground(floorZ, ceilZ, sizeZ);
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

template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
Real
get_value_safe(const Real* arr, size_t nx, size_t ny, int i, int j, Real background=0.0f) {
    int ii=i, jj=j;
    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(ii, nx);
        wrap(jj, ny);
        return get_value(arr, nx, ny, ii, jj);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(ii, nx);
        clamp(jj, ny);
        return get_value(arr, nx, ny, ii, jj);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){

	if(backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	    background = 0.f;
	}

        if (ii >= 0 && ii < nx && jj >= 0 && jj < ny)
            return get_value(arr, nx, ny, ii, jj);
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

template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
Real
get_value_safe(const Real* arr, int nx, int ny, int nz, int i, int j, int k, Real background=0.0f) {
    int ii=i, jj=j, kk=k;
    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(ii, nx);
        wrap(jj, ny);
        wrap(kk, nz);
        return get_value(arr, nx, ny, nz, ii, jj, kk);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(ii, nx);
        clamp(jj, ny);
        clamp(kk, nz);
        return get_value(arr, nx, ny, nz, ii, jj, kk);
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){

	if(backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	    background = 0.f;
	}

        if (ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >=0 && kk < nz)
            return get_value(arr, nx, ny, nz, ii, jj, kk);
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
inline __device__ bool map_point(int& floorX, int& floorY,
               int& ceilX, int& ceilY,
               size_t  sizeX, size_t  sizeY) {
	// unknown background strategy, don't allow compilation
	static_assert(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_VAL,
                      "Unknown background strategy");
    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY,
                       ceilX, ceilY,
                       sizeX, sizeY);
        return true;
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY,
                        ceilX, ceilY,
                        sizeX, sizeY);
        return true;
    }
    else {
        return isInside(floorX, floorY,
                          ceilX, ceilY,
                          sizeX, sizeY);
    }
}

template<BackgroundStrategy backgroundStrategy>
inline __device__ bool map_point(int& floorX, int& floorY, int& floorZ,
               int& ceilX, int& ceilY, int& ceilZ,
               size_t  sizeX, size_t  sizeY, size_t  sizeZ) {
	// unknown background strategy, don't allow compilation
	static_assert(backgroundStrategy== BACKGROUND_STRATEGY_WRAP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_CLAMP ||
		      backgroundStrategy== BACKGROUND_STRATEGY_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_PARTIAL_ZERO ||
		      backgroundStrategy== BACKGROUND_STRATEGY_VAL,
                      "Unknown background strategy");
    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrapBackground(floorX, floorY, floorZ,
                       ceilX, ceilY, ceilZ,
                       sizeX, sizeY, sizeZ);
        return true;
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clampBackground(floorX, floorY, floorZ,
                        ceilX, ceilY, ceilZ,
                        sizeX, sizeY, sizeZ);
        return true;
    }
    else {
        return isInside(floorX, floorY, floorZ,
                          ceilX, ceilY, ceilZ,
                          sizeX, sizeY, sizeZ);
    }
}
