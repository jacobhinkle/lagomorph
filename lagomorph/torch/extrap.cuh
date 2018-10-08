#pragma once

#include "defs.cuh"

template<typename Real>
inline __device__ Real get_pixel_3d(int x, int y, int z,
                                    const Real* d_i,
                                    size_t sizeX, size_t sizeY, size_t sizeZ)
{
    auto index = (x * sizeY + y) * sizeZ + z;
    return d_i[index];
}

template<typename Real>
__device__ Real get_pixel_2d(int x, int y,
                                    const Real* d_i,
                                    size_t sizeX, size_t sizeY)
{
    auto index =  x * sizeY + y;
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

