/* vim: set ft=cuda: */
#pragma once

#include "extrap.cuh"
#include "defs.cuh"

template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
Real
diff_x(const Real* arr,
       int nx, int ny,
       int i, int j) {
    return 0.5f*(get_value_safe<Real, backgroundStrategy>(arr, nx, ny, i+1, j)
               - get_value_safe<Real, backgroundStrategy>(arr, nx, ny, i-1, j));
}
template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
Real
diff_y(const Real* arr,
       int nx, int ny,
       int i, int j) {
    return 0.5f*(get_value_safe<Real, backgroundStrategy>(arr, nx, ny, i, j+1)
               - get_value_safe<Real, backgroundStrategy>(arr, nx, ny, i, j-1));
}

template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
Real
diff_x(const Real* arr,
       int nx, int ny, int nz,
       int i, int j, int k) {
    return 0.5f*(get_value_safe<Real, backgroundStrategy>(arr, nx, ny, nz, i+1, j, k)
               - get_value_safe<Real, backgroundStrategy>(arr, nx, ny, nz, i-1, j, k));
}
template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
Real
diff_y(const Real* arr,
       int nx, int ny, int nz,
       int i, int j, int k) {
    return 0.5f*(get_value_safe<Real, backgroundStrategy>(arr, nx, ny, nz, i, j+1, k)
               - get_value_safe<Real, backgroundStrategy>(arr, nx, ny, nz, i, j-1, k));
}
template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
Real
diff_z(const Real* arr,
       int nx, int ny, int nz,
       int i, int j, int k) {
    return 0.5f*(get_value_safe<Real, backgroundStrategy>(arr, nx, ny, nz, i, j, k+1)
               - get_value_safe<Real, backgroundStrategy>(arr, nx, ny, nz, i, j, k-1));
}


template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
void
grad_point(Real& gx, Real& gy,
        const Real* arr,
        int nx, int ny,
        int i, int j) {
    gx = diff_x<Real, backgroundStrategy>(arr, nx, ny, i, j);
    gy = diff_y<Real, backgroundStrategy>(arr, nx, ny, i, j);
}

template<typename Real, BackgroundStrategy backgroundStrategy>
inline __device__
void
grad_point(Real& gx, Real& gy, Real& gz,
        const Real* arr,
        int nx, int ny, int nz,
        int i, int j, int k) {
    gx = diff_x<Real, backgroundStrategy>(arr, nx, ny, nz, i, j, k);
    gy = diff_y<Real, backgroundStrategy>(arr, nx, ny, nz, i, j, k);
    gz = diff_z<Real, backgroundStrategy>(arr, nx, ny, nz, i, j, k);
}
