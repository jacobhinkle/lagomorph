#ifndef DIFF_CUH
#define DIFF_CUH

#include "extrap.cuh"
#include "defs.cuh"

template<BackgroundStrategy backgroundStrategy>
inline __device__
Real
diff_x(const Real* arr,
       int nx, int ny,
       int i, int j) {
    return 0.5f*(get_value_safe<backgroundStrategy>(arr, nx, ny, i+1, j)
               - get_value_safe<backgroundStrategy>(arr, nx, ny, i-1, j));
}
template<BackgroundStrategy backgroundStrategy>
inline __device__
Real
diff_y(const Real* arr,
       int nx, int ny,
       int i, int j) {
    return 0.5f*(get_value_safe<backgroundStrategy>(arr, nx, ny, i, j+1)
               - get_value_safe<backgroundStrategy>(arr, nx, ny, i, j-1));
}

template<BackgroundStrategy backgroundStrategy>
inline __device__
void
grad_point(Real& gx, Real& gy,
        const Real* arr,
        int nx, int ny,
        int i, int j) {
    gx = diff_x<backgroundStrategy>(arr, nx, ny, i, j);
    gy = diff_y<backgroundStrategy>(arr, nx, ny, i, j);
}

#endif /* DIFF_CUH */
