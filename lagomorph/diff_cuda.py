# vim: set syntax=cuda:
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
    int index = (x * sizeY + y) * sizeX + z;
    return d_i[index];
}

inline __device__ Real get_pixel_2d(int x, int y,
                                    const Real* d_i,
                                    int sizeX, int sizeY)
{
    if (x < 0 || x >= sizeX ||
        y < 0 || y >= sizeY) return 0;
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

inline __device__ void clamp(int& r, int b){
    if (r < 0) r = 0;
    else if (r >= b) r = b - 1;
}

template<BackgroundStrategy backgroundStrategy>
inline __device__
Real
get_value_safe(const Real* arr, int nx, int ny, int i, int j, Real background=0.0f) {
    int ii=i, jj=j;
    // adjust the position of the sample point if required
    if (backgroundStrategy == BACKGROUND_STRATEGY_WRAP){
        wrap(ii, nx);
        wrap(jj, ny);
        return arr[ii*ny + jj];
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_CLAMP){
        clamp(ii, nx);
        clamp(jj, ny);
        return arr[ii*ny + jj];
    }
    else if (backgroundStrategy == BACKGROUND_STRATEGY_VAL ||
	     backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	     backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){

	if(backgroundStrategy == BACKGROUND_STRATEGY_ZERO ||
	   backgroundStrategy == BACKGROUND_STRATEGY_PARTIAL_ZERO){
	    background = 0.f;
	}

        if (ii >= 0 && ii < nx && jj >= 0 && jj < ny)
            return arr[ii*ny + jj];
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

// Just a simple image gradient
template<BackgroundStrategy backgroundStrategy>
inline __device__
void
gradient_kernel(Real* out, const Real* im,
        int nn, int nx, int ny, int i, int j) {
    Real gx, gy; // gradient of im
    int nxy = nx*ny;
    const Real* imn = im; // pointer to current image
    // index of current output point (first component. add nxy for second)
    int ino = i*ny + j;
    for (int n=0; n < nn; ++n) {
        grad_point<backgroundStrategy>(gx, gy, imn, nx, ny, i, j);
        out[ino] = gx;
        ino += nxy;
        out[ino] = gy;
        ino += nxy;
        imn += nxy;
    }
}

// Just a simple vector field divergence
template<BackgroundStrategy backgroundStrategy>
inline __device__
void
divergence_kernel(Real* out, const Real* v,
        int nn, int nx, int ny, int i, int j) {
    int nxy = nx*ny;
    const Real* vn = v; // pointer to current channel
    // index of current output point (first component. add nxy for second)
    int ino = i*ny + j;
    for (int n=0; n < nn; ++n) {
        out[ino] = diff_x<backgroundStrategy>(vn, nx, ny, i, j);
        vn += nxy;
        out[ino] += diff_y<backgroundStrategy>(vn, nx, ny, i, j);
        ino += nxy;
    }
}

// Templated function to compute the Jacobian matrix of the first vector field
// and contract it with the second vector field in a point-wise fashion. The
// Jacobian will be transposed first if the template argument 'transpose' is 1
// instead of 0. If the template argument displacement is 1 then the vector
// field v will be treated as the displacement of a deformation whose jacobian
// we compute.
template<BackgroundStrategy backgroundStrategy, int transpose, int displacement>
inline __device__
void
jacobian_times_vectorfield_kernel(Real* out, const Real* v, const Real* w,
        int nn, int nx, int ny, int i, int j) {
    Real gx, gy, gx2, gy2; // gradient of component of v
    int nxy = nx*ny;
    const Real* vn = v; // pointer to current vector field v
    // index of current output point (first component. add nxy for second)
    int ino = i*ny + j;
    int inx = ino;
    int iny = nxy + ino;
    for (int n=0; n < nn; ++n) {
        if (transpose) {
            // get gradient of each component of vn
            grad_point<backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
            vn += nxy; // move to next component
            grad_point<backgroundStrategy>(gx2, gy2, vn, nx, ny, i, j);
            if (displacement) {
                out[ino] = (gx+1.0f)*w[inx] + gx2*w[iny];
                ino += nxy;
                out[ino] = gy*w[inx] + (gy2+1.0f)*w[iny];
            } else {
                out[ino] = gx*w[inx] + gx2*w[iny];
                ino += nxy;
                out[ino] = gy*w[inx] + gy2*w[iny];
            }
            vn += nxy; // move to next image
            ino += nxy;
        } else {
            // get gradient of each component of vn
            grad_point<backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
            if (displacement) {
                out[ino] = (gx+1.0f)*w[inx] + gy*w[iny];
                vn += nxy; // move to next component
                ino += nxy;
                grad_point<backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
                out[ino] = gx*w[inx] + (gy+1.0f)*w[iny];
            } else {
                out[ino] = gx*w[inx] + gy*w[iny];
                vn += nxy; // move to next component
                ino += nxy;
                grad_point<backgroundStrategy>(gx, gy, vn, nx, ny, i, j);
                out[ino] = gx*w[inx] + gy*w[iny];
            }
            vn += nxy; // move to next image
            ino += nxy;
        }
        // increment w lookups
        inx += 2*nxy;
        iny += 2*nxy;
    }
}

extern "C" {
    __global__ void gradient_2d(Real* out,
            const Real* im,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        gradient_kernel<DEFAULT_BACKGROUND_STRATEGY>(out, im, nn, nx, ny, i, j);
    }
    __global__ void divergence_2d(Real* out,
            const Real* v,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        divergence_kernel<DEFAULT_BACKGROUND_STRATEGY>(out, v, nn, nx, ny, i, j);
    }
    __global__ void jacobian_times_vectorfield_2d(Real* out,
            const Real* v, const Real* w,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        jacobian_times_vectorfield_kernel<DEFAULT_BACKGROUND_STRATEGY, 0, 0>(
            out, v, w, nn, nx, ny, i, j);
    }
    __global__ void jacobian_transpose_times_vectorfield_2d(Real* out,
            const Real* v, const Real* w,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        jacobian_times_vectorfield_kernel<DEFAULT_BACKGROUND_STRATEGY, 1, 0>(
            out, v, w, nn, nx, ny, i, j);
    }
    __global__ void jacobian_displacement_times_vectorfield_2d(Real* out,
            const Real* v, const Real* w,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        jacobian_times_vectorfield_kernel<DEFAULT_BACKGROUND_STRATEGY, 0, 1>(
            out, v, w, nn, nx, ny, i, j);
    }
    __global__ void jacobian_displacement_transpose_times_vectorfield_2d(Real* out,
            const Real* v, const Real* w,
            int nn, int nx, int ny) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        int j = blockDim.y * blockIdx.y + threadIdx.y;
        if (i >= nx || j >= ny) return;
        jacobian_times_vectorfield_kernel<DEFAULT_BACKGROUND_STRATEGY, 1, 1>(
            out, v, w, nn, nx, ny, i, j);
    }
}
''', extra_nvcc_flags=[
        '-DDEFAULT_BACKGROUND_STRATEGY=BACKGROUND_STRATEGY_CLAMP'])
gradient_2d = mod.func("gradient_2d")
divergence_2d = mod.func("divergence_2d")
jacobian_times_vectorfield_2d = mod.func("jacobian_times_vectorfield_2d")
jacobian_transpose_times_vectorfield_2d = \
        mod.func("jacobian_transpose_times_vectorfield_2d")
jacobian_displacement_times_vectorfield_2d = mod.func("jacobian_displacement_times_vectorfield_2d")
jacobian_displacement_transpose_times_vectorfield_2d = \
        mod.func("jacobian_displacement_transpose_times_vectorfield_2d")
