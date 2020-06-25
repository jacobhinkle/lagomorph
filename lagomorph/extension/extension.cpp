#include <torch/extension.h>

#include <vector>
#include <string>

#include "defs.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
bool check_contiguous(std::vector<torch::Tensor> tens) {
    for (const auto& x : tens) {
        if (!x.is_contiguous())
            return false;
    }
    return true;
}
bool check_inputs(std::vector<torch::Tensor> tens) {
    for (const auto& x : tens) {
        if (!x.is_contiguous() || !x.is_cuda())
            return false;
    }
    return true;
}

bool lagomorph_debug_mode = false;

// forward declarations of CPU entrypoints
torch::Tensor affine_interp_cpu_forward(
    torch::Tensor I,
    torch::Tensor A,
    torch::Tensor T);

// forward declarations of cuda entrypoints
torch::Tensor interp_cuda_forward(
    torch::Tensor Iv,
    torch::Tensor u,
    double dt);
std::vector<torch::Tensor> interp_cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor Iv,
    torch::Tensor u,
    double dt,
    bool need_I,
    bool need_u);
torch::Tensor interp_hessian_diagonal_image(
    torch::Tensor Iv,
    torch::Tensor u,
    double dt);
torch::Tensor affine_interp_cuda_forward(
    torch::Tensor I,
    torch::Tensor A,
    torch::Tensor T);
std::vector<torch::Tensor> affine_interp_cuda_backward(
    torch::Tensor grad_out,
    torch::Tensor I,
    torch::Tensor A,
    torch::Tensor T,
    bool need_I,
    bool need_A,
    bool need_T);
torch::Tensor regrid_forward(
    torch::Tensor I,
    std::vector<int> shape,
    std::vector<double> origin,
    std::vector<double> spacing);
torch::Tensor regrid_backward(
    torch::Tensor grad_out,
    std::vector<int> inshape,
    std::vector<int> shape,
    std::vector<double> origin,
    std::vector<double> spacing);
torch::Tensor jacobian_times_vectorfield_forward(
    torch::Tensor g,
    torch::Tensor v,
    bool displacement,
    bool transpose);
std::vector<torch::Tensor> jacobian_times_vectorfield_backward(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor w,
    bool displacement,
    bool transpose,
    bool need_v,
    bool need_w);
torch::Tensor jacobian_times_vectorfield_adjoint_forward(
    torch::Tensor g,
    torch::Tensor v);
std::vector<torch::Tensor> jacobian_times_vectorfield_adjoint_backward(
    torch::Tensor grad_out,
    torch::Tensor v,
    torch::Tensor w,
    bool need_v,
    bool need_w);
void fluid_operator_cuda(
    torch::Tensor Fmv,
    bool inverse,
    std::vector<torch::Tensor> coslut,
    std::vector<torch::Tensor> sinlut,
    double alpha,
    double beta,
    double gamma);


void set_debug_mode(bool mode) {
    lagomorph_debug_mode = mode;
}

torch::Tensor affine_interp_forward(
        torch::Tensor I,
        torch::Tensor A,
        torch::Tensor T) {
    check_contiguous({I, A, T});
    if (I.is_cuda())
        return affine_interp_cuda_forward(I, A, T);
    else
        return affine_interp_cpu_forward(I, A, T);
}

std::vector<torch::Tensor> affine_interp_backward(
        torch::Tensor grad_out,
        torch::Tensor I,
        torch::Tensor A,
        torch::Tensor T,
        bool need_I,
        bool need_A,
        bool need_T) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(I);
    CHECK_INPUT(A);
    CHECK_INPUT(T);
    return affine_interp_cuda_backward(grad_out, I, A, T, need_I, need_A, need_T);
}

torch::Tensor interp_forward(
        torch::Tensor Iv,
        torch::Tensor u,
        double dt=1.0) {
    CHECK_INPUT(Iv);
    CHECK_INPUT(u);
    // insert a dummy dimension then call the generic interpolator
    return interp_cuda_forward(Iv, u, dt);
}

std::vector<torch::Tensor> interp_backward(
        torch::Tensor grad_out,
        torch::Tensor I,
        torch::Tensor u,
        double dt,
        bool need_I,
        bool need_u) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(I);
    CHECK_INPUT(u);
    return interp_cuda_backward(grad_out, I, u, dt, need_I, need_u);
}

void fluid_operator(
    torch::Tensor Fmv,
    const bool inverse,
    const std::vector<torch::Tensor> cosluts,
    const std::vector<torch::Tensor> sinluts,
    const double alpha,
    const double beta,
    const double gamma) {
    CHECK_INPUT(Fmv);
    size_t dim = Fmv.dim() - 3; // note that pytorch fft adds a dimension (size 2)
    TORCH_CHECK(cosluts.size() == dim, "Must provide same number cosine LUTs ("
            + std::to_string(cosluts.size()) + ") as spatial dimension '" + std::to_string(dim) + "'")
    TORCH_CHECK(sinluts.size() == dim, "Must provide same number sine LUTs ("
            + std::to_string(sinluts.size()) + ") as spatial dimension '" + std::to_string(dim) + "'")
    return fluid_operator_cuda(Fmv, inverse, cosluts, sinluts, alpha, beta, gamma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_debug_mode", &set_debug_mode, "Set debug (sync) mode");
  m.def("affine_interp_forward", &affine_interp_forward, "Affine interp forward (CUDA)");
  m.def("affine_interp_backward", &affine_interp_backward, "Affine interp backward (CUDA)");
  m.def("regrid_forward", &regrid_forward, "Regrid forward (CUDA)");
  m.def("regrid_backward", &regrid_backward, "Regrid backward (CUDA)");
  m.def("fluid_operator", &fluid_operator, "Fluid forward and inverse FFT operator");
  m.def("interp_forward", &interp_forward, "Free-form interp forward (CUDA)");
  m.def("interp_backward", &interp_backward, "Free-form interp backward (CUDA)");
  m.def("interp_hessian_diagonal_image", &interp_hessian_diagonal_image, "Hessian diagonal of free-form interp forward (CUDA)");
  m.def("jacobian_times_vectorfield_forward", &jacobian_times_vectorfield_forward, "Jacobian times vector field forward (CUDA)");
  m.def("jacobian_times_vectorfield_backward", &jacobian_times_vectorfield_backward, "Jacobian times vector field backward (CUDA)");
  m.def("jacobian_times_vectorfield_adjoint_forward", &jacobian_times_vectorfield_adjoint_forward, "Jacobian times vector field adjoint forward (CUDA)");
  m.def("jacobian_times_vectorfield_adjoint_backward", &jacobian_times_vectorfield_adjoint_backward, "Jacobian times vector field adjoint backward (CUDA)");
}
