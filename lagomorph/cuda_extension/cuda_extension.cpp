#include <torch/torch.h>

#include <vector>
#include <string>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

bool lagomorph_debug_mode = false;

// forward declarations of cuda entrypoints
at::Tensor interp_cuda_forward(
    at::Tensor Iv,
    at::Tensor u,
    double dt);
std::vector<at::Tensor> interp_cuda_backward(
    at::Tensor grad_out,
    at::Tensor Iv,
    at::Tensor u,
    double dt,
    bool need_I,
    bool need_u);
at::Tensor interp_hessian_diagonal_image(
    at::Tensor Iv,
    at::Tensor u,
    double dt);
at::Tensor affine_interp_cuda_forward(
    at::Tensor I,
    at::Tensor A,
    at::Tensor T);
std::vector<at::Tensor> affine_interp_cuda_backward(
    at::Tensor grad_out,
    at::Tensor I,
    at::Tensor A,
    at::Tensor T,
    bool need_I,
    bool need_A,
    bool need_T);
at::Tensor regrid_forward(
    at::Tensor I,
    std::vector<int> shape,
    std::vector<double> origin,
    std::vector<double> spacing);
at::Tensor regrid_backward(
    at::Tensor grad_out,
    std::vector<int> inshape,
    std::vector<int> shape,
    std::vector<double> origin,
    std::vector<double> spacing);
at::Tensor jacobian_times_vectorfield_forward(
    at::Tensor g,
    at::Tensor v,
    bool displacement,
    bool transpose);
std::vector<at::Tensor> jacobian_times_vectorfield_backward(
    at::Tensor grad_out,
    at::Tensor v,
    at::Tensor w,
    bool displacement,
    bool transpose,
    bool need_v,
    bool need_w);
at::Tensor jacobian_times_vectorfield_adjoint_forward(
    at::Tensor g,
    at::Tensor v);
std::vector<at::Tensor> jacobian_times_vectorfield_adjoint_backward(
    at::Tensor grad_out,
    at::Tensor v,
    at::Tensor w,
    bool need_v,
    bool need_w);
void fluid_operator_cuda(
    at::Tensor Fmv,
    bool inverse,
    std::vector<at::Tensor> coslut,
    std::vector<at::Tensor> sinlut,
    double alpha,
    double beta,
    double gamma);


void set_debug_mode(bool mode) {
    lagomorph_debug_mode = mode;
}

at::Tensor affine_interp_forward(
        at::Tensor I,
        at::Tensor A,
        at::Tensor T) {
    CHECK_INPUT(I);
    CHECK_INPUT(A);
    CHECK_INPUT(T);
    return affine_interp_cuda_forward(I, A, T);
}

std::vector<at::Tensor> affine_interp_backward(
        at::Tensor grad_out,
        at::Tensor I,
        at::Tensor A,
        at::Tensor T,
        bool need_I,
        bool need_A,
        bool need_T) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(I);
    CHECK_INPUT(A);
    CHECK_INPUT(T);
    return affine_interp_cuda_backward(grad_out, I, A, T, need_I, need_A, need_T);
}

at::Tensor interp_forward(
        at::Tensor Iv,
        at::Tensor u,
        double dt=1.0) {
    CHECK_INPUT(Iv);
    CHECK_INPUT(u);
    // insert a dummy dimension then call the generic interpolator
    return interp_cuda_forward(Iv, u, dt);
}

std::vector<at::Tensor> interp_backward(
        at::Tensor grad_out,
        at::Tensor I,
        at::Tensor u,
        double dt,
        bool need_I,
        bool need_u) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(I);
    CHECK_INPUT(u);
    return interp_cuda_backward(grad_out, I, u, dt, need_I, need_u);
}

void fluid_operator(
    at::Tensor Fmv,
    const bool inverse,
    const std::vector<at::Tensor> cosluts,
    const std::vector<at::Tensor> sinluts,
    const double alpha,
    const double beta,
    const double gamma) {
    CHECK_INPUT(Fmv);
    size_t dim = Fmv.dim() - 3; // note that pytorch fft adds a dimension (size 2)
    AT_ASSERTM(cosluts.size() == dim, "Must provide same number cosine LUTs ("
            + std::to_string(cosluts.size()) + ") as spatial dimension '" + std::to_string(dim) + "'")
    AT_ASSERTM(sinluts.size() == dim, "Must provide same number sine LUTs ("
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
