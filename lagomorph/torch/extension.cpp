#include <torch/torch.h>

#include <vector>
#include <string>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// forward declarations of cuda entrypoints
at::Tensor affine_interp_image_cuda_forward(
    at::Tensor I,
    at::Tensor A,
    at::Tensor T);
std::vector<at::Tensor> affine_interp_image_cuda_backward(
    at::Tensor grad_out,
    at::Tensor I,
    at::Tensor A,
    at::Tensor T,
    bool need_I,
    bool need_A,
    bool need_T);

at::Tensor affine_interp_image_forward(
        at::Tensor I,
        at::Tensor A,
        at::Tensor T) {
    CHECK_INPUT(I);
    CHECK_INPUT(A);
    CHECK_INPUT(T);
    return affine_interp_image_cuda_forward(I, A, T);
}

std::vector<at::Tensor> affine_interp_image_backward(
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
    return affine_interp_image_cuda_backward(grad_out, I, A, T, need_I, need_A, need_T);
}

// forward declarations of cuda entrypoints
void fluid_operator_cuda(
    at::Tensor Fmv,
    bool inverse,
    std::vector<at::Tensor> coslut,
    std::vector<at::Tensor> sinlut,
    double alpha,
    double beta,
    double gamma);

void fluid_operator(
    at::Tensor Fmv,
    bool inverse,
    std::vector<at::Tensor> cosluts,
    std::vector<at::Tensor> sinluts,
    double alpha,
    double beta,
    double gamma) {
    CHECK_INPUT(Fmv);
    size_t dim = Fmv.dim() - 3; // note that pytorch fft adds a dimension (size 2)
    AT_ASSERTM(cosluts.size() == dim, "Must provide same number cosine LUTs ("
            + std::to_string(cosluts.size()) + ") as spatial dimension '" + std::to_string(dim) + "'")
    AT_ASSERTM(sinluts.size() == dim, "Must provide same number sine LUTs ("
            + std::to_string(sinluts.size()) + ") as spatial dimension '" + std::to_string(dim) + "'")
    return fluid_operator_cuda(Fmv, inverse, cosluts, sinluts, alpha, beta, gamma);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine_interp_image_forward", &affine_interp_image_forward, "Affine interp image forward (CUDA)");
  m.def("affine_interp_image_backward", &affine_interp_image_backward, "Affine interp image backward (CUDA)");
  m.def("fluid_operator", &fluid_operator, "Fluid forward and inverse FFT operator");
}
