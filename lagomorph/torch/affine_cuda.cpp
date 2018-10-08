#include <torch/torch.h>

#include <vector>
#include <stdexcept>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("affine_interp_image_forward", &affine_interp_image_forward, "Affine interp image forward (CUDA)");
  m.def("affine_interp_image_backward", &affine_interp_image_backward, "Affine interp image backward (CUDA)");
}
