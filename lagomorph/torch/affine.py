import torch
import lagomorph_torch_cuda

class AffineInterpImageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, A, T):
        ctx.save_for_backward(I, A, T)
        return lagomorph_torch_cuda.affine_interp_image_forward(
            I.contiguous(),
            A.contiguous(),
            T.contiguous())
    @staticmethod
    def backward(ctx, grad_out):
        I, A, T = ctx.saved_variables
        d_I, d_A, d_T = lagomorph_torch_cuda.affine_interp_image_backward(
                grad_out.contiguous(),
                I.contiguous(),
                A.contiguous(),
                T.contiguous(),
                *ctx.needs_input_grad)
        return d_I, d_A, d_T

class AffineInterpImage(torch.nn.Module):
    def __init__(self):
        super(AffineInterpImage, self).__init__()
    def forward(self, I, A, T):
        return AffineInterpImageFunction.apply(I, A, T)
