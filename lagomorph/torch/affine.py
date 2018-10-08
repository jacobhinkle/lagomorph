import torch
import lagomorph_torch_cuda

class AffineInterpImageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, I, A, T):
        ctx.save_for_backward(I, A, T)
        print(ctx, ctx.needs_input_grad)
        return lagomorph_torch_cuda.affine_interp_image_forward(
            I.contiguous(),
            A.contiguous(),
            T.contiguous())
    @staticmethod
    def backward(ctx, grad_out):
        I, A, T = ctx.saved_variables
        return lagomorph_torch_cuda.affine_interp_image_backward(
                grad_out.contiguous(),
                I.contiguous(),
                A.contiguous(),
                T.contiguous(),
                *ctx.needs_input_grad)

class AffineInterpImage(torch.nn.Module):
    def __init__(self, dim):
        super(AffineInterpImage, self).__init__()
        self.dim = dim
    def forward(self, I, A, T):
        return AffineInterpImageFunction.apply(I, A, T)
