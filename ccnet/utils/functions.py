import torch

class Heaviside(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):

        indices = torch.where(tensor >= 0.0)
        ctx.indices = indices
        output = torch.zeros(tensor.shape, dtype=tensor.dtype)
        output[indices] = 1

        return output

    @staticmethod
    def backward(ctx, grad_output):

        indices = ctx.indices
        grad_input = torch.zeros(grad_output.shape, dtype=grad_output.dtype) + 0.001
        grad_input[indices] = 1e+6

        return grad_input * grad_output
