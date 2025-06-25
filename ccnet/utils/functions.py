import torch

class Heaviside(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, k=100.0):

        ctx.k = k
        ctx.save_for_backward(tensor)

        return (tensor >= 0.0).to(tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output):

        k = ctx.k
        tensor, = ctx.saved_tensors
        sig = torch.sigmoid(k * tensor)
        grad_input = grad_output * k * sig * (1 - sig) 

        return grad_input, None
