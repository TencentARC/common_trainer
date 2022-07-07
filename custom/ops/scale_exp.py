# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import _scale_exp


class ScaleExpOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, A, scale, bias):
        output = _scale_exp.scale_exp_forward(A, scale, bias)
        ctx.save_for_backward(A)  # only save tensors
        ctx.scale = scale
        ctx.bias = bias

        return output

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()  # make it contiguous
        grad_A = _scale_exp.scale_exp_backward(grad, *ctx.saved_tensors, ctx.scale, ctx.bias)  # restore inputs

        return grad_A, None, None  # have to return grad for other inputs as well


class ScaleExp(nn.Module):
    """A torch.nn class that use the add_matrix function"""

    def __init__(self, scale=1.0, bias=0.0):
        super(ScaleExp, self).__init__()
        self.scale = scale
        self.bias = bias

    def forward(self, x):
        """
        Args:
            x: torch tensor with (B, N) shape

        Returns:
             output: torch tensor with (B, N) shape. output = scale * exp(-x) + bias
        """
        return ScaleExpOps.apply(x, self.scale, self.bias)


if __name__ == '__main__':
    func = ScaleExp()
    x = torch.rand((100, 200), dtype=torch.float32, requires_grad=True).cuda()
    out = func(x)
