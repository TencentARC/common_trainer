# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

import _add_matrix


class AddMatrixOps(torch.autograd.Function):
    """Python wrapper of the CUDA function"""

    @staticmethod
    def forward(ctx, A, B):
        output = _add_matrix.add_matrix_forward(A, B)
        ctx.save_for_backward(A, B)  # save A, B for backward

        return output

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()  # make it contiguous
        grad_A, grad_B = _add_matrix.add_matrix_backward(grad, *ctx.saved_tensors)  # restore A, B for grad calculation

        return grad_A, grad_B


class AddMatrix(nn.Module):
    """A torch.nn class that use the add_matrix function"""

    def __init__(self):
        super(AddMatrix, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x: torch tensor with (B, N) shape
            y: torch tensor with (B, N) shape

        Returns:
             output: torch tensor with (B, N) shape
        """
        return AddMatrixOps.apply(x, y)


if __name__ == '__main__':
    func = AddMatrix()
    x = torch.rand((100, 200), dtype=torch.float32, requires_grad=True).cuda()
    y = torch.rand((100, 200), dtype=torch.float32, requires_grad=True).cuda()
    out = func(x, y)
