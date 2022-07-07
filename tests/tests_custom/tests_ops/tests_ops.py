#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest
import time

import torch

from common.utils.logger import Logger
from custom.ops import AddMatrix, ScaleExp

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):
    """Only test any implementation on GPU"""

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger(path=osp.join(RESULT_DIR, './benchmark.txt'), keep_console=False)

    @staticmethod
    def get_start_time():
        torch.cuda.synchronize()
        t0 = time.time()

        return t0

    @staticmethod
    def get_end_time(t0):
        torch.cuda.synchronize()

        return time.time() - t0

    def tests_add_matrix(self):
        if not torch.cuda.is_available():
            return

        self.logger.add_log('_' * 60)
        self.logger.add_log('Add matrix: ')
        self.logger.add_log('Input dim (1000, 2000) ')

        # grad is on cpu tensor only. gpu tensor calculation
        tensor_a = torch.rand((1000, 2000), dtype=torch.float32, requires_grad=True)
        tensor_b = torch.rand((1000, 2000), dtype=torch.float32, requires_grad=True)
        tensor_a_gpu = tensor_a.cuda()
        tensor_b_gpu = tensor_b.cuda()

        # torch implementation
        t0 = self.get_start_time()
        sum_torch = tensor_a_gpu + tensor_b_gpu
        t_forward_torch = self.get_end_time(t0)
        self.logger.add_log('Torch Forward time {:.6f}s'.format(t_forward_torch))

        loss = torch.sum((1.0 - sum_torch)**2)

        t0 = self.get_start_time()
        loss.backward()
        t_backward_torch = self.get_end_time(t0)
        self.logger.add_log('Torch Backward time {:.6f}s'.format(t_backward_torch))

        grad_a_torch = tensor_a.grad.clone()
        grad_b_torch = tensor_b.grad.clone()

        # zeros grad
        tensor_a.grad.zero_()
        tensor_b.grad.zero_()

        # custom cuda implementation
        add_matrix_custom = AddMatrix()

        t0 = self.get_start_time()
        sum_custom = add_matrix_custom(tensor_a_gpu, tensor_b_gpu)
        t_forward_custom = self.get_end_time(t0)
        self.logger.add_log(
            'Custom Forward time {:.6f}s, boost x{:.2f}'.format(t_forward_custom, t_forward_torch / t_forward_custom)
        )

        loss = torch.sum((1.0 - sum_custom)**2)

        t0 = self.get_start_time()
        loss.backward()
        t_backward_custom = self.get_end_time(t0)
        self.logger.add_log(
            'Custom Backward time {:.6f}s, boost x{:.2f}'.format(
                t_backward_custom, t_backward_torch / t_backward_custom
            )
        )

        grad_a_custom = tensor_a.grad.clone()
        grad_b_custom = tensor_b.grad.clone()

        self.assertTrue(torch.allclose(sum_torch, sum_custom))
        self.assertTrue(torch.allclose(grad_a_torch, grad_a_custom))
        self.assertTrue(torch.allclose(grad_b_torch, grad_b_custom))

        self.logger.add_log('_' * 60)
        self.logger.add_log('\n')

    def tests_scale_exp(self):
        if not torch.cuda.is_available():
            return

        self.logger.add_log('_' * 60)
        self.logger.add_log('Scale exp: ')
        self.logger.add_log('Input dim (1000, 2000) ')

        # grad is on cpu tensor only. gpu tensor calculation
        tensor_a = torch.rand((1000, 2000), dtype=torch.float, requires_grad=True)
        tensor_a_gpu = tensor_a.cuda()
        scale = 3.3
        bias = 2.89

        # torch implementation
        t0 = self.get_start_time()
        out_torch = scale * torch.exp(-tensor_a_gpu) + bias
        t_forward_torch = self.get_end_time(t0)
        self.logger.add_log('Torch Forward time {:.6f}s'.format(t_forward_torch))

        loss = torch.sum((1.0 - out_torch)**2)

        t0 = self.get_start_time()
        loss.backward()
        t_backward_torch = self.get_end_time(t0)
        self.logger.add_log('Torch Backward time {:.6f}s'.format(t_backward_torch))

        grad_a_torch = tensor_a.grad.clone()

        # zeros grad
        tensor_a.grad.zero_()

        # custom cuda implementation
        scale_exp_custom = ScaleExp(scale, bias)
        t0 = self.get_start_time()
        out_custom = scale_exp_custom(tensor_a_gpu)
        t_forward_custom = self.get_end_time(t0)
        self.logger.add_log(
            'Custom Forward time {:.6f}s, boost x{:.2f}'.format(t_forward_custom, t_forward_torch / t_forward_custom)
        )

        loss = torch.sum((1.0 - out_custom)**2)

        t0 = self.get_start_time()
        loss.backward()
        t_backward_custom = self.get_end_time(t0)
        self.logger.add_log(
            'Custom Backward time {:.6f}s, boost x{:.2f}'.format(
                t_backward_custom, t_backward_torch / t_backward_custom
            )
        )

        grad_a_custom = tensor_a.grad.clone()

        self.assertTrue(torch.allclose(out_torch, out_custom))
        self.assertTrue(torch.allclose(grad_a_torch, grad_a_custom))

        self.logger.add_log('_' * 60)
        self.logger.add_log('\n')
