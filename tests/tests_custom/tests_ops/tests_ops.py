#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch
from torch.autograd import gradcheck

from common.utils.logger import Logger
from custom.ops import AddMatrix, ScaleExp
from . import log_custom_benchmark

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):
    """Only test any implementation on GPU"""

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger(path=osp.join(RESULT_DIR, './benchmark.txt'), keep_console=False)
        cls.batch_size = 1024 * 4
        cls.n_sample = 2048 * 4

    def check_output_and_grad(self, out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom, atol=1e-8):
        """Check the output and grad"""
        if out_torch is not None:
            if isinstance(out_torch, list):
                for out, _out, _out_forward in zip(out_torch, out_custom, out_custom_forward_only):
                    if isinstance(out, torch.Tensor):
                        self.assertTrue(torch.allclose(out, _out, atol=atol))
                        self.assertTrue(torch.allclose(out, _out_forward, atol=atol))
            else:
                if isinstance(out_torch, torch.Tensor):
                    self.assertTrue(torch.allclose(out_torch, out_custom, atol=atol))
                    self.assertTrue(torch.allclose(out_torch, out_custom_forward_only, atol=atol))

        if grad_torch is not None:
            if isinstance(grad_torch, list):
                for grad, _grad in zip(grad_torch, grad_custom):
                    if isinstance(grad, torch.Tensor):
                        self.assertTrue(torch.allclose(grad, _grad, atol=atol))
            else:
                if isinstance(grad_torch, torch.Tensor):
                    self.assertTrue(torch.allclose(grad_torch, grad_custom, atol=atol))

    def tests_add_matrix(self):
        inputs = [
            torch.rand((self.batch_size, self.n_sample), dtype=torch.double, requires_grad=True),
            torch.rand((self.batch_size, self.n_sample), dtype=torch.double, requires_grad=True)
        ]

        def add_matrix_torch(x, y):
            return x + y

        add_matrix_custom = AddMatrix()

        out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom = log_custom_benchmark(
            self.logger, 'Add Matrix', add_matrix_torch, add_matrix_custom, inputs
        )

        self.check_output_and_grad(out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom)

    def tests_scale_exp(self):
        inputs = [torch.rand((self.batch_size, self.n_sample), dtype=torch.double, requires_grad=True)]

        scale = 3.3
        bias = 2.89

        def scale_exp_torch(x):
            return scale * torch.exp(-x) + bias

        scale_exp_custom = ScaleExp(scale, bias)

        out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom = log_custom_benchmark(
            self.logger, 'Scale Exp', scale_exp_torch, scale_exp_custom, inputs
        )

        self.check_output_and_grad(out_torch, out_custom, out_custom_forward_only, grad_torch, grad_custom)

    def tests_gradcheck_add_matrix(self):
        if not torch.cuda.is_available():
            return

        inputs = (
            torch.rand((10, 20), dtype=torch.double,
                       requires_grad=True).cuda(), torch.rand((10, 20), dtype=torch.double, requires_grad=True).cuda()
        )

        add_matrix_custom = AddMatrix()

        self.assertTrue(gradcheck(add_matrix_custom, inputs, eps=1e-6, atol=1e-8))

    def tests_gradcheck_scale_exp(self):
        if not torch.cuda.is_available():
            return

        inputs = torch.rand((10, 20), dtype=torch.double, requires_grad=True).cuda()

        scale_exp_custom = ScaleExp(3.3, 2.89)

        self.assertTrue(gradcheck(scale_exp_custom, inputs, eps=1e-6, atol=1e-8))
