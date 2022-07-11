#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import unittest

import torch

from . import log_custom_benchmark
from common.utils.logger import Logger
from custom.ops import AddMatrix, ScaleExp

RESULT_DIR = osp.abspath(osp.join(__file__, '..', 'results'))
os.makedirs(RESULT_DIR, exist_ok=True)


class TestDict(unittest.TestCase):
    """Only test any implementation on GPU"""

    @classmethod
    def setUpClass(cls):
        cls.logger = Logger(path=osp.join(RESULT_DIR, './benchmark.txt'), keep_console=False)

    def check_output_and_grad(self, out_torch, out_custom, grad_torch, grad_custom, atol=1e-5):
        """Check the output and grad"""
        if out_torch is not None:
            if isinstance(out_torch, list):
                for out, _out in zip(out_torch, out_custom):
                    if isinstance(out, torch.Tensor):
                        self.assertTrue(torch.allclose(out, _out, atol=atol))
            else:
                if isinstance(out_torch, torch.Tensor):
                    self.assertTrue(torch.allclose(out_torch, out_custom, atol=atol))

        if grad_torch is not None:
            if isinstance(grad_torch, list):
                for grad, _grad in zip(grad_torch, grad_custom):
                    if isinstance(grad, torch.Tensor):
                        print(torch.abs(grad - _grad).max())
                        self.assertTrue(torch.allclose(grad, _grad, atol=atol))
            else:
                if isinstance(grad_torch, torch.Tensor):
                    self.assertTrue(torch.allclose(grad_torch, grad_custom, atol=atol))

    def tests_add_matrix(self):
        inputs = [
            torch.rand((1000, 2000), dtype=torch.float32, requires_grad=True),
            torch.rand((1000, 2000), dtype=torch.float32, requires_grad=True)
        ]

        def add_matrix_torch(x, y):
            return x + y

        add_matrix_custom = AddMatrix()

        out_torch, out_custom, grad_torch, grad_custom = log_custom_benchmark(
            self.logger, 'Add Matrix', add_matrix_torch, add_matrix_custom, inputs
        )

        self.check_output_and_grad(out_torch, out_custom, grad_torch, grad_custom)

    def tests_scale_exp(self):
        inputs = [torch.rand((1000, 2000), dtype=torch.float32, requires_grad=True)]

        scale = 3.3
        bias = 2.89

        def scale_exp_torch(x):
            return scale * torch.exp(-x) + bias

        scale_exp_custom = ScaleExp(scale, bias)

        out_torch, out_custom, grad_torch, grad_custom = log_custom_benchmark(
            self.logger, 'Scale Exp', scale_exp_torch, scale_exp_custom, inputs
        )

        self.check_output_and_grad(out_torch, out_custom, grad_torch, grad_custom)
