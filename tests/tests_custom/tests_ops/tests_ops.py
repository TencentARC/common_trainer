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

    def check_output_and_grad(self, out_torch, out_custom, grad_torch, grad_custom):
        """Check the output and grad"""
        if out_torch is not None:
            for out, out_custom in zip(out_torch, out_custom):
                self.assertTrue(torch.allclose(out, out_custom))

        if grad_torch is not None:
            for grad, grad_custom in zip(grad_torch, grad_custom):
                self.assertTrue(torch.allclose(grad, grad_custom))

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
