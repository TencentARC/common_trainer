#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from custom.models import build_model
from tests import setup_test_config


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.model = build_model(self.cfgs, None)
        self.batch_size = 2
        self.input = self.set_dummpy_input()

    def set_dummpy_input(self):
        """Set dummpy input for model"""
        input = torch.zeros(size=(self.batch_size, 3, 10, 10))

        return input

    def test_model(self):
        """Test model"""
        output = self.model(self.input)['img']
        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], self.batch_size, 'Batch size not match')


if __name__ == '__main__':
    unittest.main()
