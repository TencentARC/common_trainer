#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from common.loss.loss_dict import LossDictCounter
from custom.loss import build_loss
from tests import setup_test_config


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.loss_factory = build_loss(self.cfgs, None)
        self.loss_summary = LossDictCounter()
        self.batch_size = 2
        self.inputs, self.output = self.setup_dummpy_var()

    def setup_dummpy_var(self):
        """Set up dummpy inputs and output for testing"""
        inputs = {'gt': torch.zeros(size=(self.batch_size, 1, 10, 10))}
        output = {'img': torch.zeros(size=(self.batch_size, 1, 10, 10))}

        return inputs, output

    def test_loss_factory(self):
        """Test loss forward by loss factory"""
        loss = self.loss_factory(self.inputs, self.output)
        self.assertIsInstance(loss, dict)
        self.assertIn('sum', loss.keys(), 'Loss sum not inside')
        self.assertIn('names', loss.keys(), 'Loss names not inside')
        for name in loss['names']:
            self.assertIn(name, loss.keys(), '{} not in loss'.format(name))

    def test_loss_dict(self):
        """Test loss counting dict"""
        self.loss_summary.reset()
        num_iter = 20
        for _ in range(num_iter):
            loss = self.loss_factory(self.inputs, self.output)
            self.loss_summary(loss, self.batch_size)
        self.assertIsNotNone(self.loss_summary.get_summary())
        self.assertEqual(self.loss_summary.get_count(), num_iter * self.batch_size, 'Num not match')
        # cal avg
        self.loss_summary.cal_average()
        self.assertIsNotNone(self.loss_summary.get_avg_summary())
        self.assertIn('sum', self.loss_summary.get_avg_summary().keys())
        # reset
        self.loss_summary.reset()
        self.assertEqual(self.loss_summary.get_count(), 0)
        self.assertIsNone(self.loss_summary.get_avg_summary())
        self.assertIsNone(self.loss_summary.get_summary())


if __name__ == '__main__':
    unittest.main()
