#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

import torch

from common.metric.metric_dict import MetricDictCounter
from custom.metric import build_metric
from tests import setup_test_config


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.eval_metric = build_metric(self.cfgs, None)
        self.metric_summary = MetricDictCounter()
        self.batch_size = 2
        self.inputs, self.output = self.setup_dummpy_var()

    def setup_dummpy_var(self):
        """Set up dummpy inputs and output for testing"""
        inputs = {'gt': torch.zeros(size=(self.batch_size, 1, 10, 10))}
        output = {'img': torch.zeros(size=(self.batch_size, 1, 10, 10))}

        return inputs, output

    def test_eval_metric(self):
        """Test metric forward"""
        metric = self.eval_metric(self.inputs, self.output)
        self.assertIsInstance(metric, dict)
        self.assertIn('names', metric.keys(), 'Metric names not inside')
        for name in metric['names']:
            self.assertIn(name, metric.keys(), '{} not in metric'.format(name))

    def test_metric_dict(self):
        """Test metric counting dict"""
        self.metric_summary.reset()
        num_iter = 20
        for _ in range(num_iter):
            metric = self.eval_metric(self.inputs, self.output)
            self.metric_summary(metric, self.batch_size)
        self.assertIsNotNone(self.metric_summary.get_summary())
        self.assertEqual(self.metric_summary.get_count(), num_iter * self.batch_size, 'Num not match')
        # cal avg
        self.metric_summary.cal_average()
        self.assertIsNotNone(self.metric_summary.get_avg_summary())
        # reset
        self.metric_summary.reset()
        self.assertEqual(self.metric_summary.get_count(), 0)
        self.assertIsNone(self.metric_summary.get_avg_summary())
        self.assertIsNone(self.metric_summary.get_summary())


if __name__ == '__main__':
    unittest.main()
