#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest

from custom.datasets import get_dataset
from custom.datasets.transform.augmentation import get_transforms
from tests import setup_test_config


class TestDict(unittest.TestCase):

    def setUp(self):
        self.cfgs = setup_test_config()
        self.mode = 'train'
        self.dataset = self.setup_dataset()

    def setup_dataset(self):
        """Setup dataset by get_dataset and get_transforms"""
        transform, info = get_transforms(getattr(self.cfgs.dataset, self.mode))
        dataset = get_dataset(self.cfgs.dataset, self.cfgs.dir.data_dir, None, self.mode, transform)

        return dataset

    def test_dataset(self):
        """Test dataset"""
        self.assertGreater(len(self.dataset), 0, 'Empty datasets...')
        self.assertIsInstance(self.dataset[0], dict, 'Return is not a dict')


if __name__ == '__main__':
    unittest.main()
