#!/usr/bin/python
# -*- coding: utf-8 -*-

from common.utils.cfgs_utils import parse_configs
from custom.trainer.custom_trainer import CustomTrainer

if __name__ == '__main__':
    # parse args
    cfgs = parse_configs()

    # trainer
    trainer = CustomTrainer(cfgs)
    trainer.train()
