# -*- coding: utf-8 -*-

import torchvision.transforms as transforms

from common.datasets.transform.augmentation import (ColorJitter, ImgNorm, PermuteImg)
from common.utils.cfgs_utils import valid_key_in_cfgs


def get_transforms(cfgs):
    """Get a list of transformation. You can change it in your only augmentation"""
    transforms_list = []
    aug_info = ''

    if valid_key_in_cfgs(cfgs, 'augmentation'):
        if valid_key_in_cfgs(cfgs.augmentation, 'jitter'):
            transforms_list.append(ColorJitter(cfgs.augmentation.jitter))
            aug_info += '  Add ColorJitter with level {}\n'.format(cfgs.augmentation.jitter)

    transforms_list.append(ImgNorm(norm_by_255=True))
    aug_info += '  Add ImgNorm'
    transforms_list.append(PermuteImg())
    aug_info += '  Add Permute'

    return transforms.Compose(transforms_list), aug_info
