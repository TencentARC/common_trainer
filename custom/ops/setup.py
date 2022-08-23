# -*- coding: utf-8 -*-
# see: https://pytorch.org/tutorials/advanced/cpp_extension.html for details

import os
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# compile on all arch
os.environ['TORCH_CUDA_ARCH_LIST'] = ''

include_dirs = [osp.dirname(osp.abspath(__file__)) + '/include']

setup(
    name='custom_ops',
    version='1.0',
    author='leoyluo',
    author_email='lawy623@gmail.com',
    description='custom cuda ops',
    long_description='custom cuda ops examples for matrix addition and exponential',
    ext_modules=[
        CUDAExtension(
            name='_add_matrix',
            sources=['./src/add_matrix/add_matrix.cpp', './src/add_matrix/add_matrix_kernel.cu'],
            include_dirs=include_dirs
        ),
        CUDAExtension(
            name='_scale_exp',
            sources=['./src/scale_exp/scale_exp.cpp', './src/scale_exp/scale_exp_kernel.cu'],
            include_dirs=include_dirs
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
