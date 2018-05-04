#!/usr/bin/env python

"""
from setuptools import setup, find_packages

requirements = [
    'torch',
]

setup(
    name='torch_detectron',
    version='0.1',
    author='killeent',
    url='https://github.com/soumith/detectron.pytorch',
    description='object detection in pytorch',
    packages=find_packages(exclude=('test',)),
    install_requires=requirements,
)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()


# Extension modules
ext_modules = [
    Extension(
        name='utils.cython_bbox',
        sources=[
            'torch_detectron/lib/utils/cython_bbox.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
    Extension(
        name='utils.cython_nms',
        sources=[
            'torch_detectron/lib/utils/cython_nms.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    )
]

setup(
    name='torch_detectron',
    ext_modules=cythonize(ext_modules)
)
