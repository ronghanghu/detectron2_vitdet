#!/usr/bin/env python


from setuptools import setup, find_packages

requirements = [
    'torch',
]

setup(
    name='torch_detectron',
    version='0.1',
    author='fmassa',
    url='https://github.com/fairinternal/detectron.pytorch',
    description='object detection in pytorch',
    packages=find_packages(exclude=('test',)),
    # install_requires=requirements,
)
