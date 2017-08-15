#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:43
# Last modified: 2017-08-15 10:10
# Filename: setup.py
# Description:
from setuptools import setup, find_packages

VERSION = '0.0.1'
DESC = ('A High-Level training API on top of PyTorch with '
        'many useful features')
DESC_SHORT = 'A High-Level training API on top of PyTorch'

setup_info = dict(
    name='torchtools',
    version=VERSION,
    author='Youchen Du',
    author_email='youchen.du@gmail.com',
    url='https://github.com/Time1ess/torchtools',
    description=DESC_SHORT,
    long_description=DESC,
    license='MIT',
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    install_requires=[
        'numpy',
        'Pillow',
        'tqdm',
        'torch',
        'visdom'])


setup(**setup_info)
