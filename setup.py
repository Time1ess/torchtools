# coding: UTF-8
import sys
from setuptools import setup, find_packages

VERSION = '0.1.4'
DESC = ('A High-Level training API on top of PyTorch with '
        'many useful features')
DESC_SHORT = 'A High-Level training API on top of PyTorch'

install_requires = ['numpy', 'Pillow', 'tqdm', 'torch', 'tensorboardX']
if sys.version_info < (3, 3):
    install_requires.append('mock')


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
    install_requires=install_requires)


setup(**setup_info)
