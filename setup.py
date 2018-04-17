# coding: UTF-8
from setuptools import setup, find_packages

VERSION = '0.1.0'
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
        'tensorboardX'])


setup(**setup_info)
