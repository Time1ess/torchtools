#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-07 14:58
# Last modified: 2017-09-07 22:19
# Filename: __init__.py
# Description:
from .sbd import SBDClassSegmentation
from .voc import VOCClassSegmentation


__all__ = ['SBDClassSegmentation', 'VOCClassSegmentation']
