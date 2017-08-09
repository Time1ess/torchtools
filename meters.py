#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-09 11:23
# Last modified: 2017-08-09 21:52
# Filename: meters.py
# Description:
import math

import numpy as np


class Meter:
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class AverageMeter(Meter):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0
        self.var = 0

    def add(self, value, n = 1):
        self.n += n
        self.sum += value
        self.var += value * value

    def value(self):
        n = self.n
        if n == 0:
            mean, std = np.nan, np.nan
        elif n == 1:
            mean, std = self.sum, np.inf
        else:
            mean = self.sum / n
            std = math.sqrt((self.var - mean * mean * n) / (n - 1.0))
        return mean, std
