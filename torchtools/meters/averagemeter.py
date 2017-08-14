#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:18
# Last modified: 2017-08-14 21:19
# Filename: averagemeter.py
# Description:
import math

import numpy as np

from .meter import Meter


class AverageMeter(Meter):
    def reset(self):
        self.n = 0
        self.sum = 0
        self.var = 0

    def on_forward_end(self, trainer, state):
        val = state.get(self.name, None)
        if val is not None:
            self.add(state[self.name].data[0])

    def add(self, value, n=1):
        self.n += n
        self.sum += value
        self.var += value * value

    def calculate(self):
        n = self.n
        if n == 0:
            mean, std = np.nan, np.nan
        elif n == 1:
            mean, std = self.sum, np.inf
        else:
            mean = self.sum / n
            std = math.sqrt((self.var - mean * mean * n) / (n - 1.0))
        self.mean = mean
        self.std = std

    @property
    def value(self):
        self.calculate()
        return self.mean


class EpochAverageMeter(AverageMeter):
    def on_epoch_start(self, trainer, state):
        self.reset()


class BatchAverageMeter(AverageMeter):
    def on_batch_start(self, trainer, state):
        self.reset()
