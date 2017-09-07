#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:18
# Last modified: 2017-09-07 16:40
# Filename: averagemeter.py
# Description:
import numpy as np

from .meter import Meter


class AverageMeter(Meter):
    def __init__(self, *args, **kwargs):
        super(AverageMeter, self).__init__(*args, **kwargs)
        self.values = []

    def reset(self):
        self.values.clear()

    def on_forward_end(self, trainer, state):
        val = state.get(self.name, None)
        if val is not None:
            self.add(state[self.name].data[0])

    def add(self, value):
        self.values.append(value)

    def calculate(self):
        if not self.values:
            mean, std = np.nan, np.nan
        else:
            mean, std = np.nanmean(self.values), np.nanstd(self.values)
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
