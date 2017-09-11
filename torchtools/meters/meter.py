#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:17
# Last modified: 2017-09-11 15:24
# Filename: meter.py
# Description:
import numpy as np

from collections import deque

from ..callbacks import Hook

NO_RESET = 0b0
BATCH_RESET = 0b1
EPOCH_RESET = 0b10

BASE_METER = 'none'
SCALAR_METER = 'scalar'
TEXT_METER = 'text'
IMAGE_METER = 'image'
HIST_METER = 'hist'
GRAPH_METER = 'graph'
AUDIO_METER = 'audio'


class Meter(Hook):
    reset_mode = NO_RESET
    meter_type = BASE_METER

    def __init__(self, name, meter_mode, *args, **kwargs):
        self.name = name
        self.meter_mode = meter_mode
        super(Meter, self).__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        pass

    @property
    def value(self):
        pass

    @property
    def can_call(self):
        return True

    def add(self):
        pass


class EpochResetMixin(Meter):
    reset_mode = EPOCH_RESET

    def on_epoch_start(self, trainer, state):
        self.reset()


class BatchResetMixin(Meter):
    reset_mode = BATCH_RESET

    def on_batch_start(self, trainer, state):
        self.reset()


class AverageMeter(Meter):
    meter_type = SCALAR_METER

    def __init__(self, *args, **kwargs):
        self.values = deque()
        super(AverageMeter, self).__init__(*args, **kwargs)

    def reset(self):
        self.values.clear()

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


class EpochAverageMeter(EpochResetMixin, AverageMeter):
    pass


class BatchAverageMeter(BatchResetMixin, AverageMeter):
    pass


class FixSizeAverageMeter(AverageMeter):
    def __init__(self, name, meter_mode, fix_size, *args, **kwargs):
        self.fix_size = fix_size
        super(FixSizeAverageMeter, self).__init__(name, meter_mode, *args,
                                                  **kwargs)

    def add(self, value):
        if len(self.values) == self.fix_size:
            self.values.popleft()
        super(FixSizeAverageMeter, self).add(value)
