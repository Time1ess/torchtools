#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-10-16 17:28
# Last modified: 2017-10-16 17:55
# Filename: accmeter.py
# Description:
from .meter import EpochResetMixin, BatchResetMixin, SCALAR_METER
from .meter import Meter, FixSizeAverageMeter


class AccuracyMeter(Meter):
    meter_type = SCALAR_METER

    def __init__(self, name, meter_mode, *args, **kwargs):
        self.total_cnt = 1e-9
        self.correct_cnt = 0
        super(AccuracyMeter, self).__init__(name, meter_mode, *args, **kwargs)

    def on_forward_end(self, trainer, state):
        if state['mode'] != self.meter_mode:
            return
        output = state['output'].max(1)[1]
        target = state['target']
        self.total_cnt += output.size()[0]
        self.correct_cnt += target.eq(output.data.cpu()).sum()

    def reset(self):
        self.total_cnt = 1e-9
        self.correct_cnt = 0

    @property
    def value(self):
        return self.correct_cnt / self.total_cnt * self.scaling


class EpochAccuracyMeter(EpochResetMixin, AccuracyMeter):
    pass


class BatchAccuracyMeter(BatchResetMixin, AccuracyMeter):
    pass


class FixSizeAccuracyMeter(FixSizeAverageMeter, AccuracyMeter):
    pass
