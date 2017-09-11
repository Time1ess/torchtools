#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:18
# Last modified: 2017-09-11 15:24
# Filename: lossmeter.py
# Description:
from .meter import EpochAverageMeter, BatchAverageMeter, SCALAR_METER
from .meter import AverageMeter, FixSizeAverageMeter


class LossMeter(AverageMeter):
    meter_type = SCALAR_METER

    def __init__(self, name, meter_mode, loss_type=None, *args, **kwargs):
        if loss_type is None:
            loss_type = name
        assert loss_type in ('loss', 'val_loss')
        self.values = []
        self.loss_type = loss_type
        super(LossMeter, self).__init__(name, meter_mode, *args, **kwargs)

    def on_forward_end(self, trainer, state):
        if state['mode'] != self.meter_mode:
            return
        val = state.get(self.loss_type, None)
        if val is not None:
            self.add(state[self.name].data[0])


class EpochLossMeter(EpochAverageMeter, LossMeter):
    pass


class BatchLossMeter(BatchAverageMeter, LossMeter):
    pass


class FixSizeLossMeter(FixSizeAverageMeter, LossMeter):
    pass
