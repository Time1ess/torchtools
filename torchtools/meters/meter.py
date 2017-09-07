#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:17
# Last modified: 2017-09-07 22:17
# Filename: meter.py
# Description:
from ..callbacks import Hook

NO_RESET = 0b0
BATCH_RESET = 0b1
EPOCH_RESET = 0b10


class Meter(Hook):
    reset_mode = NO_RESET

    def __init__(self, name, *args, **kwargs):
        super(Meter, self).__init__(*args, **kwargs)
        self.reset()
        self.name = name

    def reset(self):
        pass

    @property
    def value(self):
        pass

    def add(self):
        pass


class EpochResetMeter(Meter):
    reset_mode = EPOCH_RESET

    def on_epoch_start(self, trainer, state):
        self.reset()


class BatchResetMeter(Meter):
    reset_mode = BATCH_RESET

    def on_batch_start(self, trainer, state):
        self.reset()
