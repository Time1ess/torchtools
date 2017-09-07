#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:19
# Last modified: 2017-09-07 21:19
# Filename: timemeter.py
# Description:
from datetime import datetime

from .meter import Meter, EpochResetMeter


class TimeMeter(EpochResetMeter):
    def on_epoch_start(self, trainer, state):
        self.tick = datetime.now()

    def on_epoch_end(self, trainer, state):
        self.tock = datetime.now()

    @property
    def value(self):
        return (self.tock - self.tick).total_seconds()
