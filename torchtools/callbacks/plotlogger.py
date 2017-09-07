#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:36
# Last modified: 2017-09-07 20:09
# Filename: plotlogger.py
# Description:
from copy import copy
from collections import defaultdict

from ..plots import VisdomPlot
from .callback import Callback
from ..exceptions import MeterNotFoundError


class PlotLogger(Callback, VisdomPlot):
    data_cache = None

    def __init__(self, mode, monitor, cache_size=None, *args, **kwargs):
        super(PlotLogger, self).__init__(*args, **kwargs)
        self.mode = mode
        self.monitor = monitor
        self.data_cache = defaultdict(list)
        if cache_size:
            self.cache_size = cache_size

    def on_train_start(self, trainer, state):
        if self.monitor not in state['meters']:
            msg = 'Meter <{}> not found'.format(self.monitor)
            raise MeterNotFoundError(msg)

    def on_terminated(self, trainer, state):
        super(PlotLogger, self)._teardown()

    def send_to_cache(self, x, y):
        self.data_cache['x'].append(x)
        self.data_cache['y'].append(y)
        if len(self.data_cache['x']) == self.cache_size:
            x = copy(self.data_cache['x'])
            y = copy(self.data_cache['y'])
            self.data_cache.clear()
            self.log(x, y)


class EpochPlotLogger(PlotLogger):
    cache_size = 1

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value
        self.send_to_cache(state['epochs'], meter_value)


class BatchPlotLogger(PlotLogger):
    cache_size = 100

    def on_batch_end(self, trainer, state):
        if state['mode'] != self.mode:  # Only Train or Test allowed.
            return
        iters = state['iters']
        meter_value = state['meters'][self.monitor].value
        self.send_to_cache(iters, meter_value)
