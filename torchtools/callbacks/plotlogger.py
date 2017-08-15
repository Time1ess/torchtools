#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:36
# Last modified: 2017-08-15 11:40
# Filename: plotlogger.py
# Description:
from ..plots import ProcessVisdomPlot
from .callback import Callback


class PlotLogger(Callback, ProcessVisdomPlot):
    def __init__(self, mode, monitor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.monitor = monitor

    def on_terminated(self, trainer, state):
        super()._teardown()


class EpochPlotLogger(PlotLogger):
    cache_size = 1

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value
        self.log(state['epochs'], meter_value)


class BatchPlotLogger(PlotLogger):
    cache_size = 100

    def on_batch_end(self, trainer, state):
        if state['mode'] != self.mode:
            return
        iters = state['iters']
        meter_value = state['meters'][self.monitor].value
        self.log(iters, meter_value)
