#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-07 20:05
# Last modified: 2017-09-07 21:54
# Filename: tblogger.py
# Description:
from .callback import Callback
from tensorboard import SummaryWriter

from ..exceptions import LogTypeError
from ..meters import meter as _meter


class TensorBoardLogger(Callback):
    _tensorboard = None

    def __init__(self, directory='logs', ignores=None, *args, **kwargs):
        super(TensorBoardLogger, self).__init__(*args, **kwargs)
        if self._tensorboard is None:
            type(self)._tensorboard = SummaryWriter(directory)
        if ignores is None:
            ignores = []
        self.ignores = ignores

    def _teardown(self):
        self.board.close()

    @property
    def board(self):
        return self._tensorboard

    def log(self, log_type, *args, **kwargs):
        if not hasattr(self, 'log_'+log_type):
            raise LogTypeError('No such log type: {}'.format(log_type))
        method = getattr(self, 'log_'+log_type)
        method(*args, **kwargs)

    def log_image(self):
        pass

    def log_scalar(self, tag, x, y):
        self.board.add_scalar(tag, y, x)

    def log_graph(self):
        pass

    def log_hist(self):
        pass

    def log_text(self):
        pass

    def on_batch_end(self, trainer, state):
        iters = state['iters']
        for name, meter in state['meters'].items():
            if meter.reset_mode == _meter.BATCH_RESET and \
                    name not in self.ignores:
                self.log('scalar', name, iters, meter.value)

    def on_epoch_end(self, trainer, state):
        epochs = state['epochs']
        for name, meter in state['meters'].items():
            if meter.reset_mode == _meter.EPOCH_RESET and \
                    name not in self.ignores:
                self.log('scalar', name, epochs, meter.value)
