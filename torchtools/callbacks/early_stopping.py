# coding: UTF-8
import numpy as np

from torchtools.exceptions import EarlyStoppingError
from torchtools.callbacks.callback import Callback


class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', mode='auto', patience=0):
        self.monitor = monitor
        self.init_patience = patience
        self.patience = patience
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.inf
            else:
                self.monitor_op = np.less
                self.best = np.inf

    def on_epoch_end(self, trainer, state):
        val = state['meters'][self.monitor].value
        if self.monitor_op(val, self.best):
            self.best = val
            self.patience = self.init_patience
        else:
            self.patience -= 1
            if self.patience < 0:
                raise EarlyStoppingError()
