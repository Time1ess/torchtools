# coding: UTF-8
import numpy as np

from torchtools.exceptions import EarlyStoppingException
from torchtools.callbacks.callback import Callback


class EarlyStopping(Callback):
    """Callback that stops training if monitor value has stopped improving."""
    def __init__(self, monitor='val_loss', mode='auto', patience=0):
        """Initialization for EarlyStopping.

        Parameters
        ----------
        monitor: str
            Value to be monitored, Default: 'val_loss'.
        mode: str
            One of 'max', 'min' and 'auto', default 'auto'. In 'max' mode,
            training will be stopped if monitor value stopped increasing. In
            'min' mode, training will be stopped if monitor value stopped
            decreasing. In 'auto' mode, the true mode will be decided by
            the name of monitor value, Default: 'auto'.
        patience: int
            Number of epochs with no improvement on monitor value which the
            training will be stopped, Default: 0.
        """
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
                raise EarlyStoppingException()
