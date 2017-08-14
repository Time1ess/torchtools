#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:26
# Last modified: 2017-08-14 22:05
# Filename: earlystopping.py
# Description:
from .callback import Callback
from .utils import reset_best, better_result


class EarlyStopping(Callback):
    def __init__(self, monitor, patience):
        self.best = reset_best(monitor)
        self.monitor = monitor
        self.init_patience = patience
        self.patience = patience

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value
        if better_result(self.monitor, self.best, meter_value):
            self.best = meter_value
            self.patience = self.init_patience
        else:
            self.patience -= 1
            if self.patience == 0:
                print('\n')
                print('EarlyStopping!')
                trainer.exit()