#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:34
# Last modified: 2017-08-15 11:11
# Filename: reducelronplateau.py
# Description:
from .lrscheduler import LRScheduler
from .utils import reset_best, better_result_thres


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 epsilon=0.0001, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()
        self.best = reset_best(monitor)
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.init_patience = patience
        self.epsilon = epsilon
        self.cd = 0
        self.init_cd = cooldown
        self.min_lr = min_lr

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value
        if self.cd != 0:  # Wait for cooling down
            self.cd -= 1
        elif better_result_thres(self.monitor, self.best, meter_value,
                                 self.epsilon):  # New best
            self.best = meter_value
            self.patience = self.init_patience
        else:  # Not cooling down and not best
            self.patience -= 1
            if self.patience < 0:
                optimizer = state['optimizer']
                init_lr = []
                for idx, d in enumerate(optimizer.param_groups):
                    d['lr'] = max(self.init_lr[idx] * self.factor, self.min_lr)
                    init_lr.append(d['lr'])

                # Reset
                self.set_lr(init_lr)  # Reset base learning rate
                self.cd = self.init_cd
                self.patience = self.init_patience
                self.best = meter_value  # Better than this bad value?
