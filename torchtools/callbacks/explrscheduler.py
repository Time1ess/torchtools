#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:33
# Last modified: 2017-08-14 21:37
# Filename: explrscheduler.py
# Description:
from .lrscheduler import LRScheduler


class ExpLRScheduler(LRScheduler):
    def __init__(self, power):
        super().__init__()
        self.power = power

    def on_train_start(self, trainer, state):
        super().on_train_start(trainer, state)
        self.max_iters = len(trainer.train_data_loader) * state['max_epoch']

    def on_batch_end(self, trainer, state):
        if state['mode'] == 'test':
            return
        iters = state['iters']
        optimizer = state['optimizer']
        for idx, d in enumerate(optimizer.param_groups):
            d['lr'] = self.init_lr[idx] * \
                (1 - 1.0 * iters / self.max_iters) ** self.power
