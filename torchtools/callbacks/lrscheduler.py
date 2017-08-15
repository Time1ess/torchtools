#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:27
# Last modified: 2017-08-15 18:27
# Filename: lrscheduler.py
# Description:
from .callback import Callback


class LRScheduler(Callback):
    init_lr = None

    def set_lr(self, init_lr):
        self.init_lr = init_lr

    def on_train_start(self, trainer, state):
        init_lr = [d['lr'] for d in state['optimizer'].param_groups]
        self.set_lr(init_lr)

    def has_hook_conflict(self, trainer):
        for hook in trainer.callbacks:
            if isinstance(hook, LRScheduler):
                msg = ('Only one LRScheduler should be used, '
                       'Already has one: {}')
                self.conflict = msg.format(hook)
                return True
        return False
