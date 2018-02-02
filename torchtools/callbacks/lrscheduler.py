#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:27
# Last modified: 2017-10-19 15:03
# Filename: lrscheduler.py
# Description:
from .callback import Callback
from .utils import reset_best, better_result_thres


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


class PolyLRScheduler(LRScheduler):
    def __init__(self, power, *args, **kwargs):
        super(PolyLRScheduler, self).__init__(*args, **kwargs)
        self.power = power

    def _decay(self, state, current):
        optimizer = state['optimizer']
        for idx, d in enumerate(optimizer.param_groups):
            d['lr'] = self.init_lr[idx] * \
                (1 - 1.0 * current / self.maximum) ** self.power


class BatchDecayMixin(object):
    def on_train_start(self, trainer, state):
        super(BatchDecayMixin, self).on_train_start(trainer, state)
        self.maximum = state['max_epoch'] * len(trainer.train_data_loader)

    def on_batch_end(self, trainer, state):
        if state['mode'] != 'train':
            return
        current = state['iters']
        self._decay(state, current)


class EpochDecayMixin(object):
    def on_train_start(self, trainer, state):
        super(EpochDecayMixin, self).on_train_start(trainer, state)
        self.maximum = state['max_epoch']

    def on_epoch_end(self, trainer, state):
        if state['mode'] != 'train':
            return
        current = state['epochs']
        self._decay(state, current)


class BatchPolyLRScheduler(BatchDecayMixin, PolyLRScheduler):
    pass


class EpochPolyLRScheduler(EpochDecayMixin, PolyLRScheduler):
    pass


class ExpLRScheduler(LRScheduler):
    def __init__(self, power):
        super(ExpLRScheduler, self).__init__()
        self.power = power

    def _decay(self, state, current):
        optimizer = state['optimizer']
        for idx, d in enumerate(optimizer.param_groups):
            d['lr'] = self.init_lr[idx] * self.power ** current


class BatchExpLRScheduler(BatchDecayMixin, ExpLRScheduler):
    pass


class EpochExpLRScheduler(EpochDecayMixin, ExpLRScheduler):
    pass


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
                for idx, d in enumerate(optimizer.param_groups):
                    d['lr'] = max(d['lr'] * self.factor, self.min_lr)

                # Reset
                self.cd = self.init_cd
                self.patience = self.init_patience
                self.best = meter_value  # Better than this bad value?
