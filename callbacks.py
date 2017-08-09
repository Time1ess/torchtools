#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-09 10:57
# Last modified: 2017-08-09 21:57
# Filename: callbacks.py
# Description:
import math
import shutil
import os

import torch


class Callback:
    """
    Abstract class.
    """
    def __init__(self):
        pass

    def on_train_start(self, state):
        pass

    def on_train_end(self, state):
        pass

    def on_epoch_start(self, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_batch_start(self, state):
        pass

    def on_batch_end(self, state):
        pass

    def on_forward_end(self, state):
        pass


class ModelCheckPoint(Callback):
    def __init__(self,
                 directory,
                 monitor='val_loss',
                 fname='checkpoint_{}_{:d}_{:d}_{:.2f}.pth.tar',
                 save_best_only=False,
                 save_weights_only=True):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.directory = directory
        self.monitor = monitor
        self.fname = fname
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        if monitor.endswith('loss'):
            self.best = math.inf
        else:
            self.best = 0

    def better_result(self, meter_value):
        if self.monitor.endswith('loss') and meter_value < self.best:
            return True
        elif self.monitor.endswith('acc') and meter_value > self.best:
            return True
        else:
            return False

    def save(self, state, meter_value):
        checkpoint = {
            'epochs': state['epochs'],
            'iters': state['iters'],
            'model_state_dict': state['model'].state_dict(),
            'optimizer_state_dict': state['optimizer'].state_dict(),
        }
        if not self.save_best_only:
            fname = os.path.join(self.directory, self.fname).format(
                state['arch'], state['epochs'], state['iters'],
                meter_value)
            torch.save(checkpoint, fname)

        if self.better_result(meter_value):
            self.best = meter_value
            best_fname = os.path.join(
                self.directory,
                'checkpoint_{}_best.pth.tar'.format(state['arch']))
            if not self.save_best_only:
                shutil.copy(fname, best_fname)
            else:
                torch.save(checkpoint, best_fname)


class EarlyStopping(Callback):
    pass


class LRScheduler(Callback):
    def __call__(self, state):
        pass


class ExpLRScheduler(LRScheduler):
    def __init__(self, optimizer, max_iters, power):
        self.init_lr = [d['lr'] for d in optimizer.param_groups]
        self.max_iters = max_iters
        self.power = power

    def __call__(self, state):
        if state['train'] is False:
            return
        iters = state['iters']
        optimizer = state['optimizer']
        for idx, d in enumerate(optimizer.param_groups):
            d['lr'] = self.init_lr[idx] * \
                (1 - 1.0 * iters / self.max_iters) ** self.power


class CSVLogger(Callback):
    pass
