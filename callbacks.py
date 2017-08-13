#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-09 10:57
# Last modified: 2017-08-13 09:45
# Filename: callbacks.py
# Description:
import math
import csv
import shutil
import os

import torch


def better_result(monitor, old_value, new_value):
    if (monitor == 'loss' or monitor == 'val_loss') and new_value < old_value:
        return True
    elif (monitor == 'acc' or monitor == 'val_acc') and new_value > old_value:
        return True
    else:
        return False


class Hook:
    """
    Abstract class.
    """
    def __init__(self):
        pass

    def on_train_start(self, trainer, state):
        pass

    def on_train_end(self, trainer, state):
        pass

    def on_epoch_start(self, trainer, state):
        pass

    def on_epoch_end(self, trainer, state):
        pass

    def on_batch_start(self, trainer, state):
        pass

    def on_batch_end(self, trainer, state):
        pass

    def on_forward_end(self, trainer, state):
        pass

    def on_test_start(self, trainer, state):
        pass

    def on_test_end(self, trainer, state):
        pass

    def __str__(self):
        return self.__class__.__name__


class Callback(Hook):
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
        if monitor == 'loss' or monitor == 'val_loss':
            self.best = math.inf
        elif monitor == 'acc' or monitor == 'val_acc':
            self.best = 0
        else:
            raise ValueError(
                'Monitor value {} is not supported'.format(monitor))
        self.directory = directory
        self.monitor = monitor
        self.fname = fname
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value[0]
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

        if better_result(self.monitor, self.best, meter_value):
            self.best = meter_value
            best_fname = os.path.join(
                self.directory,
                'checkpoint_{}_best.pth.tar'.format(state['arch']))
            if not self.save_best_only:
                shutil.copy(fname, best_fname)
            else:
                torch.save(checkpoint, best_fname)


class EarlyStopping(Callback):
    def __init__(self, monitor, patience):
        if monitor == 'loss' or monitor == 'val_loss':
            self.best = math.inf
        elif monitor == 'acc' or monitor == 'val_acc':
            self.best = 0
        else:
            raise ValueError(
                'Monitor value {} is not supported'.format(monitor))
        self.monitor = monitor
        self.init_patience = patience
        self.patience = patience

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value[0]
        if better_result(self.monitor, self.best, meter_value):
            self.best = meter_value
            self.patience = self.init_patience
        else:
            self.patience -= 1
            if self.patience == 0:
                trainer.training_end = True


class LRScheduler(Callback):
    def on_train_start(self, trainer, state):
        self.init_lr = [d['lr'] for d in state['optimizer'].param_groups]


class ExpLRScheduler(LRScheduler):
    def __init__(self, max_iters, power):
        self.max_iters = max_iters
        self.power = power

    def on_batch_end(self, trainer, state):
        if state['train'] is False:
            return
        iters = state['iters']
        optimizer = state['optimizer']
        for idx, d in enumerate(optimizer.param_groups):
            d['lr'] = self.init_lr[idx] * \
                (1 - 1.0 * iters / self.max_iters) ** self.power


class CSVLogger(Callback):
    def __init__(self,
                 fname='training_log.csv',
                 directory='logs',
                 separator=',',
                 append=False):
        super().__init__()
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.fpath = os.path.join(directory, fname)
        self.sep = separator
        self.append = append


class BaseVisdomLogger(Callback):
    pass
