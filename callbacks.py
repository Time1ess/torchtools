#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-09 10:57
# Last modified: 2017-08-14 11:05
# Filename: callbacks.py
# Description:
import math
import csv
import shutil
import os

from datetime import datetime

import torch

from .plots import ProcessVisdomPlot


def better_result(monitor, old_value, new_value):
    if (monitor == 'loss' or monitor == 'val_loss') and new_value < old_value:
        return True
    elif (monitor == 'acc' or monitor == 'val_acc') and new_value > old_value:
        return True
    else:
        return False


def better_result_thres(monitor, old_value, new_value, epsilon):
    if (monitor == 'loss' or monitor == 'val_loss'):
        new_value += epsilon
    elif (monitor == 'acc' or monitor == 'val_acc'):
        new_value -= epsilon
    return better_result(monitor, old_value, new_value)


def reset_best(monitor):
    if monitor == 'loss' or monitor == 'val_loss':
        return math.inf
    elif monitor == 'acc' or monitor == 'val_acc':
        return 0
    else:
        raise ValueError(
            'Monitor value {} is not supported'.format(monitor))


class Hook:
    """
    Abstract class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def on_terminated(self, trainer, state):
        pass

    def has_hook_conflict(self, trainer):
        pass

    def _teardown(self):
        super()._teardown()

    def __str__(self):
        return type(self).__name__


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
        self.best = reset_best(monitor)
        self.directory = directory
        self.monitor = monitor
        self.fname = fname
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value
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
                print()
                print('EarlyStopping!')
                trainer.exit()


class LRScheduler(Callback):
    @classmethod
    def set_lr(cls, init_lr):
        cls.init_lr = init_lr

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


class ReduceLROnPlateau(LRScheduler):
    def __init__(self, monitor='val_loss', factor=0.1, patience=10,
                 epsilon=0.0001, cooldown=0, min_lr=0):
        self.best = reset_best(monitor)
        super().__init__()
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


class CSVLogger(Callback):
    def __init__(self,
                 directory='logs',
                 fname='training_log',
                 ext='csv',
                 separator=',',
                 keys=None,
                 append=False):
        super().__init__()
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.fpath = os.path.join(directory, fname)+'.'+ext
        self.sep = separator
        self.writer = None
        if keys is None:
            self.keys = ['timestamp', 'epochs', 'val_loss']
        elif not (isinstance(keys, list) or isinstance(keys, str)):
            raise ValueError('keys {} is not supported'.format(repr(keys)))
        elif isinstance(keys, str):
            self.keys = [keys]
        else:
            self.keys = keys
        self.append = append
        self.append_header = True
        self.csv_file = None

    def on_train_start(self, trainer, state):
        if self.append:
            if os.path.exists(self.fpath):
                with open(self.fpath) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.fpath, 'a')
        else:
            self.csv_file = open(self.fpath, 'w')

        class CustomDialect(csv.excel):
            delimiter = self.sep

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.keys,
            dialect=CustomDialect)
        if self.append_header:
            self.writer.writeheader()


    def on_epoch_end(self, trainer, state):
        def handle_value(key):
            if key == 'timestamp':
                return datetime.now()
            elif key in state['meters']:
                return state['meters'][key].value
            elif key in state:
                return state[key]
            else:
                raise KeyError("Key {} not in state dict".format(key))

        row_dict = {key: handle_value(key) for key in self.keys}
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def _teardown(self):
        if self.csv_file:
            self.csv_file.close()
        self.writer = None

    def on_train_end(self, trainer, state):
        self._teardown()

    def on_terminated(self, trainer, state):
        self._teardown()


class PlotLogger(Callback, ProcessVisdomPlot):
    def __init__(self, mode, monitor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.monitor = monitor

    def on_terminated(self, trainer, state):
        super()._teardown()


class EpochPlotLogger(PlotLogger):

    def on_epoch_end(self, trainer, state):
        meter_value = state['meters'][self.monitor].value
        self.log(state['epochs'], meter_value)


class BatchPlotLogger(PlotLogger):
    def on_batch_end(self, trainer, state):
        if state['mode'] != self.mode:
            return
        iters = state['iters']
        meter_value = state['meters'][self.monitor].value
        self.log(iters, meter_value)
