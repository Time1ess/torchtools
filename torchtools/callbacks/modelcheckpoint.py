#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:25
# Last modified: 2017-08-14 21:26
# Filename: modelcheckpoint.py
# Description:
import os
import shutil

import torch

from .callback import Callback
from .utils import reset_best, better_result


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
