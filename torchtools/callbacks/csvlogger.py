#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:35
# Last modified: 2017-09-10 16:52
# Filename: csvlogger.py
# Description:
import os
import csv

from datetime import datetime

from .callback import Callback


class CSVLogger(Callback):
    def __init__(self,
                 directory='logs',
                 fname='training_log',
                 ext='csv',
                 separator=',',
                 keys=None,
                 append=False):
        super(CSVLogger, self).__init__()
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.fpath = os.path.join(directory, fname) + '.' + ext
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
