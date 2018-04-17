#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:23
# Last modified: 2017-10-17 21:19
# Filename: callback.py
# Description:


class Hook(object):
    """
    Abstract class.
    """
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

    def on_backward_end(self, trainer, state):
        pass

    def on_validate_start(self, trainer, state):
        pass

    def on_validate_end(self, trainer, state):
        pass

    def on_test_start(self, trainer, state):
        pass

    def on_test_end(self, trainer, state):
        pass

    def on_terminated(self, trainer, state):
        pass

    def __str__(self):
        return type(self).__name__


class Callback(Hook):
    def _callback_check(self, trainer):
        pass

    def _teardown(self):
        pass
