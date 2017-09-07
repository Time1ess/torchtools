#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:23
# Last modified: 2017-09-07 16:29
# Filename: callback.py
# Description:


class Hook(object):
    """
    Abstract class.
    """
    def __init__(self, *args, **kwargs):
        super(Hook, self).__init__(*args, **kwargs)

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

    def on_validate_start(self, trainer, state):
        pass

    def on_validate_end(self, trainer, state):
        pass

    def on_terminated(self, trainer, state):
        pass

    def has_hook_conflict(self, trainer):
        pass

    def _teardown(self):
        super(Hook, self)._teardown()

    def __str__(self):
        return type(self).__name__


class Callback(Hook):
    def __init__(self, *args, **kwargs):
        super(Callback, self).__init__(*args, **kwargs)
