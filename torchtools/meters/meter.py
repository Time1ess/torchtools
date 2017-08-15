#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:17
# Last modified: 2017-08-15 11:13
# Filename: meter.py
# Description:
from ..callbacks import Hook


class Meter(Hook):
    def __init__(self, name, *args, **kwargs):
        super(Meter, self).__init__(*args, **kwargs)
        self.reset()
        self.name = name

    def reset(self):
        pass

    @property
    def value(self):
        pass

    def add(self):
        pass
