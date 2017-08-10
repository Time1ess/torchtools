#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-10 13:56
# Last modified: 2017-08-10 14:26
# Filename: exceptions.py
# Description:
class TorchToolsException(Exception):
    pass


class EarlyStoppingError(TorchToolsException):
    pass


class HookTypeError(TorchToolsException):
    pass
