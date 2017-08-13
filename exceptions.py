#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-10 13:56
# Last modified: 2017-08-13 21:42
# Filename: exceptions.py
# Description:
class TorchToolsException(Exception):
    pass


class HookTypeError(TorchToolsException):
    pass


class HookCheckError(TorchToolsException):
    pass


class TrainerTerminated(TorchToolsException):
    pass
