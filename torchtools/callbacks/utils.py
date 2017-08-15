#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-14 21:25
# Last modified: 2017-08-15 11:21
# Filename: utils.py
# Description:
import numpy as np


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
        return np.inf
    elif monitor == 'acc' or monitor == 'val_acc':
        return 0
    else:
        raise ValueError(
            'Monitor value {} is not supported'.format(monitor))
