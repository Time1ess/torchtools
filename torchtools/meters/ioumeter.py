#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-07 21:22
# Last modified: 2017-09-07 21:29
# Filename: ioumeter.py
# Description:
import numpy as np

from .averagemeter import EpochAverageMeter

from .utils import fast_hist


class IoUMeter(EpochAverageMeter):
    def __init__(self, mode, num_classes, *args, **kwargs):
        super(IoUMeter, self).__init__(*args, **kwargs)
        self.mode = mode
        self.num_classes = num_classes
        self.reset()

    def on_forward_end(self, trainer, state):
        if state['mode'] != self.mode:
            return
        output = state['output']
        prediction = output.data.max(1)[1].cpu().numpy()[:, :, :]
        ground_truth = state['target'].cpu().numpy()
        val = 0
        n_class = self.num_classes
        for p, g in zip(prediction, ground_truth):
            hist = np.zeros((n_class, n_class))
            for lt, lp in zip(p, g):
                hist += fast_hist(lt.flatten(), lp.flatten(), n_class)
            dom = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            dom[dom == 0] = np.nan
            iu = np.diag(hist) / dom
            iou = np.nanmean(iu)
            val += iou
        self.add(val / len(ground_truth))

    @property
    def value(self):
        value = (IoUMeter, self).value
        return value * 100
