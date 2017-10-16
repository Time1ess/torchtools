#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-07 21:22
# Last modified: 2017-10-16 17:21
# Filename: ioumeter.py
# Description:
import numpy as np

from .meter import AverageMeter, EpochAverageMeter, BatchAverageMeter
from .meter import FixSizeAverageMeter, SCALAR_METER

from .utils import fast_hist


class IoUMeter(AverageMeter):
    meter_type = SCALAR_METER

    def __init__(self, name, meter_mode, num_classes, *args, **kwargs):
        self.num_classes = num_classes
        super(IoUMeter, self).__init__(name, meter_mode, *args, **kwargs)

    def on_forward_end(self, trainer, state):
        if state['mode'] != self.meter_mode:
            return
        output = state['output']
        prediction = output.data.max(1)[1].cpu().numpy()[:, :, :]
        ground_truth = state['target'].cpu().numpy()  # N x H x W
        val = 0
        n_class = self.num_classes
        for p, g in zip(prediction, ground_truth):  # For each sample, H x W
            hist = fast_hist(g.flatten(), p.flatten(), n_class)
            dom = hist.sum(axis=1) + hist.sum(axis=0) - \
                np.diag(hist).astype(float)
            dom[dom == 0] = np.nan
            iu = np.diag(hist) / dom
            iou = np.nanmean(iu)
            if np.isnan(iou):  # In case iou is nan, set iou to 0
                iou = 0
            val += iou
        self.add(val / len(ground_truth))

    @property
    def value(self):
        value = super(IoUMeter, self).value
        return value


class EpochIoUMeter(EpochAverageMeter, IoUMeter):
    pass


class BatchIoUMeter(BatchAverageMeter, IoUMeter):
    pass


class FixSizeIoUMeter(FixSizeAverageMeter, IoUMeter):
    pass
