#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 14:21
# Last modified: 2017-09-10 16:49
# Filename: test_meters.py
# Description:
import time
import math
import unittest

import numpy as np
import torch

from torch.autograd import Variable
from torchtools.meters import AverageMeter, TimeMeter, IoUMeter
from torchtools.meters import EpochAverageMeter, BatchAverageMeter
from torchtools.meters import SemSegVisualizer
from torchtools.exceptions import MeterNoValueError

from helpers import ValueObject


class TestAverageMeter(unittest.TestCase):
    def setUp(self):
        self.meter = AverageMeter('loss')

    def test_add(self):
        meter = self.meter
        self.assertIs(meter.value, np.nan)

        trainer = None
        state = {}
        state['loss'] = ValueObject(10)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 10)

        state['loss'] = ValueObject(5)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 7.5)


class TestEpochAverageMeter(unittest.TestCase):
    def setUp(self):
        self.meter = EpochAverageMeter('loss')

    def test_add(self):
        meter = self.meter
        self.assertIs(meter.value, np.nan)

        trainer = None
        state = {}
        state['loss'] = ValueObject(10)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 10)

        meter.on_epoch_start(trainer, state)

        state['loss'] = ValueObject(5)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 5)


class TestBatchAverageMeter(unittest.TestCase):
    def setUp(self):
        self.meter = BatchAverageMeter('loss')

    def test_add(self):
        meter = self.meter
        self.assertIs(meter.value, np.nan)

        trainer = None
        state = {}
        state['loss'] = ValueObject(10)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 10)

        meter.on_batch_start(trainer, state)

        state['loss'] = ValueObject(5)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 5)


class TestTimeMeter(unittest.TestCase):
    def setUp(self):
        self.meter = TimeMeter('time')

    def test_tick(self):
        trainer = None
        state = None
        self.assertIs(self.meter.on_epoch_start(trainer, state), None)

    def test_tock(self):
        trainer = None
        state = None
        self.assertIs(self.meter.on_epoch_end(trainer, state), None)

    def test_time(self):
        trainer = None
        state = None
        self.meter.on_epoch_start(trainer, state)
        time.sleep(0.1)
        self.meter.on_epoch_end(trainer, state)
        self.assertAlmostEqual(self.meter.value, 0.1, 2)


class TestIoUMeter(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.m = 3
        self.h = self.w = 10
        self.meter = IoUMeter('validate', self.num_classes, name='IoU')

    def test_iou(self):
        num_classes, m, h, w = self.num_classes, self.m, self.h, self.w
        state = {}
        trainer = None
        state['output'] = Variable(torch.zeros(m, num_classes, h, w))
        state['target'] = torch.from_numpy(np.zeros((m, h, w)).astype(int))
        state['mode'] = 'validate'
        self.meter.on_forward_end(trainer, state)
        self.assertEqual(self.meter.value, 100.0)


class TestSemSegVisualizer(unittest.TestCase):
    def setUp(self):
        self.m = 3
        self.h = self.w = 10
        self.meter = SemSegVisualizer('validate', 'voc', 2, 'seg_visual')

    def test_visual(self):
        m, h, w = self.m, self.h, self.w
        meter = self.meter
        state = {}
        trainer = None
        state['output'] = Variable(torch.zeros(m, 3, h, w).long())
        state['target'] = torch.from_numpy(np.zeros((m, h, w)).astype(int))
        state['mode'] = 'validate'
        meter.on_forward_end(trainer, state)
        with self.assertRaises(MeterNoValueError):
            meter.value
        self.assertEqual(meter.fpi, 1)
        self.assertEqual(meter.step, 0)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.fpi, 0)
        self.assertEqual(meter.step, 1)
        self.assertEqual(meter.value.dim(), 3)


if __name__ == '__main__':
    unittest.main()
