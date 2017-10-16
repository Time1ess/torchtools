#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 14:21
# Last modified: 2017-10-16 18:39
# Filename: test_meters.py
# Description:
import time
import unittest

import numpy as np
import torch

from torch.autograd import Variable
from torchtools.meters import TimeMeter, IoUMeter, FixSizeIoUMeter
from torchtools.meters import BatchLossMeter, EpochLossMeter, FixSizeLossMeter
from torchtools.meters import SemSegVisualizer
from torchtools.meters import EpochAccuracyMeter, BatchAccuracyMeter
from torchtools.exceptions import MeterNoValueError

from helpers import ValueObject


class TestEpochLossMeter(unittest.TestCase):
    def setUp(self):
        self.meter = EpochLossMeter('loss', 'train', 'loss')

    def test_add(self):
        meter = self.meter
        self.assertTrue(np.isnan(meter.value))

        trainer = None
        state = {}
        state['mode'] = 'train'
        state['loss'] = ValueObject(10)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 10)

        meter.on_epoch_start(trainer, state)

        state['loss'] = ValueObject(5)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 5)


class TestFixSizeLossMeter(unittest.TestCase):
    def setUp(self):
        self.meter = FixSizeLossMeter('loss', 'train', 2)

    def test_add(self):
        meter = self.meter
        self.assertTrue(np.isnan(meter.value))

        trainer = None
        state = {}
        state['mode'] = 'train'
        state['loss'] = ValueObject(10)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 10)

        meter.on_epoch_start(trainer, state)

        state['loss'] = ValueObject(5)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 7.5)

        state['loss'] = ValueObject(8)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 6.5)

        state['loss'] = ValueObject(20)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 14)


class TestBatchLossMeter(unittest.TestCase):
    def setUp(self):
        self.meter = BatchLossMeter('loss', 'train', 'loss')

    def test_add(self):
        meter = self.meter
        self.assertTrue(np.isnan(meter.value))

        trainer = None
        state = {}
        state['mode'] = 'train'
        state['loss'] = ValueObject(10)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 10)

        meter.on_batch_start(trainer, state)

        state['loss'] = ValueObject(5)
        meter.on_forward_end(trainer, state)
        self.assertEqual(meter.value, 5)


class TestTimeMeter(unittest.TestCase):
    def setUp(self):
        self.meter = TimeMeter('time', 'train')

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
        self.meter = IoUMeter('IoU', 'validate', self.num_classes, 100)

    def test_iou(self):
        num_classes, m, h, w = self.num_classes, self.m, self.h, self.w
        state = {}
        trainer = None
        state['output'] = Variable(torch.zeros(m, num_classes, h, w))
        state['target'] = torch.from_numpy(np.zeros((m, h, w)).astype(int))
        state['mode'] = 'validate'
        self.meter.on_forward_end(trainer, state)
        self.assertEqual(self.meter.value, 100.0)


class TestFixSizeIoUMeter(unittest.TestCase):
    def setUp(self):
        self.num_classes = 10
        self.m = 3
        self.h = self.w = 10
        self.meter = FixSizeIoUMeter('IoU', 'validate', 3, self.num_classes, 100)

    def test_iou(self):
        num_classes, m, h, w = self.num_classes, self.m, self.h, self.w
        state = {}
        trainer = None
        state['output'] = Variable(torch.zeros(m, num_classes, h, w))
        ones_target = torch.from_numpy(np.ones((m, h, w)).astype(int))
        zeros_target = torch.from_numpy(np.zeros((m, h, w)).astype(int))

        state['target'] = ones_target
        state['mode'] = 'validate'
        self.meter.on_forward_end(trainer, state)
        self.assertEqual(self.meter.value, 0.0)

        state['target'] = zeros_target
        self.meter.on_forward_end(trainer, state)
        self.assertEqual(self.meter.value, 50.0)

        state['target'] = ones_target
        state['mode'] = 'validate'
        self.meter.on_forward_end(trainer, state)
        self.assertAlmostEqual(self.meter.value, 100/3.0)

        state['target'] = zeros_target
        self.meter.on_forward_end(trainer, state)
        self.assertAlmostEqual(self.meter.value, 100/3.0*2)


class TestSemSegVisualizer(unittest.TestCase):
    def setUp(self):
        self.m = 3
        self.h = self.w = 10
        self.meter = SemSegVisualizer('seg_visual', 'validate', 'voc', 2)

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
        self.assertEqual(meter.fpi, 2)
        self.assertEqual(meter.step, 1)
        self.assertEqual(meter.value.dim(), 3)


class TestEpochAccuracyMeter(unittest.TestCase):
    def setUp(self):
        self.meter = BatchAccuracyMeter('acc', 'train')

    def test_add(self):
        meter = self.meter
        self.assertEqual(meter.value, 0)

        trainer = None
        state = {}
        state['mode'] = 'train'
        self.assertAlmostEqual(meter.value, 0)

        meter.on_batch_start(trainer, state)
        state['output'] = Variable(torch.from_numpy(
            np.arange(16).reshape(4, 4)))
        state['target'] = torch.from_numpy(np.array([[3], [7], [11], [15]]))
        meter.on_forward_end(trainer, state)
        self.assertAlmostEqual(meter.value, 0.99999999)


if __name__ == '__main__':
    unittest.main()
