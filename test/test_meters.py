#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 14:21
# Last modified: 2017-09-07 16:56
# Filename: test_meters.py
# Description:
import time
import unittest
import numpy as np

from torchtools.meters import AverageMeter, TimeMeter
from torchtools.meters import EpochAverageMeter, BatchAverageMeter

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


if __name__ == '__main__':
    unittest.main()
