# coding: UTF-8
import unittest

from random import randint

import numpy as np
import torch

from torch.autograd import Variable

from torchtools import TRAIN_MODE
from torchtools.meters import (
    AverageMeter, AccuracyMeter, ErrorMeter, LossMeter)


class TestAverageMeter(unittest.TestCase):
    def test_meter(self):
        meter = AverageMeter('loss')
        vals = [randint(0, 100) for _ in range(20)]
        for val in vals:
            meter.add(vals)
        self.assertAlmostEqual(np.nanmean(vals), meter.value)

        meter.reset()
        self.assertTrue(np.isnan(meter.value))


class TestAccuracyMeter(unittest.TestCase):
    def test_meter(self):
        meter = AccuracyMeter('acc', scaling=100)
        pred = np.array([
            [0.2, 0.5, 0.3, 0.1],
            [0.9, 0.2, 0.3, 0.2],
            [0.9, 0.4, 0.6, 0.2],
        ])
        target = np.array([
            [1],
            [0],
            [2],
        ])
        pred = Variable(torch.from_numpy(pred))
        target = torch.from_numpy(target)

        state = {}
        state['mode'] = TRAIN_MODE
        state['output'] = pred
        state['target'] = target
        meter.on_forward_end(None, state)
        self.assertAlmostEqual(meter.value, 100. * 2 / 3)


class TestErrorMeter(unittest.TestCase):
    def test_meter(self):
        meter = ErrorMeter('err', scaling=100)
        pred = np.array([
            [0.2, 0.5, 0.3, 0.1],
            [0.9, 0.2, 0.3, 0.2],
            [0.9, 0.4, 0.6, 0.2],
        ])
        target = np.array([
            [1],
            [0],
            [2],
        ])
        pred = Variable(torch.from_numpy(pred))
        target = torch.from_numpy(target)

        state = {}
        state['mode'] = TRAIN_MODE
        state['output'] = pred
        state['target'] = target
        meter.on_forward_end(None, state)
        self.assertAlmostEqual(meter.value, 100. * 1 / 3)


class TestLossMeter(unittest.TestCase):
    def test_meter(self):
        meter = LossMeter('loss')

        state = {}
        state['mode'] = TRAIN_MODE
        loss = randint(1, 100)
        state['loss'] = Variable(torch.FloatTensor([loss]))

        meter.on_forward_end(None, state)
        self.assertAlmostEqual(meter.value, loss)


if __name__ == '__main__':
    unittest.main()
