#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 14:21
# Last modified: 2017-08-15 14:29
# Filename: test_meters.py
# Description:
import unittest

from torchtools.meters import AverageMeter, TimeMeter

from helpers import ValueObject


class TestAverageMeter(unittest.TestCase):
    def setUp(self):
        self.meter = AverageMeter('loss')


    def assertMeter(self, meter, n, sum, var):
        self.assertEqual(meter.n, n)
        self.assertEqual(meter.sum, sum)
        self.assertEqual(meter.var, var)

    def test_add(self):
        meter = self.meter
        self.assertMeter(meter, 0, 0, 0)

        trainer = None
        state = {}
        state['loss'] = ValueObject(10)

        meter.on_forward_end(trainer, state)
        self.assertMeter(meter, 1, 10, 100)
        self.assertEqual(meter.value, 10)


if __name__ == '__main__':
    unittest.main()
