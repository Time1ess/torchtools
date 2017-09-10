#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-07 17:07
# Last modified: 2017-09-10 16:54
# Filename: test_transforms.py
# Description:
import unittest

import numpy as np

from PIL import Image

from torchtools.vision.transforms import PairRandomCrop, ReLabel, ToArray
from torchtools.vision.transforms import PairRandomHorizontalFlip


class TestPairRandomCrop(unittest.TestCase):
    def test_crop(self):
        x = Image.fromarray(np.arange(0, 64 * 3).reshape(8, 8, 3), 'RGB')
        y = Image.fromarray(np.arange(0, 64 * 3).reshape(8, 8, 3), 'RGB')
        crop = PairRandomCrop(4)
        crop_x = crop(x)
        crop_y = crop(y)
        self.assertEqual(crop_x, crop_y)


class TestPairRandomHorizontalFlip(unittest.TestCase):
    def test_flip(self):
        x = Image.fromarray(np.arange(0, 64 * 3).reshape(8, 8, 3), 'RGB')
        y = Image.fromarray(np.arange(0, 64 * 3).reshape(8, 8, 3), 'RGB')
        flip = PairRandomHorizontalFlip()
        for _ in range(10):
            flip_x = flip(x)
            flip_y = flip(y)
            self.assertEqual(flip_x, flip_y)
        flip_x = flip(x)
        flip.input_flip = not flip.input_flip
        flip_y = flip(y)
        self.assertNotEqual(flip_x, flip_y)


class TestReLabel(unittest.TestCase):
    def test_relabel(self):
        x = np.arange(0, 64 * 3).reshape(8, 8, 3)
        relabel = ReLabel([k for k in range(64, 64 * 3)], 0)
        relabel_x = relabel(x)
        self.assertEqual(63, x.max())

        relabel = ReLabel(5, 10)
        relabel_x = relabel(x)
        self.assertEqual(2, np.count_nonzero(relabel_x[relabel_x == 10]))


class TestToArray(unittest.TestCase):
    def test_toarray(self):
        x = Image.fromarray(np.arange(0, 64 * 3).reshape(8, 8, 3), 'RGB')
        toarray = ToArray()
        array_x = toarray(x)
        self.assertIsInstance(array_x, np.ndarray)


if __name__ == '__main__':
    unittest.main()
