# coding: UTF-8
import unittest

import torch
import numpy as np

from PIL import Image

from torchtools.vision.transforms import PairRandomCrop, ReLabel, ToArray
from torchtools.vision.transforms import PairRandomHorizontalFlip
from torchtools.vision.transforms import ToTensor, Transpose, IndexSwap


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


class TestToTensor(unittest.TestCase):
    def test_no_rescale(self):
        x = np.arange(0, 64 * 3).reshape(8, 8, 3)
        tensor = ToTensor(False)(x)
        self.assertAlmostEqual(tensor.max(), 64 * 3 - 1)

    def test_rescale(self):
        x = np.arange(0, 64 * 3).reshape(8, 8, 3)
        tensor = ToTensor()(x)
        self.assertAlmostEqual(tensor.max(), (64 * 3 - 1) / 255.0)


class TestTranspose(unittest.TestCase):
    def test_trans(self):
        x = np.arange(0, 64 * 3).reshape(8, 8, 3)
        transposed_x = np.transpose(x, (2, 0, 1))
        tensor = Transpose([(0, 2), (1, 2)])(torch.from_numpy(x))
        self.assertEqual(tensor.size(), transposed_x.shape)


class TestIndexSwap(unittest.TestCase):
    def test_swap(self):
        x = np.arange(0, 64 * 3).reshape(8, 8, 3)
        swap_x_tensor = torch.from_numpy(x[:, :, ::-1].copy())
        tensor = IndexSwap(2, [2, 1, 0])(torch.from_numpy(x))
        self.assertTrue(torch.equal(swap_x_tensor, tensor))


if __name__ == '__main__':
    unittest.main()
