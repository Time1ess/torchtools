#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-08 19:25
# Last modified: 2017-10-16 19:27
# Filename: transforms.py
# Description:
import random
import os
import numbers
import sys

import numpy as np
import torch

from PIL import ImageOps, Image
from torchvision.transforms import ToTensor as _ToTensor

if sys.version_info.major == 2:
    from collections import Iterable
else:
    from typing import Iterable


class PairRandomCrop(object):
    """Crop the given PIL.Image at a random location.
    ** This is a MODIFIED version **, which supports identical random crop for
    both image and target map in Semantic Segmentation.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    image_crop_position = {}

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        pid = os.getpid()
        if pid in self.image_crop_position:
            x1, y1 = self.image_crop_position.pop(pid)
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            self.image_crop_position[pid] = (x1, y1)
        return img.crop((x1, y1, x1 + tw, y1 + th))


class PairRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability
    of 0.5, for both input and target image
    """

    def __init__(self):
        self.flip_set = False
        self.input_flip = False

    def __call__(self, img):
        if self.flip_set:
            self.flip_set = False
            if self.input_flip:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                return img
        else:
            self.flip_set = True
            if random.random() < 0.5:
                self.input_flip = True
                return img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                self.input_flip = False
                return img


class ReLabel(object):
    """ReLabel image from src to dst"""

    def __init__(self, src, dst):
        if not isinstance(src, Iterable):
            src = [src]
        if not isinstance(dst, Iterable):
            dst = [dst]
            if len(dst) == 1:
                dst = dst * len(src)
        assert len(dst) == len(src)
        self.src = src
        self.dst = dst

    def __call__(self, img):
        for s, d in zip(self.src, self.dst):
            img[img == s] = d
        return img


class ToArray(object):
    """Convert PIL Image to Numpy array"""
    def __call__(self, img):
        return np.array(img, dtype=np.uint8)


class NoReScaleToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    to a torch.FloatTensor of shape (C x H x W) .
    """

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backard compability
            return img.float()
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


def ToTensor(rescale=True):
    if rescale:
        return _ToTensor()
    else:
        return NoReScaleToTensor()


class Transpose(object):
    def __init__(self, dim_pairs):
        self.dim_pairs = dim_pairs

    def __call__(self, tensor):
        for dim0, dim1 in self.dim_pairs:
            tensor = torch.transpose(tensor, dim0, dim1)
        return tensor.contiguous()


class IndexSwap(object):
    def __init__(self, dim, new_idx):
        self.dim = dim
        self.new_idx = new_idx

    def __call__(self, tensor):
        new_idx = self.new_idx
        dim = self.dim
        index = torch.LongTensor(new_idx)
        return torch.index_select(tensor, dim, index)
