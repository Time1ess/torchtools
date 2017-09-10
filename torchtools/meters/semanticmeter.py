#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-10 14:59
# Last modified: 2017-09-10 16:53
# Filename: semanticmeter.py
# Description:
import pickle
import os
import os.path as osp
import inspect

import torch

from PIL.ImagePalette import ImagePalette
from torchvision.utils import make_grid

from ..exceptions import MeterNoValueError
from .meter import BatchResetMeter
from .utils import build_ss_img_tensor
from .palette import palettes as default_palettes


class SemSegVisualizer(BatchResetMeter):
    """
    Create visualization for model outputs.
    """

    meter_type = 'image'

    def __init__(self, mode, palette, fpi=1, *args, **kwargs):
        super(SemSegVisualizer, self).__init__(*args, **kwargs)
        self.mode = mode
        self.palette = self._get_palette(palette)
        self.fpi = 0
        self.target_fpi = fpi
        self.step = 0
        self.reset()

    @staticmethod
    def _get_palette(palette):
        if isinstance(palette, ImagePalette):
            return palette
        elif isinstance(palette, str) and palette in default_palettes:
            return pickle.loads(default_palettes[palette])
        elif isinstance(palette, str) and osp.exists(palette):
            return pickle.load(open(palette, 'rb'))

    def on_forward_end(self, trainer, state):
        if state['mode'] != self.mode:
            return
        self.fpi += 1
        if self.fpi < self.target_fpi:
            return
        label_pred = state['output'].data.max(1)[1].cpu().numpy()  # N x H x W
        label_true = state['target'].cpu().numpy()  # N x H x W
        batch, height, width = label_true.shape
        img = torch.zeros((2 * batch, 3, height, width))
        for index in range(batch):
            img[index] = build_ss_img_tensor(label_pred[index], self.palette)
            img[batch + index] = build_ss_img_tensor(label_true[index],
                                                     self.palette)
        img_grid = make_grid(img, padding=20)
        self.img = img_grid
        self.fpi = 0  # Reset counter
        self.step += 1

    def reset(self):
        self.img = None

    @property
    def value(self):
        if self.img is None:
            raise MeterNoValueError()
        return self.img
