#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-10 14:59
# Last modified: 2017-09-11 14:31
# Filename: semanticmeter.py
# Description:
import pickle
import os
import os.path as osp
import inspect

import torch
import numpy as np

from PIL.ImagePalette import ImagePalette
from torchvision.utils import make_grid

from ..exceptions import MeterNoValueError
from .meter import BatchResetMixin
from .utils import build_ss_img_tensor
from .palette import palettes as default_palettes


class SemSegVisualizer(BatchResetMixin):
    """
    Create visualization for model outputs.
    """

    meter_type = 'image'

    def __init__(self, name, meter_mode, palette, fpi=1, *args, **kwargs):
        self.fpi = 0
        self.target_fpi = fpi
        self.step = 0
        self.palette = self._get_palette(palette)
        super(SemSegVisualizer, self).__init__(name, meter_mode, *args,
                                               **kwargs)

    @staticmethod
    def _get_palette(palette):
        if isinstance(palette, ImagePalette):
            return palette
        elif isinstance(palette, str) and palette in default_palettes:
            return pickle.loads(default_palettes[palette])
        elif isinstance(palette, str) and osp.exists(palette):
            return pickle.load(open(palette, 'rb'))

    def on_forward_end(self, trainer, state):
        if state['mode'] != self.meter_mode:
            return
        self.fpi += 1
        if self.fpi < self.target_fpi:
            return
        label_pred = state['output'].data.max(1)[1].cpu().numpy()  # N x H x W
        label_true = state['target'].cpu().numpy()  # N x H x W
        batch, height, width = label_true.shape
        imgs = []
        for index in range(batch):
            imgs.append(build_ss_img_tensor(label_pred[index], self.palette))
            imgs.append(build_ss_img_tensor(label_true[index], self.palette))
        img_grid = make_grid(imgs, padding=20)
        self.img = img_grid
        self.step += 1

    def on_batch_start(self, trainer, state):
        if self.fpi == self.target_fpi:
            self.fpi = 0

    @property
    def can_call(self):
        return self.fpi == self.target_fpi

    def reset(self):
        self.img = None

    @property
    def value(self):
        if self.img is None:
            raise MeterNoValueError()
        return self.img
