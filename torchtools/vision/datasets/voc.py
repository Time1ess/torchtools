#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-17 10:53
# Last modified: 2017-09-11 14:22
# Filename: voc.py
# Description:
import os.path as osp
import random

import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchtools.vision.transforms import PairRandomCrop, ReLabel, ToArray
from torchtools.vision.transforms import PairRandomHorizontalFlip, ToTensor


class VOCClassSegmentation(Dataset):
    mean_rgb = (122.67891434, 116.66876762, 104.00698793)
    mean_bgr = (104.00698793, 116.66876762, 122.67891434)
    mean_rgb_norm = tuple(v / 255.0 for v in mean_rgb)
    mean_bgr_norm = tuple(v / 255.0 for v in mean_bgr)

    std_rgb = (58.395, 57.12, 57.375)
    std_rgb_norm = tuple(v / 255.0 for v in std_rgb)

    def __init__(self, base_dir, phase, input_trans=None, target_trans=None,
                 pair_trans=None):
        self.base_dir = base_dir
        self.phase = phase
        phase_file = osp.join(base_dir, 'ImageSets',
                              'Segmentation', phase + '.txt')
        self.phase_list = []
        with open(phase_file, 'r') as f:
            for uid in f:
                self.phase_list.append(uid.strip('\n'))

        self.input_trans = input_trans
        self.target_trans = target_trans
        self.pair_trans = pair_trans

    def __getitem__(self, idx):
        uid = self.phase_list[idx]
        input_path = osp.join(self.base_dir, 'JPEGImages', uid + '.jpg')
        target_path = osp.join(self.base_dir, 'SegmentationClass',
                               uid + '.png')

        input = Image.open(input_path).convert('RGB')
        target = Image.open(target_path)

        if self.pair_trans:
            input = self.pair_trans(input)
            target = self.pair_trans(target)
        if self.input_trans:
            input = self.input_trans(input)
        if self.target_trans:
            target = self.target_trans(target)

        target = torch.from_numpy(target).long()

        return input, target

    def __len__(self):
        return len(self.phase_list)


def main():
    input_trans = T.Compose([
        ToTensor(True),
        T.Normalize(
            VOCClassSegmentation.mean_rgb_norm,
            VOCClassSegmentation.std_rgb_norm)])
    pair_trans = T.Compose([
        T.Scale(512),
        PairRandomCrop(512),
        PairRandomHorizontalFlip()])
    target_trans = T.Compose([
        ToArray(),
        ReLabel([k for k in range(0, 256)], 30)])
    voc = VOCClassSegmentation(
        '/share/datasets/VOCdevkit/VOC2012', 'train',
        input_trans=input_trans, target_trans=target_trans,
        pair_trans=pair_trans)
    print(len(voc))
    item = random.choice(voc)
    print(item[0].numpy().max())
    print(item[0].numpy().min())
    print(item[0].numpy().mean())
    print(item[0].numpy().std())
    print(item[1].numpy().max())


if __name__ == '__main__':
    main()
