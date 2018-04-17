#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-07 21:24
# Last modified: 2017-09-11 14:41
# Filename: utils.py
# Description:
import numpy as np

from PIL import Image
from torchvision import transforms as T


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
    - overall accuracy
    - mean accuracy
    - mean IU
    - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def build_ss_img_tensor(result, palette):
    """
    Build a Semantic result image from output with palette.

    Parameters:
        * result(torch.Tensor): H x W, pixel classification result
        * palette(PIL.ImagePalette): Palette

    Return:
        * img(torch.Tensor): 3 x H x W
    """

    img = Image.fromarray(np.uint8(result), mode='P')
    img.putpalette(palette)
    img = img.convert()
    return T.ToTensor()(img)
