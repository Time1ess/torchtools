# coding: UTF-8
import numpy as np

from torchtools import TRAIN_MODE, VALIDATE_MODE, TEST_MODE
from torchtools.callbacks import Hook
from torchtools.meters import NO_RESET, NONE_METER, EPOCH_RESET, SCALAR_METER


class Meter(Hook):
    reset_mode = NO_RESET
    meter_type = NONE_METER
    mode = None

    def __init__(self, name, alias=None, scaling=1, *args, **kwargs):
        super(Meter, self).__init__(*args, **kwargs)
        self._check_and_set_mode(name)
        self.name = name
        self.alias = name if alias is None else alias
        self.scaling = scaling
        self.reset()

    def _check_and_set_mode(self, name):
        if name.startswith('val'):
            self.mode = VALIDATE_MODE
        elif name.startswith('test'):
            self.mode = TEST_MODE
        else:
            self.mode = TRAIN_MODE

    def reset(self):
        pass

    @property
    def value(self):
        pass

    def add(self):
        pass


class EpochResetMixin(object):
    reset_mode = EPOCH_RESET

    def on_epoch_start(self, trainer, state):
        self.reset()


class EpochMeter(EpochResetMixin, Meter):
    pass


class AverageMeter(EpochMeter):
    meter_type = SCALAR_METER

    def __init__(self, *args, **kwargs):
        self.values = []
        super(AverageMeter, self).__init__(*args, **kwargs)

    def reset(self):
        self.values = []

    def add(self, value):
        self.values.append(value)

    def calculate(self):
        if not self.values:
            mean, std = np.nan, np.nan
        else:
            mean, std = np.nanmean(self.values), np.nanstd(self.values)
        self.mean = mean
        self.std = std

    @property
    def value(self):
        self.calculate()
        return self.mean * self.scaling
