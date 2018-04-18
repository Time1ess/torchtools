# coding: UTF-8
from time import time

from torchtools.meters import EpochMeter, SCALAR_METER


class TimeMeter(EpochMeter):
    """Meter that measures elapsed time for epoch."""
    meter_type = SCALAR_METER

    def on_epoch_start(self, trainer, state):
        self.tick = time()

    def on_epoch_end(self, trainer, state):
        self.tock = time()

    @property
    def value(self):
        return self.scaling * (self.tock - self.tick)
