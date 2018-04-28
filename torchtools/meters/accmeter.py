# coding: UTF-8
from torchtools.meters import EpochMeter, SCALAR_METER


class AccuracyMeter(EpochMeter):
    """Meter that measures average accuracy for epoch."""
    meter_type = SCALAR_METER

    def __init__(self, name, *args, **kwargs):
        super(AccuracyMeter, self).__init__(name, *args, **kwargs)
        self.total_cnt = 0
        self.correct_cnt = 0

    def on_forward_end(self, trainer, state):
        if state['mode'] != self.mode:
            return
        pred = state['output'].max(1)[1].data.cpu()
        target = state['target']
        if len(target.size()) != 1:
            target = target.max(1)[1].data.cpu()
        self.total_cnt += pred.size()[0]
        self.correct_cnt += target.eq(pred).sum().item()

    def reset(self):
        self.total_cnt = 0
        self.correct_cnt = 0

    @property
    def value(self):
        if self.total_cnt == 0:
            return 0
        return 1. * self.scaling * self.correct_cnt / self.total_cnt


class ErrorMeter(AccuracyMeter):
    """Meter that measures average error rate for epoch."""
    @property
    def value(self):
        return self.scaling * (1. - 1. * self.correct_cnt / self.total_cnt)
