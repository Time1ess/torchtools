# coding: UTF-8
"""Learning rate scheduler"""
from torch.optim import lr_scheduler as lrs

from torchtools.exceptions import CallbackCheckError
from torchtools.callbacks.callback import Callback


class LRScheduler(Callback):
    def _callback_check(self, trainer):
        for cb in trainer.callbacks:
            if isinstance(cb, LRScheduler):
                msg = 'Only one learning rate scheduler should be used'
                raise CallbackCheckError(msg)

    def step(self, *args, **kwargs):
        pass

    def on_epoch_start(self, trainer, state):
        self.step(trainer, state)


class _StepMixin(object):  # For PyTorch schedulers
    def __init__(self, *args, **kwargs):
        super(_StepMixin, self).__init__(*args, **kwargs)
        self._scheduler = None

    def step(self, *args, **kwargs):
        self._scheduler.step()


class LambdaLR(_StepMixin, LRScheduler):
    def __init__(self, optimizer, lr_lambda, *args, **kwargs):
        super(LambdaLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.LambdaLR(optimizer, lr_lambda)


class StepLR(_StepMixin, LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, *args, **kwargs):
        super(StepLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.StepLR(optimizer, step_size, gamma)


class MultiStepLR(_StepMixin, LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, *args, **kwargs):
        super(MultiStepLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.MultiStepLR(optimizer, milestones, gamma)


class ExponentialLR(_StepMixin, LRScheduler):
    def __init__(self, optimizer, gamma, *args, **kwargs):
        super(ExponentialLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.ExponentialLR(optimizer, gamma)


class ReduceLROnPlateau(Callback):
    def __init__(self, optimizer, monitor, mode='min', factor=0.1, patience=10,
                 threshold=0.0001, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, *args, **kwargs):
        super(ReduceLROnPlateau, self).__init__(*args, **kwargs)
        lr_kwargs = {
            'mode': mode,
            'factor': factor,
            'patience': patience,
            'threshold': threshold,
            'threshold_mode': threshold_mode,
            'cooldown': cooldown,
            'min_lr': min_lr,
            'eps': eps,
        }
        self._scheduler = lrs.ReduceLROnPlateau(optimizer, **lr_kwargs)
        self.monitor = monitor

    def on_epoch_end(self, trainer, state):
        self.step(trainer, state)

    def step(self, trainer, state):
        val = state['meters'][self.monitor].value
        self._scheduler.step(val)
