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
    """Callback that sets the learning rate with a function.

    Sets the learning rate of each parameter group to the initial lr times
    a given function.

    This callback is a wrapper for PyTorch lr_schedulers.
    """
    def __init__(self, optimizer, lr_lambda, *args, **kwargs):
        """Initialization for LambdaLR.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer for the training net.
        lr_lambda: function or list
            A function which computes a multiplicative factor given an integer
            parameter epoch, or a list of such functions, one for each group
            in optimizer.param_groups.
        """
        super(LambdaLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.LambdaLR(optimizer, lr_lambda)


class StepLR(_StepMixin, LRScheduler):
    """Callback that sets the learning rate with a decay rate.

    Sets the learning rate of each parameter group to the initial lr decayed
    by gamma every step_size epochs.

    This callback is a wrapper for PyTorch lr_schedulers.
    """
    def __init__(self, optimizer, step_size, gamma=0.1, *args, **kwargs):
        """Initialization for StepLR.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer for the training net.
        step_size: int
            Period of learning rate decay.
        gamma: float
            Multiplicative factor of learning rate decay. Default: 0.1.
        """
        super(StepLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.StepLR(optimizer, step_size, gamma)


class MultiStepLR(_StepMixin, LRScheduler):
    """Callback that sets the learning rate with epoch milestones.

    Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones.

    This callback is a wrapper for PyTorch lr_schedulers.
    """
    def __init__(self, optimizer, milestones, gamma=0.1, *args, **kwargs):
        """Initialization for MultiStepLR.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer for the training net.
        milestones: list
            List of epoch indices. Must be increasing.
        gamma: float
            Multiplicative factor of learning rate decay. Default: 0.1.
        """
        super(MultiStepLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.MultiStepLR(optimizer, milestones, gamma)


class ExponentialLR(_StepMixin, LRScheduler):
    """Callback that sets the learning rate with a decay rate.

    Set the learning rate of each parameter group to the initial lr decayed
    by gamma every epoch.

    This callback is a wrapper for PyTorch lr_schedulers.
    """
    def __init__(self, optimizer, gamma, *args, **kwargs):
        """Initialization for ExponentialLR.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer for the training net.
        gamma: float
            Multiplicative factor of learning rate decay. Default: 0.1.
        """
        super(ExponentialLR, self).__init__(*args, **kwargs)
        self._scheduler = lrs.ExponentialLR(optimizer, gamma)


class ReduceLROnPlateau(Callback):
    """Callback that reduces the learning rate if monitor value stop improving.

    Reduce learning rate when a metric has stopped improving. Models often
    benefit from reducing the learning rate by a factor of 2-10 once learning
    stagnates. This callback reads a monitor value and if no improvement
    is seen for a `patience` number of epochs, the learning rate is reduced.

    This callback is a wrapper for PyTorch lr_schedulers.
    """
    def __init__(self, optimizer, monitor, mode='min', factor=0.1, patience=10,
                 threshold=0.0001, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, *args, **kwargs):
        """Initialization for ReduceLROnPlateau.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer for the training net.
        monitor: str
            Value to be monitored. Default: `val_loss`.
        mode: str
            One of `max`, `min`. For `acc` and `val_acc`,
            mode should be `max`, for `loss` and `val_loss`, mode should be
            `min`. Default: 'min'
        factor: float
            Factor by which the learning rate will be reduced.
            new_lr = lr * factor. Default: 0.1.
        patience: int
            Number of epochs with no improvement after which learning rate
            will be reduced. Default: 10.
        threshold: float
            Threshold for measuring the new optimum, to only focus on
            significant changes. Default: 1e-4.
        threshold_mode: str
            One of 'rel', 'abs'. In 'rel' mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max' mode or
            best * ( 1 - threshold ) in 'min' mode. In abs mode,
            dynamic_threshold = best + threshold in max mode or
            best - threshold in min mode. Default: 'rel',
        cooldown: int
            Number of epochs to wait before resuming normal operation after
            lr has been reduced. Default: 0.
        min_lr: float or list
             A scalar or a list of scalars. A lower bound on the learning rate
             of all param groups or each group respectively. Default: 0.
        eps: float
             Minimal decay applied to lr. If the difference between new and
             old lr is smaller than eps, the update is ignored. Default: 1e-8.
        """
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
