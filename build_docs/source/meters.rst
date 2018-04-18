Meters
==========================

**meters** are provided to measure loss, accuracy, error rate, time in different ways.

::

    from torchtools.meters import LossMeter, AccuracyMeter
    
    loss_meter = LossMeter('loss')
    val_loss_meter = LossMeter('val_loss'))
    val_acc_meter = AccuracyMeter('val_acc')

    ...
    
    trainer.register_hooks([loss_meter, val_loss_meter, val_acc_meter])


Meter
--------------------------------

.. autoclass:: meters.meter.Meter
    :members:

EpochMeter
--------------------------------

.. autoclass:: meters.meter.EpochMeter
    :members:

AverageMeter
--------------------------------

.. autoclass:: meters.meter.AverageMeter
    :members:

AccuracyMeter
--------------------------------

.. autoclass:: meters.accmeter.AccuracyMeter
    :members:

ErrorMeter
--------------------------------

.. autoclass:: meters.accmeter.ErrorMeter
    :members:

LossMeter
--------------------------------

.. autoclass:: meters.lossmeter.LossMeter
    :members:

TimeMeter
--------------------------------

.. autoclass:: meters.timemeter.TimeMeter
    :members:
