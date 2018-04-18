Callbacks
=============================

**callbacks** provides samilar API compared with Keras_. We can have more control on our training process through callbacks.

.. _Keras: https://github.com/fchollet/keras

::

    from torchtools.callbacks import StepLR, ReduceLROnPlateau, TensorBoardLogger
    
    scheduler = StepLR(optimizer, 1, gamma=0.95)
    reduce_lr = ReduceLROnPlateau(optimizer, 'val_loss', factor=0.3, patience=3)
    logger = TensorBoardLogger(comment=name)
    
    ...
    
    trainer.register_hooks([scheduler, reduce_lr, logger])

Hook
--------------------------------------

.. autoclass:: callbacks.callback.Hook
    :members:

Callback
--------------------------------------

.. autoclass:: callbacks.callback.Callback
    :members:

ModelCheckPoint
--------------------------------------

.. autoclass:: callbacks.checkpoint.ModelCheckPoint
    :members:

CSVLogger
--------------------------------------

.. autoclass:: callbacks.csvlogger.CSVLogger
    :members:

EarlyStopping
--------------------------------------

.. autoclass:: callbacks.early_stopping.EarlyStopping
    :members:

LambdaLR
--------------------------------------

.. autoclass:: callbacks.lr_scheduler.LambdaLR
    :members:

StepLR
--------------------------------------

.. autoclass:: callbacks.lr_scheduler.StepLR
    :members:

MultiStepLR
--------------------------------------

.. autoclass:: callbacks.lr_scheduler.MultiStepLR
    :members:

ExponentialLR
--------------------------------------

.. autoclass:: callbacks.lr_scheduler.ExponentialLR
    :members:

ReduceLROnPlateau
--------------------------------------

.. autoclass:: callbacks.lr_scheduler.ReduceLROnPlateau
    :members:

TensorBoardLogger
--------------------------------------

.. autoclass:: callbacks.tensorboard_logger.TensorBoardLogger
    :members:

