Home
===========

Introduction
------------

**torchtools** is a High-Level training API on top of PyTorch_ with many useful features to simplifiy the traing process for users.

.. _PyTorch: http://pytorch.org

It was developed based on ideas from tnt_, Keras_. I wrote this tool just want to release myself, since many different training tasks share same training routine(define dataset, retrieve a batch of samples, forward propagation, backward propagation, ...).

.. _tnt: https://github.com/pytorch/tnt
.. _Keras: https://github.com/fchollet/keras

Features
--------

This API provides these follows:

* A high-level training class named `ModelTrainer`. No need to repeat yourself.
* A bunch of useful `callbacks` to inject your code in any stages during the training.
* A set of `meters` to get the performance of your model.
* Visualization in TensorBoard support(TensorBoard required).

Requirements
------------

* tqdm
* Numpy
* PyTorch_
* tensorboardX_
* `Standalone TensorBoard`_ (Optional)

.. _tensorboardX: https://github.com/lanpa/tensorboard-pytorch
.. _`Standalone TensorBoard`: https://github.com/dmlc/tensorboard

Installation
------------

torchtools has been tested on **Python 2.7+**, **Python 3.5+**.

Windows is not supported since PyTorch is only supported on Linux and OSX.

::

    pip install torchtools


Screenshots
-----------

Training Process:

.. image:: _static/images/training_process.gif

Visualization in TensorBoard:

.. image:: _static/images/visualization_in_tensorboard.png

1 Minute torchtools MNIST example
---------------------------------
::

    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torchvision.transforms as T
    
    from torch.utils.data import DataLoader
    from torch.nn.init import xavier_uniform as xavier
    from torchvision.datasets import MNIST
    
    from torchtools.trainer import Trainer
    from torchtools.meters import LossMeter, AccuracyMeter
    from torchtools.callbacks import (
        StepLR, ReduceLROnPlateau, TensorBoardLogger, CSVLogger)
    
    
    EPOCHS = 10
    BATCH_SIZE = 32
    DATASET_DIRECTORY = 'dataset'
    
    trainset = MNIST(root=DATASET_DIRECTORY, transform=T.ToTensor())
    testset = MNIST(root=DATASET_DIRECTORY, train=False, transform=T.ToTensor())
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE)
    
    
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28 * 28, 100)
            self.fc2 = nn.Linear(100, 10)
    
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    xavier(m.weight.data)
    
        def forward(self, x):
            x = x.view(-1, 28 * 28)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, criterion, optimizer, test_loader)
    
    # Callbacks
    loss = LossMeter('loss')
    val_loss = LossMeter('val_loss')
    acc = AccuracyMeter('acc')
    val_acc = AccuracyMeter('val_acc')
    scheduler = StepLR(optimizer, 1, gamma=0.95)
    reduce_lr = ReduceLROnPlateau(optimizer, 'val_loss', factor=0.3, patience=3)
    logger = TensorBoardLogger()
    csv_logger = CSVLogger(keys=['epochs', 'loss', 'acc', 'val_loss', 'val_acc'])
    
    trainer.register_hooks([
        loss, val_loss, acc, val_acc, scheduler, reduce_lr, logger, csv_logger])
    
    trainer.train(EPOCHS)

