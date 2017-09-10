#!/usr/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-09 12:15
# Last modified: 2017-09-10 16:54
# Filename: full_tests.py
# Description:
import shutil
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from torch.nn.init import xavier_uniform as xavier

from torchtools.trainer import ModelTrainer
from torchtools.callbacks import ModelCheckPoint, ExpLRScheduler, EarlyStopping
from torchtools.callbacks import CSVLogger, ReduceLROnPlateau
from torchtools.callbacks import BatchPlotLogger, EpochPlotLogger
from torchtools.meters import TimeMeter, EpochAverageMeter, BatchAverageMeter


LOG_DIRECTORY = 'tmp/log'
LOG_FILENAME = 'train_log'
CHECKPOINTS_DIRECTORY = 'tmp/checkpoints'
DATASET_DIRECTORY = '/share/datasets'
EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 1


def clean_up():
    shutil.rmtree(CHECKPOINTS_DIRECTORY, ignore_errors=True)
    try:
        os.remove(os.path.join(LOG_DIRECTORY, LOG_FILENAME))
    except OSError:
        pass


transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root=DATASET_DIRECTORY, train=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

valset = torchvision.datasets.CIFAR10(
    root=DATASET_DIRECTORY, train=False, transform=transform)

val_loader = torch.utils.data.DataLoader(
    valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
           'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier(m.weight.data)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 32 x 32 x 32
        x = F.relu(self.bn2(self.conv2(x)))  # 32 x 32 x 32
        x = self.pool(x)  # 16 x 16 x 32

        x = F.relu(self.bn3(self.conv3(x)))  # 16 x 16 x 64
        x = F.relu(self.bn4(self.conv4(x)))  # 16 x 16 x 64
        x = self.pool(x)  # 8 x 8 x 64

        x = F.relu(self.bn5(self.conv5(x)))  # 8 x 8 x 128
        x = F.relu(self.bn6(self.conv6(x)))  # 8 x 8 x 128
        x = self.pool(x)  # 4 x 4 x 128

        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


clean_up()

model = Net()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Meters
loss_meter = EpochAverageMeter('loss')
val_loss_meter = BatchAverageMeter('val_loss')
time_meter = TimeMeter('seconds_per_epoch')

# Callbacks
checkpoint = ModelCheckPoint(CHECKPOINTS_DIRECTORY, monitor='val_loss')
lr_scheduler = ExpLRScheduler(0.9)
early_stopping = EarlyStopping('val_loss', 5)
csv_logger = CSVLogger(directory=LOG_DIRECTORY, fname=LOG_FILENAME,
                       keys=['timestamp', 'epochs', 'iters', 'loss',
                             'val_loss', 'seconds_per_epoch'],
                       append=False)
lr_reduce = ReduceLROnPlateau(patience=0)
train_plot_logger = BatchPlotLogger(
    'train', 'loss', 10, 'line',
    opts={'title': 'Training Loss', 'xlabel': 'iterations', 'ylabel': 'Loss'})
val_plot_logger = EpochPlotLogger(
    'val', 'val_loss', 1, 'line',
    opts={'title': 'Test Loss', 'xlabel': 'epochs', 'ylabel': 'Loss'})
time_plot_logger = EpochPlotLogger(
    'train', 'seconds_per_epoch', 1, 'line',
    opts={'title': 'Epoch training Time',
          'xlabel': 'Epochs', 'ylabel': 'Seconds'})

trainer = ModelTrainer(model, train_loader, criterion, optimizer,
                       val_loader, use_cuda=True)

trainer.register_hooks([
    loss_meter, val_loss_meter, time_meter,
    lr_scheduler, checkpoint, early_stopping, csv_logger,
    train_plot_logger, val_plot_logger, time_plot_logger])


trainer.train(EPOCHS)
# trainer.test()
