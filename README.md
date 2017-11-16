# torchtools: A High-Level training API on top of PyTorch

[![Build Status](https://travis-ci.org/Time1ess/torchtools.svg?branch=master)](https://travis-ci.org/Time1ess/torchtools)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/Time1ess/torchtools/blob/master/LICENSE)

## Easy Training, Easy DL.

torchtools is a High-Level training API on top of [PyTorch](http://pytorch.org) with many useful features to simplifiy the traing process for users.
It was developed with some ideas from [tnt](https://github.com/pytorch/tnt), [Keras](https://github.com/fchollet/keras), and it was designed to enhance the abilities of PyTorch.

This API provides these follows:

* A high-level training tool named `ModelTrainer`.
* A bunch of `callbacks` to inject your code in any stages during the training.
* A set of `meters` to get the performance of your model.
* More `transforms` to PyTorch(such as RandomCrop for Semantic Segmentation).
* (**New**) TensorBoard support added.

torchtools supports **Python 2.7+**, **Python 3.5+**.

***Important***: This README may be overdated, please refer to source codes to get more intuition.

## Show

Training Process:

![](training_process.gif)

Traing Plot:

![](training_plot.gif)

## 1 Minute torchtools example

### Regular PyTorch setups(transforms, dataloader, model, optimizer, loss)

```Python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from torch.nn.init import xavier_uniform as xavier

from torchtools.trainer import ModelTrainer
from torchtools.callbacks import ModelCheckPoint, ExpLRScheduler, EarlyStopping
from torchtools.callbacks import CSVLogger, BatchPlotLogger, EpochPlotLogger
from torchtools.meters import TimeMeter, EpochAverageMeter, BatchAverageMeter


LOG_DIRECTORY = 'tmp/log'
LOG_FILENAME = 'train_log'
CHECKPOINTS_DIRECTORY = 'tmp/checkpoints'
DATASET_DIRECTORY = '/share/datasets'
EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 1


transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root=DATASET_DIRECTORY, train=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.CIFAR10(
    root=DATASET_DIRECTORY, train=False, transform=transform)

test_loader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 10)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier(m.weight.data)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  
        x = F.relu(self.bn2(self.conv2(x)))  
        x = self.pool(x)  

        x = F.relu(self.bn3(self.conv3(x)))  
        x = F.relu(self.bn4(self.conv4(x)))  
        x = self.pool(x)  

        x = F.relu(self.bn5(self.conv5(x))) 
        x = F.relu(self.bn6(self.conv6(x))) 
        x = self.pool(x)  

        x = x.view(-1, 128*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Net()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
criterion = nn.CrossEntropyLoss()
```

### Meters

`meters` are provided to measure `loss`, `accuracy`, `time` in different ways.

```Python
loss_meter = EpochAverageMeter('loss')
val_loss_meter = BatchAverageMeter('val_loss')
time_meter = TimeMeter('seconds_per_epoch')
```

### Callbacks

`callbacks` provides samilar API compared with [Keras](https://github.com/fchollet/keras).

```Python
checkpoint = ModelCheckPoint(CHECKPOINTS_DIRECTORY, monitor='val_loss')

lr_scheduler = ExpLRScheduler(0.9)

early_stopping = EarlyStopping('val_loss', 5)

csv_logger = CSVLogger(
	directory=LOG_DIRECTORY, fname=LOG_FILENAME,
	keys=['timestamp', 'epochs', 'iters', 'loss', 'val_loss', 'seconds_per_epoch'],
	append=False)
	
train_plot_logger = BatchPlotLogger(
    'train', 'loss', 'line',
    opts={'title': 'Training Loss', 'xlabel': 'iterations', 'ylabel': 'Loss'})
                                   
test_plot_logger = EpochPlotLogger(
    'test', 'val_loss', 'line',
    opts={'title': 'Test Loss', 'xlabel': 'epochs', 'ylabel': 'Loss'})
    
time_plot_logger = EpochPlotLogger(
    'train', 'seconds_per_epoch', 'line',
    opts={'title': 'Epoch training Time', 'xlabel': 'Epochs', 'ylabel': 'Seconds'})
```

### Put together

Now, we can put it together.

1. Create a trainer with previous variables.
2. All `callbacks` and `meters` are actually `Hook` objects, so we can use `register_hooks` to register these hooks to `ModelTrainer`.
3. Call `train` with training epoch num.
4. Done!

```Python
trainer = ModelTrainer(model, train_loader, criterion, optimizer,
                       test_loader, use_cuda=True)

trainer.register_hooks([
    loss_meter, val_loss_meter, time_meter,
    lr_scheduler, checkpoint, early_stopping, csv_logger,
    train_plot_logger, test_plot_logger, time_plot_logger])

trainer.train(EPOCHS)
```

## More to go

### callbacks

### meters


## Contribute

If there are any bugs or feature requests please [submit an issue](https://github.com/Time1ess/torchtools/issues/new), I'll see I can do.

Any new features or bug fixes please submit a PR in [Pull requests](https://github.com/Time1ess/torchtools/pulls).

If there are any other problems, please email: <a href="mailto:youche.du@gmail.com">youchen.du@gmail.com</a>

## Acknowledgement

Thanks to these people and groups:

* All PyTorch developers
* All PyTorchNet developers
* All Keras developers
