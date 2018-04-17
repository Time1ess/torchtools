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
    StepLR, ReduceLROnPlateau, TensorBoardLogger)


EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 1
DATASET_DIRECTORY = 'tmp/dataset'


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
optimizer = optim.Adam(model.parameters())
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

trainer.register_hooks([loss, val_loss, acc, val_acc,
                        scheduler, reduce_lr, logger])

trainer.train(EPOCHS)
