# coding: UTF-8
from random import randint

import torch
import torch.nn as nn



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(2, 2)
        self.l2 = nn.Linear(2, 2)

    def forward(self, x):
        return self.l2(self.l1(x))
