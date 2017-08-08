#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-08 20:06
# Last modified: 2017-08-08 20:41
# Filename: fake.py
# Description:
from time import sleep
from random import random


class FakeNetwork:
    def __init__(self, speed=0.01):
        self.speed = speed
    def __call__(self, *arg, **kwargs):
        sleep(self.speed)
        return None


class FakeDataLoader:
    def __init__(self, size=100):
        self.fake_data = [(i, i+1) for i in range(size)]

    def __iter__(self):
        return iter(self.fake_data)

    def __len__(self):
        return len(self.fake_data)


class FakeOptimizer:
    def zero_grad(self):
        pass

    def step(self, closure):
        return closure()


class FakeLoss:
    def backward(self):
        pass

    def __str__(self):
        return '{:.2f}'.format(random() * 10)


class FakeHook:
    def __init__(self, name):
        self.name = name

    def __call__(self, state):
        print('>'*10)
        print(self.name)
        print('<'*10)


def FakeCreteria(output, target):
    return FakeLoss()
