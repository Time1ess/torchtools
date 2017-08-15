#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 13:38
# Last modified: 2017-08-15 14:25
# Filename: helpers.py
# Description:
class FakeModel:
    _train = True

    def train(self, train=True):
        self._train = train

    def __call__(self, input):
        return input


class FakeDatasetLoader:
    def __len__(self):
        return 10


class FakeCriterion:
    pass


class FakeOptimizer:
    param_groups = [{'lr': 1}, {'lr': 2}]


class FakeTrainer:
    def exit(self):
        return 0


class ValueObject:
    def __init__(self, value):
        self.value = value
        self.data = [value]
