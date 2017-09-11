#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 13:38
# Last modified: 2017-09-11 12:02
# Filename: helpers.py
# Description:
from random import randint


class FakeModel(object):
    _train = True

    def __init__(self, *args, **kwargs):
        self._state_dict = [randint(0, 100) for _ in range(10)]

    def train(self, train=True):
        self._train = train

    def __call__(self, input):
        return input

    def state_dict(self):
        return self._state_dict


class FakeDatasetLoader(object):
    def __len__(self):
        return 10


class FakeCriterion(object):
    pass


class FakeOptimizer(object):
    def __init__(self):
        self.param_groups = [{'lr': 1}, {'lr': 2}]
        self._state_dict = [randint(0, 100) for _ in range(10)]

    def state_dict(self):
        return self._state_dict


class FakeTrainer(object):
    def exit(self):
        return 0


class ValueObject(object):
    def __init__(self, value):
        self.value = value
        self.data = [value]
