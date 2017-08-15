#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 12:28
# Last modified: 2017-08-15 19:38
# Filename: test_callbacks.py
# Description:
import os
import sys
import unittest
import tempfile
from random import randint

import torch

from torchtools.callbacks import EarlyStopping, CSVLogger
from torchtools.callbacks import LRScheduler, ExpLRScheduler, ReduceLROnPlateau
from torchtools.callbacks import ModelCheckPoint
from torchtools.callbacks import EpochPlotLogger, BatchPlotLogger

from helpers import FakeModel, FakeDatasetLoader
from helpers import FakeOptimizer, FakeTrainer
from helpers import ValueObject


class TestEarlyStopping(unittest.TestCase):
    def test_loss(self):
        stdout, sys.stdout = sys.stdout, None
        trainer = FakeTrainer()
        early_stopping = EarlyStopping('val_loss', 1)

        state = {'meters': {'val_loss': ValueObject(5)}}
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        state = {'meters': {'val_loss': ValueObject(2)}}
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        state['meters']['val_loss'] = ValueObject(10)
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertEqual(ret, 0)
        sys.stdout = stdout

    def test_acc(self):
        stdout, sys.stdout = sys.stdout, None
        trainer = FakeTrainer()
        early_stopping = EarlyStopping('acc', 1)

        state = {'meters': {'acc': ValueObject(5)}}
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        state = {'meters': {'acc': ValueObject(10)}}
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        state['meters']['acc'] = ValueObject(2)
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertEqual(ret, 0)
        sys.stdout = stdout


class TestCSVLogger(unittest.TestCase):
    def test_wrong_init(self):
        with self.assertRaises(ValueError):
            CSVLogger(keys=5)

    def test_wrong_key(self):
        directory = tempfile.gettempdir()
        csv_logger = CSVLogger(
            directory=directory,
            keys=['loss'])
        trainer = FakeTrainer()

        state = {}

        ret = csv_logger.on_train_start(trainer, state)
        self.assertIs(ret, None)

        with self.assertRaises(KeyError):
            csv_logger.on_epoch_end(trainer, state)

        csv_logger.on_train_end(trainer, state)

    def test_write_log(self):
        directory = tempfile.gettempdir()
        csv_logger = CSVLogger(
            directory=directory,
            keys=['loss'])
        trainer = FakeTrainer()

        state = {'meters': {}}

        ret = csv_logger.on_train_start(trainer, state)
        self.assertIs(ret, None)

        state['meters']['loss'] = ValueObject(5)
        ret = csv_logger.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        csv_logger.on_train_end(trainer, state)


class TestLRScheduler(unittest.TestCase):
    def test_init(self):
        scheduler = LRScheduler()
        trainer = FakeTrainer()
        train_data_loader = FakeDatasetLoader()
        optimizer = FakeOptimizer()
        trainer.train_data_loader = train_data_loader

        state = {}
        state['optimizer'] = optimizer
        state['max_epoch'] = 1
        state['mode'] = 'train'
        state['iters'] = 5

        scheduler.on_train_start(trainer, state)
        self.assertListEqual(scheduler.init_lr, [1, 2])


class TestExpLRScheduler(unittest.TestCase):
    def test_schedule(self):
        scheduler = ExpLRScheduler(0.9)
        trainer = FakeTrainer()
        train_data_loader = FakeDatasetLoader()
        optimizer = FakeOptimizer()
        trainer.train_data_loader = train_data_loader

        state = {}
        state['optimizer'] = optimizer
        state['max_epoch'] = 1
        state['mode'] = 'train'
        state['iters'] = 5

        scheduler.on_train_start(trainer, state)
        self.assertEqual(scheduler.max_iters, 10)

        scheduler.on_batch_end(trainer, state)
        gt_lrs = [0.535889, 1.07177]
        lrs = [d['lr'] for d in optimizer.param_groups]
        for lr, gt_lr in zip(lrs, gt_lrs):
            self.assertAlmostEqual(lr, gt_lr, 5)


class TestModelCheckPoint(unittest.TestCase):
    def test_save_and_load(self):
        tempdir = tempfile.gettempdir()
        checkpoint = ModelCheckPoint(tempdir, monitor='loss',
                                     save_best_only=True)
        trainer = FakeTrainer()

        state = {}
        state['meters'] = {}
        state['meters']['loss'] = ValueObject(5)
        state['arch'] = 'Fake'
        state['epochs'] = randint(0, 100)
        state['iters'] = randint(0, 100) * 100
        state['model'] = FakeModel()
        state['optimizer'] = FakeOptimizer()

        checkpoint.on_epoch_end(trainer, state)

        path = os.path.join(
            tempdir, 'checkpoint_{}_best.pth.tar'.format(state['arch']))
        self.assertTrue(os.path.exists(path))

        loaded_state = torch.load(path)
        self.assertListEqual(state['model'].state_dict(),
                             loaded_state['model_state_dict'])
        self.assertListEqual(state['optimizer'].state_dict(),
                             loaded_state['optimizer_state_dict'])
        self.assertEqual(state['epochs'], loaded_state['epochs'])
        self.assertEqual(state['iters'], loaded_state['iters'])


class TestPlotLogger(unittest.TestCase):
    def test_epoch_plot_logger(self):
        plot_logger = EpochPlotLogger('train', 'loss', 'line')

        trainer = FakeTrainer()
        state = {}
        state['meters'] = {}
        state['meters']['loss'] = ValueObject(5)
        state['epochs'] = randint(0, 100)
        plot_logger.on_epoch_end(trainer, state)
        self.assertEqual(dict(plot_logger.data_cache), {})

    def test_batch_plot_logger(self):
        plot_logger = BatchPlotLogger('test', 'val_loss', 'line')
        plot_logger.cache_size = 2

        trainer = FakeTrainer()
        state = {}
        state['meters'] = {}
        state['mode'] = 'test'
        state['meters']['val_loss'] = ValueObject(randint(0, 100))
        state['iters'] = randint(0, 100)
        gt = {'x': [state['iters']], 'y': [state['meters']['val_loss'].value]}
        plot_logger.on_batch_end(trainer, state)
        self.assertEqual(dict(plot_logger.data_cache), gt)

        plot_logger.on_batch_end(trainer, state)
        self.assertEqual(dict(plot_logger.data_cache), {})


class TestReduceLROnPlateau(unittest.TestCase):
    def test_reduce(self):
        reducer = ReduceLROnPlateau(patience=0)
        trainer = FakeTrainer()
        train_data_loader = FakeDatasetLoader()
        optimizer = FakeOptimizer()
        trainer.train_data_loader = train_data_loader

        state = {}
        state['optimizer'] = optimizer
        state['meters'] = {}
        state['meters']['val_loss'] = ValueObject(1)

        reducer.on_train_start(trainer, state)
        reducer.on_epoch_end(trainer, state)

        state['meters']['val_loss'] = ValueObject(5)  # worse result
        reducer.on_epoch_end(trainer, state)

        gt_lrs = [0.1, 0.2]
        lrs = [d['lr'] for d in optimizer.param_groups]
        for lr, gt_lr in zip(lrs, gt_lrs):
            self.assertAlmostEqual(lr, gt_lr, 2)


if __name__ == '__main__':
    unittest.main()
