#!python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-15 12:28
# Last modified: 2017-08-15 14:40
# Filename: test_callbacks.py
# Description:
import unittest
import tempfile

from torchtools.callbacks import EarlyStopping, CSVLogger, ExpLRScheduler

from helpers import FakeModel, FakeDatasetLoader
from helpers import FakeCriterion, FakeOptimizer, FakeTrainer
from helpers import ValueObject


class TorchToolsTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        trainer = FakeTrainer()
        trainer.model = FakeModel()
        trainer.train_data_loader = FakeDatasetLoader()
        trainer.test_data_loader = FakeDatasetLoader()
        trainer.criterion = FakeCriterion
        trainer.optimizer = FakeOptimizer()
        cls.trainer = trainer


class TestEarlyStopping(TorchToolsTestBase):
    def test_loss(self):
        trainer = self.trainer
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

    def test_acc(self):
        trainer = self.trainer
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


class TestCSVLogger(TorchToolsTestBase):
    def test_wrong_init(self):
        with self.assertRaises(ValueError):
            CSVLogger(keys=5)

    def test_wrong_key(self):
        directory = tempfile.gettempdir()
        csv_logger = CSVLogger(
            directory=directory,
            keys=['loss'])
        trainer = self.trainer

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
        trainer = self.trainer

        state = {'meters': {}}

        ret = csv_logger.on_train_start(trainer, state)
        self.assertIs(ret, None)

        state['meters']['loss'] = ValueObject(5)
        ret = csv_logger.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        csv_logger.on_train_end(trainer, state)


class TestExpLRScheduler(TorchToolsTestBase):
    def test_schedule(self):
        scheduler = ExpLRScheduler(0.9)
        trainer = self.trainer
        optimizer = trainer.optimizer
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


if __name__ == '__main__':
    unittest.main()
