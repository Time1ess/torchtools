# coding: UTF-8
import sys
import os
import os.path as osp
import re
import unittest
import tempfile

from random import randint, random
if sys.version_info >= (3, 3):
    from unittest.mock import Mock, patch
else:
    from mock import Mock, patch

import torch
import torch.optim as optim

from torchtools import VALIDATE_MODE
from torchtools.meters import EPOCH_RESET, SCALAR_METER
from torchtools.exceptions import EarlyStoppingException
from torchtools.callbacks import (
    ModelCheckPoint, CSVLogger, EarlyStopping,
    LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau,
    TensorBoardLogger)

from helpers import Net


class TestModelCheckPoint(unittest.TestCase):
    def test_save_and_load(self):
        tempdir = tempfile.gettempdir()
        checkpoint = ModelCheckPoint(tempdir, monitor='val_loss')
        trainer = Mock()

        state = {}
        state['meters'] = {}
        state['meters']['loss'] = Mock(value=5)
        state['meters']['val_loss'] = Mock(value=5)
        state['arch'] = 'Fake'
        state['epochs'] = randint(0, 100)
        state['iters'] = randint(0, 100) * 100
        net = Net()
        state['model'] = net
        state['optimizer'] = optim.SGD(net.parameters(), lr=1e-3)

        checkpoint.on_epoch_end(trainer, state)

        path = os.path.join(
            tempdir,
            'best_{arch}_{epochs:05d}_{val_loss:.2f}.pt'.format(**state))

        self.assertTrue(os.path.exists(path))

        loaded_state = torch.load(path)
        net = Net()
        net.load_state_dict(loaded_state['model_state_dict'])
        for param1, param2 in zip(
                state['model'].state_dict().values(),
                net.state_dict().values()):
            self.assertTrue(torch.equal(param1, param2))
        optimizer = optim.SGD(net.parameters(), lr=1e-3)
        optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
        self.assertEqual(state['epochs'], loaded_state['epochs'])
        self.assertEqual(state['iters'], loaded_state['iters'])


class TestCSVLogger(unittest.TestCase):
    def test_wrong_keys(self):
        with self.assertRaises(AssertionError):
            CSVLogger(keys=5)

    def test_wrong_key(self):
        log_dir = tempfile.gettempdir()
        csv_logger = CSVLogger(
            log_dir=log_dir,
            keys=['loss'])
        trainer = Mock()

        state = {}

        ret = csv_logger.on_train_start(trainer, state)
        self.assertIs(ret, None)

        with self.assertRaises(KeyError):
            csv_logger.on_epoch_end(trainer, state)

        csv_logger.on_train_end(trainer, state)

    def test_write_log(self):
        log_dir = tempfile.gettempdir()
        csv_logger = CSVLogger(
            log_dir=log_dir,
            keys=['loss'])
        trainer = Mock()

        state = {'meters': {}}

        ret = csv_logger.on_train_start(trainer, state)
        self.assertIs(ret, None)

        loss = Mock(value=randint(0, 100))
        state['meters']['loss'] = loss
        ret = csv_logger.on_epoch_end(trainer, state)
        fpath = osp.join(log_dir, 'training_log.csv')
        with open(fpath, 'r') as f:
            data = ''.join(f.readlines())
        pat = re.compile(
            r'timestamp,loss\r*\n\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d+,')
        self.assertIsNot(pat.match(data), None)
        csv_logger.on_train_end(trainer, state)


class TestEarlyStopping(unittest.TestCase):
    def test_loss(self):
        trainer = Mock()
        early_stopping = EarlyStopping('val_loss', patience=1)

        state = {'meters': {'val_loss': Mock(value=10)}}
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        state = {'meters': {'val_loss': Mock(value=2)}}
        ret = early_stopping.on_epoch_end(trainer, state)
        self.assertIs(ret, None)

        state['meters']['val_loss'] = Mock(value=5)
        ret = early_stopping.on_epoch_end(trainer, state)

        state['meters']['val_loss'] = Mock(value=3)
        with self.assertRaises(EarlyStoppingException):
            ret = early_stopping.on_epoch_end(trainer, state)

    def test_acc(self):
        trainer = Mock()
        early_stopping = EarlyStopping('acc', patience=1)

        state = {'meters': {'acc': Mock(value=5)}}
        early_stopping.on_epoch_end(trainer, state)

        state = {'meters': {'acc': Mock(value=10)}}
        early_stopping.on_epoch_end(trainer, state)

        state['meters']['acc'] = Mock(value=2)
        early_stopping.on_epoch_end(trainer, state)

        state['meters']['acc'] = Mock(value=3)
        with self.assertRaises(EarlyStoppingException):
            early_stopping.on_epoch_end(trainer, state)


class TestLambdaLR(unittest.TestCase):
    def test_step(self):
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=1)
        random_lr = random()
        scheduler = LambdaLR(optimizer, lambda epoch: random_lr)
        state = {}
        state['optimizer'] = optimizer

        scheduler.on_epoch_start(None, state)
        for param in optimizer.param_groups:
            self.assertEqual(param['lr'], random_lr)


class TestStepLR(unittest.TestCase):
    def test_step(self):
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=1)
        scheduler = StepLR(optimizer, 2)
        state = {}
        state['optimizer'] = optimizer

        scheduler.on_epoch_start(None, state)
        scheduler.on_epoch_start(None, state)
        scheduler.on_epoch_start(None, state)
        for param in optimizer.param_groups:
            self.assertEqual(param['lr'], 0.1)


class TestMultiStepLR(unittest.TestCase):
    def test_step(self):
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=1)
        scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
        state = {}
        state['optimizer'] = optimizer

        for epoch in range(100):
            scheduler.on_epoch_start(None, state)
            for param in optimizer.param_groups:
                lr = param['lr']
                if epoch < 30:
                    self.assertAlmostEqual(lr, 1)
                elif epoch < 80:
                    self.assertAlmostEqual(lr, 0.1)
                else:
                    self.assertAlmostEqual(lr, 0.01)


class TestExponentialLR(unittest.TestCase):
    def test_step(self):
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=1)
        scheduler = ExponentialLR(optimizer, 0.5)
        state = {}
        state['optimizer'] = optimizer

        scheduler.on_epoch_start(None, state)
        scheduler.on_epoch_start(None, state)
        for param in optimizer.param_groups:
            self.assertEqual(param['lr'], 0.5)


class TestReduceLROnPlateau(unittest.TestCase):
    def test_step(self):
        net = Net()
        optimizer = optim.SGD(net.parameters(), lr=1)
        scheduler = ReduceLROnPlateau(optimizer, 'val_loss', patience=0)
        state = {}
        state['optimizer'] = optimizer
        state['meters'] = {}

        state['meters']['val_loss'] = Mock(value=5)
        scheduler.on_epoch_end(None, state)
        state['meters']['val_loss'] = Mock(value=8)
        scheduler.on_epoch_end(None, state)

        for param in optimizer.param_groups:
            self.assertEqual(param['lr'], 0.1)


class TestTensorBoardLogger(unittest.TestCase):
    @patch('torchtools.callbacks.TensorBoardLogger.log_scalar')
    def test_log_scalar(self, mocked_log_scalar):
        log_dir = tempfile.gettempdir()
        tb = TensorBoardLogger(log_dir)

        state = {}
        state['meters'] = {}
        state['epochs'] = 10
        state['mode'] = VALIDATE_MODE
        val_loss_meter = Mock(value=5)
        val_loss_meter.mode = VALIDATE_MODE
        val_loss_meter.alias = 'val_loss'
        val_loss_meter.reset_mode = EPOCH_RESET
        val_loss_meter.meter_type = SCALAR_METER
        state['meters']['val_loss'] = val_loss_meter

        tb.on_epoch_end(None, state)
        mocked_log_scalar.assert_called_with(
            val_loss_meter.alias, val_loss_meter.value, state['epochs'])


if __name__ == '__main__':
    unittest.main()
