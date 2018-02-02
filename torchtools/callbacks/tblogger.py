#!/usr/bin/env python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-09-07 20:05
# Last modified: 2017-10-20 18:43
# Filename: tblogger.py
# Description:
from .callback import Callback
from tensorboard import SummaryWriter

from ..exceptions import LogTypeError
from ..meters import meter as _meter


class TensorBoardLogger(Callback):
    _tensorboard = None

    def __init__(self, directory='logs', log_param_hists=False,
                 log_model_graph=False, ignores=None, log_param_interval=1000,
                 *args, **kwargs):
        super(TensorBoardLogger, self).__init__(*args, **kwargs)
        if self._tensorboard is None:
            type(self)._tensorboard = SummaryWriter(directory)
        if ignores is None:
            ignores = []
        self.ignores = ignores
        self.log_param_hists = log_param_hists
        self.log_model_graph = log_model_graph
        self.log_param_interval = log_param_interval
        self.counter = 0
        self.epochs = 0

    def _teardown(self):
        self.board.close()

    @property
    def board(self):
        return self._tensorboard

    def log(self, global_step, meter):
        log_type = meter.meter_type
        if not hasattr(self, 'log_' + log_type):
            raise LogTypeError('No such log type: {}'.format(log_type))
        method = getattr(self, 'log_' + log_type)
        step = getattr(meter, 'step', global_step)
        method(meter.name, meter.value, step)

    def log_image(self, tag, img_tensor, step=None):
        self.board.add_image(tag, img_tensor, step)

    def log_scalar(self, tag, scalar_value, step=None):
        self.board.add_scalar(tag, scalar_value, step)

    def log_graph(self, model, res):
        self.board.add_graph(model, res)

    def log_hist(self, tag, value, step=None, bins='tensorflow'):
        self.board.add_histogram(tag, value, step, bins)

    def log_text(self):
        pass

    def log_audio(self):
        pass

    def on_forward_end(self, trainer, state):
        if state['mode'] != 'train':
            return

        if self.log_model_graph:
            model = state['model']
            output = state['output']
            self.log_graph(model, output)
            self.log_model_graph = False

        self.counter += 1
        if self.counter == self.log_param_interval:
            if self.log_param_hists:
                model = state['model']
                step = state['iters']
                for name, params in model.named_parameters():
                    self.log_hist(name,
                                  params.clone().cpu().data.numpy(), step)
            self.counter = 0

    def on_batch_end(self, trainer, state):
        iters = state['iters']
        mode = state['mode']
        for name, meter in state['meters'].items():
            if meter.meter_mode != mode:
                continue
            if meter.reset_mode == _meter.BATCH_RESET and \
                    name not in self.ignores and meter.can_call:
                self.log(iters, meter)

    def on_epoch_end(self, trainer, state):
        epochs = state.get('epochs', self.epochs)
        self.epochs = epochs
        mode = state['mode']
        for name, meter in state['meters'].items():
            if meter.meter_mode != mode:
                continue
            if meter.reset_mode == _meter.EPOCH_RESET and \
                    name not in self.ignores and meter.can_call:
                self.log(epochs, meter)

    def on_validate_end(self, trainer, state):
        self.on_epoch_end(trainer, state)
