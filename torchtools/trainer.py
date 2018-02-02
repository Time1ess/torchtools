#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-08 19:34
# Last modified: 2017-10-19 19:25
# Filename: trainer.py
# Description:
import functools

import torch

from torch.autograd import Variable
from tqdm import tqdm, trange

from .exceptions import HookTypeError, HookCheckError, TrainerTerminated
from .callbacks import Hook
from .meters import Meter


def trainer_wraps(func):
    @functools.wraps(func)
    def _wraps(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except TrainerTerminated:
            return None
        except KeyboardInterrupt:
            self.terminate(raise_exception=False)
        except Exception:
            raise
    return _wraps


class ModelTrainer(object):
    hook_entries = [
        'on_train_start', 'on_train_end',
        'on_epoch_start', 'on_epoch_end',
        'on_forward_end', 'on_backward_end',
        'on_batch_start', 'on_batch_end',
        'on_validate_start', 'on_validate_end',
        'on_test_start', 'on_test_end',
        'on_terminated']
    trainer_ended = False

    def __init__(self, model, train_data_loader, criterion, optimizer,
                 val_data_loader=None, test_data_loader=None,
                 use_cuda=True, device_id=None):
        self.model = model
        self.train_data_loader = train_data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.device_id = device_id
        self.use_cuda = use_cuda and torch.cuda.is_available()

        self.callback_hooks = {k: [] for k in self.hook_entries}
        self.meter_hooks = {k: [] for k in self.hook_entries}
        self.meters = {}
        self.callbacks = []

    def register_hooks(self, hooks):
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook):
        if not isinstance(hook, Hook):
            raise HookTypeError('{} is not a valid Hook type'.format(hook))
        elif hook.has_hook_conflict(self):
            raise HookCheckError(hook.conflict)
        elif isinstance(hook, Meter):
            container = self.meter_hooks
            self.meters[hook.name] = hook
        else:
            container = self.callback_hooks
            self.callbacks.append(hook)

        for name in self.hook_entries:
            entry = getattr(hook, name)
            container[name].append(entry)

    def unregister_hook(self, hook):
        if not isinstance(hook, Hook):
            raise HookTypeError('{} is not a valid Hook type'.format(hook))
        elif isinstance(hook, Meter):
            container = self.meter_hooks
            self.meters.pop(hook.name, None)
        else:
            container = self.callback_hooks

        for name in self.hook_entries:
            entry = getattr(hook, name, None)
            if entry is not None:
                container[name].remove(entry)

    def terminate(self, raise_exception=True):
        self.on_hook('on_terminated', None)
        if raise_exception:
            raise TrainerTerminated()

    def exit(self):
        self.trainer_ended = True
        return 0

    def on_hook(self, name, state):
        for hook in self.meter_hooks[name]:
            hook(self, state)
        # TODO: Maybe there is a better way to call validate after meter reset?
        if name == 'on_epoch_end':
            self.validate()
        for hook in self.callback_hooks[name]:
            hook(self, state)

    def restore_state(self, state, checkpoint):
        print('Restore from checkpoint:'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        state['model'].load_state_dict(checkpoint['model_state_dict'])
        state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
        state['epochs'] = checkpoint['epochs']
        state['iters'] = checkpoint['iters']

    def train(self, max_epoch, checkpoint=None):
        model = self.model.train(True)
        data_loader = self.train_data_loader
        criterion = self.criterion
        optimizer = self.optimizer
        meters = self.meters
        use_cuda = self.use_cuda
        device_id = self.device_id
        self.trainer_ended = False

        state = {
            'model': model,
            'arch': type(model).__name__,
            'max_epoch': max_epoch,
            'epochs': 0,
            'iters': 0,
            'optimizer': optimizer,
            'mode': 'train',
            'meters': meters,
        }
        if checkpoint is not None:
            self.restore_state(state, checkpoint)
        self.on_hook('on_train_start', state)
        iter_epoch = trange(max_epoch, initial=state['epochs'], unit='epoch')
        iter_epoch.set_description('Train')
        for epoch in iter_epoch:
            model.train(True)
            state['epochs'] = epoch + 1
            self.on_hook('on_epoch_start', state)

            iter_data = tqdm(data_loader, unit=' batches')
            iter_data.set_description('Epoch ' + str(epoch))
            for batch in iter_data:
                input, target = batch[0], batch[1]
                state['iters'] += 1
                state['input'] = input
                state['target'] = target
                self.on_hook('on_batch_start', state)

                if use_cuda:
                    input = input.cuda(device_id, async=True)
                    target = target.cuda(device_id, async=True)
                input = Variable(input)
                target = Variable(target)

                def closure():
                    state['optimizer'].zero_grad()
                    output = state['model'](input)
                    loss = criterion(output, target)
                    iter_data.set_postfix(iters=state['iters'],
                                          loss=loss.data[0])
                    state['output'] = output
                    state['loss'] = loss
                    self.on_hook('on_forward_end', state)
                    loss.backward()
                    self.on_hook('on_backward_end', state)
                    # Free memory
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].step(closure)
                self.on_hook('on_batch_end', state)
                if self.trainer_ended:
                    break
            self.on_hook('on_epoch_end', state)
            if self.trainer_ended:
                break
        self.trainer_ended = True
        self.on_hook('on_train_end', state)
        return state

    def validate(self):
        if self.val_data_loader is None:
            return {}
        model = self.model.train(False)
        data_loader = self.val_data_loader
        criterion = self.criterion
        meters = self.meters
        use_cuda = self.use_cuda
        device_id = self.device_id

        state = {
            'model': model,
            'arch': type(model).__name__,
            'mode': 'validate',
            'iters': 0,
            'meters': meters,
        }
        self.on_hook('on_validate_start', state)
        iter_data = tqdm(data_loader, unit=' batches')
        iter_data.set_description('Validate')
        for batch in iter_data:
            input, target = batch[0], batch[1]
            state['iters'] += 1
            state['input'] = input
            state['target'] = target
            self.on_hook('on_batch_start', state)

            if use_cuda:
                input = input.cuda(device_id, async=True)
                target = target.cuda(device_id, async=True)
            input = Variable(input, volatile=True)
            target = Variable(target, volatile=True)

            def closure():
                output = state['model'](input)
                loss = criterion(output, target)
                iter_data.set_postfix(loss=loss.data[0])
                state['output'] = output
                state['val_loss'] = loss
                self.on_hook('on_forward_end', state)
                # Free memory
                state['output'] = None
                state['val_loss'] = None
                return loss

            closure()
            self.on_hook('on_batch_end', state)
        self.on_hook('on_validate_end', state)
        return state

    def test(self, test_data_loader=None):
        if test_data_loader is None and self.test_data_loader is None:
            return {}
        model = self.model.train(False)
        if test_data_loader:
            data_loader = test_data_loader
        else:
            data_loader = self.test_data_loader
        criterion = self.criterion
        meters = self.meters
        use_cuda = self.use_cuda
        device_id = self.device_id

        state = {
            'model': model,
            'arch': type(model).__name__,
            'mode': 'test',
            'iters': 0,
            'meters': meters,
        }
        self.on_hook('on_test_start', state)
        iter_data = tqdm(data_loader, unit=' batches')
        iter_data.set_description('Test')
        for batch in iter_data:
            input, target = batch[0], batch[1]
            state['iters'] += 1
            state['input'] = input
            state['target'] = target
            self.on_hook('on_batch_start', state)

            if use_cuda:
                input = input.cuda(device_id, async=True)
                target = target.cuda(device_id, async=True)
            input = Variable(input, volatile=True)
            target = Variable(target, volatile=True)

            def closure():
                output = state['model'](input)
                loss = criterion(output, target)
                iter_data.set_postfix(loss=loss.data[0])
                state['output'] = output
                state['test_loss'] = loss
                self.on_hook('on_forward_end', state)
                # Free memory
                state['output'] = None
                state['test_loss'] = None
                return loss

            closure()
            self.on_hook('on_batch_end', state)
        self.on_hook('on_test_end', state)
        return state
