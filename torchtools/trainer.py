# coding: UTF-8
import torch

from torch.autograd import Variable
from tqdm import tqdm, trange

from torchtools import TRAIN_MODE, VALIDATE_MODE, TEST_MODE
from torchtools.callbacks import Hook, Callback
from torchtools.exceptions import (
    HookTypeError, TrainerTerminated)
from torchtools.meters import Meter


class Trainer(object):
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
            raise HookTypeError('{} is not a valid hook'.format(hook))
        if isinstance(hook, Callback):
            hook._callback_check(self)

        if isinstance(hook, Meter):
            container = self.meter_hooks
            self.meters[hook.name] = hook
        else:
            container = self.callback_hooks
            self.callbacks.append(hook)

        for name in self.hook_entries:
            entry = getattr(hook, name)
            container[name].append(entry)

    def unregister_hooks(self, hooks):
        for hook in hooks:
            self.unregister_hook(hook)

    def unregister_hook(self, hook):
        if not isinstance(hook, Hook):
            raise HookTypeError('{} is not a valid hook'.format(hook))

        if isinstance(hook, Meter):
            container = self.meter_hooks
            self.meters.pop(hook.name, None)
        else:
            container = self.callback_hooks
            self.callbacks.remove(hook)

        for name in self.hook_entries:
            entry = getattr(hook, name, None)
            if entry is not None:
                container[name].remove(entry)

    def terminate(self, raise_exception=True):
        self.notify_registered_hooks('on_terminated', None)
        if raise_exception:
            raise TrainerTerminated()

    def exit(self):
        self.trainer_ended = True
        return 0

    def notify_registered_hooks(self, name, state):
        for hook in self.meter_hooks[name]:
            hook(self, state)
        if name == 'on_epoch_end':
            self.validate(epochs=state['epochs'])
        for hook in self.callback_hooks[name]:
            hook(self, state)

    def restore_state(self, state, checkpoint):
        checkpoint = torch.load(checkpoint)
        state['model'].load_state_dict(checkpoint['model_state_dict'])
        state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
        state['epochs'] = checkpoint['epochs']

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
            'iters': 0,  # Total iterations
            'optimizer': optimizer,
            'mode': TRAIN_MODE,
            'meters': meters,
        }
        if checkpoint is not None:
            self.restore_state(state, checkpoint)
        self.notify_registered_hooks('on_train_start', state)
        iter_epoch = trange(max_epoch, initial=state['epochs'], unit='epoch')
        iter_epoch.set_description('Train')
        for epoch in iter_epoch:
            model.train(True)
            state['epochs'] = epoch + 1
            self.notify_registered_hooks('on_epoch_start', state)

            iter_data = tqdm(data_loader, unit=' batches')
            iter_data.set_description('Epoch ' + str(epoch))
            for batch in iter_data:
                input, target = batch[0], batch[1]
                state['iters'] += 1
                state['input'] = input
                state['target'] = target
                self.notify_registered_hooks('on_batch_start', state)

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
                    self.notify_registered_hooks('on_forward_end', state)
                    loss.backward()
                    self.notify_registered_hooks('on_backward_end', state)
                    # Free memory
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].step(closure)
                self.notify_registered_hooks('on_batch_end', state)
                if self.trainer_ended:
                    break
            self.notify_registered_hooks('on_epoch_end', state)
            if self.trainer_ended:
                break
        self.trainer_ended = True
        self.notify_registered_hooks('on_train_end', state)
        return state

    def validate(self, epochs=-1):
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
            'mode': VALIDATE_MODE,
            'epochs': epochs,
            'iters': 0,
            'meters': meters,
        }
        self.notify_registered_hooks('on_validate_start', state)
        iter_data = tqdm(data_loader, unit=' batches')
        iter_data.set_description('Validate')
        for batch in iter_data:
            input, target = batch[0], batch[1]
            state['iters'] += 1
            state['input'] = input
            state['target'] = target
            self.notify_registered_hooks('on_batch_start', state)

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
                self.notify_registered_hooks('on_forward_end', state)
                # Free memory
                state['output'] = None
                state['val_loss'] = None
                return loss

            closure()
            self.notify_registered_hooks('on_batch_end', state)
        self.notify_registered_hooks('on_validate_end', state)
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
            'mode': TEST_MODE,
            'iters': 0,
            'meters': meters,
        }
        self.notify_registered_hooks('on_test_start', state)
        iter_data = tqdm(data_loader, unit=' batches')
        iter_data.set_description('Test')
        for batch in iter_data:
            input, target = batch[0], batch[1]
            state['iters'] += 1
            state['input'] = input
            state['target'] = target
            self.notify_registered_hooks('on_batch_start', state)

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
                self.notify_registered_hooks('on_forward_end', state)
                # Free memory
                state['output'] = None
                state['test_loss'] = None
                return loss

            closure()
            self.notify_registered_hooks('on_batch_end', state)
        self.notify_registered_hooks('on_test_end', state)
        return state
