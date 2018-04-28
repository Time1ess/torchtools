# coding: UTF-8
import torch

from tqdm import tqdm, trange

from torchtools import TRAIN_MODE, VALIDATE_MODE, TEST_MODE
from torchtools.callbacks import Hook, Callback
from torchtools.exceptions import (
    HookTypeError, TrainerTerminatedException)
from torchtools.meters import Meter


class Trainer(object):
    """A class to handle the whole training, validating and testing process"""
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
                 device='cpu'):
        """
        Instantiate a trainer object.

        Parameters
        ----------
        model: torch.nn.Module
            A model to train.
        train_data_loader: torch.utils.data.DataLoader
            An instance of DataLoader to load train data.
        criterion: torch.nn.Module
            A loss function.
        optimizer: torch.optim.Optimizer
            Optimizer responsible for optimizing model
        val_data_loader(optional): torch.utils.data.DataLoader
            An instance of DataLoader to load validate data.
        test_data_loader(optional): torch.utils.data.DataLoader
            An instance of DataLoader to load test data.
        device: str
            Which device should be used if use_cuda is True, should be
            formatted like 'cuda:0' or 'cpu'. Default: 'cpu'.
        """
        self.model = model
        self.train_data_loader = train_data_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.device = torch.device(device)

        self.callback_hooks = {k: [] for k in self.hook_entries}
        self.meter_hooks = {k: [] for k in self.hook_entries}
        self.meters = {}
        self.callbacks = []

    def register_hooks(self, hooks):
        """
        Register multiple hooks at the same time.

        Parameters
        ----------
        hooks: Iterable of Hook
            A iterable of hooks need to be registered.
        """
        for hook in hooks:
            self.register_hook(hook)

    def register_hook(self, hook):
        """
        Register a hook.

        Parameters
        ----------
        hook: Hook
            A Hook object need to be registered.
        """
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
        """
        Unregister multiple hooks at the same time.

        Parameters
        ----------
        hooks: Iterable of Hook
            A iterable of hooks need to be registered.
        """
        for hook in hooks:
            self.unregister_hook(hook)

    def unregister_hook(self, hook):
        """
        Unregister a hook.

        Parameters
        ----------
        hook: Hook
            A hook object need to be unregistered.
        """
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
        """
        Terminate training process, trigger `on_terminated` event.

        Parameters
        ----------
        raise_exception: bool
            whether to raise TrainerTerminated exception after event,
            default `True`.
        """
        self.notify_registered_hooks('on_terminated', None)
        if raise_exception:
            raise TrainerTerminatedException()

    def exit(self):
        """
        Set trainer_ended flag to True.
        """
        self.trainer_ended = True
        return 0

    def notify_registered_hooks(self, name, state):
        """
        Event dispatcher for all registered hooks.

        Parameters
        ----------
        name: str
            Event name.
        state: dict
            Current state dictionary.
        """
        for hook in self.meter_hooks[name]:
            hook(self, state)
        if name == 'on_epoch_end':
            self.validate(epochs=state['epochs'])
        for hook in self.callback_hooks[name]:
            hook(self, state)

    def restore_state(self, state, checkpoint):
        """
        Restore from checkpoint.

        Parameters
        ----------
        state: dict
            Current state dictionary.
        checkpoint: str
            Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint)
        state['model'].load_state_dict(checkpoint['model_state_dict'])
        state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
        state['epochs'] = checkpoint['epochs']

    def train(self, max_epoch, checkpoint=None):
        """
        Train model with `max_epoch` epochs.

        Parameters
        ----------
        max_epoch: int
            Num of epochs the model need to be trained.
        checkpoint: str
            Path to checkpoint, default `None`.
        """
        model = self.model.train(True)
        data_loader = self.train_data_loader
        criterion = self.criterion
        optimizer = self.optimizer
        meters = self.meters
        device = self.device
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
        with torch.set_grad_enabled(True):
            for epoch in iter_epoch:
                model = self.model.train(True)
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

                    input, target = input.to(device), target.to(device)

                    def closure():
                        state['optimizer'].zero_grad()
                        output = state['model'](input)
                        loss = criterion(output, target)
                        loss_val = loss.item()
                        iter_data.set_postfix(iters=state['iters'],
                                              loss=loss_val)
                        state['loss'] = loss_val
                        state['output'] = output
                        self.notify_registered_hooks('on_forward_end', state)
                        loss.backward()
                        self.notify_registered_hooks('on_backward_end', state)
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
        """
        Validate model(val_date_loader needed).

        Parameters
        ----------
        epochs: int
            Which epoch the validation process is in, default `-1`.
        """
        if self.val_data_loader is None:
            return {}
        model = self.model.train(False)
        data_loader = self.val_data_loader
        criterion = self.criterion
        meters = self.meters
        device = self.device

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
        with torch.set_grad_enabled(False):
            for batch in iter_data:
                input, target = batch[0], batch[1]
                state['iters'] += 1
                state['input'] = input
                state['target'] = target
                self.notify_registered_hooks('on_batch_start', state)

                input, target = input.to(device), target.to(device)

                def closure():
                    output = state['model'](input)
                    loss = criterion(output, target)
                    loss_val = loss.item()
                    iter_data.set_postfix(loss=loss_val)
                    state['output'] = output
                    state['val_loss'] = loss_val
                    self.notify_registered_hooks('on_forward_end', state)
                    return loss

                closure()
                self.notify_registered_hooks('on_batch_end', state)
            self.notify_registered_hooks('on_validate_end', state)
        return state

    def test(self, test_data_loader=None):
        """
        Test model(test_data_loader needed).

        Parameters
        ----------
        test_data_loader: torch.utils.data.DataLoader
            An instance of DataLoader to load test data, default `None`.
        """
        if test_data_loader is None and self.test_data_loader is None:
            return {}
        if test_data_loader:
            data_loader = test_data_loader
        else:
            data_loader = self.test_data_loader
        model = self.model.train(False)
        criterion = self.criterion
        meters = self.meters
        device = self.device

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
        with torch.set_grad_enabled(False):
            for batch in iter_data:
                input, target = batch[0], batch[1]
                state['iters'] += 1
                state['input'] = input
                state['target'] = target
                self.notify_registered_hooks('on_batch_start', state)

                input, target = input.to(device), target.to(device)

                def closure():
                    output = state['model'](input)
                    loss = criterion(output, target)
                    loss_val = loss.item()
                    iter_data.set_postfix(loss=loss_val)
                    state['output'] = output
                    state['test_loss'] = loss_val
                    self.notify_registered_hooks('on_forward_end', state)
                    return loss

                closure()
                self.notify_registered_hooks('on_batch_end', state)
            self.notify_registered_hooks('on_test_end', state)
        return state
