#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-08 19:34
# Last modified: 2017-08-09 19:06
# Filename: trainer.py
# Description:
import torch

from torch.autograd import Variable
from tqdm import tqdm, trange


class ModelTrainer:
    def __init__(self, use_cuda):
        self.hooks = {}
        self.use_cuda = use_cuda

    def register_hook(self, name, hook):
        self.hooks[name] = hook

    def unregister_hook(self, name):
        self.hooks.pop(name, None)

    def on_hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def restore_state(self, state, checkpoint):
        print('Restore from checkpoint:'.format(checkpoint))
        checkpoint = torch.load(checkpoint)
        state['model'].load_state_dict(checkpoint['model_state_dict'])
        state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
        state['epochs'] = checkpoint['epochs']
        state['iters'] = checkpoint['iters']

    def train(self, model, data_loader, creteria, optimizer, max_epoch,
            checkpoint=None):
        state = {
            'model': model,
            'arch': model.__class__.__name__,
            'data_loader': data_loader,
            'max_epoch': max_epoch,
            'epochs': 0,
            'iters': 0,
            'optimizer': optimizer,
            'train': True,
        }
        if checkpoint is not None:
            self.restore_state(state, checkpoint)
        self.on_hook('on_train_start', state)
        iter_epoch = trange(max_epoch, initial=state['epochs'], unit='epoch')
        iter_epoch.set_description('Train')
        for epoch in iter_epoch:
            state['epochs'] = epoch
            self.on_hook('on_epoch_start', state)

            iter_data = tqdm(data_loader, unit=' batches')
            iter_data.set_description('Epoch ' + str(epoch))
            for batch in iter_data:
                if self.use_cuda:
                    input = Variable(batch[0].cuda())
                    target = Variable(batch[1].cuda())
                else:
                    input = Variable(batch[0])
                    target = Variable(batch[1])

                state['input'] = batch[0]
                state['target'] = batch[1]
                self.on_hook('on_batch_start', state)

                def closure():
                    state['optimizer'].zero_grad()
                    output = state['model'](input)
                    loss = creteria(output, target)
                    iter_data.set_postfix(iters=state['iters'],
                                          loss=loss.data[0])
                    state['output'] = output
                    state['loss'] = loss
                    self.on_hook('on_forward_end', state)
                    loss.backward()
                    # Free memory
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].step(closure)
                self.on_hook('on_batch_end', state)
                state['iters'] += 1
            self.on_hook('on_epoch_end', state)
        self.on_hook('on_train_end', state)
        return state

    def test(self, model, data_loader, creteria):
        state = {
            'model': model,
            'arch': model.__class__.__name__,
            'data_loader': data_loader,
            'iters': 0,
            'train': False,
        }
        self.on_hook('on_test_start', state)
        iter_data = tqdm(data_loader, unit=' batches')
        iter_data.set_description('Test')
        for batch in iter_data:
            if self.use_cuda:
                input = Variable(batch[0].cuda())
                target = Variable(batch[1].cuda())
            else:
                input = Variable(batch[0])
                target = Variable(batch[1])
            state['input'] = batch[0]
            state['target'] = batch[1]
            self.on_hook('on_batch_start', state)

            def closure():
                output = state['model'](input)
                loss = creteria(output, target)
                iter_data.set_postfix(loss=loss.data[0])
                state['output'] = output
                state['loss'] = loss
                self.on_hook('on_forward_end', state)
                # Free memory
                state['output'] = None
                state['loss'] = None
                return loss

            closure()
            self.on_hook('on_batch_end', state)
            state['iters'] += 1
        self.on_hook('on_test_end', state)
        return state
