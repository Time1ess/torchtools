#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-08 19:34
# Last modified: 2017-08-08 20:41
# Filename: trainer.py
# Description:
from tqdm import tqdm, trange


class ModuleTrainer:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, hook):
        self.hooks[name] = hook

    def unregister_hook(self, name):
        self.hooks.pop(name, None)

    def on_hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, data_loader, creteria, optimizer, max_epoch):
        state = {
            'network': network,
            'data_loader': data_loader,
            'max_epoch': max_epoch,
            'epoch': 0,
            'iters': 0,
            'optimizer': optimizer,
            'train': True,
        }
        self.on_hook('on_train_start', state)
        iter_epoch = trange(max_epoch, unit=' epochs')
        iter_epoch.set_description('Train')
        for epoch in iter_epoch:
            state['epoch'] = epoch
            self.on_hook('on_epoch_start', state)

            iter_data = tqdm(data_loader, unit=' batches')
            iter_data.set_description('Epoch ' + str(epoch))
            for (input, target) in iter_data:
                state['input'] = input
                state['target'] = target
                self.on_hook('on_batch_start', state)

                def closure():
                    output = state['network'](input)
                    loss = creteria(output, target)
                    iter_data.set_postfix(loss=loss)
                    state['output'] = output
                    state['loss'] = loss
                    self.on_hook('on_forward_end', state)
                    loss.backward()
                    # Free memory
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                self.on_hook('on_batch_end', state)
                state['iters'] += 1
            self.on_hook('on_epoch_end', state)
        self.on_hook('on_train_end', state)
        return state

    def test(self, network, data_loader, creteria):
        state = {
            'network': network,
            'data_loader': data_loader,
            'optimizer': optimizer,
            'iters': 0,
            'train': False,
        }
        self.on_hook('on_test_start', state)
        iter_data = tqdm(data_loader, unit=' batches')
        iter_data.set_description('Test')
        for (input, target) in iter_data:
            state['input'] = input
            state['target'] = target
            self.on_hook('on_batch_start', state)

            def closure():
                output = state['network'](input)
                loss = creteria(output, target)
                iter_data.set_postfix(loss=loss)
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


if __name__ == '__main__':
    from fake import FakeNetwork, FakeDataLoader, FakeCreteria, FakeOptimizer
    network = FakeNetwork(0.01)
    data_loader = FakeDataLoader()
    optimizer = FakeOptimizer()
    trainer = ModuleTrainer()
    trainer.train(network, data_loader, FakeCreteria, optimizer, 10)
    trainer.test(network, data_loader, FakeCreteria)
