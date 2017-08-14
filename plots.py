#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-13 13:43
# Last modified: 2017-08-14 09:35
# Filename: plots.py
# Description:
import signal
import os

from multiprocessing import Process
from multiprocessing import Queue as PQueue

import numpy as np
import visdom


class BaseVisdom:
    _viz = visdom.Visdom()

    def __init__(self, win=None, env=None, opts=None):
        super().__init__()
        self.win = win
        self.env = env
        self.opts = opts

    @property
    def viz(self):
        return type(self)._viz

    def log(self, *args, **kwargs):
        raise NotImplementedError('log should be implemented in subclass')


class VisdomPlot(BaseVisdom):
    def __init__(self, plot_type, win=None, env=None, opts=None):
        super().__init__(win, env, opts)
        self.plot_type = plot_type
        self.chart = getattr(self.viz, plot_type)  # raise AttributeError

    def plot(self, x, y):
        self.viz.updateTrace(
            X=np.array([x]),
            Y=np.array([y]),
            win=self.win,
            env=self.env,
            opts=self.opts)

    def log(self, *args, **kwargs):
        if self.win is None:
            if self.plot_type == 'scatter':
                chart_args = {'X': np.array([args])}
            else:
                chart_args = {'X': np.array([args[0]]),
                              'Y': np.array([args[1]])}
            self.win = self.chart(
                win=self.win,
                env=self.env,
                opts=self.opts,
                **chart_args)
        if len(args) != 2:
            raise ValueError('Wrong parameters number, Found: {}'.format(
                len(args)))
        x, y = args
        self.plot(x, y)


def consumer(pid, plotter, data_queue):
    # Block SIGINT
    if pid != os.getpid():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    win = data_queue.get()
    env = data_queue.get()
    opts = data_queue.get()
    while True:
        item = data_queue.get()
        if item is None:
            break
        x, y = item
        plotter.viz.updateTrace(
            X=np.array([x]),
            Y=np.array([y]),
            win=win,
            env=env,
            opts=opts)


class ProcessVisdomPlot(VisdomPlot):
    handler_core = Process
    queue_core = PQueue
    context_send = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_queue = self.queue_core()
        self.data_queue = data_queue

        self.data_handler = self.handler_core(
            target=consumer, args=(os.getpid(), self, data_queue),
            daemon=True)
        self.data_handler.start()

    def _teardown(self):
        self.data_handler.terminate()
        self.data_handler.join()

    def plot(self, x, y):
        if self.context_send is False:
            self.data_queue.put(self.win)
            self.data_queue.put(self.env)
            self.data_queue.put(self.opts)
            self.context_send = True
        self.data_queue.put((x, y))
