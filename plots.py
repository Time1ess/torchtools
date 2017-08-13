#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-13 13:43
# Last modified: 2017-08-13 21:52
# Filename: plots.py
# Description:
import signal

from threading import Thread
from multiprocessing import Process, Queue

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


class ProcessVisdomPlot(VisdomPlot):
    handler_core = Process
    context_send = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_queue = Queue()
        self.data_queue = data_queue

        def consumer(plotter, data_queue):
            # Block SIGINT
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

        self.data_handler = self.handler_core(
            target=consumer, args=(self, data_queue))
        self.data_handler.start()

    def _teardown(self):
        for i in range(4):
            self.data_queue.put(None)
        self.data_queue.close()
        self.data_queue.join_thread()
        self.data_handler.join()
        super()._teardown()

    def plot(self, x, y):
        if self.context_send is False:
            self.data_queue.put(self.win)
            self.data_queue.put(self.env)
            self.data_queue.put(self.opts)
            self.context_send = True
        self.data_queue.put((x, y))


class ThreadVisdomPlot(ProcessVisdomPlot):
    handler_core = Thread
