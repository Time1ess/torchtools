#!/usr/local/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-13 13:43
# Last modified: 2017-08-15 11:40
# Filename: plots.py
# Description:
import signal
import os

from functools import partial
from collections import defaultdict
from threading import Thread
from queue import Queue as TQueue
from multiprocessing import Process
from multiprocessing import Queue as PQueue

import numpy as np
import visdom


def cls_ref(cls):
    cls.cls = cls
    return cls


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


def consumer(pid, viz, data_queue):
    # Block SIGINT
    if pid != os.getpid():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    context = {}
    while True:
        item = data_queue.get()
        if item is None:
            break
        elif len(item) == 4:
            uid, win, env, opts = item
            context[uid] = (win, env, opts)
            continue
        else:
            uid, x, y = item
            win, env, opts = context[uid]
        viz.updateTrace(
            X=np.array(x),
            Y=np.array(y),
            win=win,
            env=env,
            opts=opts)


@cls_ref
class AsyncVisdomPlot(VisdomPlot):
    data_handler = None
    data_queue = None
    handler_core = None
    queue_core = None
    context_send = False
    data_cache = defaultdict(partial(defaultdict, list))
    cache_size = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls = self.cls
        if self.data_handler is None:
            if self.handler_core is None or self.queue_core is None:
                raise AttributeError(
                    'AsyncVisdomPlot should not be initialized')
            data_queue = self.queue_core()
            cls.data_queue = data_queue
            cls.data_handler = self.handler_core(
                target=consumer, args=(os.getpid(), self._viz, data_queue),
                daemon=True)
            cls.data_handler.start()

    def send_to_cache(self, uid, x, y):
        self.data_cache[uid]['x'].append(x)
        self.data_cache[uid]['y'].append(y)
        if len(self.data_cache[uid]['x']) == self.cache_size:
            x = self.data_cache[uid]['x'].copy()
            y = self.data_cache[uid]['y'].copy()
            self.data_queue.put((uid, x, y))
            self.data_cache[uid]['x'].clear()
            self.data_cache[uid]['y'].clear()

    def plot(self, x, y):
        if self.context_send is False:
            self.data_queue.put((id(self), self.win, self.env, self.opts))
            self.context_send = True
        self.send_to_cache(id(self), x, y)

    def _teardown(self):
        cls = self.cls
        if hasattr(cls.data_handler, 'terminate'):
            cls.data_handler.terminate()
        else:
            cls.data_queue.put(None)
        cls.data_handler.join()


class ProcessVisdomPlot(AsyncVisdomPlot):
    handler_core = Process
    queue_core = PQueue


class ThreadVisdomPlot(AsyncVisdomPlot):
    handler_core = Thread
    queue_core = TQueue
