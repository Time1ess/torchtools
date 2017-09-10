#!/usr/bin/python3
# coding: UTF-8
# Author: David
# Email: youchen.du@gmail.com
# Created: 2017-08-13 13:43
# Last modified: 2017-09-07 22:18
# Filename: visdom_plot.py
# Description:
import numpy as np
import visdom


class BaseVisdom(object):
    _viz = visdom.Visdom()

    def __init__(self, win=None, env=None, opts=None, *args, **kwargs):
        super(BaseVisdom, self).__init__(*args, **kwargs)
        self.win = win
        self.env = env
        self.opts = opts

    @property
    def viz(self):
        return type(self)._viz

    def log(self, *args, **kwargs):
        raise NotImplementedError('log should be implemented in subclass')

    def _teardown(self):
        pass


class VisdomPlot(BaseVisdom):
    def __init__(self, plot_type, win=None, env=None, opts=None,
                 *args, **kwargs):
        super(VisdomPlot, self).__init__(win, env, opts, *args, **kwargs)
        self.plot_type = plot_type
        self.chart = getattr(self.viz, plot_type)  # raise AttributeError

    def log(self, *args, **kwargs):
        if len(args) != 2:
            raise ValueError('Wrong parameters number, Found: {}'.format(
                len(args)))
        if self.win is None:
            if self.plot_type == 'scatter':
                chart_args = {'X': np.array(args)}
            else:
                chart_args = {'X': np.array(args[0]),
                              'Y': np.array(args[1])}
            self.win = self.chart(
                win=self.win,
                env=self.env,
                opts=self.opts,
                **chart_args)
        else:
            x, y = args
            self.viz.updateTrace(
                X=np.array(x),
                Y=np.array(y),
                win=self.win,
                env=self.env,
                opts=self.opts)