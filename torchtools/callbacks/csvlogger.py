# coding: UTF-8
import os
import csv

from datetime import datetime

from torchtools.callbacks.callback import Callback


class CSVLogger(Callback):
    def __init__(self, log_dir=None, comment='', separator=',',
                 keys=None, header=True, timestamp=True, datetime_fmt=None):
        super(CSVLogger, self).__init__()
        if not log_dir:
            import socket
            from datetime import datetime
            now = datetime.now().strftime('%b%d_%H-%M-%S')
            hostname = socket.gethostname()
            log_dir = os.path.join('logs', now + '_' + hostname + comment)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.fpath = os.path.join(log_dir, 'training_log.csv')
        self.sep = separator
        assert isinstance(keys, list) or isinstance(keys, tuple)
        if timestamp:
            self.keys = ['timestamp']
        else:
            self.keys = []
        if keys is not None:
            self.keys.extend(keys)
        self.header = header
        self.datetime_fmt = datetime_fmt

    def on_train_start(self, trainer, state):
        self.csv_file = open(self.fpath, 'w')

        class CustomDialect(csv.excel):
            delimiter = self.sep

        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=self.keys,
            dialect=CustomDialect)

        if self.header:
            self.writer.writeheader()

    def on_epoch_end(self, trainer, state):
        def handle_value(key):
            if key == 'timestamp':
                now = datetime.now()
                if self.datetime_fmt is not None:
                    return now.strftime(self.datetime_fmt)
                return now
            elif key in state['meters']:
                return state['meters'][key].value
            elif key in state:
                return state[key]
            else:
                raise KeyError("Key {} not in state dict".format(key))

        row_dict = {key: handle_value(key) for key in self.keys}
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def _teardown(self):
        if self.csv_file:
            self.csv_file.close()
        self.writer = None

    def on_train_end(self, trainer, state):
        self._teardown()

    def on_terminated(self, trainer, state):
        self._teardown()
