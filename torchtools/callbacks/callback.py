# coding: UTF-8


class Hook(object):
    """Base class for all callbacks and meters"""
    def on_train_start(self, trainer, state):
        pass

    def on_train_end(self, trainer, state):
        pass

    def on_epoch_start(self, trainer, state):
        pass

    def on_epoch_end(self, trainer, state):
        pass

    def on_batch_start(self, trainer, state):
        pass

    def on_batch_end(self, trainer, state):
        pass

    def on_forward_end(self, trainer, state):
        pass

    def on_backward_end(self, trainer, state):
        pass

    def on_validate_start(self, trainer, state):
        pass

    def on_validate_end(self, trainer, state):
        pass

    def on_test_start(self, trainer, state):
        pass

    def on_test_end(self, trainer, state):
        pass

    def on_terminated(self, trainer, state):
        pass

    def __str__(self):
        return type(self).__name__


class Callback(Hook):
    """Base class for all callbacks."""
    def _callback_check(self, trainer):
        pass

    def _teardown(self):
        pass
