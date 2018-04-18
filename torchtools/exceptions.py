# coding: UTF-8


class TorchToolsException(Exception):
    """Base class for all exceptions in torchtools"""
    pass


class HookTypeError(TorchToolsException):
    """Invalid Hook type error"""
    pass


class CallbackCheckError(TorchToolsException):
    """Callback check failed error"""
    pass


class TrainerTerminatedException(TorchToolsException):
    """Trainer terminated exception"""
    pass


class EarlyStoppingException(TorchToolsException):
    """Early stopping exception"""
    pass
