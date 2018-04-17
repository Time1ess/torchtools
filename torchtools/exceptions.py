# coding: UTF-8


class TorchToolsException(Exception):
    pass


class CallbackTypeError(TorchToolsException):
    pass


class CallbackCheckError(TorchToolsException):
    pass


class TrainerTerminated(TorchToolsException):
    pass


class MeterNotFoundError(TorchToolsException):
    pass


class MeterNoValueError(TorchToolsException):
    pass


class LogTypeError(TorchToolsException):
    pass


class EarlyStoppingError(TorchToolsException):
    pass
