import warnings


message = 'You are using a deprecated package, which will be removed soon.'
warnings.simplefilter('always', DeprecationWarning)
warnings.warn(message, DeprecationWarning)
