import wandb
from unittest.mock import Mock
from argparse import Namespace

class MockWandb(object):
    def __init__(self, print_to_console, config):
        self.print = print_to_console
        self.config = config

    def __getattr__(self, attr):
        try:
            return super(MockWandb, self).__getattr__(attr)
        except AttributeError:
            return self.__get_global_handler(attr)

    def __get_global_handler(self, name):
        # Do anything that you need to do before simulating the method call
        handler = self.__global_handler
        #handler.name = name # Change the method's name
        if name == "config":
            return self.config
        return handler

    def __global_handler(self, *args, **kwargs):
        # Do something with these arguments
        #function_name = self.__global_handler.im_func.func_name
        if self.print:
            print(str(args))
            print(str(kwargs))


class Logger:
    def __init__(self, enabled=True, print_logs_to_console=False, config={}, **kwargs):
        self.enabled = enabled
        if self.enabled:
            wandb.init(config=config, **kwargs)

        config = Namespace(**config)
        self.mock = MockWandb(print_to_console=print_logs_to_console, config=config)

    def __call__(self, *args, **kwargs):
        if self.enabled:
            return wandb
        else:
            return self.mock