import inspect

def get_default_parameters(function):
    sig = inspect.signature(function)

    defaults = {
        name: param.default
        for name, param in sig.parameters.items()
        if param.default is not inspect.Parameter.empty
    }

    return defaults