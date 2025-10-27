import inspect
import json
import functools
from datetime import datetime
import papylio

def function_arguments(function, function_locals):
    signature_values = inspect.signature(function).parameters.values()
    all_argument_values = {
        parameter.name:
            function_locals[parameter.name] for parameter
        in signature_values if parameter.name is not 'self' and parameter.name in function_locals
    }

    if list(all_argument_values.keys())==['configuration']:
        all_argument_values = all_argument_values['configuration']

    return all_argument_values

def function_arguments_json(function, function_locals):
    return json.dumps(function_arguments(function, function_locals))

def get_current_datetime():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d %H:%M:%S")

def add_configuration_to_dataarray(dataarray, function=None, function_locals=None, units=None):
    dataarray.attrs['version'] = papylio.__version__
    if function is not None:
        dataarray.attrs['configuration'] = function_arguments_json(function, function_locals)
    dataarray.attrs['datetime'] = get_current_datetime()
    if units is not None:
        dataarray.attrs['units'] = units
    return dataarray

# def log_call(method):
#     """Decorator that logs a method call with its arguments using self.logger."""
#     @functools.wraps(method)
#     def wrapper(self, *args, **kwargs):
#         if hasattr(self, "logger"):
#             # Format args/kwargs into a readable string
#             sig = inspect.signature(method)
#             bound = sig.bind(self, *args, **kwargs)
#             bound.apply_defaults()
#             # Drop `self` for clarity
#             call_args = ", ".join(f"{k}={v!r}" for k, v in list(bound.arguments.items())[1:])
#             self.logger.info(f"Called {method.__name__}({call_args})")
#         return method(self, *args, **kwargs)
#     return wrapper
#
# def log_call(method):
#     """Log only if the method was called from outside the class."""
#     @functools.wraps(method)
#     def wrapper(self, *args, **kwargs):
#         # Get the call stack
#         stack = inspect.stack()
#
#         # Caller is the frame one level above this decorator
#         # We'll look a few levels up to find the first non-File caller
#         called_from_inside = False
#         for frame_info in stack[1:]:
#             caller_locals = frame_info.frame.f_locals
#             # If 'self' exists and is the same instance, it's an internal call
#             if caller_locals.get("self") is self:
#                 called_from_inside = True
#                 break
#
#         # Log only if called externally
#         if not called_from_inside and hasattr(self, "logger"):
#             arg_str = ", ".join(
#                 [f"{k}={v!r}" for k, v in {**dict(zip(method.__code__.co_varnames[1:], args)), **kwargs}.items()]
#             )
#             self.logger.info(f"Called {method.__name__}({arg_str})")
#
#         return method(self, *args, **kwargs)
#     return wrapper

def log_call(method):
    """Log external calls first, then log exceptions if they occur."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        # Determine if this is an external call
        stack = inspect.stack()
        called_from_inside = any(frame.frame.f_locals.get("self") is self for frame in stack[1:])

        # Prepare argument string
        arg_str = ", ".join(
            f"{k}={v!r}" for k, v in {**dict(zip(method.__code__.co_varnames[1:], args)), **kwargs}.items()
        )

        # 1️⃣ Log method call first if external
        if not called_from_inside and hasattr(self, "logger"):
            self.logger.info(f"Called {method.__name__}({arg_str})")

        # 2️⃣ Execute the method and catch exceptions
        try:
            return method(self, *args, **kwargs)
        except Exception:
            if hasattr(self, "logger"):
                self.logger.exception(f"Exception in {method.__name__}({arg_str})")
            raise  # re-raise after logging
    return wrapper

# import time
# def log_call(method):
#     @functools.wraps(method)
#     def wrapper(self, *args, **kwargs):
#         start = time.time()
#         result = method(self, *args, **kwargs)
#         elapsed = time.time() - start
#         if hasattr(self, "logger"):
#             self.logger.info(f"{method.__name__} executed in {elapsed:.3f}s")
#         return result
#     return wrapper

#
# def log_all_methods(cls):
#     """Class decorator that applies @log_call to all public methods."""
#     for name, method in cls.__dict__.items():
#         if callable(method) and not name.startswith("_"):
#             setattr(cls, name, log_call(method))
#     return cls

def log_all_methods(cls):
    """Decorates all public methods and property setters/getters of a class."""
    for name, attr in cls.__dict__.items():
        if name.startswith("_"):
            continue  # skip private

        # Regular callable method
        if callable(attr):
            setattr(cls, name, log_call(attr))

        # Property
        elif isinstance(attr, property):
            fs = []
            for x in ['fget', 'fset', 'fdel']:
                f = getattr(attr, x)
                if f:
                    f.__name__ = name  # Otherwise it would print "wrapper()" due to the return_none_when_executed_by_pycharm decorator
                    if x == 'fget':
                        fs.append(f)
                    else:
                        fs.append(log_call(f))
                else:
                    fs.append(None)

            setattr(cls, name, property(*fs, attr.__doc__))
            # # Wrap getter
            # fget = log_call(attr.fget) if attr.fget else None
            # # Wrap setter
            # fset = log_call(attr.fset) if attr.fset else None
            # # Wrap deleter
            # fdel = log_call(attr.fdel) if attr.fdel else None
            # # Replace property with wrapped one
            # setattr(cls, name, property(fget, fset, fdel, attr.__doc__))

    return cls