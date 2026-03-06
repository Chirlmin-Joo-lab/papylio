"""Utility decorators for Papylio functions."""
import sys

def return_none_when_executed_by_pycharm(func):
    """Decorator to return None when function is executed by PyCharm's import tip generator.

    This is useful for avoiding issues with PyCharm's automatic import suggestions
    which may trigger unintended function execution or side effects.

    Parameters
    ----------
    func : callable
        Function to decorate

    Returns
    -------
    callable
        Wrapped function that returns None if called from PyCharm's import tip generator,
        otherwise returns the normal function result
    """
    def wrapper(*args, **kwargs):
        # print(sys._getframe(1))
        if sys._getframe(1).f_code.co_name=='generate_imports_tip_for_module':
            # print('return None')
            return None
        else:
            # print('return normal')
            return func(*args, **kwargs)
    return wrapper