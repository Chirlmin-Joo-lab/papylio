import sys

def return_none_when_executed_by_pycharm(func):
    def wrapper(*args, **kwargs):
        # print(sys._getframe(1))
        if sys._getframe(1).f_code.co_name=='generate_imports_tip_for_module':
            # print('return None')
            return None
        else:
            # print('return normal')
            return func(*args, **kwargs)
    return wrapper