"""Plugin management system for dynamic class extension.

Provides mechanisms for loading plugins and dynamically extending core classes
with plugin functionality through class composition.
"""
import importlib
import pkgutil
from papylio.log_functions import log_all_methods

class PluginManager:
    """Manager for discovering and loading plugin modules.

    Automatically discovers and imports plugin modules from the papylio.plugins package.
    Provides methods to retrieve plugin classes that extend core classes.
    """
    def __init__(self):
        """Initialize PluginManager and discover available plugins."""
        self.plugins_module = importlib.import_module('papylio.plugins')
        self.plugin_names = [pluginname for _, pluginname, ispkg in pkgutil.walk_packages(self.plugins_module.__path__) if ispkg]
        self.plugins = [importlib.import_module('papylio.plugins.'+plugin_name) for plugin_name in self.plugin_names]

    def get_class_plugins(self, class_name):
        """Get plugin classes that extend a specific core class.

        Parameters
        ----------
        class_name : str
            Name of the core class to find plugins for

        Returns
        -------
        tuple
            Tuple of plugin classes extending the specified class
        """
        return tuple([getattr(plugin, class_name) for plugin in self.plugins if hasattr(plugin, class_name)])


def plugins(cls):
    """Decorator that extends a class with compatible plugin classes.

    Creates a new class that mixes in plugin classes with the decorated class.
    Plugin classes have priority in method resolution order.

    Initial comments:
    The function is ment as a decorator for the main classes and makes sure that for the class it is applied to a new class is created with the same name.
    This new class mixes-in the plugin classes with the old class, where the plugin classes have priority and thus more or less inherit from the old class.

    Parameters
    ----------
    cls : type
        The class to decorate

    Returns
    -------
    type
        The adapted class with the same name as the input class, however now with the plugin classes mixed in.
        New class with same name but extended with compatible plugin classes
    """
    classes = PluginManager().get_class_plugins(cls.__name__) + (cls,)
    if cls.__name__ == 'File':
        classes = tuple(log_all_methods(c) for c in classes)

    slots = ()
    try:
        for c in classes:
            if type(c.slots) is tuple:
                slots += c.slots
            else:
                slots += (c.slots,)
    except AttributeError:
        pass

    # For using multiprocessing.Pool add '__module__': classes[-1].__module__}
    # return type(cls.__name__, classes, {'__slots__': slots})
    return type(cls.__name__, classes, {'__slots__': slots, '__module__': classes[-1].__module__})



# class PluginMetaClass(type):
#     def __new__(cls, clsname, bases_base, attrs):
#         # bases_base = tuple(base for base in bases if not base.__name__ is clsname)
#         attrs_base = attrs.copy()
#         attrs_base.pop('__qualname__')
#         attrs_base.pop('__module__')
#         cls_base = type(clsname+'_base', bases_base, attrs_base)
#         #cls_base = type(clsname, bases_base, attrs)
#         added_bases = PluginManager().get_class_plugins(clsname)
#         bases_main = added_bases + (cls_base,)
#         return super().__new__(cls, clsname, bases_main, {'__module__': attrs['__module__']})
#
#         # bases_base = tuple(base for base in bases if not base.__name__ is clsname)
#         # attrs.pop('__qualname__')
#         # cls_base = type(clsname + '_base', bases_base, attrs)
#         # bases_main = tuple(base for base in bases if base.__name__ is clsname) + (cls_base,)
#         # return super().__new__(cls, clsname, bases_main, {})
#
#
#
# class B:
#     def __init__(self):
#         print('Badd')
#         super().__init__()
#
#
#
# class A:
#     def __init__(self):
#         print('A')
#
#
# def test(c):
#     return type(c.__name__, (c,B),{})
#
#
# @test
# class Bo(A):
#     def __init__(self):
#         print('Bo')
#         super().__init__()
