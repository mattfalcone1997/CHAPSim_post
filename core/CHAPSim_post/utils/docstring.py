"""
## docstring.py
A module to provide docstring utilities to CHAPSim_post
"""
import inspect
import functools

from matplotlib import docstring

interpd = docstring.Substitution()

docstring_sub = interpd
docstring_copy = docstring.copy


class docInherit():
    pass

def copy_fromattr(attr):
    """Copy a docstring from another source function (if present)."""
    def decorator(func):   
        @functools.wraps(func)
        def from_attr(*args,**kwargs):
            self = args[0]
            attr_func = getattr(self,attr)
            __doc__ = attr_func.__doc__

            return func(*args,**kwargs)
        return from_attr
    return decorator


