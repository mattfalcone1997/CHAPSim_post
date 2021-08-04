"""
## docstring.py
A module to provide docstring utilities to CHAPSim_post 
to avoid duplication
"""
import inspect
from functools import wraps

import numpy as np
import CHAPSim_post.plot as cplt

from matplotlib import docstring

handle = docstring.Substitution()

sub = handle
copy = docstring.copy

short_name={
    "ndarray": np.ndarray.__name__,
    "ax" : cplt.AxesCHAPSim.__name__,
    "fig" : cplt.CHAPSimFigure.__name__
}
handle.update(**short_name)

def inherit(method):

    @wraps(method)
    def func(self,*args,**kwargs):
        for parent in self.__class__.__mro__[1:]:
            source = getattr(parent,method.__name__,None)
            if source is not None:
                __doc__ = source.__doc__ ; break

        return method(self,*args,**kwargs)

    return func
                


# inherit = docInherit

def copy_fromattr(attr):
    """Copy a docstring from another source function (if present)."""
    def decorator(func):   
        @wraps(func)
        def from_attr(*args,**kwargs):
            self = args[0]
            attr_func = getattr(self,attr)
            __doc__ = attr_func.__doc__

            return func(*args,**kwargs)
        return from_attr
    return decorator


