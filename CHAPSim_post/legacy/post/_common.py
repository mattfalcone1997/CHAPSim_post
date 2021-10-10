"""
## _common.py
A module to create base level visualisation classes 
providing functionality common to several classes
"""

from abc import ABC, abstractproperty

import sys
class classproperty():
    def __init__(self,func):
        self.f = func
    def __get__(self,obj,cls):
        return self.f(cls)
    

class Common(ABC):

    @classproperty
    def _module(cls):
        return sys.modules[cls.__module__]
    @property
    def _coorddata(self):
        return self._meta_data.coord_data

    # @_coorddata.setter
    # def _coorddata(self,value):
    #     if isinstance(value,coorddata):
    #         self._meta_data._coorddata = value
    #     else:
    #         msg = "This value can only be set with an object of type coorddata"
    #         raise TypeError(msg)

    @property
    def Domain(self):
        return self._meta_data.coord_data._domain_handler

    @property
    def CoordDF(self):
        return self._meta_data.CoordDF

    @property
    def metaDF(self):
        return self._meta_data.metaDF

    @property
    def NCL(self):
        return self._meta_data.NCL

    @abstractproperty
    def shape(self):
        pass