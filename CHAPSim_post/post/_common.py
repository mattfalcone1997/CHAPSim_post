"""
## _common.py
A module to create base level visualisation classes 
providing functionality common to several classes
"""
import numpy as np
import matplotlib as mpl
import sys
import pyvista

import itertools
from abc import abstractmethod, ABC, abstractproperty

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd
from CHAPSim_post.utils import indexing, misc_utils,gradient
from ._meta import CHAPSim_meta, coorddata

  
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
    def Coord_ND_DF(self):
        return self._meta_data.Coord_ND_DF

    @property
    def metaDF(self):
        return self._meta_data.metaDF

    @property
    def NCL(self):
        return self._meta_data.NCL

    @abstractproperty
    def shape(self):
        pass


class postArray(ABC):
    _type = None
    def __init__(self,*args,from_file=False,**kwargs):
        
        if self._type is None:
            msg = "This class requires a _type variable to be set"
            raise TypeError(msg)

        if from_file:
            self._hdf_extract(*args,**kwargs)
        else:
            self._dict_ini(*args,**kwargs)

    def __getitem__(self,key):
        if key not in self._data_dict.keys():
            msg = f"Key provided {key} not in {self.__class.__name__} object"
            raise KeyError(msg)
        return self._data_dict[key]

    def __setitem__(self,key,value):
        if not isinstance(value,self._type):
            msg = f"This class can only be used values of type {self._type}"
            raise TypeError(msg)
    
    @classproperty
    def _module(cls):
        return sys.modules[cls.__module__]

    @classmethod
    def from_hdf(cls,filename,key=None):
        return cls(filename,from_file=True,key=key)

    def _dict_ini(self,data_dict):
        if not all(isinstance(val,self._type) for val in data_dict.values()):
            msg = "Not all dict_values are of the correct type"
            raise TypeError(msg)

        self._data_dict = data_dict

    def save_hdf(self,file_name,mode,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,mode=mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        for data_key, data_obj in self._data_dict:
            hdf_key = "/".join([key,data_key])
            data_obj.save_hdf(file_name,mode,key=hdf_key)


    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        labels = list(hdf_obj.keys())
        hdf_keys = ["/".join([key,h5_key]) for h5_key in labels]

        avg_list = [self._type.from_hdf(file_name,key=h5_key)\
                        for h5_key in hdf_keys]
        self._data_dict = {label : avg for label, avg in zip(labels,avg_list)}

