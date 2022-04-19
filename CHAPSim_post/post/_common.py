"""
## _common.py
A module to create base level visualisation classes 
providing functionality common to several classes
"""
import numpy as np
import matplotlib as mpl
import sys
import pyvista
import copy
import itertools
import warnings
from abc import abstractmethod, ABC, abstractproperty, abstractclassmethod

import CHAPSim_post.dtypes as cd
from CHAPSim_post import rcParams
from CHAPSim_post.utils import misc_utils

  
class classproperty():
    def __init__(self,func):
        self.f = func
    def __get__(self,obj,cls):
        return self.f(cls)

class _classfinder:
    def __init__(self,cls):
        self._cls = cls
    
    def __getattr__(self,attr):
        mro = self._cls.mro()
        
        for c in mro:
            module = sys.modules[c.__module__]
            if hasattr(c,attr):
                return getattr(self._cls,attr)
            elif hasattr(module, attr) and hasattr(c,'_module'):
                if c in mro[1:]:
                    warn_msg = (f"Attribute {attr} being inherited "
                                f"from parent class ({c.__module__}.{c.__name__})"
                                 ". This behavior may be undesired")
                    warnings.warn(warn_msg)
                    
                return getattr(module,attr)
        msg = f"Attribute {attr} was not found for class {mro[0].__name__} in module {mro[0].__module__}"
        raise ModuleNotFoundError(msg)              

class Common(ABC):

    @classproperty
    def _module(cls):
        return _classfinder(cls)
    
    @property
    def _coorddata(self):
        return self._meta_data.coord_data

    def _get_hdf_key(self,key):
        if key is None:
            key = self.__class__.__name__
        return key
    
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
    
    def copy(self):
        return copy.deepcopy(self)

class temporal_base(ABC):
    
    def phase_average(self,*others):
        # check others
        
        if not all(type(x)==type(self) for x in others):
            msg = (f"All objects to be averaged must be of type {type(self).__name}"
                    f" not {[type(x).__name__ for x in others]}")
            raise TypeError(msg)
        
        if not all(hasattr(x,'_time_shift') for x in [self,*others]):
            msg = "To use the phase averaging functionality the class variable _time_shift must be present"
            raise AttributeError(msg)
        
        self_copy = self.copy() 
        for i, other in enumerate(others):
            other_copy = other.copy()
            
            for k, v in self_copy.__dict__.items():
                if isinstance(v,temporal_base):
                    setattr(self_copy,k,v._phase_average(getattr(other_copy,k)))
                elif isinstance(v,cd.FlowStructND):
                    
                    coe1 = (i+1)/(i+2)
                    coe2 = 1/(i+2)

                    v_shifted = v.shift_times(self._time_shift)
                    
                    other_fstruct = getattr(other_copy,k)
                    fstruct_shifted = other_fstruct.shift_times(other_copy._time_shift)
                    
                    time_intersect = set(v_shifted.times).intersection(set(fstruct_shifted.times))
                    
                    for time in v_shifted.times:
                        if time not in time_intersect:
                            v_shifted.remove_time(time)
                            
                    for time in fstruct_shifted.times:
                        if time not in time_intersect:
                            fstruct_shifted.remove_time(time)
                            
                    new_v = coe1*v_shifted + coe2*fstruct_shifted
                    setattr(self_copy,k,new_v)
            
            return self_copy
                    
    @classmethod
    def _get_times_phase(cls,paths,PhyTimes=None):
        times_shifts = cls._get_times_shift(paths)
        if PhyTimes is None:
            times_list = [ set(np.array(misc_utils.time_extract(path)) + shift)\
                        for shift, path in zip(times_shifts,paths)]
            times_shifted = list(times_list[0].intersection(*times_list[1:]))
            times_shifted = np.array(times_shifted)
        else:
            times_shifted = PhyTimes    
        
        
        return [times_shifted - shift for shift in times_shifts]

    def _get_times_shift(cls,paths):
        return NotImplemented
    
    def _time_shift(self):
        return NotImplemented
    
    def _test_times_shift(self,path):
        
        time_shift1 = self._get_times_shift([path])[0]
        time_shift2 = self._time_shift
        
        if time_shift1 is NotImplemented or time_shift2 is NotImplemented:
            msg = "_get_times_shift and _time_shift methods must be implemented"
            raise NotImplementedError(msg)
        
        if time_shift1 != time_shift2:
            msg = ("methods _get_times_shift and"
                   " _time_shift must return the same value")
            raise RuntimeError(msg)
        
        
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

