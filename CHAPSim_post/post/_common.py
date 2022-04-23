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
from functools import wraps
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

def require_override(func):
    
    @wraps(func)
    def _return_func(*args,**kwargs):
       msg = (f"Method {__name__} must be overriden to be used. There is no "
              "need to override it if this function is not called")
       raise NotImplementedError(msg)
   
    return _return_func
class temporal_base(ABC):
    
    @classmethod
    def phase_average(cls,*objects_temp,items=None):
        # check others
        
        if not all(type(x)==cls for x in objects_temp):
            msg = (f"All objects to be averaged must be of type {cls.__name.__}"
                    f" not {[type(x).__name__ for x in objects_temp]}")
            raise TypeError(msg)
        
        if not all(hasattr(x,'_time_shift') for x in objects_temp):
            msg = "To use the phase averaging functionality the class variable _time_shift must be present"
            raise AttributeError(msg)
        
        if items is not None:
            if len(items) != len(objects_temp):
                msg = ("If items is present, it must be the same"
                       " length as the inputs to be phased averaged")
                raise ValueError(msg)
            items = np.array(items)
        else:
            items = np.ones(len(objects_temp))
            
            
        object_attrs = objects_temp[0].__dict__.keys()
        starter_obj = objects_temp[0].copy()
        
        for attr in object_attrs:
            
            vals = [getattr(ob,attr) for ob in objects_temp]
            val_type = type(vals[0])
            
            if not all(type(val) == val_type for val in vals):
                msg = ("Not all attributes of object "
                       "to be phased averaged are of the same type")
                raise TypeError(msg)
            
            if issubclass(val_type,temporal_base):
                setattr(starter_obj,
                        attr,
                        val_type.phase_average(*vals,
                                               items=items))
            elif issubclass(val_type,cd.FlowStructND):
                
                time_shifts = [x._time_shift for x in objects_temp]
                vals = [val.copy().shift_times(shift) \
                            for val,shift in zip(vals,time_shifts)]
                
                times_list = [set(val.times) for val in vals]
                intersect_times = sorted(set.intersection(*times_list))
                
                for val in vals:
                    for time in val.times:
                        if time not in intersect_times:
                            val.remove_time(time)
                
                coeffs = items/np.sum(items)
                phase_val = sum(coeffs*vals)
                
                setattr(starter_obj,attr,phase_val)
                
                    
        return starter_obj
            
            
            
        # for i, other in enumerate(objects_temp[1:]):
        #     other_copy = other.copy()
            
        #     for k, v in object_copy_ini.__dict__.items():
                
        #         items_sub1 = sum(items[:i+1])
        #         items_sub2
                
        #         if isinstance(v,temporal_base):
        #             sub_items = []
        #             setattr(object_copy_ini,k,v.phase_average(v,getattr(other_copy,k),items=items))
        #         elif isinstance(v,cd.FlowStructND):
                    
        #             if items:
        #                 coe1 = sum(items[:i+1])/sum(items[i+2])
        #                 coe2 = items[i+1]/sum(items[i+2])
        #             else:
        #                 coe1 = (i+1)/(i+2)
        #                 coe2 = 1/(i+2)

        #             v_shifted = v.shift_times(self._time_shift)
                    
        #             other_fstruct = getattr(other_copy,k)
        #             fstruct_shifted = other_fstruct.shift_times(other_copy._time_shift)
                    
        #             time_intersect = set(v_shifted.times).intersection(set(fstruct_shifted.times))

        #             for time in v_shifted.times:
        #                 if time not in time_intersect:
        #                     v_shifted.remove_time(time)
                            
        #             for time in fstruct_shifted.times:
        #                 if time not in time_intersect:
        #                     fstruct_shifted.remove_time(time)
                            
        #             new_v = coe1*v_shifted + coe2*fstruct_shifted
        #             setattr(self_copy,k,new_v)
            
        #     return self_copy
        
    def _del_times(self,times):
        for  v in self.__dict__.values():
            if isinstance(v,cd.FlowStructND):
                for time in times:
                    v.remove_time(time)
                    
    def _shift_times(self,time):
        for  k, v in self.__dict__.items():
            if isinstance(v,cd.FlowStructND):
                v.shift_times(time)
                setattr(self,k,v.shift_times(time))
                
                
            
    @classmethod
    def _get_times_phase(cls,paths,PhyTimes=None):
        times_shifts = [cls._get_time_shift(path) for path in paths]
        
        if PhyTimes is None:
            times_list = [ np.array(misc_utils.time_extract(path)) + shift\
                        for shift, path in zip(times_shifts,paths)]
            print([max(time) for time in times_list])
            
            times_shifted = cls._get_intersect(paths[0],times_list)
            
        else:
            times_shifted = PhyTimes    
                
        return [times_shifted - shift for shift in times_shifts]
    
    @classmethod
    def _get_intersect(cls,path,times_list):
        meta_data = cls._module._meta_class(path,tgpost=True)
        DT = meta_data.metaDF['DT']
        
        _intersect = lambda times: [any(np.isclose(x,times,atol=DT)) for x in times_list[0]]

        intersection = np.array( [_intersect(times) for times in times_list[1:]]).all(axis=0)
        print(times_list[0][~intersection])
        print(times_list[1:])
        return times_list[0][intersection]
        
            
    @require_override
    def _get_time_shift(cls,path):
        pass
    
    @require_override
    def _time_shift(self):
        pass
            
    def _test_times_shift(self,path):
        
        time_shift1 = self._get_time_shift(path)
        time_shift2 = self._time_shift
        
        if time_shift1 != time_shift2:
            msg = ("methods _get_times_shift and"
                   " _time_shift must return the same value."
                   f" Current_values: {time_shift1} {time_shift2}")
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

