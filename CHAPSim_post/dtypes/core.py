"""
# CHAPSim_dtypes
A module for the CHAPSim_post postprocessing and visualisation library. This 
experimental library contains additional classes to store data from the module.
The data types are built from the pandas DataFrame and are designed to superseed 
them for CHAPSim_post to enable some additional high level functionality to the
use and the other modules to allow data to be automatically reshaped when the 
__getitem__ method is used
"""

import itertools
import warnings
import numpy as np
import h5py
import h5py
import numbers
import os
import operator
import weakref
import CHAPSim_post as cp
import copy

from pandas._libs.index import ObjectEngine
import pandas as pd
from pandas.core.indexes.multi import MultiIndexPyIntEngine, MultiIndexUIntEngine
from abc import abstractmethod, abstractproperty, ABC
from .io import hdfHandler
# from .coords import AxisData

class IndexBase(ABC):
    @abstractproperty
    def is_MultiIndex(self):
        pass

    @abstractproperty
    def _mapping(self):
        pass
    
    def get_loc(self,key):
        return self._mapping.get_loc(key)

    @classmethod
    def _item_handler(cls,item):
        
        if isinstance(item,tuple):
            return tuple(cls._item_handler(k) for k in item)
        if isinstance(item,numbers.Number):
            return "%.9g"%item
        elif isinstance(item,str):
            return "%.9g"%float(item) if item.isnumeric() else item
        else:
            return str(item)

    @abstractmethod
    def to_array(self,string=True):
        pass
    
    def __contains__(self,key):
        key = self._item_handler(key)
        return key in self._index

    def remove(self,key):
        if key not in self:
            msg = f"Key {key} not present in indexer" 
            raise KeyError(msg)

        self._index.remove(key)
        self._update_internals()

    @abstractmethod
    def get_inner_index(self):
        pass
    
    @abstractmethod
    def _update_internals(self):
        pass

    def update_key(self,old_key,new_key):
        old_key = self._item_handler(old_key)
        new_key = self._item_handler(new_key)

        if old_key not in self:
            msg = f"The key {old_key} must be an existing key in the indexer"
            raise KeyError(msg)

        for i, x in enumerate(self._index):
            if old_key == x:
                self._index[i] = new_key
                break

        self._update_internals()

    def get_index(self):
        return self._index

    def __iter__(self):
        for index in self._index:
            yield index 
    
    def __len__(self):
        return len(self._index)

    def __getitem__(self,key):
        return self._index[key]

    def __repr__(self):
        name = self.__class__.__name__
        return "%s(%s)"%(name,self._index.__str__())

    def __str__(self):
        return self._index.__str__()

    def __setitem__(self,key,value):
        self._index[key] = value

    def __eq__(self,other_index):
        return self._index == other_index._index

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self,memo):
        return self.__class__(self._index)

    @staticmethod
    def is_listkey(key):
        if isinstance(key,tuple):
            if any([isinstance(k,(list,Index)) for k in key]):
                return True
        elif isinstance(key,(list,Index)):
            return True
        return False

    @staticmethod
    def is_multikey(key):
        if isinstance(key,tuple):
            if len(key) > 1:
                return True
        return False

    @staticmethod
    def _getitem_process_list(key):
        if isinstance(key,tuple):
            if not isinstance(key[0],(list,Index)):
                inner_key = [key[0]]
            else:
                inner_key = list(key[0])
            if not isinstance(key[1],(list,Index)):
                outer_key = [key[1]]
            else:
                outer_key = list(key[1])

            key_list = list(itertools.product(inner_key,outer_key))
        else:
            if not isinstance(key,list):
                msg = "This function should only be called on keys containing lists"
                raise TypeError(msg)
            key_list = key

        return key_list

    def check_inner(self,key,err):
        if key not in self.get_inner_index():
            raise err from None
        
        return key

    def _check_indices(self,indices):
        indices = list(indices)
        
        def _is_unique(lst):
            seen = set()
            return  not any(i in seen or seen.add(i) for i in lst)
        
        if not _is_unique(indices) and len(indices) > 1:
            msg = "Indices must be unique"
            raise ValueError(msg)

        return indices

    def extend(self,other_index):
        if type(other_index) != self.__class__:
            msg = "The type of merging index must be the same as the current index"
            raise TypeError(msg)

        self._index.extend([self._item_handler(key) for key in other_index._index])
        self._update_internals()

    def append(self,key):
        key = self._item_handler(key)
        self._index.append(key)
        self._update_internals()

    def __getstate__(self):
        return self._index

    def __setstate__(self,d):
        obj = self.__class__(d)
        self.__dict__ = obj.__dict__

class Index(IndexBase):
    def __init__(self,indices):

        if not all(isinstance(i,str) for i in indices):
            msg = "All elements of indices must be strings"
            raise TypeError(msg)

        self._index = self._check_indices(indices)

        self._update_internals()

    def to_array(self,string=True):
        type = np.string_ if string else object
        array = []
        for index in self:
            array.append(type(index))
        return np.array(array)

    def get_inner_index(self):
        return self.get_index()

    @property
    def _mapping(self):
        return self.__engine

    @property
    def is_MultiIndex(self):
        return False

    def _update_internals(self):
        values = np.array(self._index,dtype=object)
        if pd.__version__ < '1.4': 
            self.__engine = ObjectEngine(lambda: values,len(values))
        else:
            self.__engine = ObjectEngine(values)
            
class MultiIndex(IndexBase):
    def __init__(self,indices):
        if not all(isinstance(index,tuple) for index in indices):
            msg = "All indices must be tuples"
            raise TypeError(msg)
        
        if not all(all(isinstance(i,str) for i in index) for index in indices):
            msg = "All elements for the tuples must be strings"
            raise TypeError(msg)

        self._index = self._check_indices(indices)
    
        self._update_internals()

    def _update_internals(self):
        self._outer_index = Index(set([x[0] for x in self.get_index()]))
        self._inner_index = Index(set([x[1] for x in self.get_index()]))

        self.__engine = self._create_mapping() 
    
    def update_inner_key(self,old_key,new_key):
        old_key = self._item_handler(old_key)
        new_key = self._item_handler(new_key)

        for x in self._index:
            if x[1] == old_key:
                new_total_key = (x[0],new_key)
                self.update_key(x,new_total_key)

        self._update_internals()

    def update_outer_key(self,old_key,new_key):
        old_key = self._item_handler(old_key)
        new_key = self._item_handler(new_key)
        for x in self._index:
            if x[0] == old_key:
                new_total_key = (new_key,x[1])
                self.update_key(x,new_total_key)

        self._update_internals()

    def __contains__(self, key):
        levels = self._levels
        return key[0] in levels[0] and key[1] in levels[1]

    @property
    def _levels(self):
        return [Index(self.get_outer_index()),Index(self.get_inner_index())]

    @property
    def _codes(self):
        outer_level = self._levels[0]
        inner_level = self._levels[1]

        outer_code_map = dict(zip(outer_level,range(len(outer_level))))
        inner_code_map = dict(zip(inner_level,range(len(inner_level))))

        inner_index = [x[1] for x in self.get_index()]
        outer_index = [x[0] for x in self.get_index()]
        
        outer_code = [outer_code_map[x] for x in outer_index]
        inner_code = [inner_code_map[x] for x in inner_index]

        return [outer_code, inner_code]

    @property
    def outer_index(self):
        return self._outer_index

    @property
    def inner_index(self):
        return self._inner_index

    def to_array(self,string=True):
        type = np.string_ if string else object
        
        array = []
        for index in self:
            array.append(np.array(index,dtype=type))
        return np.array(array)

    def check_outer(self,key,err,warn=None):
        key = self._item_handler(key)

        # outer_index = self.get_outer_index()
        if key not in self.get_outer_index(): 
            if len(self.get_outer_index()) > 1:
                raise err from None
            else:
                if key != 'None' and warn is not None:
                    warnings.warn(warn,stacklevel=4)
                key = self.get_outer_index()[0]

        return key

    def get_inner_index(self):
        return self._inner_index
    
    def get_outer_index(self):
        return self._outer_index

    def _create_mapping(self):
        sizes = np.ceil(np.log2([len(level) + 1 for level in self._levels]))

        lev_bits = np.cumsum(sizes[::-1])[::-1]
        offsets = np.concatenate([lev_bits[1:], [0]]).astype("uint64")

        if lev_bits[0] > 64:
            return MultiIndexPyIntEngine(self._levels, self._codes, offsets)
        return MultiIndexUIntEngine(self._levels, self._codes, offsets)
        
    @property
    def _mapping(self):
        return self.__engine

    @property
    def is_MultiIndex(self):
        return True

class structIndexer:

    def __new__(cls,index):
        index = cls.index_constructor(index)
        if all(isinstance(ind,tuple) for ind in index):
            return MultiIndex(index)
        else:
            return Index(index)

    @staticmethod
    def _construct_arrays_check(index):
        if len(index) != 2:
            return False

        if not all(isinstance(ind,(list,Index)) for ind in index):
            return False

        if len(index[0]) != len(index[1]):
            msg = "Invalid iterable used for indexing"
            raise TypeError(msg)
        
        return True

    @staticmethod
    def _construct_tuples_check(index):
        if not all(isinstance(ind,tuple) for ind in index):
            return False

        if not all(len(x) == 2 for x in index):
            msg = "The length of each tuple must be 2"
            raise ValueError(msg)

        return True

    @staticmethod
    def _construct_1D_check(index):
        return all(isinstance(ind,(numbers.Number,str)) for ind in index)
    
    @classmethod
    def index_constructor(cls,index):
        # two options for index construction
        # list with two list => [outer_index, inner_index]
        # list of tuples => each tuple an index in the data struct
        if cls._construct_arrays_check(index):

            outer_index = list(IndexBase._item_handler(k) for k in index[0] ) 
            inner_index = list(IndexBase._item_handler(k) for k in index[1] ) 
            index = list(zip(outer_index,inner_index))


        elif cls._construct_tuples_check(index):
            inner_index = list(IndexBase._item_handler(k[1]) for k in index )
            outer_index = list(IndexBase._item_handler(k[0]) for k in index )
            index = list(zip(outer_index,inner_index))

        elif cls._construct_1D_check(index):
            index = list(IndexBase._item_handler(k) for k in index )
        else:
            msg = "The index passed is invalid"
            raise ValueError(msg)

        return index


class datastruct:

    def __init__(self,*args,from_hdf=False,**kwargs):

        from_array=False; from_dict = False; from_dstruct=False

        if isinstance(args[0],np.ndarray):
            from_array=True

        elif isinstance(args[0],dict):
            from_dict=True

        elif isinstance(args[0],datastruct):
            from_dstruct=True
            
        elif not from_hdf:
            msg = (f"No valid initialisation method for the {datastruct.__name__}"
                    " type has been found from arguments")
            raise ValueError(msg)

        if from_array:
            self._array_ini(*args,**kwargs)

        elif from_dict:
            self._dict_ini(*args,**kwargs)

        elif from_dstruct:
            self._dstruct_ini(*args,**kwargs)

        elif from_hdf:
            self._file_extract(*args,**kwargs)
    
    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(*args,from_hdf=True,**kwargs)

    def from_internal(self,*args,**kwargs):
        return self.__class__(*args,**kwargs)

    @classmethod
    def from_concat(cls,struct_list):
        if not all(isinstance(struct,cls) for struct in struct_list):
            msg = f"All list elements must be isinstancces of the class {cls.__name__}"
            raise TypeError
        
        dstruct = struct_list[0].copy()
        for struct in struct_list[1:]:
            dstruct.concat(struct)

        return dstruct
    def _check_array_index(self,array,index):
        if len(array) != len(index):
            msg = "The length of the input array must match the index"
            raise ValueError(msg)

    def _array_ini(self,array,index=None,dtype=None,copy=False):
        if index is None:
            index = range(len(array))
        
        self._indexer = structIndexer(index)
        self._check_array_index(array,self._indexer)

        self._data = list(self._get_data(array,copy,dtype))

    def _dict_ini(self,dict_data,dtype=None,copy=False):
        if not all([isinstance(val,np.ndarray) for val in dict_data.values()]):
            msg = "The type of the values of the dictionary must be a numpy array"
            raise TypeError(msg)
        
        index = list(dict_data.keys())
        self._data = [self._get_data(data,copy,dtype) for data in dict_data.values()]

        self._indexer = structIndexer(index)

    def _dstruct_ini(self,dstruct,copy=False):
        return self._dict_ini(dstruct.to_dict(),copy=copy)
    
    def _get_dtype(self,data,dtype=None):
        if dtype is None:
            if issubclass(data.dtype.type,np.floating):
                dtype = cp.rcParams['dtype']
            else:
                dtype = data.dtype.type
        else:
            if isinstance(dtype,str):
                dtype = np.dtype(str)
            
            super_dtype = data.dtype.type.mro()[1]
            if not issubclass(dtype,super_dtype):
                msg = (f"Cannot set the dtype to {dtype.__name__}. "
                       f"Must be subclass of {super_dtype.__name__}")
                raise TypeError(msg)         
        return dtype
    
    def _get_data(self,data,copy=False,dtype=None):
        dtype = self._get_dtype(data,dtype=dtype)         
        
        return data.astype(dtype,copy=copy)
        
    def _file_extract(self,filename,key=None):
        hdf_obj = hdfHandler(filename,mode='r',key=key)

        hdf_obj.check_type_id(self.__class__)
        data_array = list(self._get_data(hdf_obj['data'][:],copy=True))
        index_array = hdf_obj.contruct_index_from_hdfkey('index')
        shapes_array = hdf_obj['shapes'][:]
        
        self._data = [None for _ in range(len(data_array))] 
        for i, (data,shape) in enumerate(zip(data_array,shapes_array)):
            self._data[i] = data[~np.isnan(data)].reshape(shape).squeeze()

        self._indexer = structIndexer(index_array)
        return hdf_obj

    def to_hdf(self,filepath,mode='a',key=None):
        hdf_obj =hdfHandler(filepath,mode=mode,key=key)

        hdf_array = self._construct_data_array()
        hdf_shapes = self._construct_shapes_array()
        hdf_indices = self._indexer.to_array(string=True)

        hdf_obj.create_dataset('data',data=hdf_array)
        hdf_obj.create_dataset('shapes',data=hdf_shapes)
        hdf_obj.create_dataset('index',data=hdf_indices)

        return hdf_obj

    def to_dict(self):
        return dict(self)
    
    def __mathandle__(self):
        out = dict()
        if self._is_multidim():
            for i in self.inner_index:
                full_array = np.stack([ self[t,i] for t in self.outer_index ],axis=0)
                out[i] = full_array
            
            out['times'] = np.array(self.outer_index)
        else:
            for i in self.inner_index:
                out[i] = self[i]
        return out
                
            
            
        
    def _construct_shapes_array(self):
        shapes =  [x.shape for _,x in self]
        max_dim = max([len(x) for x in shapes])
        for i,shape in enumerate(shapes):
            shape_mismatch = max_dim - len(shape)
            if shape_mismatch != 0:
                assert shape_mismatch >= 0 
                extra_shape = [1]*shape_mismatch
                shapes[i] = [*shape,*extra_shape]
        return np.array(shapes)

    def _construct_data_array(self):
        array = [x.flatten() for _,x in self]
        sizes = [x.size for x in array]
        max_size = max(sizes)
        outer_size = len(array)

        dtype = self._get_dtype(array[0])
        data_array = np.full((outer_size,max_size),np.nan,dtype=dtype)
        for i, arr in enumerate(array):
            data_array[i,:arr.size] = arr

        return data_array

    def _is_multidim(self):
        return self._indexer.is_MultiIndex

    def equals(self,other_datastruct):
        if not isinstance(other_datastruct,datastruct):
            msg = "other_datastruct must be of type datastruct"
            raise TypeError(msg)

        for key,val1 in self:
            if key not in other_datastruct.index:
                return False
            if not np.allclose(val1,other_datastruct[key]):
                return False
        return True

    @property
    def index(self):
        return self._indexer

    @property
    def outer_index(self):
        return self._indexer.get_outer_index()

    @property
    def inner_index(self):
        return self._indexer.get_inner_index()

    @property
    def values(self):
        shape_list = [x.shape for x in self._data]
        if not all(x==shape_list[0] for x in shape_list):
            msg = "To use this function all the arrays in the datastruct must be the same shape"
            raise AttributeError(msg)

        return np.stack(self._data,axis=0)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__str__()
    
    def get_key(self, key):
        loc = self._indexer.get_loc(key)
        return self._data[loc]

    def __getitem__(self,key):
        if self._indexer.is_listkey(key):
            return self._getitem_process_list(key)
        elif self._indexer.is_multikey(key):
            return self._getitem_process_multikey(key)
        else:
            return self._getitem_process_singlekey(key)


    def _getitem_process_multikey(self,key):
        if not self._indexer.is_MultiIndex:
            msg = "A multidimensional index passed but a single dimensional datastruct"
            raise KeyError(msg)

        key = self._indexer._item_handler(key)

        try:
            return self.get_key(key)
        except KeyError:
            err = KeyError((f"The key provided ({key}) to the datastruct is"
                            " not present and cannot be corrected internally."))
            warn = UserWarning((f"The outer index provided is incorrect ({key[0]})"
                        f" that is present (there is only one value present in the"
                        f" datastruct ({self._indexer.get_outer_index()[0]})"))
            
            inner_key = self.check_inner(key[1],err)
            outer_key = self.check_outer(key[0],err,warn)

            return self.get_key((outer_key, inner_key))

    def _getitem_process_singlekey(self,key):
        key = self._indexer._item_handler(key)

        err_msg = (f"The key provided ({key}) to ""the datastruct is "
                    "not present and cannot be corrected internally.")
        err = KeyError(err_msg)

        try:
            return self.get_key(key)
        except KeyError:
            if self.index.is_MultiIndex:
                outer_key = self.check_outer(None,err)
                return self.get_key((outer_key,key))
            else:
                raise KeyError(err_msg) from None

    def _getitem_process_list(self,key):
        key_list = self._indexer._getitem_process_list(key)

        struct_dict = {k : self[k] for k in key_list}
        return self.from_internal(struct_dict)

    def check_outer(self,key,err,warn=None):
        return self._indexer.check_outer(key,err,warn=warn)

    def check_inner(self,key,err):
        return self._indexer.check_inner(key,err)
    
    def __setitem__(self,key,value):
        if not isinstance(value,np.ndarray):
            msg = f"The input array must be an instance of {np.ndarray.__name__}"
            raise TypeError(msg)

        self.set_value(key,value)
    
    def set_value(self,key, value):
        key = self._indexer._item_handler(key)
        if key in self.index:
            
            loc = self._indexer.get_loc(key)
            self._data[loc] = value
        else:
            self.create_new_item(key,value)

    def create_new_item(self,key,value):
        if key in self.index:
            msg = "Cannot create new key, the key is already present"
            raise KeyError(msg)
        
        if self._indexer.is_multikey(key) and not self._indexer.is_MultiIndex:
            msg = "Multi-key given, but the indexer is not a multiindex"
            raise TypeError(msg)

        if not self._indexer.is_multikey(key) and self._indexer.is_MultiIndex:
            msg = "single-key given, but the indexer is a multiindex"
            raise TypeError(msg)

        self._data.append(value)
        self._indexer.append(key)
                
    def __delitem__(self,key):
        print(key)
        print(self._is_multidim(),not self._indexer.is_multikey(key))
        
        if self._is_multidim() and not self._indexer.is_multikey(key):
            self.delete_inner_key(key)
            return
        
        if not key in self.index:
            raise KeyError(f"Key {key} not present in "
                           f"{self.__class__.__name__}")
        key = self._indexer._item_handler(key)
        loc = self._indexer.get_loc(key)

        self._data.pop(loc)
        self._indexer.remove(key)

    def delete_inner_key(self,inner_key):
        if not inner_key in self.inner_index:
            msg = "Only inner keys in the inner index can be removed"
            raise KeyError(msg)
        outer_indices =  self.outer_index
        for outer in outer_indices:
            del self[outer,inner_key]

    def delete_outer_key(self,outer_key):
        if not outer_key in self.outer_index:
            msg = "Only outer keys in the outer index can be removed"
            raise KeyError(msg)
        inner_indices =  self.inner_index
        for inner in inner_indices:
            del self[outer_key,inner]

    def __iter__(self):
        for key, val in zip(self._indexer,self._data):
            yield (key, val)

    def iterref(self):
        return zip(self._indexer,self._data)

    def concat(self,arr_or_data):
        msg = f"`arr_or_data' must be of type {self.__class__.__name__} or an iterable of it"
        
        if isinstance(arr_or_data,self.__class__):
            if any(index in self.index for index in arr_or_data.index):
                e_msg = ("Key exists in current datastruct cannot concatenate")
                raise ValueError(e_msg)

            self._indexer.extend(arr_or_data._indexer)

            self._data.extend(arr_or_data._data)

        elif all([isinstance(arr,self.__class) for arr in arr_or_data]):
            for arr in arr_or_data:
                self.concat(arr)
        else:
            raise TypeError(msg)
        

    def append(self,arr,key=None,axis=0):
        if isinstance(arr,np.ndarray):
            msg = f"If the type of arr is {np.ndarray.__name__}, key must be provided"
            # raise TypeError(msg)
            if key is None:
                raise ValueError(msg)
            if len(self[key].shape) == 1:
                self[key] = np.stack([self[key],arr],axis=axis)
            else:
                self[key] = np.append(self[key],[arr],axis=axis)
        elif isinstance(arr,datastruct):
            if key is None:
                key = self.index
            if hasattr(key,"__iter__") and not isinstance(key,str):
                for k in key:
                    self.append(arr[k],key=k,axis=axis)
            else:
                self.append(arr[key],key=key,axis=axis)   
        else:
            msg = f"Type of arr must be either {np.ndarray.__name__} or {datastruct.__name__}"
            raise TypeError(msg)

    def _arith_binary_op(self,other_obj,func):
        if isinstance(other_obj,datastruct):
            if not self.index==other_obj.index:
                msg = "This can only be used if the indices in both datastructs are the same"
                raise ValueError(msg)
            new_data = {}
            for key, val in self:
                new_data[key] = func(val,other_obj[key])

        else:
            try:
                new_data = {key :func(val,other_obj) for key, val in self}
            except TypeError:
                msg = (f"Cannot use operation {func.__name__} datastruct by "
                        f"object of type {type(other_obj)}")
                raise TypeError(msg) from None

        return datastruct(new_data) 

    def __add__(self,other_obj):
        return self._arith_binary_op(other_obj,operator.add)

    def __radd__(self,other_obj):
        return self.__add__(other_obj)

    def __sub__(self,other_obj):
        return self._arith_binary_op(other_obj,operator.sub)

    def __rsub__(self,other_obj):
        self_neg = operator.neg(self)
        return operator.add(self_neg,other_obj)

    def __mul__(self,other_obj):
        return self._arith_binary_op(other_obj,operator.mul)

    def __rmul__(self,other_obj):
        return self.__mul__(other_obj)

    def __truediv__(self,other_obj):
        return self._arith_binary_op(other_obj,operator.truediv)
    
    def _arith_unary_op(self,func):

        new_data = {key :func(val) for key, val in self}
        return datastruct(new_data) 

    def __abs__(self):
        return self._arith_unary_op(operator.abs)

    def __neg__(self):
        return self._arith_unary_op(operator.neg)

    def __eq__(self,other_datastruct):
        return self.equals(other_datastruct)
    
    def __ne__(self,other_datastruct):
        return not self.equals(other_datastruct)

    def copy(self):
        cls = self.__class__
        return cls(copy.deepcopy(self.to_dict()))

    def __deepcopy__(self,memo):
        return self.copy()
    
    def __contains__(self,key):
        return key in self._indexer

    def symmetrify(self,dim=None):
        
        slicer = slice(None,None,None)
        indexer = [slicer for _ in range(self.values.ndim-1)]
        if dim is not None:
            indexer[dim] = slice(None,None,-1)

        data = {}
        for index,vals in self:
           
            comp = index[1] if self._is_multidim() else index
            count = comp.count('v') + comp.count('y')
            data[index] = vals.copy()[tuple(indexer)]*(-1)**count

        return datastruct(data)


class metastruct():
    def __init__(self,*args,from_hdf=False,**kwargs):

        from_list = False; from_dict=False
        if isinstance(args[0],list):
            from_list = True
        elif isinstance(args[0],dict):
            from_dict = True
        elif not from_hdf:
            msg = (f"{self.__class__.__name__} can be instantiated by list,"
                    " dictionary or the class method from_hdf")
            raise TypeError(msg)

        if from_list:
            self._list_extract(*args,**kwargs)
        elif from_dict:
            self._dict_extract(*args,**kwargs)
        elif from_hdf:
            self._file_extract(*args,**kwargs)

    def _conversion(self,old_key,*replacement_keys):
        """
        Converts the old style metadata to the new style metadata
        """
        if old_key not in self._meta.keys():
            return

        if not isinstance(self._meta[old_key],list):
            item = [self._meta[old_key]]
        else:
            item = self._meta[old_key]
        for i, key in enumerate(replacement_keys):
            self._meta[key] = item[i]

        del self._meta[old_key]

    
    def _update_keys(self):
        update_dict = {
            'icase' : ['iCase'],
            'thermlflg' : ['iThermoDynamics'],
            'HX_tg_io' :['HX_tg','HX_io'],
            'NCL1_tg_io': ['NCL1_tg','NCL1_io'],
            'REINI_TIME' : ['ReIni','TLgRe'],
            'FLDRVTP' : ['iFlowDriven'],
            'CF' :['Cf_Given'],
            'accel_start_end':['temp_start_end'],
            'HEATWALLBC' : ['iThermalWallType'],
            'WHEAT0' : ['thermalWallBC_Dim'],
            'RSTflg_tg_io' : ['iIniField_tg','iIniField_io'],
            'RSTtim_tg_io' : ['TimeReStart_tg','TimeReStart_io'],
            'RST_type_flg' : ['iIniFieldType','iIniFieldTime'],
            'CFL' : ['CFLGV'],
            'visthemflg' : ['iVisScheme'],
            'Weightedpressure' : ['iWeightedPre'],
            'DTSAVE1' : ['dtSave1'],
            'TSTAV1' : ['tRunAve1','tRunAve_Reset'],
            'ITPIN' : ['iterMonitor'],
            'MGRID_JINI' : ['MGRID','JINI'],
            'pprocessonly': ['iPostProcess'],
            'ppinst' : ['iPPInst'],
            'ppspectra' : ['iPPSpectra'],
            'ppdim' : ['iPPDimension'],
            'ppinstnsz' : ['pp_instn_sz'],
            'grad' : ['accel_grad']
        }

        if 'NCL1_tg_io' in self._meta.keys() and 'iDomain' not in self._meta.keys():
            if self._meta['NCL1_tg_io'][1] < 2:
                self._meta['iDomain'] = 1
            else:
                self._meta['iDomain'] = 3

        if 'iCHT' not in self._meta.keys():
            self._meta['iCHT'] = 0

        if 'BCY12' not in self._meta.keys():
            self._meta['BCY12'] = [1,1]

        if'loc_start_end' in self._meta.keys():
            loc_list = [self._meta['loc_start_end'][0]*self._meta['HX_tg_io'][1],
                        self._meta['loc_start_end'][1]*self._meta['HX_tg_io'][1]]
            self._meta['location_start_end'] = loc_list
            del self._meta['loc_start_end']

        if 'DT' in self._meta.keys():
            if isinstance(self._meta['DT'],list):
                update_dict['DT'] = ['DT','DtMin']

        for key, val in update_dict.items():
            self._conversion(key,*val)
        
    def _list_extract(self,list_vals,index=None):
        if index is None:
            index = list(range(len(list_vals)))
        
        if len(index) != len(list_vals):
            msg = "The length of the index must be the same as list_vals"
            raise ValueError(msg)

        self._meta = {i:val for i, val in zip(index,list_vals)}
        self._update_keys()

    def _dict_extract(self,dictionary):
        self._meta = dictionary
        self._update_keys()

    def to_hdf(self,filename,key=None,mode='a'):
        hdf_obj = hdfHandler(filename,mode=mode,key=key)
        hdf_obj.set_type_id(self.__class__)
        str_items = hdf_obj.create_group('meta_str')

        for k, val in self._meta.items():

            if not hasattr(val,"__iter__") and not isinstance(val,str):
                hdf_obj.create_dataset(k,data=np.array([val]))
            else:
                if isinstance(val,str):
                    str_items.attrs[key] = val.encode('utf-8')
                else:
                    hdf_obj.create_dataset(k,data=np.array(val))

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(*args,from_hdf=True,**kwargs)

    def _file_extract(self,filename,*args,key=None,**kwargs):
        hdf_obj = hdfHandler(filename,mode='r',key=key)

        index = list(key for key in hdf_obj.keys() if key != 'meta_str')
        list_vals=[]
        for k in index:
            val = list(hdf_obj[k])
            if len(val)==1:
                val = val[0]
            list_vals.append(val)

        index.extend(hdf_obj.attrs.keys())

        str_items = hdf_obj['meta_str'] if 'meta_str' in hdf_obj.keys() else hdf_obj
        for k in str_items.attrs.keys():
            list_vals.append(str_items.attrs[k])
        
        self._meta = {i:val for i, val in zip(index,list_vals)}


        self._update_keys()
        
    def __mathandle__(self):
        return self._meta
    
    @property
    def index(self):
        return self._meta.keys()

    def __getitem__(self,key):
        if key not in self._meta.keys():
            msg = "key not found in metastruct"
            raise KeyError(msg)
        return self._meta[key]

    def __setitem__(self,key,value):
        warn_msg = "item in metastruct being manually overriden, this may be undesireable"
        warnings.warn(warn_msg)

        self._meta[key] = value

    def copy(self):
        cls = self.__class__
        index = list(self._meta.keys()).copy()
        values = list(self._meta.values()).copy()
        return cls(values,index=index)

    def __deepcopy__(self,memo):
        return self.copy()