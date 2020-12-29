"""
# CHAPSim_dtypes
A module for the CHAPSim_post postprocessing and visualisation library. This 
experimental library contains additional classes to store data from the module.
The data types are built from the pandas DataFrame and are designed to superseed 
them for CHAPSim_post to enable some additional high level functionality to the
use and the other modules to allow data to be automatically reshaped when the 
__getitem__ method is used
"""

import warnings
from numpy.core.numeric import array_equal, indices, outer
import pandas as pd
import numpy as np
import h5py
import sys
import h5py
import numbers
import os

from . import CHAPSim_Tools as CT

class datastruct:
    def __init__(self,*args,array=False,dict=False,hdf=False,**kwargs):
        if not array and not dict and not hdf:
            if isinstance(args[0],np.ndarray):
                array=True
            elif isinstance(args[0],dict):
                dict=True
            else:
                msg = "No extract type selected"
                raise ValueError(msg)
        
        self.__rmul__=self.__mul__
        if array:
            self._array_ini(*args,**kwargs)
        elif dict:
            self._dict_ini(*args,*kwargs)
        elif hdf:
            self._file_extract(*args,**kwargs)
        else:
            msg = f"This is not a valid initialisation method for the {datastruct.__name__} type"
            raise ValueError(msg)
        
    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(*args,hdf=True,**kwargs)

    def _file_extract(self,filename,*args,key=None,**kwargs):
        hdf_file = h5py.File(filename,mode='r')
        if key is not None:
            hdf_data = hdf_file[key]
        else:
            hdf_data= hdf_file
        
        pd_DF_keys = ['axis0', 'block0_items', 'block0_values']
        if all([key in list(hdf_data.keys()) for key in pd_DF_keys]):
            self._extract_pd_DataFrame(filename,*args,key=key,**kwargs)
        else:
            self._hdf_extract(filename,key=key)
        hdf_file.close()
        
    def _hdf_extract(self,filename,key=None):

        def construct_key(hdf_obj,key=None):
            keys_list =[]
            outer_key = hdf_obj.keys()
            for key in outer_key:
                if hasattr(hdf_obj[key],'keys'):
                    inner_keys = ["/".join([key,ikey]) for ikey in hdf_obj[key].keys()]
                    keys_list.extend(inner_keys)
                else:
                    keys_list = outer_key
            if key is not None:
                for k in keys_list:
                    k = "/".join([key,k])
            return keys_list

        try:
            hdf_file = h5py.File(filename,mode='r')
            if key is not None:
                hdf_data = hdf_file[key]
            else:
                hdf_data= hdf_file
                    
            keys = construct_key(hdf_data)
            self._data = {}
            self._index = []
            for key in keys:
                if key.count("/")>0:
                    index = tuple(key.split("/"))
                else:
                    index = key
                self._index.append(index)
                self._data[index] = np.array(hdf_data[key][:])

            if self._is_multidim():
                self._outer_index = list(set([i[0] for i in self._index]))
            else:
                self._outer_index = [None]
        finally:
            hdf_file.close()

    # @classmethod
    # def from_DataFrame(cls,filename,shapes=None,key=None):
    #     return cls(filename,shapes=shapes,from_DF=True,key=key)

    def _extract_pd_DataFrame(self,filename,shapes=None,key=None):
        indices = None
        if shapes is not None:
            dataFrame = pd.read_hdf(filename,key=key).data(shapes)
            indices = dataFrame.index
            
        else:
            dataFrame = pd.read_hdf(filename,key=key).coord()
            indices = dataFrame.columns

        self._index = list(indices)
        if self._is_multidim():
            for i,index in enumerate(self._index):
                if isinstance(self._index[i][0],numbers.Number):
                    if np.isnan(self._index[i][0]):
                        self._index[i] = (None,*self._index[i][1:])
        
        if shapes is not None:
            self._data = {i : dataFrame.data[i] for i in self._index}
        else:
            self._data = {i : dataFrame.coord[i] for i in self._index}
        self._outer_index = list(set([x[0] for x in self._index]))


    @classmethod
    def from_array(cls,*args,**kwargs):
        return cls(*args,array=True,**kwargs)
    
    def _array_ini(self,array,index=None,copy=False):

        self._index, self._outer_index = self._index_construct(index,array)
        if self._index is None:
            self._index = list(range(array.shape[0]))

        if len(self._index) != len(array):
            msg = "The length of the indices must be the same as the outer dimension of the array"
            raise ValueError(msg)
        if copy:
            self._data = {i : value.copy() for i, value in zip(self._index,array)}
        else:
            self._data = {i : value for i, value in zip(self._index,array)}
    
    @staticmethod
    def _index_construct(index,array):
        if len(index) == len(array):
            outer_index = list(set([x[0] for x in index]))
        elif len(index[0]) == len(array):
            outer_index = None
            if all([hasattr(x,"__iter__") and not isinstance(x,str) for x in index]):
                if not len(index) == 2:
                    msg = "This class can only currently handle up to two dimensional indices"
                    raise ValueError(msg)
                outer_index = list(str(k) if not isinstance(k,numbers.Number) else "%g"%k for k in index[0] ) 
                inner_index = list(str(k) if not isinstance(k,numbers.Number) else "%g"%k for k in index[1] ) 
                index = [outer_index,inner_index]
                outer_index = list(set(outer_index))
                index = list(zip(*index))
            elif index is not None:
                index = list(str(k) if not isinstance(k,numbers.Number) else "%g"%k for k in index ) 
                outer_index = [None]
            else:
                index = None
                outer_index = [None]
        else:
            msg = "The index list is the wrong size"
            raise ValueError(msg)
        
        return index, outer_index

        
    @classmethod
    def from_dict(cls,*args,**kwargs):
        return cls(*args,dict=True,**kwargs)
    
    def _dict_ini(self,dict_data,copy=False):
        if not all([isinstance(val,np.ndarray) for val in dict_data.values()]):
            msg = "The type of the values of the dictionary must be a numpy array"
            raise TypeError(msg)
        if copy:
            self._data = {key : val.copy() for key, val in dict_data.items()}
        else:
            self._data = dict_data
        self._index = dict_data.keys()
        if self._is_multidim():
            self._outer_index = list(set([i[0] for i in self._index]))
            if all(self._outer_index == None):
                self._outer_index=None
        else:
            self._outer_index = None



    def to_hdf(self,filepath,key=None,mode='a'):
        hdffile=h5py.File(filepath,mode=mode)
        def convert2str(index):
            if hasattr(index,"__iter__") and not isinstance(index,str):
                key = [str(x) for x in index]
            else:
                key = str(index)
            return key
        for k, val in self: 
            k = convert2str(k)
            hdf_key = None
            if hasattr(k,"__iter__") and not isinstance(k,str):
                hdf_key =   "/".join(k)
            else:
                hdf_key = k
            hdf_key = "/".join([key,hdf_key])
            hdffile.create_dataset(hdf_key,data=val)
        hdffile.close()

    def _is_multidim(self):
        return all([hasattr(i,"__iter__") and not isinstance(i,str) for i in self._index])

    def equals(self,other_datastruct):
        if not isinstance(other_datastruct,datastruct):
            msg = "other_datastruct must be of type datastruct"
            raise TypeError(msg)

        for key,val1 in self:
            if key not in other_datastruct.index:
                return False
            if not np.array_equal(val1,other_datastruct[key]):
                return False
        return True

    @property
    def index(self):
        return self._index

    # @index.setter
    # def index(self,index):
    #     if len(index) != len(self._data):
    #         msg = "The index given must be the same as the length of the data" 
    #         raise ValueError(msg)
    #     values = self._obj.values()

    #     self._index = index

    @property
    def times(self):
        if self._is_multidim():
            return self._outer_index
        else:
            msg = "This method cannot be used on datastructs with single dimensional indices"
            raise AttributeError(msg)

    # @times.setter
    # def times(self,times):
    #     self._times = times

    @property
    def values(self):
        shape_list = [x.shape for x in self._data.values()]
        if not all(x==shape_list[0] for x in shape_list):
            msg = "To use this function all the arrays in the datastruct must be the same shape"
            raise AttributeError(msg)

        return np.stack(self._data.values(),axis=0)

    def __str__(self):
        return self._data.__str__()
        
    def __getitem__(self,key):
        key = self._getitem_check_key(key)
        return self._data[key]
        
    def _getitem_check_key(self,key,err_msg=None,warn_msg=None):
        if isinstance(key,tuple):
            if len(key) > 1:
                return self._getitem_process_multikey(key,err_msg,warn_msg)
            else:
                return self._getitem_process_singlekey(*key,err_msg=err_msg)
        else:
            return self._getitem_process_singlekey(key,err_msg=err_msg)

    def _getitem_process_multikey(self,key,err_msg=None,warn_msg=None):
        if not self._is_multidim():
            msg = "A multidimensional index passed but a single dimensional datastruct"
            raise KeyError(msg)
        key = tuple(str(k) if not isinstance(k,numbers.Number) else "%g"%k for k in key )

        if key in self._data.keys():
            return key
        elif len(self._outer_index) == 1:
            if warn_msg is None:
                warn_msg = f"The outer index provided is incorrect ({key[0]})"+\
                    f" that is present (there is only one value present in the"+\
                    f" datastruct ({self._outer_index[0]}))"
            warnings.warn(warn_msg,stacklevel=2)
            key = (self._outer_index[0],*key[1:])
            return key
        else:
            if err_msg is None:
                err_msg = f"The key provided ({key}) to the datastruct is not present and cannot be corrected internally."
            raise KeyError(err_msg)

    def _getitem_process_singlekey(self,key,err_msg=None):

        if isinstance(key, numbers.Number):
            key = "%g"%key
        else:
            key = str(key)

        if key in self._data.keys():
            return key
        elif len(self._outer_index) == 1:
            key = (self._outer_index[0],key)
            return key
        else:
            if err_msg is None:
                err_msg = f"The key provided ({key}) to the datastruct is not present and cannot be corrected internally."
            raise KeyError(err_msg)

    def check_index(self,*key,exc_type=None,err_msg=None,warn_msg=None,inner=False,outer=False):
        
        if not isinstance(exc_type,(Exception,Warning)):
            exc_type=None

        if inner and outer:
            set_key = key
        elif inner:
            set_key = (self._outer_index[0],*key)
        elif outer:
            inner_index = [x[1] for x in self.index]
            set_key = (*key,inner_index[0])
        else:
            raise ValueError("Not index selected for testing")

        try:
            set_key = self._getitem_check_key(set_key,err_msg,warn_msg)
        except KeyError as e:
            if exc_type is None:
                exc_type = type(e)
            raise exc_type(e.args[0]) from None
        else:
            return set_key

    def __setitem__(self,key,value):
        if not isinstance(value,np.ndarray):
            msg = f"The input array must be an instance of {np.ndarray.__name__}"
            raise TypeError(msg)

        if isinstance(key,tuple):
            if len(key) > 1:
                self._setitem_process_multikey(key,value)
            else:
                self._setitem_process_singlekey(*key,value)
        else:
            self._setitem_process_singlekey(key,value)
    
    def _setitem_process_multikey(self,key,value):
        if not self._is_multidim():
            msg = "A multidimensional index passed but the datastruct is single dimensional"
            raise KeyError(msg)
        key = tuple(str(k) if not isinstance(k,numbers.Number) else "%g"%k for k in key )

        if key not in self._index:
            msg = f"The key {key} is not present in the datastruct's indices, if you want to "+\
                "add this key create new datastruct and use the concat method"
            raise KeyError(msg)
        
        self._data[key] = value

    def _setitem_process_singlekey(self,key,value):
        if isinstance(key, numbers.Number):
            key = "%g"%key
        else:
            key = str(key)

        if key not in self._index:
            msg = f"The key {key} is not present in the datastruct's indices, if you want to "+\
                "add this key create new datastruct and use the concat method"
            raise KeyError(msg)
        
        self._data[key] = value
        

    def __iter__(self):
        return self._data.items().__iter__()

    def concat(self,arr_or_data):
        msg = f"`arr_or_data' must be of type {datastruct.__name__} or an iterable of it"
        if isinstance(arr_or_data,datastruct):
            indices = arr_or_data.index
            for index in indices:
                if index in self.index:
                    if not np.array_equal(arr_or_data[index],self[index]):
                        e_msg = f"Key exists and arrays are not identical, you"+\
                            " may be looking for the method {self.append.__name__}"
                        raise ValueError(e_msg)
                else:
                    self._index += index
                    self._data[index] = arr_or_data[index]

        elif hasattr(arr_or_data,"__iter__"):
            if not all([isinstance(type(arr),datastruct) for arr in arr_or_data]):
                raise TypeError(msg)
            for arr in arr_or_data:
                self.concat(arr)
        else:
            raise TypeError(msg)
        
        if self._is_multidim():
            self._outer_index = list(set([i[0] for i in self._index]))

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
    
    def __add__(self,other_datastruct):
        if not self.index==other_datastruct.index:
            msg = "This can only be used if the indices in both datastructs are the same"
            raise ValueError(msg)
        
        return_array = self.values + other_datastruct.values
        if self._is_multidim():
            outer_indices = [x[0] for x in self.index]
            inner_indices = [x[1] for x in self.index]
            indices = [outer_indices,inner_indices]
        else:
            indices = self.index

        return self.__class__(return_array,index=indices)

    def __mul__(self,object_ins):
        
        if isinstance(object_ins,datastruct):
            if not all(self.index==object_ins.index):
                msg = "This can only be used if the indices in both datastructs are the same"
                raise ValueError(msg)
            return_array = self.values*object_ins.values
        else:
            try:
                return_array = self.values*object_ins
            except Exception:
                msg = "Cannot multiply datastruct by this value"
                raise ValueError(msg)

        if self._is_multidim():
            outer_indices = [x[0] for x in self.index]
            inner_indices = [x[1] for x in self.index]
            indices = [outer_indices,inner_indices]
        else:
            indices = self.indices
        return self.__class__(return_array,index=indices)        

    def __eq__(self,other_datastruct):
        return self.equals(other_datastruct)
    
    def __ne__(self,other_datastruct):
        return not self.equals(other_datastruct)
    

class metastruct():
    def __init__(self,*args,from_list=False,from_hdf=False,from_DF=False,**kwargs):
        if not from_list and not from_hdf:
            from_list = True

        if from_list:
            self._list_extract(*args,**kwargs)
        elif from_hdf:
            self._file_extract(*args,**kwargs)
        

    def _list_extract(self,list_vals,index=None):
        if index is None:
            index = list(range(len(list_vals)))
        
        if len(index) != len(list_vals):
            msg = "The length of the index must be the same as list_vals"
            raise ValueError(msg)

        self._meta = {i:val for i, val in zip(index,list_vals)}

    def to_hdf(self,filename,key=None,mode='a'):
        # hdf_key1 = 'meta_vals'
        # hdf_key2 = 'meta_index'
        # if key is not None:
        #     hdf_key1 = "/".join([key,hdf_key1])
        #     hdf_key2 = "/".join([key,hdf_key2])

        hdf_file = h5py.File(filename,mode=mode)
        for k, val in self._meta.items():
            if key is not None:
                k = "/".join([key,k])
            if not hasattr(val,"__iter__") and not isinstance(val,str):
                hdf_file.create_dataset(k,data=np.array([val]))
            else:
                if isinstance(val,str):
                    hdf_file.attrs[key] = val.encode('utf-8')
                else:
                    hdf_file.create_dataset(k,data=np.array(val))
        # hdf_file.create_dataset(hdf_key2,data=index)

        hdf_file.close()


    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(*args,from_hdf=True,**kwargs)

    def _file_extract(self,filename,*args,key=None,**kwargs):
        hdf_file = h5py.File(filename,mode='r')
        if key is not None:
            hdf_data = hdf_file[key]
        else:
            hdf_data= hdf_file
        
        pd_DF_keys = ['axis0', 'block0_items', 'block0_values']
        if all([key in list(hdf_data.keys()) for key in pd_DF_keys]):
            self._extract_pd_DataFrame(filename,*args,key=key,**kwargs)
        else:
            self._hdf_extract(filename,key=key) 
        hdf_file.close()

    def _hdf_extract(self,filename,key=None):
        try:
            hdf_file = h5py.File(filename,mode='r')

            if key is None:
                hdf_data = hdf_file
            else:
                hdf_data=hdf_file[key]
            
            index = list(hdf_data.keys())
            list_vals=[]
            for k in index:
                val = list(hdf_data[k][:])
                if len(val)==1:
                    val = val[0]
                list_vals.append(val)
            index.extend(hdf_data.attrs.keys())
            for k in hdf_data.attrs.keys():
                list_vals.append(hdf_data.attrs[key].decode('utf-8'))
        finally:
            hdf_file.close()
        self._meta = {i:val for i, val in zip(index,list_vals)}

    def _extract_pd_DataFrame(self,filename,key=None):
        dataFrame = pd.read_hdf(filename,key=key)
        list_vals = []
        for index in dataFrame.index:
            val = list(dataFrame.loc[index].dropna().values) 
            if len(val) ==1:
                list_vals.append(val[0])
            else:
                list_vals.append(val)
        self._list_extract(list_vals,index=list(dataFrame.index))

    @property
    def index(self):
        return self._meta.keys()

    def __getitem__(self,key):
        return self._meta[key]

warnings.filterwarnings('ignore',category=UserWarning)
@pd.api.extensions.register_dataframe_accessor("data")
class DataFrame():
    def __init__(self,pandas_obj):
        self._obj = pandas_obj
        self._reshape = None
        self._active=False

    def __call__(self, shape):
        self._validate()
        self.FrameShape = shape
        return self._obj

    def _validate(self):
        cols = self._obj.columns
        if not all([type(col)==int for col in cols]):
            msg = "The columns must be integers the use this attribute"
            raise ValueError(msg)
    @property
    def FrameShape(self):
        return self._reshape

    @FrameShape.setter
    def FrameShape(self,shape):
        self._FrameShapeHelper(shape)
        self._reshape = shape
    
    def _FrameShapeHelper(self,shape):
        msg = "The shape provided to this function must be able to"+\
             f" reshape an array of the appropriate size {self._obj.shape}."+\
            f" Shape provided {shape}"
        if hasattr(shape[0],"__iter__"):

            size_list = [series.dropna().size for _,series in self._obj.iterrows()]
            for shape_i in shape:
                num_true = list(size_list == np.prod(shape_i)).count(True)
                if  num_true ==0:
                    raise ValueError(msg)
                elif num_true > 1:
                    warn_msg = "The array of this size appears more than "+\
                                "once data attribute should not be used"
                    warnings.warn(warn_msg,stacklevel=2)
                    break
                else:
                    self._active=True
                
        else:
            if np.prod(shape) not in self._obj.shape:
                raise ValueError(msg)
            self._active=True
        


    def __getitem__(self,key):
        if not self._active:
            raise RuntimeError("This functionality cannot be used here")
        if self._reshape is None:
            msg = "The shape has not been set, this function cannot be used"
            raise TypeError(msg)
        try:
            shape = self._getitem_helper(key)
            return self._obj.loc[key].dropna().values.reshape(shape)
        except KeyError as msg:
            if self._obj.index.nlevels==2:
                times = list(set([float(x[0]) for x in self._obj.index]))
                if len(times) ==1 or all(np.isnan(np.array(times))):
                    return self._obj.loc[times[0],key].dropna().values.reshape(shape)
                else:
                    tb = sys.exc_info()[2]
                    raise KeyError(msg.args[0]).with_traceback(tb)
            else:
                tb = sys.exc_info()[2]
                raise KeyError(msg.args[0]).with_traceback(tb)

    def _getitem_helper(self,key):
        if hasattr(self.FrameShape[0],"__iter__"):
            arr_size = self._obj.loc[key].dropna().values.size
            for shape in self.FrameShape:
                if np.prod(shape) == arr_size:
                    return shape
        else:
            return self.FrameShape


@pd.api.extensions.register_dataframe_accessor("coord")
class CoordFrame():
    def __init__(self,pandas_obj):
        self._obj = pandas_obj
        self._active = False

    def _validate(self):
        cols = self._obj.columns.to_list()
        msg = f"The columns of the DataFrame must be {['x','y','z']}"
        if not cols == ['x','y','z']:
            raise ValueError(msg)

    def __call__(self):
        self._validate()
        self._active = True
        return self._obj
    def __getitem__(self,key):
        if self._active:
            return self._obj[key].dropna().values
        else:
            msg = "This DataFrame extension is not active, the __call__ "+\
                        "special method needs to be called on the DataFrame"
            raise AttributeError(msg)
            
warnings.resetwarnings()