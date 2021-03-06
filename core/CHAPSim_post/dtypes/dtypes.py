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
from numpy.linalg.linalg import _raise_linalgerror_eigenvalues_nonconvergence
import pandas as pd
import numpy as np
import h5py
import sys
import h5py
import numbers
import os
import operator

import CHAPSim_post as cp
import sys


class datastruct:
    def __init__(self,*args,from_hdf=False,**kwargs):

        from_array=False; from_dict = False
        if isinstance(args[0],np.ndarray):
            from_array=True
        elif isinstance(args[0],dict):
            from_dict=True
        elif not from_hdf:
            msg = "No extract type selected"
            raise ValueError(msg)
        
        if from_array:
            self._array_ini(*args,**kwargs)
        elif from_dict:
            self._dict_ini(*args,*kwargs)
        elif from_hdf:
            self._file_extract(*args,**kwargs)
        else:
            msg = f"This is not a valid initialisation method for the {datastruct.__name__} type"
            raise ValueError(msg)
        
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
    
    def _construct_key(self,hdf_obj,key=None):
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

    def _hdf_extract(self,filename,key=None):

        

        try:
            hdf_file = h5py.File(filename,mode='r')
            if key is not None:
                hdf_data = hdf_file[key]
            else:
                hdf_data= hdf_file
                    
            keys = self._construct_key(hdf_data)
            if 'order' in hdf_data.attr.keys():
                order = list(hdf_data.attr['order'][:])
                keys = keys[order]
                
            self._data = {}
            self._index = []
            for key in keys:
                if key.count("/")>0:
                    index = tuple(key.split("/"))
                else:
                    index = key
                self._index.append(index)
                self._data[index] = np.array(hdf_data[key][:]).astype(cp.rcParams['dtype'])

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
            self._data = {i : dataFrame.data[i].astype(cp.rcParams['dtype']) for i in self._index}
        else:
            self._data = {i : dataFrame.coord[i].astype(cp.rcParams['dtype']) for i in self._index}
        self._outer_index = list(set([x[0] for x in self._index]))

    
    def _array_ini(self,array,index=None,copy=False):

        self._index, self._outer_index = self._index_construct(index,array)
        if self._index is None:
            self._index = list(range(array.shape[0]))

        if len(self._index) != len(array):
            msg = "The length of the indices must be the same as the outer dimension of the array"
            raise ValueError(msg)

        self._data = {i : value.astype(cp.rcParams['dtype'],copy=copy) for i, value in zip(self._index,array)}

    @staticmethod
    def _index_construct(index,array):
        if all(isinstance(x,tuple) for x in index):
            if len(index) != len(array):
                msg = "The index is not the correct size"
                raise ValueError(msg)
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

    
    def _dict_ini(self,dict_data,copy=False):
        if not all([isinstance(val,np.ndarray) for val in dict_data.values()]):
            msg = "The type of the values of the dictionary must be a numpy array"
            raise TypeError(msg)

        self._data = {key : val.astype(cp.rcParams['dtype'],copy=copy) for key, val in dict_data.items()}


        self._index = list(dict_data.keys())

        if self._is_multidim():
            self._outer_index = list(set([i[0] for i in self._index]))
            if None in self._outer_index and len(self._outer_index) == 1:
                self._outer_index=None
        else:
            self._outer_index = None



    def to_hdf(self,filepath,key=None,mode='a'):
        hdffile=h5py.File(filepath,mode=mode)
        def convert2str(index):
            if hasattr(index,"__iter__") and not isinstance(index,str):
                k = [str(x) for x in index]
            else:
                k = str(index)
            return k

        hdf_data = hdffile if key is None else hdffile[key]
        order =[]
        for k, val in self: 
            k = convert2str(k)
            hdf_key = None
            if hasattr(k,"__iter__") and not isinstance(k,str):
                hdf_key =   "/".join(k)
            else:
                hdf_key = k

            
            order.append(np.str(hdf_key))
            hdf_data.create_dataset(hdf_key,data=val)

        order_num = np.argsort(np.array(order,dtype=np.str))
        hdf_order = np.zeros_like(order_num,dtype=np.int32)
        for i,num in enumerate(order_num):
            hdf_order[num] = i
        hdf_data.attr['order'] = hdf_order
        hdffile.close()

    def _is_multidim(self):
        return all([isinstance(i,tuple) for i in self._index])

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
        return list(self._data.keys())

    # @index.setter
    # def index(self,index):
    #     if len(index) != len(self._data):
    #         msg = "The index given must be the same as the length of the data" 
    #         raise ValueError(msg)
    #     values = self._obj.values()

    #     self._index = index

    @property
    def outer_index(self):
        if self._is_multidim():
            return self._outer_index
        else:
            msg = "This method cannot be used on datastructs with single dimensional indices"
            raise AttributeError(msg)

    @property
    def inner_index(self):
        if self._is_multidim():
            return [x[1] for x in self._index]
        else:
            return self.index

    # @times.setter
    # def times(self,times):
    #     self._times = times

    @property
    def values(self):
        shape_list = [x.shape for x in self._data.values()]
        if not all(x==shape_list[0] for x in shape_list):
            msg = "To use this function all the arrays in the datastruct must be the same shape"
            raise AttributeError(msg)
        stack_list = list(self._data.values())
        return np.stack(stack_list,axis=0)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
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

    def _getitem_process_multikey(self,key,err=None,warn=None):
        if not self._is_multidim():
            msg = "A multidimensional index passed but a single dimensional datastruct"
            raise KeyError(msg)
        key = tuple(str(k) if not isinstance(k,numbers.Number) else "%g"%k for k in key )

        if key in self._data.keys():
            return key
        else:
            err = KeyError((f"The key provided ({key}) to the datastruct is"
                            " not present and cannot be corrected internally."))
            warn = UserWarning((f"The outer index provided is incorrect ({key[0]})"
                        f" that is present (there is only one value present in the"
                        f" datastruct ({self._outer_index[0]})"))
            
            inner_key = self.check_inner(key[1],err)
            outer_key = self.check_outer(key[0],err,warn)

            return (outer_key, inner_key)

    def _getitem_process_singlekey(self,key,err_msg=None):

        if isinstance(key, numbers.Number):
            key = "%g"%key
        else:
            key = str(key)

        err_msg = (f"The key provided ({key}) to ""the datastruct is "
                    "not present and cannot be corrected internally.")
        err = KeyError(err_msg)

        if key in self.index:
            return key
        else:
            outer_key = self.check_outer(None,err)
            return (outer_key,key)

    def check_outer(self,key,err,warn=None):
        if isinstance(key, (float,int)):
            key = "%g"%key
        else:
            key = str(key)

        if not self._is_multidim():
            raise err

        if key not in self.outer_index: 
            if len(self.outer_index) > 1:
                raise err
            else:
                if warn is None:
                    warnings.warm(warn,stacklevel=4)
                key = self.outer_index[0]

        return key

    def check_inner(self,key,err):
        if key not in self.inner_index:
            raise err
        
        return key
    
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
                    self._index.append(index)
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

        return self.__class__(new_data) 

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
        return self.__class__(new_data) 

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
        return cls(self._data,copy=True)

    def __deepcopy__(self,memo):
        return self.copy()
    
    def symmetrify(self,dim=None):
        if self._is_multidim():
            comp_list = self.inner_index
        else:
            comp_list=self.index

        symm_count=[]
        for comp in comp_list:
            symm_count.append(comp.count('v') + comp.count('y'))

        slicer = slice(None,None,None)
        indexer = [slicer for _ in range(self.values.ndim-1)]
        if dim is not None:
            indexer[dim] = slice(None,None,-1)

        data = {}
        for index,count in zip(self.index,symm_count):

            data[index] = self._data[index][tuple(indexer)]*(-1)**count

        return datastruct(data)

class coordstruct:
    def __init__(self,path,abs_path,from_hdf=False,key=None,metaDF=None):
        if from_hdf:
            self._CoordDF = self._coord_extract()

    def _coord_extract(self,path_to_folder,abs_path,tgpost,ioflg):
        if os.path.isdir(os.path.join(path_to_folder,'0_log_monitors')):
            return self._coord_extract_new(path_to_folder,abs_path,tgpost,ioflg)
        else:
            return self._coord_extract_old(path_to_folder,abs_path,tgpost,ioflg)

    def _coord_extract_new(self,path_to_folder,abs_path,tgpost,ioflg):
        full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
                                                            '.')

        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(full_path,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(full_path,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(full_path,'CHK_COORD_ZND.dat')
    #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        x_coord=np.loadtxt(file,comments='#',usecols=1)
        
        if tgpost and not ioflg:
            XND = x_coord[:-1]
        else:
            index = int(self.metaDF['NCL1_tg']) + 1 
            if tgpost and ioflg:
                XND = x_coord[:index]
                XND -= XND[0]
            else:
                XND = x_coord[index:]
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        y_coord=np.loadtxt(file,usecols=1,skiprows=1)
        index = int(self.metaDF['NCL1_tg']) + 1 
        YCC=y_coord[index:]
        y_size = YCC.size
        file.close()
        #============================================================
    
        file=open(z_coord_file,'rb')
        ZND=np.loadtxt(file,comments='#',usecols=1)

        #============================================================
        XCC, ZCC = self._coord_interp(XND,ZND)

        z_size = ZCC.size
        x_size = XCC.size
        y_size = YCC.size
        file.close()

        CoordDF = datastruct({'x':XCC,'y':YCC,'z':ZCC})
        return CoordDF


    def _coord_extract_old(self,path_to_folder,abs_path,tgpost,ioflg):
        """ Function to extract the coordinates from their .dat file """
        
        full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
                                                            '.')

        if not abs_path:
            x_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_XND.dat'))
            y_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_YND.dat'))
            z_coord_file = os.path.abspath(os.path.join(full_path,'CHK_COORD_ZND.dat'))
        else:
            x_coord_file = os.path.join(full_path,'CHK_COORD_XND.dat')
            y_coord_file = os.path.join(full_path,'CHK_COORD_YND.dat')
            z_coord_file = os.path.join(full_path,'CHK_COORD_ZND.dat')
        #===================================================================
        #Extracting XND from the .dat file
    
        file=open(x_coord_file,'rb')
        x_coord=np.loadtxt(file,comments='#')
        x_size=  int(x_coord[0])
        x_coord=np.delete(x_coord,0)
        
        if tgpost and not ioflg:
            XND = x_coord[:-1]
        else:
            for i in range(x_size):
                if x_coord[i] == 0.0:
                    index=i
                    break
            if tgpost and ioflg:
                XND = x_coord[:index+1]
                XND -= XND[0]
            else:
                XND = x_coord[index+1:]
        file.close()
        #===========================================================
    
        #Extracting YCC from the .dat file
        file=open(y_coord_file,'rb')
        y_coord=np.loadtxt(file,comments='#',usecols=1)
        y_size = int(y_coord[0])
        for i in range(y_coord.size):
            if y_coord[i] == 1.0:
                index=i
                break
        YCC=np.delete(y_coord,np.arange(index+1))
        y_size = YCC.size
        file.close()
        #============================================================
    
        file=open(z_coord_file,'rb')
        z_coord=np.loadtxt(file,comments='#')
        z_size = int(z_coord[0])
        ZND=np.delete(z_coord,0)
        #============================================================
        XCC, ZCC = self._coord_interp(XND,ZND)

        z_size = ZCC.size
        x_size = XCC.size
        y_size = YCC.size
        file.close()

        CoordDF = datastruct({'x':XCC,'y':YCC,'z':ZCC})
        return CoordDF
        
    def _coord_interp(self,XND, ZND):
        """ Interpolate the coordinates to give their cell centre values """
        XCC=np.zeros(XND.size-1)
        for i in range(XCC.size):
            XCC[i] = 0.5*(XND[i+1] + XND[i])
    
        ZCC=np.zeros(ZND.size-1)
        for i in range(ZCC.size):
            ZCC[i] = 0.5*(ZND[i+1] + ZND[i])
    
        return XCC, ZCC

    @classmethod
    def from_hdf(cls,file_name,key=None,metaDF=None):
        return cls(file_name,key=key,metaDF=metaDF)
    @property
    def staggered(self):
        pass

class flowstruct_base:
    pass

class flowstruct2D(flowstruct_base):
    pass

class flowstruct3D(datastruct):
    def __init__(self,CoordDF,*args,from_hdf=False,**kwargs):
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        self._set_coords(CoordDF,from_hdf,*args,**kwargs)

    def _set_coords(self,CoordDF,from_hdf,*args,**kwargs):
        if from_hdf:
            file_name = args[0]
            hdf_file = h5py.File(file_name,'r')
            sub_key = "coordDF"
            if len(args)>1:
                key=args[1]
            else:
                key = kwargs.get('key',None)

            if key is None:
                key = sub_key
            else:
                key = "/".join([key,sub_key])
            if key in hdf_file.keys():
                self._CoordDF = datastruct.from_hdf(file_name,key=key)
            elif CoordDF is not None:
                self._CoordDF = CoordDF
            else:
                msg = "If CoordDF datastruct is not present in HDF file, a datastruct must be provided"
                raise KeyError(msg)
        else:
            self._CoordDF = CoordDF

    @classmethod
    def from_hdf(cls,*args,CoordDF=None,**kwargs):
        return cls(CoordDF,*args,from_hdf=True,**kwargs)

    def to_hdf(self,filepath,key=None,mode='a'):
        super().to_hdf(filepath,key=key,mode=mode)
        self._CoordDF.to_hdf(filepath,key=key+"/coordDF",mode='a')
    
    @property
    def times(self):
        return self.outer_index

    @property
    def comp(self):
        return self.inner_index

    def check_times(self,*args,**kwargs):
        return self.check_outer(*args,**kwargs)

    def check_comp(self,*args,**kwargs):
        return self.check_inner(*args,**kwargs)

    def concat(self,arr_or_data):
        msg= "The coordinate data of the flowstructs must be the same"
        if isinstance(arr_or_data,self.__class__):
            if not self._CoordDF != arr_or_data._CoordDF:
                raise ValueError(msg)
        elif hasattr(arr_or_data,"__iter__"):
            if not all([self._CoordDF != arr._CoordDF for arr in arr_or_data]):
                raise ValueError(msg)
        super().concat(arr_or_data)
    
    def append(self,*args,**kwargs):
        msg = "This method is not available for this class"
        raise NotImplementedError(msg)

    def _arith_binary_op(self,other_obj,func):
        if isinstance(other_obj,self.__class__):
            msg= "The coordinate data of the flowstructs must be the same"
            if isinstance(other_obj,self.__class__):
                if not self._CoordDF != other_obj._CoordDF:
                    raise ValueError(msg)
        super()._arith_binary_op(other_obj,func)

    def copy(self):
        cls = self.__class__
        return cls(self._CoordDF,self._data,copy=True)

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

        self._update_keys()

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

    def copy(self):
        cls = self.__class__
        index = list(self._meta.keys()).copy()
        values = list(self._meta.values()).copy()
        return cls(values,index=index)

    def __deepcopy__(self,memo):
        return self.copy()

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