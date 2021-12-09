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
import itertools
import weakref
from functools import wraps
import CHAPSim_post as cp
import sys
import copy
try:
    from pyvista import StructuredGrid
    import vtk
except ImportError:
    pass

from CHAPSim_post.legacy.utils import misc_utils,indexing
import CHAPSim_post.legacy.plot as cplt

class structIndexer:
    def __init__(self, dstruct):
        self.__parent_dstruct = weakref.ref(dstruct) 
    
    @property
    def _struct_index(self):
        if self.__parent_dstruct() is None:
            msg = "The instance of datastruct this {self.__class__.__name__} references has been destroyed"
            raise AttributeError(msg)

        return list(self.__parent_dstruct()._data.keys())

    @property
    def is_MultiIndex(self):
        return all([isinstance(i,tuple) for i in self._struct_index])

    def _item_handler(self,item):
        
        if isinstance(item,tuple):
            return tuple(self._item_handler(k) for k in item)
        if isinstance(item,numbers.Number):
            return "%g"%item
        elif isinstance(item,str):
            return "%g"%float(item) if item.isnumeric() else item
        else:
            return str(item)

    def _construct_op_1_check(self,index,array):
        return all(len(i) == len(array) for i in index)

    def _construct_op_2_check(self,index,array):
        return len(index) == len(array)

    def _check_index_type(self,index):
        return all([isinstance(x,(tuple,list)) for x in index])

    def index_constructor(self,index,array):
        # two options for index construction
        # list with two list => [outer_index, inner_index]
        # list of tuples => each tuple an index in the data struct
        if index is None:
            index = list(range(array.shape[0]))
        elif self._construct_op_1_check(index,array):
            outer_index = None

            if self._check_index_type(index):
                if not len(index) == 2:
                    msg = "This class can only currently handle up to two dimensional indices"
                    raise ValueError(msg)

                outer_index = list(self._item_handler(k) for k in index[0] ) 
                inner_index = list(self._item_handler(k) for k in index[1] ) 
                index = list(zip(outer_index,inner_index))
            elif index is not None:
                index = list(self._item_handler(k) for k in index ) 
            else:
                index = None

        elif self._construct_op_2_check(index,array):
            outer_index = None
            if all(isinstance(x,tuple) for x in index):
                if all(len(x) == 2 for x in index):
                    inner_index = list(self._item_handler(k[1]) for k in index )
                    outer_index = list(self._item_handler(k[0]) for k in index )
                    index = list(zip(outer_index,inner_index))
                else:
                    msg = "This class can only currently handle up to two dimensional indices"
                    raise ValueError(msg)
            else:
                index = list(self._item_handler(k) for k in index )
        else:
            msg = "The index list is the wrong size"
            raise ValueError(msg)

        return index

    def get_index_order(self):

        order = []
        for x in self._struct_index:
            hdf_key = self.hdf_key_contructor(x)
            order.append(np.str(hdf_key))

        order_num = np.argsort(np.array(order,dtype=np.str))
        hdf_order = np.zeros_like(order_num,dtype=np.int32)
        for i,num in enumerate(order_num):
            hdf_order[num] = i

        return hdf_order

    def hdf_key_contructor(self,index,key=None):
        if self.is_MultiIndex:
            return "/".join([str(x) for x in index])
        else:
            return str(index)
    
    def construct_keys_from_hdf(self,hdf_obj,key=None):
        keys_list =[]
        outer_key = hdf_obj.keys()
        for key in outer_key:
            if hasattr(hdf_obj[key],'keys'):
                inner_keys = ["/".join([key,ikey]) for ikey in hdf_obj[key].keys()]
                keys_list.extend(inner_keys)
            else:
                keys_list.append(key)
        if key is not None:
            for k in keys_list:
                k = "/".join([key,k])
        return keys_list

    def get_index(self):
        return self._struct_index

    def get_outer_index(self):
        if self.is_MultiIndex:
            return list(set([x[0] for x in self._struct_index]))
        else:
            msg = "This method cannot be used on datastructs with single dimensional indices"
            raise AttributeError(msg)

    def get_inner_index(self):
        if self.is_MultiIndex:
            return list(set([x[1] for x in self._struct_index]))
        else:
            return self.index

    def getitem_check_key(self,key,err_msg=None,warn_msg=None):
        if isinstance(key,tuple):
            if len(key) > 1:
                return self._getitem_process_multikey(key)
            else:
                return self._getitem_process_singlekey(*key)
        else:
            return self._getitem_process_singlekey(key)

    def _getitem_process_multikey(self,key):
        if not self.is_MultiIndex:
            msg = "A multidimensional index passed but a single dimensional datastruct"
            raise KeyError(msg)
        key = self._item_handler(key)
        if key in self._struct_index:
            return key
        else:
            err = KeyError((f"The key provided ({key}) to the datastruct is"
                            " not present and cannot be corrected internally."))
            warn = UserWarning((f"The outer index provided is incorrect ({key[0]})"
                        f" that is present (there is only one value present in the"
                        f" datastruct ({self.get_outer_index()[0]})"))
            
            inner_key = self.check_inner(key[1],err)
            outer_key = self.check_outer(key[0],err,warn)

            return (outer_key, inner_key)

    def _getitem_process_singlekey(self,key):
        key = self._item_handler(key)

        err_msg = (f"The key provided ({key}) to ""the datastruct is "
                    "not present and cannot be corrected internally.")
        err = KeyError(err_msg)

        if key in self._struct_index:
            return key
        else:
            outer_key = self.check_outer(None,err)
            return (outer_key,key)
        
    def check_outer(self,key,err,warn=None):
        key = self._item_handler(key)

        if not self.is_MultiIndex:
            raise err

        if key not in self.get_outer_index(): 
            if len(self.get_outer_index()) > 1:
                raise err
            else:
                if warn is None:
                    warnings.warn(warn,stacklevel=4)
                key = self.get_outer_index()[0]

        return key

    def check_inner(self,key,err):
        if key not in self.get_inner_index():
            raise err
        
        return key

    def setitem_process_multikey(self,key):
        if not self.is_MultiIndex:
            msg = "A multidimensional index passed but the datastruct is single dimensional"
            raise KeyError(msg)

        key = self._item_handler(key)

        if key not in self._struct_index:
            msg = f"The key {key} is not present in the datastruct's indices, if you want to "+\
                "add this key create new datastruct and use the concat method"
            raise KeyError(msg)

        return key
        
    def setitem_process_singlekey(self,key):
        key = self._item_handler(key)

        if key not in self._struct_index:
            msg = f"The key {key} is not present in the datastruct's indices, if you want to "+\
                "add this key create new datastruct and use the concat method"
            raise KeyError(msg)

        return key

    def __getstate__(self):
        d = self.__dict__
        d['_structIndexer__parent_dstruct'] = d['_structIndexer__parent_dstruct']()
        return d

    def __setstate__(self,state):
        state['_structIndexer__parent_dstruct'] = weakref.ref(state['_structIndexer__parent_dstruct'])
        self.__dict__ = state
class hdfHandler:

    _available_ext = ['.h5','.hdf5']
    _write_modes = ['r+','w','w-','x','a']
    _file_must_exist_modes = ['r','w-','x']
    
    def __init__(self,filename,mode='r',key=None):
        
        filename = self._check_h5_file_name(filename,mode)

        self._hdf_file = h5py.File(filename,mode=mode)

        self._hdf_data = self._check_h5_key(self._hdf_file,key,mode=mode)

    def __getattr__(self,attr):
        if attr == '_hdf_data':
            msg = ("The __getattr__ special method for this function can only be"
                    " called after member _hdf_data has been set")
            raise AttributeError(msg)

        if hasattr(self._hdf_data,attr):
            return getattr(self._hdf_data,attr)
        else:
            msg = f"The {hdfHandler.__name__} class nor its HDF data object as attribute {attr}"
            raise AttributeError(msg)

    def __getitem__(self,key):
        if not self.check_key(key,groups_only=False):
            msg = (f"The key {key} doesn't exist in the HDF file "
                    f"{self._hdf_file.filename} at location {self._hdf_data.name}")
            raise KeyError(msg)

        return self._hdf_data.__getitem__(key)

    def __setitem__(self,key,value):
        self._hdf_data.__setitem__(key,value)

    def extract_array_by_key(self,key):
        return np.array(self._hdf_data[key][:])

    def get_index_from_key(self,key):
        if key.count("/")>0:
            return tuple(key.split("/"))
        else:
            return key

    def _check_h5_file_name(self,filename,mode):

        ext = os.path.splitext(filename)[-1]

        if ext == '':
            ext = '.h5'
            filename = filename + ext

        if ext not in self._available_ext:
            msg = "The file extention for HDF5 files must be %s not %s"%(self._available_ext,ext)
            raise ValueError(msg)

        if self._file_must_exist(mode):
            if not os.path.isfile(filename):
                msg = f"Using file mode {mode} requires the file to exist"
                raise ValueError(msg)

        return filename

    def _write_allowed(self,mode):
        return bool(set(self._write_modes).intersection(mode))

    def _file_must_exist(self,mode):
        return bool(set(self._file_must_exist_modes).intersection(mode))

    def check_type_id(self,class_obj):
        if "type_id" not in self.attrs.keys():
            return

        if cp.rcParams["relax_HDF_type_checks"]:
            return

        id = str(self._generate_type_id(class_obj))

        module, class_name = id.split("/")

        type_id = str(self.attrs['type_id'])

        type_m, type_c = id.split("/")

        if class_name != type_c:
            msg = ("The class_names do not match"
                    " for the HDF type id check")
            raise ValueError(msg)

        if module != type_m:
            msg = "The calling modules do match the HDF type id"
            warnings.warn(msg)
            

    def set_type_id(self,class_obj):
        id = self._generate_type_id(class_obj)

        self.attrs["type_id"] = id


    def _generate_type_id(self,class_obj):
        if not isinstance(class_obj,type):
            msg = "class needs to be of instance type to generate type id"
            raise TypeError(msg)

        module = class_obj.__module__
        class_name = class_obj.__name__
        return np.string_("/".join([module,class_name]))

    def check_key(self,key,groups_only=True):
        return self._check_key_recursive(self._hdf_data,key,groups_only)

    def _check_h5_key(self,h5_obj,key,mode=None,groups_only=True):
        if mode is None:
            mode = self._hdf_file.mode

        avail_keys = self._get_key_recursive(h5_obj,groups_only)

        if self._check_key_recursive(h5_obj,key,groups_only):
            return h5_obj[key]
        else:
            if key is None:
                return h5_obj
            elif self._write_allowed(mode):
                return h5_obj.create_group(key)
            else:
                msg = (f"Key {key} not found in file HDF "
                        f"file, keys available are {avail_keys}")
                raise KeyError(msg)

    def _check_key_recursive(self,h5_obj,key,groups_only=True):
        avail_keys = self._get_key_recursive(h5_obj,groups_only)

        key = os.path.join(h5_obj.name,key)

        return key in avail_keys
    
    def _get_key_recursive(self,h5_obj,groups_only=True):

        if groups_only:
            keys = [key for key in h5_obj.keys() if hasattr(h5_obj[key],'keys')]
        else:
            keys= list(h5_obj.keys()) if hasattr(h5_obj,'keys') else []

        all_key = []
        for key in keys:
            all_key.append(h5_obj[key].name)
            all_key.extend(self._get_key_recursive(h5_obj[key],groups_only))

        return all_key
        
    def construct_keys_from_hdf(self,key=None):
        keys_list =[]
        outer_key = self._hdf_data.keys()
        for key in outer_key:
            if hasattr(self._hdf_data[key],'keys'):
                inner_keys = ["/".join([key,ikey]) for ikey in self._hdf_data[key].keys()]
                keys_list.extend(inner_keys)
            else:
                keys_list.append(key)
        if key is not None:
            for k in keys_list:
                k = "/".join([key,k])
        return keys_list

    def __del__(self):
        if hasattr(self,'_hdf_file'):
            self._hdf_file.close()

class datastruct:


    def __init__(self,*args,from_hdf=False,**kwargs):

        self._indexer = structIndexer(self)

        from_array=False; from_dict = False
        if isinstance(args[0],np.ndarray):
            from_array=True

        elif isinstance(args[0],dict):
            from_dict=True
            
        elif not from_hdf:
            msg = (f"No valid initialisation method for the {datastruct.__name__}"
                    " type has been found from arguments")
            raise ValueError(msg)

        if from_array:
            self._array_ini(*args,**kwargs)
        elif from_dict:
            self._dict_ini(*args,**kwargs)
        elif from_hdf:
            self._file_extract(*args,**kwargs)

        
    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(*args,from_hdf=True,**kwargs)

    def _file_extract(self,filename,key=None):
        hdf_obj = hdfHandler(filename,mode='r',key=key)

        hdf_obj.check_type_id(self.__class__)
                    
        keys = hdf_obj.construct_keys_from_hdf()

        if 'order' in hdf_obj.attrs.keys():
            order = list(hdf_obj.attrs['order'][:])
            keys = [keys[i] for i in order]
            
        self._data = {}
        for key in keys:
            index = hdf_obj.get_index_from_key(key)
            self._data[index] = hdf_obj.extract_array_by_key(key).astype(cp.rcParams['dtype'])

    def _array_ini(self,array,index=None,copy=False):
        
        index = self._indexer.index_constructor(index,array)

        self._data = {i : value.astype(cp.rcParams['dtype'],copy=copy) for i, value in zip(index,array)}

    def _dict_ini(self,dict_data,copy=False):
        if not all([isinstance(val,np.ndarray) for val in dict_data.values()]):
            msg = "The type of the values of the dictionary must be a numpy array"
            raise TypeError(msg)
        
        index = list(dict_data.keys())
        array = list(dict_data.values())

        index = self._indexer.index_constructor(index,array)

        self._data = {i : value.astype(cp.rcParams['dtype'],copy=copy)\
                     for i, value in zip(index,array)}



    def to_hdf(self,filepath,key=None,mode='a'):
        hdf_obj =hdfHandler(filepath,mode=mode,key=key)
        def convert2str(index):
            if hasattr(index,"__iter__") and not isinstance(index,str):
                k = [str(x) for x in index]
            else:
                k = str(index)
            return k

        hdf_order = self._indexer.get_index_order()
        for k, val in self: 
            hdf_key = self._indexer.hdf_key_contructor(k)
            
            hdf_obj.create_dataset(hdf_key,data=val)

        hdf_obj.attrs['order'] = hdf_order

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
        return self._indexer.get_index()

    @property
    def outer_index(self):
        return self._indexer.get_outer_index()

    @property
    def inner_index(self):
        return self._indexer.get_inner_index()

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
        key = self._indexer.getitem_check_key(key)
        return self._data[key]

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
        if isinstance(key,tuple):
            if len(key) > 1:
                key = self._indexer.setitem_process_multikey(key)
            else:
                key = self._indexer.setitem_process_singlekey(*key)
        else:
            key = self._indexer.setitem_process_singlekey(key)

        self._data[key] = value

        
    def __delitem__(self,key):
        key = self._indexer._item_handler(key)

        del self._data[key]
            
    def __iter__(self):
        return self._data.items().__iter__()

    def concat(self,arr_or_data):
        msg = f"`arr_or_data' must be of type {datastruct.__name__} or an iterable of it"
        if isinstance(arr_or_data,datastruct):
            indices = arr_or_data.index
            for index in indices:
                if index in self.index:
                    if not np.array_equal(arr_or_data[index],self[index]):
                        e_msg = ("Key exists and arrays are not identical, you"
                            f" may be looking for the method {self.append.__name__}")
                        raise ValueError(e_msg)
                else:
                    self._data[index] = arr_or_data[index]

        elif hasattr(arr_or_data,"__iter__"):
            if not all([isinstance(type(arr),datastruct) for arr in arr_or_data]):
                raise TypeError(msg)
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
        
        return cls(copy.deepcopy(self._data))

    def __deepcopy__(self,memo):
        return self.copy()
    
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



class coordstruct(datastruct):
    
    def set_domain_handler(self,GeomHandler):
        self._domain_handler = GeomHandler

    @property
    def DomainHandler(self):
        if hasattr(self,"_domain_handler"):
            return self._domain_handler
        else: 
            return None

    def _get_subdomain_lims(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        if xmin is None:
            xmin = np.amin(self['x'])
        if xmax is None:
            xmax = np.amax(self['x'])
        if ymin is None:
            ymin = np.amin(self['y'])
        if ymax is None:
            ymax = np.amax(self['y'])
        if zmin is None:
            zmin = np.amin(self['z'])
        if zmax is None:
            zmax = np.amax(self['z'])
            
        xmin_index, xmax_index = (self.index_calc('x',xmin)[0],
                                    self.index_calc('x',xmax)[0])
        ymin_index, ymax_index = (self.index_calc('y',ymin)[0],
                                    self.index_calc('y',ymax)[0])
        zmin_index, zmax_index = (self.index_calc('z',zmin)[0],
                                    self.index_calc('z',zmax)[0])
        return xmin_index,xmax_index,ymin_index,ymax_index,zmin_index,zmax_index

    def create_subdomain(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        (xmin_index,xmax_index,
        ymin_index,ymax_index,
        zmin_index,zmax_index) = self._get_subdomain_lims(xmin,xmax,ymin,ymax,zmin,zmax)

        xcoords = self['x'][xmin_index:xmax_index]
        ycoords = self['y'][ymin_index:ymax_index]
        zcoords = self['z'][zmin_index:zmax_index]

        return self.__class__({'x':xcoords, 'y':ycoords,'z':zcoords})

    # def vtkStructuredGrid(self):
    #     x_coords = self.staggered['x']
    #     y_coords = self.staggered['y']
    #     z_coords = self.staggered['z']

    #     Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)

    #     grid = StructuredGrid(X,Z,Y)
    #     return grid

    def index_calc(self,comp,vals):
        return indexing.coord_index_calc(self,comp,vals)
    
    def check_plane(self,plane):
        if plane not in ['xy','zy','xz']:
            plane = plane[::-1]
            if plane not in ['xy','zy','xz']:
                msg = "The contour slice must be either %s"%['xy','yz','xz']
                raise KeyError(msg)
        slice_set = set(plane)
        coord_set = set(list('xyz'))
        coord = "".join(coord_set.difference(slice_set))
        return plane, coord

class flowstruct_base(datastruct):
    def __init__(self,coord_data,*args,from_hdf=False,**kwargs):
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        
        self._set_coords(coord_data,from_hdf,*args,**kwargs)   

    @property
    def Domain(self):
        return self._coorddata._domain_handler

    @property
    def CoordDF(self):
        return self._coorddata.centered

    @property
    def Coord_ND_DF(self):
        return self._coorddata.staggered

    def _set_coords(self,coord_data,from_hdf,*args,**kwargs):
        if from_hdf:
            filename = args[0]
            if len(args)>1:
                key=args[1]
            else:
                key = kwargs.get('key',None)
                
            from CHAPSim_post.legacy.post._meta import coorddata
            self._coorddata = coorddata.from_hdf(filename,key=key+"/coorddata")
        else:
            self._coorddata = coord_data.copy()

    @classmethod
    def from_hdf(cls,*args,coorddata=None,**kwargs):
        return cls(coorddata,*args,from_hdf=True,**kwargs)

    def to_hdf(self,filepath,key=None,mode='a'):
        super().to_hdf(filepath,key=key,mode=mode)
        self._coorddata.to_hdf(filepath,key=key+"/coorddata",mode='a')

        
    @property
    def times(self):
        return self.outer_index

    @property
    def comp(self):
        return self.inner_index

    @property
    def data_shape(self):
        return self._data[self.index[0]].shape

    def check_times(self,key,err=None,warn=None):
        if err is None:
            msg = f"flowstruct3D soes not have time {key}"
            err = KeyError(msg)
        return self.check_outer(key,err,warn=warn)

    def check_comp(self,key,err=None):
        if err is None:
            msg = f"Component {key} not in {self.__class__.__name__}"
            err = KeyError(msg)
        return self.check_inner(key,err)

    def concat(self,arr_or_data):
        msg= "The coordinate data of the flowstructs must be the same"
        if isinstance(arr_or_data,self.__class__):
            if self.CoordDF != arr_or_data.CoordDF:
                raise ValueError(msg)
        elif hasattr(arr_or_data,"__iter__"):
            if not all([self.CoordDF != arr.CoordDF for arr in arr_or_data]):
                raise ValueError(msg)
        super().concat(arr_or_data)

    def append(self,*args,**kwargs):
        msg = "This method is not available for this class"
        raise NotImplementedError(msg)

    def _arith_binary_op(self,other_obj,func):
        if isinstance(other_obj,self.__class__):
            msg= "The coordinate data of the flowstructs must be the same"
            if isinstance(other_obj,self.__class__):
                if not self.CoordDF != other_obj.CoordDF:
                    raise ValueError(msg)
        super()._arith_binary_op(other_obj,func)

    def copy(self):
        cls = self.__class__
        return cls(self._coorddata,self._data,copy=True)

class VTKstruct:
    def __init__(self,flowstruct_obj):
        if not isinstance(flowstruct_obj,flowstruct3D):
            msg = "This class can only be used on objects of type flowstruct3D"
            raise TypeError(msg)

        self._flowstruct = flowstruct_obj
        self._grid = self._flowstruct._coorddata.create_vtkStructuredGrid()

    def __getattr__(self,attr):
        if hasattr(self._grid,attr):
            inner_list = self._flowstruct.inner_index
            outer_list = self._flowstruct.outer_index


            grid = self[outer_list,inner_list]
            return getattr(grid,attr)

        elif hasattr(self._flowstruct, attr):
            return getattr(self._flowstruct,attr)

        else:
            msg = ("Method must be either an attribute "
                f"of VTKstruct, {self._grid.__class__},"
                f" or {self._flowstruct.__class__}")
            raise AttributeError(msg)
    def __deepcopy__(self,memo):
        new_flow_struct = self._flowstruct.copy()
        return self.__class__(new_flow_struct)
    def copy(self):
        return copy.deepcopy(self)
    @property
    def flowstruct(self):
        return self._flowstruct
        
    @property
    def Grid(self):
        inner_list = self._flowstruct.inner_index
        outer_list = self._flowstruct.outer_index


        return self[outer_list,inner_list]
        
    def __getitem__(self,key):
        return_grid = self._grid.copy()

        if isinstance(key[0],slice):
            start = key[0].start
            stop = key[0].stop
            for i,i_index in enumerate(self._flowstruct.outer_index):
                if i_index == start:
                    start_i = i
                
                if i_index == stop:
                    stop_i = i
            
            if stop_i < start_i:
                tmp = stop_i
                stop_i = start_i
                start_i= tmp

            outer_list = self._flowstruct.outer_index[start_i:stop_i+1]
        elif isinstance(key[0],list):
            outer_list = key[0]
        else:
            outer_list = [key[0]]

        if isinstance(key[1],slice):
            start = key[1].start
            stop = key[1].stop
            for i,i_index in enumerate(self._flowstruct.inner_index):
                if i_index == start:
                    start_i = i
                
                if i_index == stop:
                    stop_i = i
            
            if stop_i < start_i:
                tmp = stop_i
                stop_i = start_i
                start_i= tmp

            inner_list = self._flowstruct.inner_index[start_i:stop_i+1]
        elif isinstance(key[1],list):
            inner_list = key[1]
        else:
            inner_list = [key[1]]

        keys = list(itertools.product(outer_list,inner_list))
        if len(keys) > 1:
            keys = [key for key in keys if key in self._flowstruct.index]

        for k in keys:

            data = self._flowstruct[k]
            if len(set(outer_list)) < 2:
                k = k[1]
            return_grid.cell_arrays[np.str_(k)] = data.flatten()
        return return_grid

    def __iadd__(self,other_VTKstruct):
        if not isinstance(other_VTKstruct,self.__class__):
            msg = "This operation can only be used with other VTKstruct's"
            raise TypeError(msg)
        
        if not np.allclose(self._grid.points,other_VTKstruct._grid.points):
            msg = "The grids of the VTKstruct's must be allclose"
            raise ValueError(msg)

        self._flowstruct.concat(other_VTKstruct._flowstruct)

        return self

class flowstruct3D(flowstruct_base):

    def _set_coords(self,CoordDF,from_hdf,*args,**kwargs):
        super()._set_coords(CoordDF,from_hdf,*args,**kwargs)
        if len(self.CoordDF.index) != 3:
            msg = ("for a 3D flowstruct the number of keys in the "
                    f"coordstruct should be 3 not {len(self._CoordDF.index)}")
            raise ValueError(msg)

    @property
    def VTK(self):
        return VTKstruct(self)

    def get_unit_figsize(self,plane):
        plane, coord = self.CoordDF.check_plane(plane)

        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]

        x_size = 1.5*(np.amax(x_coords) - np.amin(x_coords))
        z_size = 1.2*(np.amax(z_coords) - np.amin(z_coords))

        return x_size,z_size
        
    def create_subdomain(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        (xmin_index,xmax_index,
        ymin_index,ymax_index,
        zmin_index,zmax_index) = self.CoordDF._get_subdomain_lims(xmin,xmax,ymin,ymax,zmin,zmax)

        new_coorddata = self._coorddata.create_subdomain(xmin,xmax,ymin,ymax,zmin,zmax)
        
        shape = (len(self.index),zmax_index-zmin_index,
                ymax_index-ymin_index,
                xmax_index-xmin_index )

        vals_array = np.zeros(shape)
        for i,vals in enumerate(self.values):
            vals_array[i] = vals[zmin_index:zmax_index,
                                ymin_index:ymax_index,
                                xmin_index:xmax_index]

        return self.__class__(new_coorddata,vals_array,index=self.index)


    def create_slice(self,index):
        slice_dict = {index : self[index]}
        return self.__class__(self._coorddata,slice_dict)

    def to_vtk(self,file_name):
        
        for i,time in enumerate(self.times):
            file_base, file_ext = os.path.splitext(file_name)
            if file_ext == ".vtk":
                writer = vtk.vtkStructuredGridWriter()
            elif file_ext == ".vts":
                writer = vtk.vtkXMLStructuredGridWriter()
            elif file_ext == "":
                file_name = file_base +".vts"
                writer = vtk.vtkXMLStructuredGridWriter()
            else:
                msg = "This function can only use the vtk or vts file extension not %s"%file_ext
                raise ValueError(msg)

            grid = self._coorddata.create_vtkStructuredGrid()
            if len(self.times) > 1:
                num_zeros = int(np.log10(len(self.times)))+1
                ext = str(num_zeros).zfill(num_zeros)
                file_name = os.path.join(file_name,".%s"%ext)

            for comp in self.comp:
                grid.cell_arrays[np.str_(comp)] = self[time,comp].flatten()
            # pyvista.save_meshio(file_name,grid,file_format="vtk")

            
            writer.SetFileName(file_name)

            if vtk.vtkVersion().GetVTKMajorVersion() <= 5:
                grid.Update()
                writer.SetInput(grid)
            else:
                writer.SetInputData(grid)
                
            writer.Write()


    def plot_contour(self,comp,plane,axis_val,time=None,fig=None,ax=None,pcolor_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        plane, coord = self.CoordDF.check_plane(plane)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = indexing.contour_indexer(self[time,comp],axis_index,coord)

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        ax = ax.pcolormesh(X,Y,flow_slice.squeeze(),**pcolor_kw)

        return fig, ax

    def plot_vector(self,plane,axis_val,time=None,spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        quiver_kw = cplt.update_quiver_kw(quiver_kw)

        time = self.check_times(time)

        plane, coord = self.CoordDF.check_plane(plane)

        coord_1 = self.CoordDF[plane[0]][::spacing[0]]
        coord_2 = self.CoordDF[plane[1]][::spacing[1]]
        UV_slice = [chr(ord(x)-ord('x')+ord('u')) for x in plane]
        U = self[time,UV_slice[0]]
        V = self[time,UV_slice[1]]

        axis_index = self.CoordDF.index_calc(coord,axis_val)

        U_space, V_space = indexing.vector_indexer(U,V,axis_index,coord,spacing[0],spacing[1])
        U_space = U_space.squeeze(); V_space = V_space.squeeze()
        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        scale = np.amax(U_space[:,:])*coord_1.size/np.amax(coord_1)/scaling
        ax = ax.quiver(coord_1_mesh, coord_2_mesh,U_space[:,:].T,V_space[:,:].T,angles='uv',scale_units='xy', scale=scale,**quiver_kw)

        return fig, ax

    def plot_isosurface(self,comp,Value,time=None,y_limit=None,x_split_pair=None,fig=None,ax=None,surf_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,projection='3d',**kwargs)
        surf_kw = cplt.update_mesh_kw(surf_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        z_coords = self.CoordDF['z']
        y_coords = self.CoordDF['y']
        x_coords = self.CoordDF['x']

        if x_split_pair is None:
            x_split_pair = [np.amin(x_coords),np.amax(x_coords)]
        
        x_index_list = self.CoordDF.index_calc('x',x_split_pair)

        x_coords = x_coords[x_index_list[0]:x_index_list[1]]

        if y_limit is not None:
            y_index = self.CoordDF.index_calc('y',y_limit)[0]
            y_coords = y_coords[:y_index]
        else:
            y_index = y_coords.size

        Z,Y,X = misc_utils.meshgrid(z_coords,y_coords,x_coords)

        flow_array = self[time,comp][:,:y_index,x_index_list[0]:x_index_list[1]]

        ax = ax.plot_isosurface(Z,X,Y,flow_array,Value,**surf_kw)
        coord_lims = [np.amax(Z) - np.amin(Z),np.amax(X) - np.amin(X),np.amax(Y) - np.amin(Y) ]
        ax.axes.set_box_aspect(coord_lims)
        return fig, ax


    def plot_surf(self,comp,plane,axis_val,time=None,x_split_pair=None,fig=None,ax=None,surf_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,projection='3d',**kwargs)
        surf_kw = cplt.update_mesh_kw(surf_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        plane, coord = self.CoordDF.check_plane(plane)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = indexing.contour_indexer(self[time,comp],axis_index,coord)

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        if x_split_pair is None or plane[0] != 'x':
            x_split_pair = [np.amin(x_coord),np.amax(x_coord)]
        
        x_index_list = self.CoordDF.index_calc('x',x_split_pair)

        x_coord = x_coord[x_index_list[0]:x_index_list[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = indexing.contour_indexer(self[time,comp],axis_index,coord)
        flow_slice = flow_slice[:,x_index_list[0]:x_index_list[1]]

        ax = ax.plot_surface( Y,X,flow_slice,**surf_kw)

        return fig, ax


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

    # def _hdf_extract(self,filename,key=None):
    #     try:
    #         hdf_file = h5py.File(filename,mode='r')

    #         if key is None:
    #             hdf_data = hdf_file
    #         else:
    #             hdf_data=hdf_file[key]
            
    #         index = list(hdf_data.keys())
    #         list_vals=[]
    #         for k in index:
    #             val = list(hdf_data[k][:])
    #             if len(val)==1:
    #                 val = val[0]
    #             list_vals.append(val)
    #         index.extend(hdf_data.attrs.keys())
    #         for k in hdf_data.attrs.keys():
    #             list_vals.append(hdf_data.attrs[key].decode('utf-8'))
    #     finally:
    #         hdf_file.close()
    #     self._meta = {i:val for i, val in zip(index,list_vals)}

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

        