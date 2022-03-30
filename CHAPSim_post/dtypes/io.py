import h5py
import os
import numpy as np
import copy
import warnings
import CHAPSim_post as cp
from scipy import io
from numbers import Number
class hdfHandler:

    _available_ext = ['.h5','.hdf5']
    _write_modes = ['r+','w','w-','x','a']
    _file_must_exist_modes = ['r','w-','x']
    
    def __init__(self,filename,mode='r',key=None):
        
        filename = self._check_h5_file_name(filename,mode)

        self._hdf_file = h5py.File(filename,mode=mode)

        self._hdf_data = self._check_h5_key(self._hdf_file,key,mode=mode)
    
    def access_key(self,key):
        new_h5obj = self.copy()
        mode = self._hdf_file.mode
        new_h5obj._hdf_data = self._check_h5_key(self._hdf_data,key,mode=mode)
        return new_h5obj

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
        try:
            return self._hdf_data.__getitem__(key)
        except KeyError:
            msg = (f"The key {key} doesn't exist in the HDF file "
                    f"{self._hdf_file.filename} at location {self._hdf_data.name}")
            raise KeyError(msg) from None

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
        return bool(set(self._write_modes).intersection([mode]))

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
        
        if key is None:
            return h5_obj
        
        try:
            return h5_obj[key]
        except KeyError:
            if self._write_allowed(mode):
                return h5_obj.create_group(key)
            else:
                avail_keys = self._get_key_recursive(h5_obj,groups_only)
                msg = (f"Key {key} not found in file HDF "
                        f"file {self._hdf_file.name},"
                        f"keys available are {avail_keys}")
                raise KeyError(msg) from None
            
        # avail_keys = self._get_key_recursive(h5_obj,groups_only)
        # if self._check_key_recursive(h5_obj,key,groups_only):
        #     return h5_obj[key]
        # else:
        #     if key is None:
        #         return h5_obj
        #     elif self._write_allowed(mode):
        #         return h5_obj.create_group(key)
        #     else:
        #         msg = (f"Key {key} not found in file HDF "
        #                 f"file, keys available are {avail_keys}")
        #         raise KeyError(msg)

    def _check_key_recursive(self,h5_obj,key,groups_only=True):
        if key is None:
            return False

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
    def contruct_index_from_hdfkey(self,key):
        index_array = self._hdf_data[key]

        if index_array.ndim == 1:
            return [index.decode('utf-8') for index in index_array]
        else:
            return [tuple(x.decode('utf-8') for x in index) for index in index_array]
    def set_h5order(self,order):
        self._hdf_data.attrs['order'] = order

    def get_h5order(self,keys):
        if 'order' in self._hdf_data.attrs.keys():
            order = list(self._hdf_data.attrs['order'][:])
        else:
            return keys

        return [keys[i] for i in order]

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self,memo):
        key = self._hdf_data.name
        filename = self._hdf_file.filename
        mode = self._hdf_file.mode
        return self.__class__(filename,mode,key)

    def __del__(self):
        try:
            if hasattr(self,'_hdf_file'):
                if self._hdf_file:
                    self._hdf_file.close()
        except Exception:
            pass
        
class matHandler:
    def __init__(self, file_name):
        self._file_name = os.path.basename(file_name) + ".mat"
        self.__mat_dict = dict()
        
    def __setitem__(self,key,value):
        
        v = self._convert(value)
        if not isinstance(key,str):
            msg = "Key must be a str"
            raise TypeError(msg)
        
        self.__mat_dict[key] = v
    
    def __mathandle__(self):
        return self.__mat_dict
    
    def _convert(self,value):
        if hasattr(value,"__mathandle__"):
            v = value.__mathandle__()
            if not isinstance(v,(dict,np.ndarray,Number)):
                msg = (f"Class {value.__class__.__name__} does not have"
                       " __mathandle__ class that returns a dict or numpy array")
                raise AttributeError(msg)
            return v
        if isinstance(value, (np.ndarray,dict)):
            return value
        
        msg = ("inputs must be classes with __mathandle__ function,"
               " numpy arrays, or dictionaries")
        raise TypeError(msg)
    
    def save(self):
        io.savemat(self._file_name,self.__mat_dict,appendmat=True)
        
        
        