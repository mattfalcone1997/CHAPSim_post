"""
flowstruct
^^^^^^^^^^

This sub-module contains the primary storage and visualisation 
classes of CHAPSim_post. It is built around the ``FlowStructND`` 
class which can store and visualise structured data of artibrary 
dimension. Derived classes are used to store must of the data such as the ``FlowStruct3D`` class for the instantaneous data. This class is designed to have its data components accessed by keys. These classes can be output to HDF5 file format and are derived from the ``datastruct`` class found in the core module. 

"""

from abc import abstractproperty
from scipy import interpolate
from typing import (NewType,
                    Iterable,
                    Union,
                    Callable,
                    List,
                    Tuple,
                    Any)

from numbers import Number
import CHAPSim_post.plot as cplt
# from CHAPSim_post.post._meta import coorddata
from CHAPSim_post.utils import misc_utils, docstring
import matplotlib as mpl
from .vtk import VTKstruct2D, VTKstruct3D
from .coords import coordstruct, AxisData
from .core import *

FlowStructType =  NewType('FlowStructType',datastruct)

class _FlowStruct_base(datastruct):
    """
    Base class for the FlowStruct classes
    """
    def __init__(self,*args,from_hdf=False,**kwargs):
        
        self._set_coorddata(args[0],**kwargs)
            
        if isinstance(args[0],AxisData):
            args = args[1:]
            
        
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        
        self.check_shape()

    @property
    def Domain(self):
        """
        Property for accessing the FlowStruct's DomainHandler
        """
        return self._coorddata._domain_handler

    @property
    def CoordDF(self):
        """
        Property for accessing the FlowStruct's centered data
        """
        return self._coorddata.centered

    @property
    def Coord_ND_DF(self):
        """
        Property for accessing the FlowStruct's staggered data is present
        """
        return self._coorddata.staggered

    def remove_time(self,time: Number ) -> None:
        """
        Removes specified time from FlowStruct
        """
        del_index = [ index for index in self.index if index[0] == time]
        for index in del_index:
            del self[index]
    
    def shift_times(self,time: Number) -> FlowStructType:
        """
        Shifts all the FlowStruct's times by a scalar
        """
        new_index = []
        for index in self.index:
            new_time = float(index[0]) + time
            new_index.append(self._indexer._item_handler((new_time,*index[1:])))

        return self.from_internal(dict(zip(new_index, self._data)))
            

    def _get_identity_transform(self) ->  Callable:
        """
        For use on ``plot_line`` internals if data transform is given as None
        """
        def identity_transform(data):
            return data
        
        return identity_transform

    def _check_datatransforms(self,
                              transform_xdata:  Union[ Callable,None],
                              transform_ydata:  Union[ Callable,None])->  Union[ Callable,  Callable] :  
        """
        Checks validity of transforms  on ``plot_line`` functions and replaces ``None`` with the identity transform

        Parameters
        ----------
        transform_xdata : Callable or None
            Transform for line x data or None
        transform_ydata : Callable
            Transform for line y data 

        Returns
        -------
        Callable, Callable
            

        Raises
        ------
        TypeError
            If either input is not callable or not None

        """
        
        if transform_xdata is None:
            transform_xdata = self._get_identity_transform()
        if transform_ydata is None:
            transform_ydata = self._get_identity_transform()

        if not hasattr(transform_xdata,'__call__'):
            msg = "transform_xdata must be None or callable"
            raise TypeError(msg)
        if not hasattr(transform_ydata,'__call__'):
            msg = "transform_xdata must be None or callable"
            raise TypeError(msg)

        return transform_xdata, transform_ydata

        
    @property
    def times(self) ->  Union[ List[float],None]:
        """
        
        Returns a list of the FlowStruct's times or None
        
        Returns
        -------
        list of float or None
            list of the FlowStruct's times
            
        """
        if 'None' in self.index.outer_index:
            return None
        else:
            return sorted([float(x) for x in self.outer_index])

    @property
    def comp(self) -> Index:
        """
        Property returning the FlowStruct's components

        Returns
        -------
        Index
            Components of the FlowStruct
        """
        return self.inner_index

    @abstractproperty
    def VTK(self):
        pass

    @property
    def shape(self)-> tuple:
        """
        Returns the shape of the FlowStruct's underlying Numpy arrays

        Returns
        -------
        Tuple
            Shape of the array
        """
        key = self.index[0]
        return self[key].shape

    def _dstruct_ini(self, coorddata, dstruct, copy=False):
        self._dict_ini(coorddata, dstruct.to_dict(), copy=copy)

    def _set_coorddata(self,file_or_coorddata,**kwargs):
        if isinstance(file_or_coorddata,AxisData):
            self._coorddata = file_or_coorddata.copy()
        else:
            path = os.fspath(file_or_coorddata)
            key = kwargs.get('key',None)
            if key is None:
                coord_key = '/coorddata'
            else:
                coord_key = os.path.join(key,'coorddata')

            self._coorddata = AxisData.from_hdf(path,key=coord_key)

    def check_times(self,key:  Union[Number,None],
                    err: Exception =None,
                    warn: str = None) -> str:
        """
        Checks whether time is present in FlowStruct and
        corrects if possible

        Parameters
        ----------
        key :  Union[Number,None]
            Input key
        err : Exception, optional
            Error to be raised if there is an issue, by default None
        warn : str, optional
            Warning to be raised if issue can be fixed, by default None

        Returns
        -------
        str
            The correct key
        """
        
        if err is None:
            msg = f"{self.__class__.__name__} object does not have time {key}"
            err = KeyError(msg)
        if warn is None:
            warn = (f"{self.__class__.__name__} object does not have time {key}"
                    f" only one {key} present. This time will be used")
        
        return self.check_outer(key,err,warn=warn)

    def check_comp(self,
                   key: str,
                   err: Exception =None) -> str:
        """
        Checks whether component is present in FloStruct and raises error 'err' if not else a defaut KeyError is raised

        Parameters
        ----------
        key : str
            INput component
        err : Exception, optional
            Custom exception to be raised, by default None

        Returns
        -------
        str
            Valid component
        """
        if err is None:
            msg = f"Component {key} not in {self.__class__.__name__}"
            err = KeyError(msg)
        return self.check_inner(key,err)

    def check_line_labels(self,items: list,
                          labels:  List[str]):
        """
        Checks that appropriate numbers and types of line labels are given

        Parameters
        ----------
        items : list
            items to be plotted
        labels :  List[str]
            list of labels

        Raises
        ------
        TypeError
            Ensures input types are valid
        ValueError
            Ensures to correct number of labels are present
        """
        if labels is None:
            return

        if not isinstance(labels,(tuple,list)):
            msg = "The labels provided must be a list or tuple"
            raise TypeError(msg)

        if len(items) != len(labels):
            msg = "The number of labels must be equal the number of items"
            raise ValueError(msg)

    def is_fully_compatible(self,data: FlowStructType) -> bool:
        """
        Checks whether a FlowStruct is compatible with this instance. THis is primarily used for arithmetic between FlowStructs

        Parameters
        ----------
        data : FlowStructType
            Another FlowStruct

        Returns
        -------
        bool
            bool signifying whether the FlowStruct is compatable
        """
        if not self.is_shape_compatible(data):
            return False

        if not self.is_coords_compatible(data):
            return False

        if not self.is_type_compatible(data):
            return False

        return True

    def is_type_compatible(self,data: FlowStructType) -> bool:
        """
        Checks types

        Parameters
        ----------
        data : FlowStructType
            Another FlowStruct

        Returns
        -------
        bool
            returns true if the types are the compatible
        """
        return  isinstance(data,self.__class__)
    
    def is_coords_compatible(self,data: FlowStructType) -> bool:
        """
        Checks that the coorddata for the FlowStructs are compatible

        Parameters
        ----------
        data : FlowStructType
            Another FlowStruct

        Returns
        -------
        bool
            returns True if the Flowstructs have equivalent coorddata
        """
        return self._coorddata == data._coorddata

    def is_shape_compatible(self,data: FlowStructType) -> bool:
        """
        Checks that the shapes are the same

        Parameters
        ----------
        data : FlowStructType
            Another FlowStruct

        Returns
        -------
        bool
            returns True if their shapes are the same
        """
        return self.shape == data.shape

    def check_shape(self) -> bool:
        """
        Checks that the shape of the input numpy arrays are valid

        Returns
        -------
        bool
            returns True if the shapes are valid

        Raises
        ------
        ValueError
            raised if not all arrays have the same shape
        ValueError
            raised if the shapes are not in the coordinate data
        ValueError
            Checks all arrays have the same dimension

        """
        if not all([arr.shape == self.shape for _, arr in self]):
            msg = f"All arrays must have the same shape in {self.__class__.__name__} class"
            raise ValueError(msg)
        
        coord_shapes = tuple([coord.size for _,coord in self.CoordDF])

        if not all(s in coord_shapes for s in self.shape):
            msg = "The coordinate data is of different shape %s to the array data %s"%(coord_shapes,self.shape)
            raise ValueError(msg)

        dims = [arr.ndim for _, arr in self]
        if not all(dim == self._dim for dim in dims):
            msg = f"All array ndims in {self.__class__.__name__} must be equal to {self._dim}"
            raise ValueError(msg)
    
    def concat(self,arr_or_data:  Union[FlowStructType, Iterable[FlowStructType]]) -> FlowStructType:
        """
        Concatenates FlowStructs or iterables of FlowStructs

        Parameters
        ----------
        arr_or_data : FlowStructType
            FlowStruct or iterables of FlowStructs

        Raises
        ------
        ValueError
            Raised if FlowStructs are not compatible

        """

        msg= "The flowstructs must be the compatible"
        if isinstance(arr_or_data,self.__class__):
            if not self.is_fully_compatible(arr_or_data):
                raise ValueError(msg)
        elif hasattr(arr_or_data,'__iter__'):
            if not all(self.is_fully_compatible(data) for data in arr_or_data):
                raise ValueError(msg)
        else:
            msg = ("For concatenation the object must a "
               f"{self.__class__.__name__} or an iterable of them")
            raise ValueError(msg)

        super().concat(arr_or_data)

    def append(self,*args,**kwargs):
        msg = "This method is not available for this class"
        raise NotImplementedError(msg)

    def symmetrify(self,dim: int =None) -> FlowStructType:
        """
        Flips FlowStruct along a dimension

        Parameters
        ----------
        dim : int, optional
            Dimension to be flipped, by default None

        Returns
        -------
        FlowStructType
            Flipped FlowStruct
        """
        symm_dstruct = super().symmetrify(dim=dim)
        return self.from_internal(symm_dstruct)

    def _arith_binary_op(self,other_obj,func):
        if isinstance(other_obj,np.ndarray):
            pass
        elif isinstance(other_obj,numbers.Number):
            pass
        else:
            msg= "The flowstructs must be compatible in type, shape and coordinate data"
            if not self.is_fully_compatible(other_obj):
                raise ValueError(msg)

        dstruct = super()._arith_binary_op(other_obj,func)
        return self.from_internal(dstruct)

    def _arith_unary_op(self,func):
        dstruct = super()._arith_unary_op(func)
        return self.from_internal(dstruct)

    def copy(self) -> FlowStructType:
        """
        Returns a deep copy of the FlowStruct

        Returns
        -------
        FlowStructType
            deep copy of the FlowStruct
        """
        cls = self.__class__
        return cls(self._coorddata,self.to_dict(),copy=True)

class _FlowStruct_slicer:
    def __init__(self,flowstruct_obj):
        self._ref = weakref.ref(flowstruct_obj)

    def get_slicer(self,key):
        output_slice, _ = self._get_index_slice(key)
        return output_slice
    
    def __getitem__(self,key):
        if self._ref()._dim == 1:
            key = (key)
        
        flow_struct = self._ref().copy()
        output_slice, output_slice_nd = self._get_index_slice(key)

        new_coorddata, data_layout,\
        wall_normal_line, polar_plane = self._get_new_coorddata(output_slice,
                                                                 output_slice_nd)
        
        new_array = []
        for array in flow_struct._data:
            new_array.append(array[output_slice].squeeze())
        new_array = np.stack(new_array,axis=0)
        
        return FlowStructND(new_coorddata,
                            new_array,
                            data_layout = data_layout,
                            wall_normal_line = wall_normal_line,
                            polar_plane = polar_plane,
                            index=flow_struct.index)

    def _get_index_slice(self,key):
        output_slicer = []
        output_slicer_nd = []
        flow_struct = self._ref()
        data_layout = flow_struct._data_layout

        extra_slices =[slice(None)]*(len(data_layout) - len(key))
        for k, data in zip(key,data_layout):
            if isinstance(k,slice):
                if k.start is None:
                    start = None
                else:
                    start = flow_struct.CoordDF.index_calc(data,k.start)[0]
                if k.stop is None:
                    stop = None; stop_nd = None
                else:
                    stop = flow_struct.CoordDF.index_calc(data,k.stop)[0]
                    stop_nd = stop + 1

                if k.step is not None:
                    msg = "Patterns cannot be provided to the flowstruct slicer"
                output_slicer.append(slice(start,stop))
                output_slicer_nd.append(slice(start,stop_nd))

            else:
                stop = flow_struct.CoordDF.index_calc(data,k)[0]
                stop_nd = flow_struct.Coord_ND_DF.index_calc(data,k)[0]
                
                output_slicer.append(slice(stop,stop+1))
                output_slicer_nd.append(slice(stop_nd,stop_nd+1))
        
        output_slicer.extend(extra_slices)
        output_slicer_nd.extend(extra_slices)
        return tuple(output_slicer), tuple(output_slicer_nd)


    def _get_new_coorddata(self,output_slice, output_slice_nd):
        flow_struct = self._ref()
        CoordDF = flow_struct.CoordDF
        Coord_ND_DF = flow_struct.Coord_ND_DF

        new_nd_dict = {}
        new_dict = {}
        data_layout = []
        for i, data in enumerate(flow_struct._data_layout):
            coord_nd = Coord_ND_DF[data][output_slice_nd[i]]
            coord = CoordDF[data][output_slice[i]]
            if len(coord) == 1:
                continue
            
            data_layout.append(data)
            new_nd_dict[data] = coord_nd
            new_dict[data] = coord
        
        if flow_struct._wall_normal_line not in data_layout:
            wall_normal_line = None
        else:
            wall_normal_line = flow_struct._wall_normal_line

        if flow_struct._polar_plane is not None:
            if all(axis in data_layout for axis in flow_struct._polar_plane):
                polar_plane = flow_struct._polar_plane
            else:
                polar_plane = None
        else:
            polar_plane = None

        
        new_CoordDF = coordstruct(new_dict)
        new_Coord_ND_DF = coordstruct(new_nd_dict)
        new_coorddata = AxisData(flow_struct.Domain,new_CoordDF,new_Coord_ND_DF)

        return new_coorddata, data_layout, wall_normal_line, polar_plane

slicer_example = """
Example using FlowStructND.slice
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``fstruct`` is a 3D FlowStructND with a domain (x, y, z) of (0:30,-1:1,0:4)
in its three dimensions. If I wanted to create a plane at x = 15:
.. code-block:: python
    fstruct_plane = fstruct.slice[15,:,:]
    
If I wanted to create a FlowStruct with only y_values from -1 to 0:

.. code-block:: python
    fstruct_half =fstruct.slice[:,:0,:]
    
"""

docstring.sub.update(slicer=slicer_example)


class FlowStructND(_FlowStruct_base):
    """
    The class that does much of the heavy lifting for the module. It 
    contains the functionality to return FlowStructND 'sub-domains' 
    such as planes of 3D data as well as visualising data if the 
    method is available  to the dimension of the FlowStruct. It is 
    base class to the more commonly used FlowStruct3D, FlowStruct2D,
    FlowStruct1D, FlowStruct1D_time  classes.
    """
    
    def _array_ini(self, array, index=None,data_layout=None,wall_normal_line=None,polar_plane= None, copy=False):
        super()._array_ini(array, index=index, copy=copy)
        self._set_data_layout(data_layout, wall_normal_line, polar_plane)

    def _dict_ini(self, dict_data, data_layout=None,wall_normal_line=None,polar_plane= None, copy=False):
        super()._dict_ini(dict_data, copy=copy)
        self._set_data_layout(data_layout, wall_normal_line, polar_plane)

    def _dstruct_ini(self, dstruct, data_layout=None,wall_normal_line=None,polar_plane= None, copy=False):
        self._dict_ini( dstruct.to_dict(),
                        data_layout=data_layout,
                        wall_normal_line=wall_normal_line,
                        polar_plane= polar_plane, 
                        copy=copy)
    @property
    def VTK(self) ->  Union[VTKstruct2D,VTKstruct3D]:
        """
        Returns a VTKStruct if the dimensions allow 

        Returns
        -------
         Union[VTKstruct2D,VTKstruct3D]
            Allows inter-operability with VTK based tools such as 
            Pyvista

        Raises
        ------
        Exception
            raised if the FlowStruct dimension is not 2 or 3
        """
        use_cell_data = self._coorddata.contains_staggered
        if self._dim == 2:
            return VTKstruct2D(self,cell_data = use_cell_data)
        elif self._dim ==3:
            return VTKstruct3D(self,cell_data = use_cell_data)
        else:
            raise Exception
        
    def Translate(self,translation):
        if not len(translation) == self._dim:
            msg = ("Length of translation vector must"
                   " be the same as the FlowStruct dimension")
            raise ValueError(msg)
        
        reorder = np.argsort(self._data_layout)
        coord_order = np.arange(self._dim)[np.argsort(self.CoordDF.index)]

        translation = np.array(translation)
        self.CoordDF.Translate(translation[reorder][coord_order])
        self.Coord_ND_DF.Translate(translation[reorder][coord_order])
            
    def to_vtk(self,file_name: str):
        """
        Outputs .vts file of FlowStruct

        Parameters
        ----------
        file_name : str
            file path for saving
        """
        self.VTK.save_vtk(file_name)

    def to_hdf(self,filepath: str,
               mode: str ='a',
               key: Union[str,None] =None) -> hdfHandler:
        """
        Saves FlowStruct into the HDF5file format

        Parameters
        ----------
        filepath : str
            path to be saved
        mode : str, optional
            write mode, look at h5py documentation for more information, by default 'a'
        key :  Union[str,None], optional
            key location for the HDF5 file, by default None

        Returns
        -------
        hdfHandler
            for easy handling of hdf data, this is returned if more keys need to be added
        """
        
        hdf_obj = super().to_hdf(filepath,mode=mode,key=key)
        key = hdf_obj.name

        coord_key = os.path.join(key,'coorddata')
        self._coorddata.to_hdf(filepath,key=coord_key,mode='a')

        if self._polar_plane is not None:
            hdf_obj.attrs['polar_plane'] = np.array(self._polar_plane,dtype=np.string_)
        if self._wall_normal_line is not None:
            hdf_obj.attrs['wall_normal_line'] = str(self._wall_normal_line)
        hdf_obj.attrs['data_layout'] = np.array(self._data_layout,dtype=np.string_)
        
        
        return hdf_obj
    
    def __mathandle__(self):
        data = super().__mathandle__()
        coords = self._coorddata.__mathandle__()
        return dict(data=data,
                    coords=coords)
        
    def _file_extract(self, filename, key=None):
        hdf_obj =  super()._file_extract(filename, key=key)
        key = hdf_obj.name

        self._set_coorddata(filename,key=key)

        data_layout = list(x.decode('utf-8') for x in hdf_obj.attrs['data_layout'])
        if 'polar_plane' in hdf_obj.attrs:
            polar_plane = list(x.decode('utf-8') for x in hdf_obj.attrs['polar_plane'])
        else:
            polar_plane = None

        if 'wall_normal_line' in hdf_obj.attrs:
            wall_normal_line = hdf_obj.attrs['wall_normal_line']
        else:
            wall_normal_line = None

        self._set_data_layout(data_layout, wall_normal_line, polar_plane)

    def update_coord(self,old_coord,new_coord,array=None):
        if old_coord not in self._data_layout:
            msg = "Old coord key must be in the data layout"
            raise KeyError(msg)
        
        if self.Coord_ND_DF is not None:
            raise Exception
        
        if array is None:
            self.CoordDF[new_coord] = array
        else:
            self.CoordDF[new_coord] = self.CoordDF[old_coord]
            
        del self.CoordDF[old_coord]
        
        index = self._data_layout.find(old_coord)
        self._data_layout.remove(old_coord)
        self._data_layout.insert(index,new_coord)
        
        if old_coord == self._wall_normal_line:
            self._wall_normal_line = new_coord
            
        if self._polar_plane is not None:
            if old_coord in self._polar_plane:
                self._polar_plane.replace(old_coord,new_coord)
        
    def from_internal(self, *args, **kwargs) -> FlowStructType:
        """
        Gives datastruct type interface for creating a FlowStruct of the
        same type

        Returns
        -------
        FlowStruct
            Resulting FlowStruct
        """
        kwarg_dict = dict(data_layout=self._data_layout,
                            wall_normal_line = self._wall_normal_line,
                            polar_plane = self._polar_plane)

        kwargs.update(kwarg_dict)
        return FlowStructND(self._coorddata,*args,**kwargs)

    def copy(self):
        return self.from_internal(self.to_dict(),copy=True)

    def _set_data_layout(self,data_layout,wall_normal_line,polar_plane):
        if data_layout is None:
            raise Exception
        
        

            
        for i, data in enumerate(data_layout):
            if not data in self.CoordDF.index:
                msg = "Coord index must be in the coordstruct"
                raise ValueError(msg)
            
            if self.CoordDF[data].size != self.shape[i]:
                coord_shape = tuple(self.CoordDF[d].size for d in data_layout)
                msg = ("There is an issue with the data layout\n"
                        f"Coordinate shape: {coord_shape}. Data Shape: {self.shape}.")
                raise ValueError(msg)
        


        if polar_plane is not None:
            if not all(axis in data_layout for axis in polar_plane):
                msg = "The polar plane must be in the data layout"
                raise ValueError(msg)
        if polar_plane is None:
            self._polar_plane = None
        else:
            self._polar_plane = list(polar_plane)
        self._data_layout = list(data_layout)
        self._wall_normal_line = wall_normal_line
        self._dim = self._data[0].ndim


        # remove coordinates not in data layout
        for comp in self.CoordDF.index:
            if comp not in self._data_layout:
                del self.CoordDF[comp]
                del self.Coord_ND_DF[comp]
        
    def plot_line(self,comp: str,
                  time: Number =None,
                  label: str =None,
                  channel_half: bool =False,
                  transform_ydata:  Callable=None,
                  transform_xdata:  Callable=None, 
                  fig: cplt.CHAPSimFigure =None,
                  ax: cplt.AxesCHAPSim =None,
                  line_kw: dict =None,
                  **kwargs) ->  Union[cplt.CHAPSimFigure,cplt.AxesCHAPSim ]:
        """
        Plots a line if the FlowStruct is 1D

        Parameters
        ----------
        comp : str
            Component of FlowStruct to be plotted
        time : Number, optional
            time to be plotted, by default None
        label : str, optional
            label to be based to line object, by default None
        channel_half : bool, optional
            If the flow is a channel, whether to plot only half, by default False
        transform_ydata :  Callable, optional
            Function applied to ydata, by default None
        transform_xdata :  Callable, optional
            Function applied to xdata, by default None
        fig : cplt.CHAPSimFigure, optional
            Input figure, it is created if None, by default None
        ax : cplt.AxesCHAPSim, optional
            Input axes, it is created if None, by default None
        line_kw : dict, optional
            dictionary passed to plotting function, by default None
        **kwargs :
            keyword arguments passed to figure creation routine
            
        Returns
        -------
         Union[cplt.CHAPSimFigure,cplt.AxesCHAPSim ]
            returns figure and axes associated with line plot
        """
        time = self.check_times(time)
        comp = self.check_comp(comp)

        self._check_dim(1)

        data = self[time,comp].squeeze()

        fig, ax = self.plot_line_data(data,label=label,channel_half=channel_half,transform_ydata=transform_ydata,
                                         transform_xdata=transform_xdata, fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        return fig, ax

    def _check_line_channel(self):
        return self._wall_normal_line and self.Domain.is_channel

    def get_dim_from_axis(self,axis: str)-> int:
        """
        Gets the array dimension from given coord 

        Parameters
        ----------
        axis : str
            The axis present in the data_layout: e.g. 'x'

        Returns
        -------
        int
            array dimension

        Raises
        ------
        ValueError
            Raises exception if the axis is not present in the FlowStruct
        """
        if len(axis) > 1:
            return tuple([self.get_dim_from_axis(a) for a in axis])
        if self._data_layout.count(axis) > 1:
            msg = "The index axis cannot appear more than once"
            raise ValueError(msg)
        elif self._data_layout.count(axis) == 0:
            msg = "The index axis provided does not appear in the data layout"
            raise ValueError(msg)

        return "".join(self._data_layout).find(axis)

    def reduce(self,numpy_op:  Callable,axis: str) -> FlowStructType:
        """
        Produces FlowStruct by applying numpy operation along axes of a FlowStruct

        Parameters
        ----------
        numpy_op :  Callable
            Numpt operation e.g. numpy.amax
        axis : str
            FlowStruct Axis e.g. 'x'

        Returns
        -------
        FlowStructType
            FlowStruct from reduction operation
        """
        array_axis = self.get_dim_from_axis(axis)
        new_datalayout = self._data_layout.copy()
        new_coord = self.CoordDF.copy()
        new_coord_nd = None if self.Coord_ND_DF is None else self.Coord_ND_DF.copy()

        for a in axis:
            new_datalayout.remove(a)
            del new_coord[a]
            if new_coord_nd is not None:
                del new_coord_nd[a]

        if self._wall_normal_line not in new_datalayout:
            wall_normal_line = None
        else:
            wall_normal_line = self._wall_normal_line

        if self._polar_plane is not None:
            if all(axis in new_datalayout for axis in self._polar_plane):
                polar_plane = self._polar_plane
            else:
                polar_plane = None
        else:
            polar_plane = None

        new_coorddata = AxisData(self.Domain,
                                new_coord,
                                new_coord_nd)
        new_data = []
        for data in self._data:
            new_data.append(numpy_op(data,axis=array_axis))

        new_dict = dict(zip(self.index,new_data))
        
        return FlowStructND(new_coorddata,
                            new_dict,
                            data_layout=new_datalayout,
                            wall_normal_line=wall_normal_line,
                            polar_plane=polar_plane)

    def plot_line_data(self,data: np.ndarray,
                       label: str =None,
                       channel_half: bool =False,
                       transform_ydata: Callable =None,
                       transform_xdata: Callable =None, 
                       fig: cplt.CHAPSimFigure =None,
                       ax: cplt.AxesCHAPSim =None,
                       line_kw: dict =None,
                       **kwargs)->  Union[cplt.CHAPSimFigure,cplt.AxesCHAPSim ]:
        """
        Plots a line plot on 1D data based on the parameters
        from the FlowStruct

        Parameters
        ----------
        data : np.ndarray
            Array to be plotted
        label : str, optional
            label to be based to line object, by default None
        channel_half : bool, optional
            If the flow is a channel, whether to plot only half, by default False
        transform_ydata :  Callable, optional
            Function applied to ydata, by default None
        transform_xdata :  Callable, optional
            Function applied to xdata, by default None
        fig : cplt.CHAPSimFigure, optional
            Input figure, it is created if None, by default None
        ax : cplt.AxesCHAPSim, optional
            Input axes, it is created if None, by default None
        line_kw : dict, optional
            dictionary passed to plotting function, by default None
        **kwargs :
            keyword arguments passed to figure creation routine
            
        Returns
        -------
         Union[cplt.CHAPSimFigure,cplt.AxesCHAPSim ]
            returns figure and axes associated with line plot
        """
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
                                                                      transform_ydata)
        self._check_dim(1)
        line = self._data_layout[0]
        coord_data = self.CoordDF[line]

        if channel_half and self._check_line_channel():
            mid_index = coord_data.size // 2
            coord_data = coord_data[:mid_index]
            data = data[:mid_index]
            
        if label is not None:
            line_kw['label'] = label

        ax.cplot(transform_xdata(coord_data),transform_ydata(data),**line_kw)

        return fig, ax

    def plot_contour(self,comp: str,
                        time: Number =None,
                        rotate: bool=False,
                        fig: cplt.CHAPSimFigure =None,
                        ax: cplt.AxesCHAPSim =None,
                        contour_kw: dict =None,
                        **kwargs) ->  Union[cplt.CHAPSimFigure,mpl.collections.PolyCollection ]:
        """
        If 2D the FlowStructND will plot a contour plot of the data in the FlowStructND

        Parameters
        ----------
        comp : str
            Component of FlowStruct to be plotted
        time : Number, optional
            time to be plotted, by default None
        rotate : bool, optional
            whether the plot should be rotated, by default False
        fig : cplt.CHAPSimFigure, optional
            Input figure, it is created if None, by default None
        ax : cplt.AxesCHAPSim, optional
            Input axes, it is created if None, by default None
        contour_kw : dict, optional
            dictionary passed to pcolormesh, by default None

        Returns
        -------
        Union[cplt.CHAPSimFigure,mpl.collections.PolyCollection ]
            returned figure and matplotlib mesh plot
        """
        self._check_dim(2)
        if self._polar_plane is not None:
            subplot_kw = {'projection' : 'polar'}
            if 'subplot_kw' in kwargs:
                kwargs['subplot_kw'].update(subplot_kw)
            else:
                kwargs['subplot_kw'] = subplot_kw
        
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        plot_func, contour_kw = cplt.get_contour_func_params(ax,contour_kw)


        time = self.check_times(time)
        comp = self.check_comp(comp)
        plane = self._data_layout

        if rotate:
            flow = self[time,comp].T
            x_coord = self.CoordDF[plane[0]]
            y_coord = self.CoordDF[plane[1]]
        else:
            flow = self[time,comp]
            x_coord = self.CoordDF[plane[1]]
            y_coord = self.CoordDF[plane[0]]

        X,Y = np.meshgrid(x_coord,y_coord)

        ax = plot_func(X,Y,flow.squeeze(),**contour_kw)

        return fig, ax

    def plot_vector(self,comps: str,
                        time: Number=None,
                        rotate: bool=False,
                        spacing: Tuple[int] =(1,1),
                        scaling: int =1,
                        fig: cplt.CHAPSimFigure=None,
                        ax: cplt.AxesCHAPSim =None,
                        quiver_kw: dict =None,
                        **kwargs)-> Union[cplt.CHAPSimFigure,mpl.quiver.Quiver]:
        """
        Plots a vector plot of a two-dimensional FlowStruct

        Parameters
        ----------
        comp : str
            Component of FlowStruct to be plotted
        time : Number, optional
            time to be plotted, by default None
        rotate : bool, optional
            [description], by default False
        spacing : Tuple[int], optional
            Spacing between the arrows, by default (1,1)
        scaling : int, optional
            arrow scaling factor, by default 1
        fig : cplt.CHAPSimFigure, optional
            Input figure, it is created if None, by default None
        ax : cplt.AxesCHAPSim, optional
            Input axes, it is created if None, by default None
        quiver_kw : dict, optional
            dictionary passed to quiver function, by default None

        Returns
        -------
        Union[cplt.CHAPSimFigure,mpl.quiver.Quiver]
            [description]
        """    
        
        self._check_dim(2)
        if self._polar_plane is not None:
            subplot_kw = {'projection' : 'polar'}
            if 'subplot_kw' in kwargs:
                kwargs['subplot_kw'].update(subplot_kw)
            else:
                kwargs['subplot_kw'] = subplot_kw

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        quiver_kw = cplt.update_quiver_kw(quiver_kw)

        time = self.check_times(time)
        plane = self._data_layout

        if rotate:
            comps[1] = self.check_comp(comps[0])
            comps[0] = self.check_comp(comps[1])
            coord_2 = self.CoordDF[plane[0]][::spacing[0]]
            coord_1 = self.CoordDF[plane[1]][::spacing[1]]
        else:
            comps[0] = self.check_comp(comps[0])
            comps[1] = self.check_comp(comps[1])
            coord_1 = self.CoordDF[plane[0]][::spacing[0]]
            coord_2 = self.CoordDF[plane[1]][::spacing[1]]

        U = self[time,comps[0]][::spacing[0],::spacing[1]]
        V = self[time,comps[1]][::spacing[0],::spacing[1]]

        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        scale = np.amax(U[:,:])*coord_1.size/np.amax(coord_1)/scaling
        
        ax = ax.quiver(coord_1_mesh, coord_2_mesh,U.T,V.T,angles='uv',scale_units='xy', scale=scale,**quiver_kw)

        return fig, ax

    def plot_isosurface(self,comp: str,
                            Value: Number,
                            time: Number =None,
                            fig: cplt.CHAPSimFigure=None,
                            ax: cplt.AxesCHAPSim =None,
                            surf_kw: dict =None,
                            **kwargs) -> Union[cplt.CHAPSimFigure,cplt.Axes3DCHAPSim]:
        """
        

        Parameters
        ----------
        comp : str
            [description]
        Value : Number
            [description]
        time : Number, optional
            [description], by default None
        fig : cplt.CHAPSimFigure, optional
            Input figure, it is created if None, by default None
        ax : cplt.AxesCHAPSim, optional
            Input axes, it is created if None, by default None
        surf_kw : dict, optional
            dictionary passed to Poly3DCollection, by default None

        Returns
        -------
        Union[cplt.CHAPSimFigure,cplt.Axes3DCHAPSim]
            [description]
        """
        self._check_dim(3)

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,projection='3d',**kwargs)
        surf_kw = cplt.update_mesh_kw(surf_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        field = self._data_layout
        coords1 = self.CoordDF[field[0]]
        coords2 = self.CoordDF[field[1]]
        coords3 = self.CoordDF[field[2]]

        Z,Y,X = misc_utils.meshgrid(coords3,coords2,coords1)
        
        flow_array = self[time,comp]

        ax = ax.plot_isosurface(Z,X,Y,flow_array,Value,**surf_kw)
        
        coord_lims = [np.amax(Z) - np.amin(Z),np.amax(X) - np.amin(X),np.amax(Y) - np.amin(Y) ]
        ax.axes.set_box_aspect(coord_lims)
        return fig, ax
    
    def plot_surf(self,comp: str,
                  time: Number=None,
                  fig: cplt.CHAPSimFigure =None,
                  ax: cplt.Axes3DCHAPSim =None,
                  surf_kw: dict =None,
                  **kwargs):
        """


        Parameters
        ----------
        comp : str
            Component of FlowStruct to be plotted
        time : Number, optional
            time to be plotted, by default None
        fig : cplt.CHAPSimFigure, optional
            Input figure, it is created if None, by default None
        ax : cplt.AxesCHAPSim, optional
            Input axes, it is created if None, by default None
        surf_kw : dict, optional
            dictionary passed to plotting function, by default None
        **kwargs :
            keyword arguments passed to figure creation routine


        Returns
        -------
        [type]
            [description]
        """
        self._check_dim(2)

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,projection='3d',**kwargs)
        surf_kw = cplt.update_mesh_kw(surf_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        plane = self._data_layout

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        flow_slice = self[time,comp]
        ax = ax.plot_surface( Y,X,flow_slice,**surf_kw)

        return fig, ax

    def _check_dim(self,dim):
        if self._dim != dim:
            raise Exception
    
    def transform(self,axis,coord_transform,data_transform):
        dim = self.get_dim_from_axis(axis)
        Coord_ND_DF = self.Coord_ND_DF.copy()
        CoordDF = self.CoordDF.copy()

        Coord_ND_DF[axis] = coord_transform(self.Coord_ND_DF[axis])
        size = Coord_ND_DF[axis].size

        coord_nd_indices  = np.arange(size)
        coord_indices = np.arange(0.5,size-1)

        interp = interpolate.interp1d(coord_nd_indices,Coord_ND_DF[axis])
        CoordDF[axis] = interp(coord_indices)
        new_coorddata = AxisData(self.Domain,
                                CoordDF,
                                Coord_ND_DF)
        
        fstruct_copy = self.copy()
        for index , val in fstruct_copy:
            fstruct_copy[index] = np.apply_along_axis(data_transform,dim,val)

        return FlowStructND(new_coorddata,
                            fstruct_copy.to_dict(),
                            data_layout = self._data_layout,
                            polar_plane = self._polar_plane,
                            wall_normal_line=self._wall_normal_line)


    @docstring.sub
    @property
    def slice(self) -> _FlowStruct_slicer:
        """
        Returns an object that can be sliced using the coordinates 
        of the data:
        
        %(slicer)s
        
        Returns
        -------
        _FlowStruct_slicer
            [description]
        """
        return _FlowStruct_slicer(self)

    @property
    def location(self) -> Number:
        """
        Returns location if present

        Returns
        -------
        Number
            retuns location of FlowStruct
        """
        if not hasattr(self,'_location'):
            self._location = 0.
        return self._location

    @location.setter
    def location(self,value):
        self._location = value

    def _check_rotate(self,plane,rotate_axes):
        check_plane = plane in ['yz','yx','zx']
        
        if rotate_axes is None:
            return check_plane
        else:
            return rotate_axes
        
    @classmethod
    def join(cls,new_axis,axis_vals,flowstructs,index=None,wall_normal=False):
        if not all(f._data_layout == flowstructs[0]._data_layout\
                                     for f in flowstructs):
            msg = "All flowstructs require the same data_layout"
            raise ValueError(msg)

        if not all(f.shape == flowstructs[0].shape\
                                     for f in flowstructs):
            msg = "All flowstructs require the shape"
            raise ValueError(msg)

        data_layout = [*flowstructs[0]._data_layout,new_axis]
        if flowstructs[0]._wall_normal_line is not None and wall_normal:
            msg = "new axis cannot be a wall normal line as one already exists"
            raise ValueError(msg)
        elif wall_normal:
            wall_normal_line = new_axis
        else:
            wall_normal_line = None
        
        polar_plane = flowstructs[0]._polar_plane

        coord =  flowstructs[0].CoordDF.copy()
        coord_nd =  flowstructs[0].Coord_ND_DF.copy()

        coord[new_axis] = np.array(axis_vals)
        coord_nd[new_axis] = np.array(cls._estimate_grid_from_points(axis_vals))

        new_coorddata = AxisData(flowstructs[0].Domain,
                                                    coord,
                                                    coord_nd)
        new_data = [None]*len(flowstructs[0]._data)
        
        for i, d in enumerate(new_data):
            new_data[i] = [f._data[i] for f in flowstructs]
            new_data[i] = np.stack(new_data[i],axis=-1)

        return cls(new_coorddata, 
                    np.array(new_data),
                    data_layout=data_layout,
                    polar_plane=polar_plane,
                    wall_normal_line=wall_normal_line,
                    index=index)

    @staticmethod
    def _estimate_grid_from_points(axis_vals):
        diff = np.diff(axis_vals)
        diff = np.array([*diff,diff[-1]])
        half_diff = 0.5*diff[0]
        start = axis_vals[0] - half_diff
        return [start,*(diff+axis_vals)]


class FlowStructND_time(FlowStructND):
    """
    A class for FlowStructs which evolve in time 
    """
    def to_ND(self) -> FlowStructND:
        """
        Converts FlowStructND_time to FlowStruct_ND with 
        time as an additional axis

        Returns
        -------
        FlowStructND
            FlowStruct_ND with time as an additional axis
        """
        
        array = []
        for comp in self.inner_index:
            comp_array = []
            for time in self.times:
                comp_array.append(self[time,comp])

            array.append(np.stack(comp_array,axis = -1))

        array = np.stack(array,axis=0)
        index = [[None]*len(self.inner_index),self.inner_index]

        data_layout = self._data_layout + ['t']
        coordstruct_c = self._coorddata.coord_centered.copy()
        coordstruct_c['t'] = np.array(self.times)
        
        new_coorddata = AxisData(self.Domain,
                                                    coordstruct_c,
                                                    None)
        return FlowStructND(new_coorddata,
                            array,
                            index=index,
                            data_layout=data_layout,
                            wall_normal_line = self._wall_normal_line,
                            polar_plane = self._polar_plane)

    def __getitem__(self,key: Any) -> np.ndarray:
        """
        If a single key is passed, the array returned 
        contains time along the outer dimension

        Parameters
        ----------
        key : Any
            key for FlowStruct

        Returns
        -------
        np.ndarray
            Numpy array

        Raises
        ------
        TypeError
            If a listkey is used
        """
        if self._indexer.is_multikey(key):
            return self._getitem_process_multikey(key)
        
        else:
            if self._indexer.is_listkey(key):
                msg = f"In {self.__class__.__name__}, lists can only index with multikeys"
                raise TypeError(msg)
            return self._getitem_process_singlekey(key)

    def _getitem_process_singlekey(self,key):
        fstruct = self.to_ND()
        return fstruct[None,key]

    def _getitem_process_multikey(self,key):
        if key[0] is None:
            if self._indexer.is_listkey(key):
                msg = f"In {self.__class__.__name__}, lists can only index with multikeys"
                raise TypeError(msg)

            return self._getitem_process_singlekey(key[1])
        else:
            if self._indexer.is_listkey(key):
                return self._getitem_process_list(key)

            return super()._getitem_process_multikey(key)

    def set_value(self,key,value):
        msg = "Flowstruct1D_time elements can only be set for with multiple keys"
        if not isinstance(key,tuple):
            raise KeyError(msg)
        elif len(key) < 2:
            raise KeyError(msg)

        if key in self.index:
            super().set_value(key,value)

        elif value.shape[-1] == len(self.times) and key[0] is None:
            index_times = self.times
            index_comp = [key[1]]*len(self.times)
            indices = zip(index_times,index_comp)

            for i,index in enumerate(indices):
                super().set_value(index,value[:,i])
                
    def from_internal(self, *args, **kwargs):
        kwarg_dict = dict(data_layout=self._data_layout,
                            wall_normal_line = self._wall_normal_line,
                            polar_plane = self._polar_plane)

        kwargs.update(kwarg_dict)
        return FlowStructND_time(self._coorddata,*args,**kwargs)
    
class FlowStruct3D(FlowStructND):  
    """
    A core class for storing and visualising full 3D datasets 
    with the capability to output to vtkStructuredGrid and HDF5
    """  
    def _array_ini(self,array, index=None,  copy=False):
        data_layout = 'zyx'
        wall_normal_line = 'y'
        
        if self._coorddata._domain_handler.is_polar:
            polar_plane = 'zy'
        else:
            polar_plane = None

        super()._array_ini(array, index=index, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dict_ini(self, dict_data, copy=False):
        data_layout = 'zyx'
        wall_normal_line = 'y'
        
        if self._coorddata._domain_handler.is_polar:
            polar_plane = 'zy'
        else:
            polar_plane = None

        super()._dict_ini(dict_data, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dstruct_ini(self, dstruct,copy=False):
        self._dict_ini( dstruct.to_dict(), copy=copy)

    def from_internal(self, *args, **kwargs):
        return self.__class__(self._coorddata,
                                *args,**kwargs)

    @property
    def VTK(self) -> VTKstruct3D:
        """
        returns VTKStruct based on the data in the flow struct


        Returns:
            VTKstruct3D: class that can use the
        """
        return VTKstruct3D(self)

    def get_unit_figsize(self,plane):
        plane, coord = self.CoordDF.check_plane(plane)

        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]

        x_size = 1.5*(np.amax(x_coords) - np.amin(x_coords))
        z_size = 1.2*(np.amax(z_coords) - np.amin(z_coords))

        return x_size,z_size
        
    def create_subdomain(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        x_slice = slice(xmin,xmax)
        y_slice = slice(ymin,ymax)
        z_slice = slice(zmin,zmax)
        
        new_flowstruct = self.slice[z_slice,y_slice, x_slice]
        

        return new_flowstruct


    def to_vtk(self,file_name):
        self.VTK.to_vtk(file_name)


    def plot_contour(self,comp,plane,axis_val,time=None,rotate_axes=None,fig=None,ax=None,contour_kw=None,**kwargs):
        
        slicer = self._plane_calculate(plane,axis_val)
        rotate = self._check_rotate(plane,rotate_axes)
        
        flowstruct = self[time,[comp]]
        flowstruct = flowstruct.slice[slicer]

        fig, ax = flowstruct.plot_contour(comp,time=time,rotate=rotate,
                                            fig=fig,ax=ax,contour_kw=contour_kw,
                                            **kwargs)


        return fig, ax

    def _plane_calculate(self,plane,axis_val):
        if plane == 'zy' or plane == 'yz':
            return (slice(None), slice(None),axis_val)
        elif plane == 'xy' or plane == 'yx':
            return (axis_val, slice(None),slice(None))
        elif plane == 'xz' or plane == 'zx':
            return (slice(None),axis_val, slice(None))


    def plot_vector(self,comps,plane,axis_val,time=None,spacing=(1,1),scaling=1,rotate_axes=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        slicer = self._plane_calculate(plane,axis_val)
        rotate = self._check_rotate(plane,rotate_axes)
        flowstruct = self.slice[slicer]

        fig, ax = flowstruct.plot_vector(comps,time=time,rotate=rotate,spacing=spacing,
                                            scaling=scaling,fig=fig,ax=ax,quiver_kw=quiver_kw,
                                            **kwargs)
        return fig, ax



    def plot_isosurface(self,comp,Value,time=None,y_limit=None,x_split_pair=None,fig=None,ax=None,surf_kw=None,**kwargs):
        
        flowstruct = self
        if x_split_pair is not None:
            flowstruct = flowstruct.slice[:,:,x_split_pair[0],x_split_pair[1]]

        if y_limit is not None:
            flowstruct = flowstruct.slice[:,:y_limit]

        
        fig, ax = flowstruct.plot_isosurface(comp,Value,time=time,
                                            fig=fig,ax=ax,surf_kw=surf_kw,
                                            **kwargs)

        return fig, ax


    def plot_surf(self,comp,plane,axis_val,time=None,x_split_pair=None,fig=None,ax=None,surf_kw=None,**kwargs):
        flowstruct = self
        if x_split_pair is not None:
            flowstruct = flowstruct.slice[:,:,x_split_pair[0],x_split_pair[1]]

        slicer = self._plane_calculate(plane,axis_val)
        flowstruct = flowstruct.slice(slicer)

        fig, ax = flowstruct.plot_surf(comp,time=time,
                                    fig=fig,ax=ax,surf_kw=surf_kw,
                                    **kwargs)

class FlowStruct2D(FlowStructND):
    def _array_ini(self, array, index=None,  copy=False):
        data_layout = 'yx'
        wall_normal_line = 'y'
        polar_plane = None

        super()._array_ini(array, index=index, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dict_ini(self, dict_data, copy=False):
        data_layout = 'yx'
        wall_normal_line = 'y'
        polar_plane = None

        super()._dict_ini( dict_data, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dstruct_ini(self,  dstruct,copy=False):
        self._dict_ini(dstruct.to_dict(), copy=copy)

    def from_internal(self, *args, **kwargs):
        return self.__class__(self._coorddata,
                                *args,**kwargs)

    def _calculate_line(self,line,axis_val):
        if line == 'y':
            return (slice(None),axis_val)
        elif line == 'x':
            return (axis_val,slice(None))

    @property
    def VTK(self):
        return VTKstruct2D(self)


    def to_vtk(self,file_name):
        raise NotImplementedError

    def plot_line(self,comp,axis,coords,time=None,labels=None,transform_ydata=None, transform_xdata=None, channel_half=False,fig=None,ax=None,line_kw=None,**kwargs):
        coords = misc_utils.check_list_vals(coords)
        if labels is None:
            labels = [None]*len(coords)
        for coord,label in zip(coords,labels):
           slicer = self._calculate_line(axis,coord)  
           flowstruct = self.slice[slicer]

           fig, ax = flowstruct.plot_line(comp,time=time,label=label,
                                          transform_xdata=transform_xdata,
                                          transform_ydata=transform_ydata,
                                          channel_half = channel_half,
                                          fig=fig,ax=ax,
                                          line_kw=line_kw,
                                          **kwargs)  

        return fig, ax
    
    def plot_line_data(self,data,axis,coords,labels=None,transform_ydata=None, transform_xdata=None, channel_half=False,fig=None,ax=None,line_kw=None,**kwargs):
        coords = misc_utils.check_list_vals(coords)
        if labels is None:
            labels = [None]*len(coords)
        for coord,label in zip(coords,labels):
            slicer = self._calculate_line(axis,coord)  

            flowstruct = self.slice[slicer]
            
            slice_data = data[self.slice.get_slicer(slicer)]
            
            fig, ax = flowstruct.plot_line_data(slice_data,
                                                label=label,
                                                transform_xdata=transform_xdata,
                                                transform_ydata=transform_ydata,
                                                channel_half = channel_half,
                                                fig=fig,ax=ax,
                                                line_kw=line_kw,
                                                **kwargs)  
        
        return fig, ax
    
    def plot_line_max(self,comp,axis,transform_ydata=None, transform_xdata=None, time=None,fig=None,ax=None,line_kw=None,**kwargs):
        max_axis = self._data_layout.copy()
        max_axis.remove(axis)
        flowstruct = self.reduce(np.amax,max_axis[0])

        fig, ax = flowstruct.plot_line(comp, time=time,
                                        transform_ydata=transform_ydata,
                                        transform_xdata=transform_xdata,
                                        fig=fig,
                                        ax=ax,
                                        line_kw=line_kw,
                                        **kwargs)

        return fig, ax


    # def plot_line_data_max(self,data,axis,transform_ydata=None, transform_xdata=None, time=None,fig=None,ax=None,line_kw=None,**kwargs):
    #     max_axis = self._plane.replace(axis,'')

    #     axis_dim = self.get_dim_from_axis(max_axis)

    #     transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
    #                                                                   transform_ydata)

    #     fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
    #     line_kw = cplt.update_line_kw(line_kw)

    #     max_data = np.amax(data,axis=axis_dim)
    #     coords = self.CoordDF[axis]

    #     ax.cplot(transform_xdata(coords),transform_ydata(max_data),**line_kw)

    #     return fig, ax

        


    def plot_contour(self,comp,time=None,rotate_axes=None,fig=None,ax=None,contour_kw=None,**kwargs):
        
        data_layout = ''.join(self._data_layout)
        rotate = self._check_rotate(data_layout,rotate_axes)

        return super().plot_contour(comp,time=time,rotate=rotate,fig=fig, ax=ax,contour_kw=contour_kw,**kwargs)

    def plot_vector(self,comps,time=None,spacing=(1,1),scaling=1,rotate_axes=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        data_layout = ''.join(self._data_layout)
        rotate = self._check_rotate(data_layout,rotate_axes)

        return super().plot_vector(comps, time=time, rotate=rotate,
                                    spacing=spacing,scaling=scaling,
                                    fig=fig, ax=ax,quiver_kw=quiver_kw,
                                    **kwargs)

class FlowStruct1D(FlowStructND):
    def _array_ini(self, array, index=None,  copy=False):
        data_layout = 'y'
        wall_normal_line = 'y'
        polar_plane = None

        super()._array_ini(array, index=index, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dict_ini(self, dict_data, copy=False):
        data_layout = 'y'
        wall_normal_line = 'y'
        polar_plane = None

        super()._dict_ini(dict_data, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)
    
    def _dstruct_ini(self, dstruct,copy=False):
        self._dict_ini( dstruct.to_dict(), copy=copy)


    def from_internal(self, *args, **kwargs):
        return self.__class__(self._coorddata,
                                *args,**kwargs)
        
class FlowStruct1D_time(FlowStruct1D,FlowStructND_time):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self._is_multidim():
            msg = "This flow structure must be mutlidimensional"
            raise ValueError(msg)

    def from_internal(self,*args,**kwargs):
        if isinstance(args[0],np.ndarray):
            return self._handle_none_ndarray(*args,**kwargs)
                    
        elif isinstance(args[0],dict):
            return self._handle_none_dict(*args,**kwargs)

        return super().from_internal(*args,**kwargs)

    def _handle_none_ndarray(self,*args,**kwargs):
        index = kwargs.get('index',None)
        array = args[0]

        if array.shape[1] == len(self.times) and index[0][0] is None:
            array = array.T
            times = self.times
            inner = [index[0][1]]*len(times)
            index = list(zip(times,inner))

            args = [array,*args[1:]]; kwargs['index'] = index

        return super().from_internal(*args, **kwargs)



    def _handle_none_dict(self,*args,**kwargs):
        diction = args[0]

        if all(key[0] is None for key in diction.keys()):

            indices = [[key] for key in diction.keys()]
            arrays = diction.values()

            struct_list= []
            for array, index in zip(arrays,indices):
                struct_list.append(self.from_internal(array,index=index))

            return self.from_concat(struct_list)
        else:
            return super().from_internal(*args,**kwargs)


    def plot_line_time(self,comp,coords,transform_ydata=None, transform_xdata=None, labels=None,fig=None,ax=None,line_kw=None,**kwargs):
        fstruct_t = self.to_ND()

        for i, y in enumerate(coords):
            fstruct_y = fstruct_t.slice[y]
            label = labels[i] if labels is not None else None
            fig, ax = fstruct_y.plot_line(comp,
                                       transform_ydata=transform_ydata,
                                       transform_xdata=transform_xdata, 
                                       label=label,
                                       fig=fig,
                                       ax=ax,
                                       line_kw=line_kw,
                                       **kwargs)

        return fig, ax

    def plot_line_time_data(self,data,coords,transform_ydata=None, transform_xdata=None, labels=None, fig=None,ax=None,line_kw=None,**kwargs):
        fstruct_t = self.to_ND()

        for i, y in enumerate(coords):
            fstruct_y = fstruct_t.slice[y]
            label = labels[i] if labels is not None else None
            fig, ax = fstruct_y.plot_line_data(data,
                                       transform_ydata=transform_ydata,
                                       transform_xdata=transform_xdata, 
                                       label=label,
                                       fig=fig,
                                       ax=ax,
                                       line_kw=line_kw,
                                       **kwargs)
        
        return fig, ax

    def plot_line_time_max(self,comp,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):
        fstruct_t = self.to_ND().reduce(np.amax,'y')

        fig, ax = fstruct_t.plot_line(comp, time=None,
                                        transform_ydata=transform_ydata,
                                        transform_xdata=transform_xdata,
                                        fig=fig,
                                        ax=ax,
                                        line_kw=line_kw,
                                        **kwargs)

        return fig, ax

        
    # def plot_line_time_data_max(self,data,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):

    #     fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
    #     line_kw = cplt.update_line_kw(line_kw)

    #     transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
    #                                                                   transform_ydata)


    #     max_data = np.amax(data,axis=0)
    #     times = self.times

    #     ax.cplot(transform_xdata(times),transform_ydata(max_data),**line_kw)

    #     return fig, ax
