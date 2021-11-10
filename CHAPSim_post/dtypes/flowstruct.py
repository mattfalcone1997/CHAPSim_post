from abc import abstractproperty
from .dtypes import *
from . import utils
import vtk
from CHAPSim_post.post._meta import coorddata
import CHAPSim_post.plot as cplt
from pyvista import StructuredGrid
from CHAPSim_post.utils import misc_utils, indexing

import CHAPSim_post as cp
from scipy import interpolate
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

    def check_line(self,line):
        if line not in self.index:
            msg = f"The line must be in {self.index}"
            raise KeyError(msg)

        return line

class FlowStruct_base(datastruct):
    def __init__(self,*args,from_hdf=False,**kwargs):
        
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        
        self.check_shape()

    @property
    def Domain(self):
        return self._coorddata._domain_handler

    @property
    def CoordDF(self):
        return self._coorddata.centered

    @property
    def Coord_ND_DF(self):
        return self._coorddata.staggered

    def remove_time(self,time):
        del_index = [ index for index in self.index if index[0] == time]
        for index in del_index:
            del self[index]
    
    def shift_times(self,time):
        new_index = []
        for index in self.index:
            new_time = float(index[0]) + time
            new_index.append(self._indexer._item_handler((new_time,*index[1:])))

        return self.from_internal(dict(zip(new_index, self._data)))
            

    def get_identity_transform(self):
        def identity_transform(data):
            return data
        
        return identity_transform

    def _check_datatransforms(self,transform_xdata,transform_ydata):
        
        if transform_xdata is None:
            transform_xdata = self.get_identity_transform()
        if transform_ydata is None:
            transform_ydata = self.get_identity_transform()

        if not hasattr(transform_xdata,'__call__'):
            msg = "transform_xdata must be None or callable"
            raise TypeError(msg)
        if not hasattr(transform_ydata,'__call__'):
            msg = "transform_xdata must be None or callable"
            raise TypeError(msg)

        return transform_xdata, transform_ydata

        
    @property
    def times(self):
        if 'None' in self.index.outer_index:
            return None
        else:
            return sorted([float(x) for x in self.outer_index])

    @property
    def comp(self):
        return self.inner_index

    @abstractproperty
    def VTK(self):
        pass

    @property
    def shape(self):
        key = self.index[0]
        return self[key].shape

    def _dstruct_ini(self, coorddata, dstruct, copy=False):
        self._dict_ini(coorddata, dstruct.to_dict(), copy=copy)

    def _set_coorddata(self,file_or_coorddata,**kwargs):
        if isinstance(file_or_coorddata,coorddata):
            self._coorddata = file_or_coorddata.copy()
        else:
            path = os.fspath(file_or_coorddata)
            key = kwargs.get('key',None)
            if key is None:
                coord_key = '/coorddata'
            else:
                coord_key = os.path.join(key,'coorddata')

            self._coorddata = coorddata.from_hdf(path,key=coord_key)

    def check_times(self,key,err=None,warn=None):
        if err is None:
            msg = f"{self.__class__.__name__} object does not have time {key}"
            err = KeyError(msg)
        if warn is None:
            warn = (f"{self.__class__.__name__} object does not have time {key}"
                    f" only one {key} present. This time will be used")
        return self.check_outer(key,err,warn=warn)

    def check_comp(self,key,err=None):
        if err is None:
            msg = f"Component {key} not in {self.__class__.__name__}"
            err = KeyError(msg)
        return self.check_inner(key,err)

    def check_line_labels(self,items,labels):
        if labels is None:
            return

        if not isinstance(labels,(tuple,list)):
            msg = "The labels provided must be a list or tuple"
            raise TypeError(msg)

        if len(items) != len(labels):
            msg = "The number of labels must be equal the number of items"
            raise ValueError(msg)

    def is_fully_compatible(self,data):
        if not self.is_shape_compatible(data):
            return False

        if not self.is_coords_compatible(data):
            return False

        if not self.is_type_compatible(data):
            return False

        return True

    def is_type_compatible(self,data):
        return  isinstance(data,self.__class__)
    
    def is_coords_compatible(self,data):
        return self._coorddata == data._coorddata

    def is_shape_compatible(self,data):
        return self.shape == data.shape

    def check_shape(self):
        if not all([arr.shape == self.shape for _, arr in self]):
            msg = f"All arrays must have the same shape in {self.__class__.__name__} class"
            raise ValueError(msg)
        
        coord_shapes = tuple([coord.size for _,coord in self.CoordDF])

        if not all(s in coord_shapes for s in self.shape):
            msg = "The coordinate data is of different shape %s to the array data %s"%(coord_shapes,self.shape)
            raise ValueError(msg)

        shapes = [arr.shape for _, arr in self]
        dims = [arr.ndim for _, arr in self]
        if not all(s == shapes[0] for s in shapes):
            msg = f"All array shapes in {self.__class__.__name__} must be the same"
            raise ValueError(msg)

        if not all(dim == self._dim for dim in dims):
            msg = f"All array ndims in {self.__class__.__name__} must be equal to {self._dim}"
            raise ValueError(msg)
    
    def concat(self,arr_or_data):

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

    def symmetrify(self,dim=None):
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

    def copy(self):
        cls = self.__class__
        return cls(self._coorddata,self.to_dict(),copy=True)

class _FlowStruct_slicer:
    def __init__(self,flowstruct_obj):
        self._ref = weakref.ref(flowstruct_obj)

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
        new_coorddata = coorddata.from_coordstructs(flow_struct.Domain,new_CoordDF,new_Coord_ND_DF)

        return new_coorddata, data_layout, wall_normal_line, polar_plane
                 

class FlowStructND(FlowStruct_base):

    def _array_ini(self, coorddata, array, index=None,data_layout=None,wall_normal_line=None,polar_plane= None, copy=False):
        self._set_coorddata(coorddata)
        super()._array_ini(array, index=index, copy=copy)
        self._set_data_layout(data_layout, wall_normal_line, polar_plane)

    def _dict_ini(self, coorddata, dict_data, data_layout=None,wall_normal_line=None,polar_plane= None, copy=False):
        self._set_coorddata(coorddata)
        super()._dict_ini(dict_data, copy=copy)
        self._set_data_layout(data_layout, wall_normal_line, polar_plane)

    def _dstruct_ini(self, coorddata, dstruct, data_layout=None,wall_normal_line=None,polar_plane= None, copy=False):
        self._dict_ini(coorddata, dstruct.to_dict(),
                        data_layout=data_layout,
                        wall_normal_line=wall_normal_line,
                        polar_plane= polar_plane, 
                        copy=copy)
    @property
    def VTK(self):
        use_cell_data = self._coorddata.contains_staggered
        if self._dim == 2:
            return VTKstruct2D(self,cell_data = use_cell_data)
        elif self._dim ==3:
            return VTKstruct3D(self,cell_data = use_cell_data)
        else:
            raise Exception

    def to_vtk(self,file_name):
        self.VTK.to_vtk(file_name)

    def to_hdf(self,filepath,mode='a',key=None):

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

        
    def from_internal(self, *args, **kwargs):
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
            if self.CoordDF[data].size != self.shape[i]:
                coord_shape = tuple(d.size for _,d in self.CoordDF)
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

    def plot_line(self,comp,time=None,label=None,channel_half=False,
                    transform_ydata=None, transform_xdata=None, 
                    fig=None,ax=None,line_kw=None,**kwargs):

        time = self.check_times(time)
        comp = self.check_comp(comp)

        self._check_dim(1)

        input_kw = cplt.update_line_kw(line_kw).copy()
        if label is not None:
            input_kw['label'] = label

        data = self[time,comp].squeeze()

        fig, ax = self.plot_line_data(data,channel_half=channel_half,transform_ydata=transform_ydata,
                                         transform_xdata=transform_xdata, fig=fig,ax=ax,line_kw=input_kw,**kwargs)

        return fig, ax

    def _check_line_channel(self):
        return self._wall_normal_line and not self.Domain.is_cylind

    def get_dim_from_axis(self,axis):

        if len(axis) > 1:
            return tuple([self.get_dim_from_axis(a) for a in axis])
        if self._data_layout.count(axis) > 1:
            msg = "The index axis cannot appear more than once"
            raise ValueError(msg)
        elif self._data_layout.count(axis) == 0:
            msg = "The index axis provided does not appear in the data layout"
            raise ValueError(msg)

        return "".join(self._data_layout).find(axis)

    def reduce(self,numpy_op,axis):
        array_axis = self.get_dim_from_axis(axis)
        new_datalayout = self._data_layout.copy()
        new_coord = self.CoordDF.copy()
        new_coord_nd = self.Coord_ND_DF.copy()

        for a in axis:
            new_datalayout.remove(a)
            del new_coord[a]
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

        new_coorddata = coorddata.from_coordstructs(self.Domain,new_coord,
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

    def plot_line_data(self,data,channel_half=False,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):
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

        ax.cplot(transform_xdata(coord_data),transform_ydata(data),**line_kw)

        return fig, ax

    def plot_contour(self,comp,time=None,rotate=False,fig=None,ax=None,pcolor_kw=None,**kwargs):
        self._check_dim(2)
        if self._polar_plane is not None:
            subplot_kw = {'projection' : 'polar'}
            if 'subplot_kw' in kwargs:
                kwargs['subplot_kw'].update(subplot_kw)
            else:
                kwargs['subplot_kw'] = subplot_kw
        
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

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

        ax = ax.pcolormesh(X,Y,flow.squeeze(),**pcolor_kw)

        return fig, ax

    def plot_vector(self,comps,time=None,rotate=False,spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
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

    def plot_isosurface(self,comp,Value,time=None,fig=None,ax=None,surf_kw=None,**kwargs):
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
    
    def plot_surf(self,comp,time=None,fig=None,ax=None,surf_kw=None,**kwargs):
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
        new_coorddata = coorddata.from_coordstructs(self.Domain,
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


    @property
    def slice(self):
        return _FlowStruct_slicer(self)

    @property
    def location(self):
        if not hasattr(self,'_location'):
            self._location = 0.
        return self._location

    @location.setter
    def location(self,value):
        self._location = value

    def _check_rotate(self,plane,rotate_axes):
        if plane in ['yz','yx','zx'] or rotate_axes:
            return True
        else:
            return False
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

        new_coorddata = coorddata.from_coordstructs(flowstructs[0].Domain,
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
    def to_ND(self):
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
        
        new_coorddata = coorddata.from_coordstructs(self.Domain,
                                                    coordstruct_c,
                                                    None)
        return FlowStructND(new_coorddata,
                            array,
                            index=index,
                            data_layout=data_layout,
                            wall_normal_line = self._wall_normal_line,
                            polar_plane = self._polar_plane)

    def __getitem__(self,key):
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
    def _array_ini(self, coorddata,array, index=None,  copy=False):
        data_layout = 'zyx'
        wall_normal_line = 'y'
        if coorddata._domain_handler.is_cylind:
            polar_plane = 'zy'
        else:
            polar_plane = None

        super()._array_ini(coorddata,array, index=index, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dict_ini(self,coorddata, dict_data, copy=False):
        data_layout = 'zyx'
        wall_normal_line = 'y'
        if coorddata._domain_handler.is_cylind:
            polar_plane = 'zy'
        else:
            polar_plane = None

        super()._dict_ini(coorddata, dict_data, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dstruct_ini(self, coorddata, dstruct,copy=False):
        self._dict_ini(coorddata, dstruct.to_dict(), copy=copy)

    def from_internal(self, *args, **kwargs):
        return self.__class__(self._coorddata,
                                *args,**kwargs)

    @property
    def VTK(self):
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


    def plot_contour(self,comp,plane,axis_val,time=None,rotate_axes=False,fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        slicer = self._plane_calculate(plane,axis_val)
        rotate = self._check_rotate(plane,rotate_axes)
        flowstruct = self.slice[slicer]

        fig, ax = flowstruct.plot_contour(comp,time=time,rotate=rotate,
                                            fig=fig,ax=ax,pcolor_kw=pcolor_kw,
                                            **kwargs)


        return fig, ax

    def _plane_calculate(self,plane,axis_val):
        if plane == 'zy' or plane == 'yz':
            return (slice(None), slice(None),axis_val)
        elif plane == 'xy' or plane == 'yx':
            return (axis_val, slice(None),slice(None))
        elif plane == 'xz' or plane == 'zx':
            return (slice(None),axis_val, slice(None))


    def plot_vector(self,comps,plane,axis_val,time=None,spacing=(1,1),scaling=1,rotate_axes=False,fig=None,ax=None,quiver_kw=None,**kwargs):
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
    def _array_ini(self, coorddata, array, index=None,  copy=False):
        data_layout = 'yx'
        wall_normal_line = 'y'
        polar_plane = None

        super()._array_ini(coorddata, array, index=index, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dict_ini(self,coorddata,  dict_data, copy=False):
        data_layout = 'yx'
        wall_normal_line = 'y'
        polar_plane = None

        super()._dict_ini(coorddata, dict_data, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dstruct_ini(self, coorddata, dstruct,copy=False):
        self._dict_ini(coorddata, dstruct.to_dict(), copy=copy)

    def from_internal(self, *args, **kwargs):
        return self.__class__(self._coorddata,
                                *args,**kwargs)

    # def to_hdf(self,filepath,key=None,mode='a'):
        
    #     super().to_hdf(filepath,key=key,mode=mode)

    #     hdf_obj = hdfHandler(filepath,mode='a',key=key)

    #     hdf_obj.attrs['plane'] = self._plane.encode('utf-8')
    #     if self._location is not None:
    #         hdf_obj.attrs['location'] = self._location.encode('utf-8')
    #     hdf_obj.attrs['data_layout'] = self._data_layout.encode('utf-8')

        

    # @classmethod
    # def from_hdf(cls,filename,key=None):

    #     hdf_obj = hdfHandler(filename,mode='r',key=key)
    #     plane = hdf_obj.attrs['plane']
    #     data_layout = hdf_obj.attrs['data_layout']

    #     if 'location' in hdf_obj.attrs.keys():
    #         location = hdf_obj.attrs['location'][0]
    #     else:
    #         location = None

    #     return  cls(filename,key=key,plane=plane,location=location,data_layout=data_layout,from_hdf=True)





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

        


    def plot_contour(self,comp,time=None,fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        data_layout = ''.join(self._data_layout)
        rotate = self._check_rotate(data_layout,False)

        return super().plot_contour(comp,time=time,rotate=rotate,fig=fig, ax=ax,pcolor_kw=pcolor_kw,**kwargs)

    def plot_vector(self,comps,time=None,spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
        data_layout = ''.join(self._data_layout)
        rotate = self._check_rotate(data_layout,False)

        return super().plot_vector(comps, time=time, rotate=rotate,
                                    spacing=spacing,scaling=scaling,
                                    fig=fig, ax=ax,quiver_kw=quiver_kw,
                                    **kwargs)

class FlowStruct1D(FlowStructND):
    def _array_ini(self, coorddata, array, index=None,  copy=False):
        data_layout = 'y'
        wall_normal_line = 'y'
        polar_plane = None

        super()._array_ini(coorddata, array, index=index, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)

    def _dict_ini(self, coorddata, dict_data, copy=False):
        data_layout = 'y'
        wall_normal_line = 'y'
        polar_plane = None

        super()._dict_ini(coorddata, dict_data, data_layout=data_layout, wall_normal_line=wall_normal_line, polar_plane=polar_plane, copy=copy)
    def _dstruct_ini(self, coorddata, dstruct,copy=False):
        self._dict_ini(coorddata, dstruct.to_dict(), copy=copy)


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

class VTKstruct_base:
    def __getattr__(self,attr):
        _grid = self._grid
        if hasattr(_grid,attr):
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
    
    @abstractproperty
    def _grid(self):
        pass

    def __deepcopy__(self,memo):
        new_flow_struct = self._flowstruct.copy()
        return self.__class__(new_flow_struct)

    def copy(self):
        return copy.deepcopy(self)
    
    def to_vtk(self,file_name):
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

        times = self._flowstruct.times
        if times is None:
            times = [None]
        for i,time in enumerate(times):

            grid = self._grid
            if len(times) > 1:
                num_zeros = int(np.log10(len(self._flowstruct.times)))+1
                ext = str(i).zfill(num_zeros)
                file_name = os.path.join(file_name,".%s"%ext)

            if self._use_cell_data:
                for comp in self._flowstruct.comp:
                    grid.cell_data[np.str_(comp)] = self._flowstruct[time,comp].flatten()
            else:
                for comp in self._flowstruct.comp:
                    grid.point_data[np.str_(comp)] = self._flowstruct[time,comp].flatten()
            
            writer.SetFileName(file_name)

            if vtk.vtkVersion().GetVTKMajorVersion() <= 5:
                grid.Update()
                writer.SetInput(grid)
            else:
                writer.SetInputData(grid)
                
            writer.Write()
            
    @property
    def flowstruct(self):
        return self._flowstruct

    def __getitem__(self,key):
        return_grid = self._grid.copy()

        if not self._flowstruct._indexer.is_listkey(key):
            if not self._flowstruct._indexer.is_multikey(key):
                key = [key]
            else:
                key = ([k] for k in key)

        new_flowstruct = self._flowstruct[key]

        for k, array in new_flowstruct:

            if len(new_flowstruct.outer_index) < 2:
                k = k[1]
            return_grid.cell_arrays[np.str_(k)] = array.flatten()
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

class VTKstruct3D(VTKstruct_base):
    def __init__(self,flowstruct_obj,cell_data=True):
        if flowstruct_obj._dim != 3 :
            msg = "This class can only be used on objects of type FlowStruct3D"
            raise TypeError(msg)

        self._flowstruct = flowstruct_obj
        self._use_cell_data = cell_data

    @property
    def _grid(self):
        self._flowstruct._coorddata.create_vtkStructuredGrid(self._use_cell_data)


    def __iadd__(self,other_VTKstruct):
        if not isinstance(other_VTKstruct,self.__class__):
            msg = "This operation can only be used with other VTKstruct's"
            raise TypeError(msg)
        
        if not np.allclose(self._grid.points,other_VTKstruct._grid.points):
            msg = "The grids of the VTKstruct's must be allclose"
            raise ValueError(msg)

        self._flowstruct.concat(other_VTKstruct._flowstruct)

        return self


class VTKstruct2D(VTKstruct_base):
    def __init__(self,flowstruct_obj,cell_data=True):
        if flowstruct_obj._dim != 2 :
            msg = "This class can only be used on objects of type FlowStruct2D"
            raise TypeError(msg)

        self._flowstruct = flowstruct_obj
        self._use_cell_data = cell_data
        
    @property
    def _grid(self):
        plane = self._flowstruct._data_layout
        if self._use_cell_data:
            coord_1 = self._flowstruct.Coord_ND_DF[plane[0]]
            coord_2 = self._flowstruct.Coord_ND_DF[plane[1]]
        else:
            coord_1 = self._flowstruct.CoordDF[plane[0]]
            coord_2 = self._flowstruct.CoordDF[plane[1]]

        location = self._flowstruct.location
        coord_3 = [location]

        Y,X,Z = np.meshgrid(coord_1,coord_2,coord_3)

        grid = StructuredGrid(X,Z,Y)
        return grid


    def __iadd__(self,other_VTKstruct):
        if not isinstance(other_VTKstruct,self.__class__):
            msg = "This operation can only be used with other VTKstruct's"
            raise TypeError(msg)
        
        if not np.allclose(self._grid.points,other_VTKstruct._grid.points):
            msg = "The grids of the VTKstruct's must be allclose"
            raise ValueError(msg)

        self._flowstruct.concat(other_VTKstruct._flowstruct)

        return self
