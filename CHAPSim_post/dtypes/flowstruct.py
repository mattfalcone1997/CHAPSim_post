from abc import abstractproperty
from .dtypes import *
from . import utils
import vtk
from CHAPSim_post.post._meta import coorddata
import CHAPSim_post.plot as cplt
from pyvista import StructuredGrid
from CHAPSim_post.utils import misc_utils, indexing

import CHAPSim_post as cp

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
        
        self._set_coorddata(*args,**kwargs)
        if isinstance(args[0],coorddata):
            args = args[1:]
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        
        self.check_shape()

    def from_internal(self, *args, **kwargs):
        return self.__class__(self._coorddata,*args,**kwargs)

    def _set_coorddata(self,file_or_coorddata,*args,**kwargs):
        if isinstance(file_or_coorddata,coorddata):
            self._coorddata = file_or_coorddata.copy()
        else:
            path = os.fspath(file_or_coorddata)
            key = kwargs.get('key',None)
            if key is None:
                coord_key = 'coorddata'
            else:
                coord_key = os.path.join(key,'coorddata')

            self._coorddata = coorddata.from_hdf(path,key=coord_key)

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
            
    def to_hdf(self,filepath,key=None,mode='a'):
        super().to_hdf(filepath,key=key,mode=mode)
        self._coorddata.to_hdf(filepath,key=key+"/coorddata",mode='a')

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

    def _array_ini(self,array,index=None,copy=False):
        if array.ndim != (self._dim + 1):
            msg = f"The array must be dimension {self._dim +1}"
            raise ValueError(msg)

        return super()._array_ini(array,index=index,copy=copy)

    def _dict_ini(self, dict_data, copy=False):
        if not all([val.ndim == self._dim for val in dict_data.values()]):
            msg = f"The array must be dimension {self._dim}"
            raise ValueError(msg)

        return super()._dict_ini(dict_data,copy=copy)

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
        if not isinstance(other_obj,numbers.Number):
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

class FlowStructND(FlowStruct_base):
    pass

class FlowStruct3D(FlowStruct_base):
    _dim = 3

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


    def to_vtk(self,file_name):
        
        self.VTK.to_vtk(file_name)


    def plot_contour(self,comp,plane,axis_val,time=None,rotate_axes=False,fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        plane, coord = self.CoordDF.check_plane(plane)

        if plane == 'zy' and self.Domain.is_cylind:
            subplot_kw = {'projection' : 'polar'}
            if 'subplot_kw' in kwargs:
                kwargs['subplot_kw'].update(subplot_kw)
            else:
                kwargs['subplot_kw'] = subplot_kw


        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

       

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = utils.contour_indexer(self[time,comp],axis_index,coord)

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        ax = ax.pcolormesh(X,Y,flow_slice.squeeze(),**pcolor_kw)

        return fig, ax

    def plot_vector(self,comp,plane,axis_val,time=None,spacing=(1,1),scaling=1,rotate_axes=False,fig=None,ax=None,quiver_kw=None,**kwargs):
        plane, coord = self.CoordDF.check_plane(plane)

        if plane == 'zy' and self.Domain.is_cylind:
            subplot_kw = {'projection' : 'polar'}
            if 'subplot_kw' in kwargs:
                kwargs['subplot_kw'].update(subplot_kw)
            else:
                kwargs['subplot_kw'] = subplot_kw

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        quiver_kw = cplt.update_quiver_kw(quiver_kw)

        time = self.check_times(time)


        coord_1 = self.CoordDF[plane[0]][::spacing[0]]
        coord_2 = self.CoordDF[plane[1]][::spacing[1]]
        UV_slice = [chr(ord(x)-ord('x')+ord('u')) for x in plane]
        U = self[time,UV_slice[0]]
        V = self[time,UV_slice[1]]

        axis_index = self.CoordDF.index_calc(coord,axis_val)

        U_space, V_space = utils.vector_indexer(U,V,axis_index,coord,spacing[0],spacing[1])
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
        flow_slice = utils.contour_indexer(self[time,comp],axis_index,coord)

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        if x_split_pair is None or plane[0] != 'x':
            x_split_pair = [np.amin(x_coord),np.amax(x_coord)]
        
        x_index_list = self.CoordDF.index_calc('x',x_split_pair)

        x_coord = x_coord[x_index_list[0]:x_index_list[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = utils.contour_indexer(self[time,comp],axis_index,coord)
        flow_slice = flow_slice[:,x_index_list[0]:x_index_list[1]]

        ax = ax.plot_surface( Y,X,flow_slice,**surf_kw)

        return fig, ax

class FlowStruct2D(FlowStruct_base):
    _dim = 2
    def __init__(self,*args,plane =None,data_layout=None,location=None,**kwargs):

        super().__init__(*args,**kwargs)

        self._set_plane_loc(plane,location,data_layout)

    def _set_plane_loc(self,plane,location,data_layout):

        if plane is None:
            self._plane = 'xy'
            self._location = None
            self._data_layout = 'yx'
        else:
            self._plane, _ = self.CoordDF.check_plane(plane)
            self._location = location
            self._data_layout = data_layout

            if self._data_layout is None:
                msg = "Data layout must be provided if plane is provided"
                raise TypeError(msg)

    def _extract_plane_loc(self,filename,key=None):
        hdf_obj = hdfHandler(filename,mode='r',key=key)
        plane = hdf_obj.attrs['plane']
        data_layout = hdf_obj.attrs['data_layout']

        if 'location' in hdf_obj.attrs.keys():
            location = hdf_obj.attrs['location'][0]
        else:
            location = None
        return plane, location, data_layout

    def to_hdf(self,filepath,key=None,mode='a'):
        
        super().to_hdf(filepath,key=key,mode=mode)

        hdf_obj = hdfHandler(filepath,mode='a',key=key)

        hdf_obj.attrs['plane'] = self._plane.encode('utf-8')
        if self._location is not None:
            hdf_obj.attrs['location'] = self._location.encode('utf-8')
        hdf_obj.attrs['data_layout'] = self._data_layout.encode('utf-8')

        

    @classmethod
    def from_hdf(cls,filename,key=None):

        hdf_obj = hdfHandler(filename,mode='r',key=key)
        plane = hdf_obj.attrs['plane']
        data_layout = hdf_obj.attrs['data_layout']

        if 'location' in hdf_obj.attrs.keys():
            location = hdf_obj.attrs['location'][0]
        else:
            location = None

        return  cls(filename,key=key,plane=plane,location=location,data_layout=data_layout,from_hdf=True)


    def from_internal(self, *args, **kwargs):
        kwargs['plane'] = self._plane
        kwargs['location'] = self._location
        kwargs['data_layout'] = self._data_layout
        return self.__class__(self._coorddata,*args,**kwargs)

    def get_dim_from_axis(self,axis):
        if self._data_layout.count(axis) > 1:
            msg = "The index axis cannot appear more than once"
            raise ValueError(msg)
        elif self._data_layout.count(axis) == 0:
            msg = "The index axis provided does not appear in the data layout"
            raise ValueError(msg)

        return self._data_layout.find(axis)

    @property
    def VTK(self):
        return VTKstruct2D(self)

    def create_slice(self,index):
        slice_dict = {index : self[index]}
        return self.__class__(self._coorddata,slice_dict)

    def to_vtk(self,file_name):
        raise NotImplementedError

    def plot_line(self,comp,axis,coords,time=None,labels=None,transform_ydata=None, transform_xdata=None, channel_half=False,fig=None,ax=None,line_kw=None,**kwargs):
        time = self.check_times(time)
        comp = self.check_comp(comp)

        data = self[time,comp]

        return self.plot_line_data(data,axis,coords,labels=labels,transform_ydata=transform_ydata, transform_xdata=transform_xdata, channel_half=channel_half,
                                fig=fig,ax=ax,line_kw=line_kw,**kwargs)
    
       
    def plot_line_data(self,data,axis,coords,transform_ydata=None, transform_xdata=None,labels=None,channel_half=False,fig=None,ax=None,line_kw=None,**kwargs):
        
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        self.check_line_labels(coords,labels)

        transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
                                                                      transform_ydata)
        coord_axis = self._plane.replace(axis,'')
        axis_dim = self.get_dim_from_axis(coord_axis)

        axis_index = self.CoordDF.index_calc(coord_axis,coords)
        line_data = utils.line_indexer(data,axis_index,axis_dim)

        coord_data = self.CoordDF[axis]

        if axis =='y' and channel_half and not self.Domain.is_cylind:
            mid_index = int(0.5*coord_data.size)
            coord_data = coord_data[:mid_index]
            line_data = [line[:mid_index] for line in line_data]


        for i, line in enumerate(line_data):
            if labels is not None:
                label = labels[i]
            else:
                label = line_kw.pop('label',None)

            ax.cplot(transform_xdata(coord_data),transform_ydata(line),label=label,**line_kw)

        return fig, ax
    
    def plot_line_max(self,comp,axis,transform_ydata=None, transform_xdata=None, time=None,fig=None,ax=None,line_kw=None,**kwargs):
        time = self.check_times(time)
        comp = self.check_comp(comp)

        data = self[time,comp]

        return self.plot_line_data_max(data,axis,transform_ydata=transform_ydata, transform_xdata=transform_xdata, fig=fig,ax=ax,line_kw=line_kw,**kwargs)


    def plot_line_data_max(self,data,axis,transform_ydata=None, transform_xdata=None, time=None,fig=None,ax=None,line_kw=None,**kwargs):
        max_axis = self._plane.replace(axis,'')

        axis_dim = self.get_dim_from_axis(max_axis)

        transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
                                                                      transform_ydata)

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        max_data = np.amax(data,axis=axis_dim)
        coords = self.CoordDF[axis]

        ax.cplot(transform_xdata(coords),transform_ydata(max_data),**line_kw)

        return fig, ax

        


    def plot_contour(self,comp,time=None,fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        plane = self._location
        if plane == 'zy' and self.Domain.is_cylind:
            subplot_kw = {'projection' : 'polar'}
            if 'subplot_kw' in kwargs:
                kwargs['subplot_kw'].update(subplot_kw)
            else:
                kwargs['subplot_kw'] = subplot_kw

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        if self._location != self._data_layout:
            flow = self[time,comp].T
        else:
            flow = self[time,comp]

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        ax = ax.pcolormesh(X,Y,flow.squeeze(),**pcolor_kw)

        return fig, ax

    def plot_vector(self,comp,time=None,spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
        raise NotImplementedError

class FlowStruct1D(FlowStruct_base):
    _dim = 1
    def __init__(self,*args,line='y',location=None,**kwargs):
        super().__init__(*args,**kwargs)
        self._set_line_location(line,location)
    
    def _set_line_location(self,line,location):
        self._line = line
        self._location = location

    def _save_line_location(self,hdf_obj):
        hdf_obj.attrs['line'] = self._line.encode('utf-8')

        if self._location is not None:
            hdf_obj.attrs['location'] = self._location.encode('utf-8')

    def to_hdf(self,filepath,key=None,mode='a'):
        super().to_hdf(filepath,key=key,mode=mode)

        hdf_obj = hdfHandler(filepath,mode='a',key=key)
        self._save_line_location(hdf_obj)
        
    @classmethod
    def from_hdf(cls,filename,key=None):
        
        hdf_obj = hdfHandler(filename,mode='r',key=key)

        line = hdf_obj.attrs['line']
        if 'location' in hdf_obj.attrs.keys():
            location = hdf_obj.attrs['location'][0]
        else:
            location = None


        return cls(filename,key=key,line=line,location=location,from_hdf=True)

    def from_internal(self, *args, **kwargs):
        kwargs['line'] = self._line
        kwargs['location'] = self._location

        return self.__class__(self._coorddata,*args,**kwargs)

    def plot_line(self,comp,time=None,label=None,channel_half=False,
                    transform_ydata=None, transform_xdata=None, 
                    fig=None,ax=None,line_kw=None,**kwargs):
        
        time = self.check_times(time)
        comp = self.check_comp(comp)

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        input_kw = cplt.update_line_kw(line_kw).copy()
        if label is not None:
            input_kw['label'] = label
        
        data = self[time,comp]
        fig, ax = self.plot_line_data(data,channel_half=channel_half,transform_ydata=transform_ydata,
                                         transform_xdata=transform_xdata, fig=fig,ax=ax,line_kw=input_kw,**kwargs)

        return fig, ax

    def plot_line_data(self,data,channel_half=False,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
                                                                      transform_ydata)


        coord_data = self.CoordDF[self._line]

        if self._line == 'y' and channel_half and not self.Domain.is_cylind:
            mid_index = int(0.5*coord_data.size)
            coord_data = coord_data[:mid_index]
            data = data[:mid_index]

        ax.cplot(transform_xdata(coord_data),transform_ydata(data),**line_kw)

        return fig, ax



class FlowStruct1D_time(FlowStruct1D):
    def __init__(self,*args,line='y',location=None,**kwargs):
        super().__init__(*args,line=line,location=location,**kwargs)
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

    @property
    def times(self):
        return sorted([float(x) for x in self.outer_index])

    # def plot_line(self,comp,times=None,labels=None,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):

    #     return super().plot_line(comp,outer_indices=times,labels=labels,transform_ydata=transform_ydata, transform_xdata=transform_xdata, fig=fig,ax=ax,line_kw=line_kw,**kwargs)

    def plot_line_time(self,comp,coords,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):
        data = self[None,comp]
        return self.plot_line_time_data(data,coords,transform_ydata=transform_ydata, transform_xdata=transform_xdata, fig=fig,ax=ax,line_kw=line_kw,**kwargs)

    def plot_line_time_data(self,data,y_vals,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
                                                                      transform_ydata)

        y_index = self.CoordDF.index_calc('y',y_vals)
        data_slice = data[y_index]
        for d in data_slice:
            ax.cplot(transform_xdata(self.times),transform_ydata(d),**line_kw)

        return fig, ax

    def plot_line_time_max(self,comp,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):
        comp = self.check_comp(comp)

        data = self[None,comp]

        return self.plot_line_time_data_max(data,transform_ydata=transform_ydata, transform_xdata=transform_xdata, fig=fig,ax=ax,line_kw=line_kw,**kwargs)


    def plot_line_time_data_max(self,data,transform_ydata=None, transform_xdata=None, fig=None,ax=None,line_kw=None,**kwargs):

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        transform_xdata, transform_ydata = self._check_datatransforms(transform_xdata,
                                                                      transform_ydata)


        max_data = np.amax(data,axis=0)
        times = self.times

        ax.cplot(transform_xdata(times),transform_ydata(max_data),**line_kw)

        return fig, ax

    def __getitem__(self,key):
        if self._indexer.is_multikey(key):
            return self._getitem_process_multikey(key)
        
        else:
            if self._indexer.is_listkey(key):
                msg = f"In {self.__class__.__name__}, lists can only index with multikeys"
                raise TypeError(msg)
            return self._getitem_process_singlekey(key)

    def _getitem_process_singlekey(self,key):
        times = self.times

        key_list = [(time,key) for time in times]
        array = np.array([self[k] for k in key_list])
        return array.T


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

        elif value.shape[1] == len(self.times) and key[0] is None:
            index_times = self.times
            index_comp = [key[1]]*len(self.times)
            indices = zip(index_times,index_comp)

            for i,index in enumerate(indices):
                super().set_value(index,value[:,i])


class VTKstruct_base:
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

        for i,time in enumerate(self._flowstruct.times):

            grid = self._grid
            if len(self.times) > 1:
                num_zeros = int(np.log10(len(self.times)))+1
                ext = str(i).zfill(num_zeros)
                file_name = os.path.join(file_name,".%s"%ext)

            for comp in self.comp:
                grid.cell_arrays[np.str_(comp)] = self[time,comp].flatten()
            
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

class VTKstruct3D:
    def __init__(self,flowstruct_obj):
        if not isinstance(flowstruct_obj,FlowStruct3D):
            msg = "This class can only be used on objects of type FlowStruct3D"
            raise TypeError(msg)

        self._flowstruct = flowstruct_obj

    @property
    def _grid(self):
        self._flowstruct._coorddata.create_vtkStructuredGrid()


    def __iadd__(self,other_VTKstruct):
        if not isinstance(other_VTKstruct,self.__class__):
            msg = "This operation can only be used with other VTKstruct's"
            raise TypeError(msg)
        
        if not np.allclose(self._grid.points,other_VTKstruct._grid.points):
            msg = "The grids of the VTKstruct's must be allclose"
            raise ValueError(msg)

        self._flowstruct.concat(other_VTKstruct._flowstruct)

        return self


class VTKstruct2D:
    def __init__(self,flowstruct_obj):
        if not isinstance(flowstruct_obj,FlowStruct2D):
            msg = "This class can only be used on objects of type FlowStruct2D"
            raise TypeError(msg)

        self._flowstruct = flowstruct_obj

    @property
    def _grid(self):
        plane = self._flowstruct._data_layout
        coord_1 = self._flowstruct.Coord_ND_DF[plane[0]]
        coord_2 = self._flowstruct.Coord_ND_DF[plane[1]]
        location = self._flowstruct._location
        coord_3 = [0.] if location is None else [location]

        Y,X,Z = np.meshgrid(coord_2,coord_1,coord_3)

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
