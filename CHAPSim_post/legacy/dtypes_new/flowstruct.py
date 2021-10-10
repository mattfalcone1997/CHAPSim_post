from abc import abstractproperty
from .dtypes import *
from . import utils

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

    def check_line(self,line):
        if line not in self.index:
            msg = f"The line must be in {self.index}"
            raise KeyError(msg)

        return line

class FlowStruct_base(datastruct):
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

    @abstractproperty
    def VTK(self):
        pass

    def check_times(self,key,err=None,warn=None):
        if err is None:
            msg = f"FlowStruct3D soes not have time {key}"
            err = KeyError(msg)
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

class FlowStruct3D(FlowStruct_base):

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


    def plot_contour(self,comp,plane,axis_val,time=None,rotate_axes=False,fig=None,ax=None,pcolor_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        plane, coord = self.CoordDF.check_plane(plane)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = utils.contour_indexer(self[time,comp],axis_index,coord)

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        ax = ax.pcolormesh(X,Y,flow_slice.squeeze(),**pcolor_kw)

        return fig, ax

    def plot_vector(self,comp,plane,axis_val,time=None,spacing=(1,1),scaling=1,rotate_axes=False,fig=None,ax=None,quiver_kw=None,**kwargs):
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
    def __init__(self,*args,plane =None,data_layout=None,location=None,**kwargs):
        super().__init__(*args,**kwargs)

        if plane is None:
            self._location = 'xy'
            self._data_layout = 'yx'
        else:
            self._location, _ = self.CoordDF.check_plane(plane)
            self._data_layout = data_layout

            if self._data_layout is None:
                msg = "Data layout must be provided if plane is provided"
    
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

    def plot_line(self,comp,axis,coords,time=None,labels=None,fig=None,ax=None,pcolor_kw=None,**kwargs):
        time = self.check_times(time)
        comp = self.check_comp(comp)

        data = self[time,comp]

        return self.plot_line_data(data,axis,coords,labels=labels,
                        fig=fig,ax=ax,pcolor_kw=pcolor_kw,**kwargs)
    
       
    def plot_line_data(self,data,axis,coords,labels=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        self.check_line_labels(coords,labels)

        axis_dim = self.get_dim_from_axis(axis)

        coord_axis = self._location.replace(axis,'')
        axis_index = self.CoordDF.index_calc(coord_axis,coords)
        line_data = utils.line_indexer(data,axis_index,axis_dim)

        coord_data = self.CoordDF[axis]

        for i, line in enumerate(line_data):
            if labels is not None:
                label = labels[i]
            else:
                label = line_kw.pop('label',None)

            ax.cplot(coord_data,line,label=label,**line_kw)

        return fig, ax
    
    def plot_contour(self,comp,time=None,fig=None,ax=None,pcolor_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        time = self.check_times(time)
        comp = self.check_comp(comp)

        plane = self._location

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
    def __init__(self,*args,line='y',location=None,**kwargs):
        super().__init__(*args,**kwargs)
        self._line = self.CoordDF.check_line(line)
        self._location = location
    
    def plot_line(self,comp,outer_indices=None,labels=None,fig=None,ax=None,line_kw=None,**kwargs):
        if outer_indices is not None:
            outer_indices = [self.check_outer(x) for x in outer_indices]
        else:
            outer_indices = [None]

        comp = self.check_comp(comp)
        self.check_line_labels(outer_indices,labels)

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        for i, outer_index in enumerate(outer_indices):
            input_kw = cplt.update_line_kw(line_kw).copy()
            if labels is not None:
                input_kw['label'] = labels[i]
            
            data = self[outer_index,comp]
            fig, ax = self.plot_line_data(data,fig=fig,ax=ax,line_kw=input_kw,**kwargs)


        return fig, ax

    def plot_line_data(self,data,fig=None,ax=None,line_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)
        coord_data = self.CoordDF[self._line]

        ax.cplot(coord_data,data,**line_kw)

        return fig, ax



class FlowStruct1D_time(FlowStruct1D):
    def __init__(self,*args,times=None,**kwargs):
        super().__init__(*args,line='y',location=None,**kwargs)
        if not self._is_multidim():
            msg = "This flow structure must be mutlidimensional"
            raise ValueError(msg)
        
    
    @property
    def times(self):
        return [float(x) for x in self.inner_index]

    def plot_line(self,comp,times=None,labels=None,fig=None,ax=None,line_kw=None,**kwargs):

        return super().plot_line(comp,outer_indices=times,labels=labels,fig=fig,ax=ax,line_kw=line_kw,**kwargs)

    def plot_line_time(self,comp,coords,fig=None,ax=None,line_kw=None,**kwargs):
        data = self[None,comp]
        return self.plot_line_time_data(data,coords,fig=fig,ax=ax,line_kw=line_kw,**kwargs)

    def plot_line_time_data(self,data,y_vals,fig=None,ax=None,line_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        line_kw = cplt.update_line_kw(line_kw)

        y_index = self.CoordDF.index_calc('y',y_vals)
        data_slice = data[y_index]
        for d in data_slice:
            ax.cplot(self.times,d,**line_kw)

        return fig, ax
        
    def __getitem__(self,key):
        if isinstance(key,tuple):
            if len(key) > 1:
                return self._getitem_multikey(key)
            else:
                return self._getitem_singlekey(*key)
        else:
            return self._getitem_singlekey(key)

    def _getitem_singlekey(self,key):
        times = self.times

        key_list = [(time,key) for time in times]
        array = np.array([self[k] for k in key_list])
        return array.T


    def _getitem_multikey(self,key):
        if key[0] is None:
            key = self._getitem_process_singlekey(key[1])
        else:
            key = self._indexer._getitem_process_multikey(key)

        return self._data[key]
        

    def set_value(self,key,value):
        msg = "Flowstruct1D_time elements can only be set for with multiple keys"
        if not isinstance(key,tuple):
            raise KeyError(msg)
        elif len(key) < 2:
            raise KeyError(msg)
            
        key = self._indexer.setitem_process_multikey(key)

        self._data[key] = value


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
    @property
    def flowstruct(self):
        return self._flowstruct

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
        if not isinstance(flowstruct_obj,FlowStruct3D):
            msg = "This class can only be used on objects of type FlowStruct3D"
            raise TypeError(msg)

        self._flowstruct = flowstruct_obj

    @property
    def _grid(self):
        plane = self._flowstruct._data_layout
        coord_1 = self._flowstruct.Coord_ND_DF[plane[0]]
        coord_2 = self._flowstruct.Coord_ND_DF[plane[1]]
        coord_3 = [0.]

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
