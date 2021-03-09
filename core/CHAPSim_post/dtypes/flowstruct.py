
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
from CHAPSim_post.utils import misc_utils,indexing
import CHAPSim_post.plot as cplt
from .dtypes import *
import sys
import pyvista



class coordstruct(datastruct):
    def set_domain_handler(self,GeomHandler):
        self._domain_handler = GeomHandler

    @property
    def staggered(self):
        XCC = self['x']
        YCC = self['y']
        ZCC = self['z']

        XND = np.zeros(XCC.size+1) 
        YND = np.zeros(YCC.size+1)
        ZND = np.zeros(ZCC.size+1)

        XND[0] = 0.0
        YND[0] = -1.0 if self.metaDF['iCase'] == 1 else 0
        ZND[0] = 0.0

        for i in  range(1,XND.size):
            XND[i] = XND[i-1] + 2*XCC[i-1]-XND[i-1]
        
        for i in  range(1,YND.size):
            YND[i] = YND[i-1] + 2*YCC[i-1]-YND[i-1]

        for i in  range(1,ZND.size):
            ZND[i] = ZND[i-1] + 2*ZCC[i-1]-ZND[i-1]

        return self.__class__({'x':XND,'y':YND,'z':ZND})

    def vtkStructuredGrid(self):
        x_coords = self.staggered['x']
        y_coords = self.staggered['y']
        z_coords = self.staggered['z']
        Y,Z,X = np.meshgrid(y_coords,z_coords,x_coords)
        grid = pyvista.StructuredGrid(Y,Z,X)
        return grid

    def index_calc(self,comp,vals):
        return indexing.coord_index_calc(self,comp,vals)
    
    def check_plane(plane):
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
    def __init__(self,CoordDF,*args,from_hdf=False,**kwargs):
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        self._set_coords(CoordDF,from_hdf,*args,**kwargs)   

    @property
    def CoordDF(self):
        return self._CoordDF

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
                self._CoordDF = coordstruct.from_hdf(file_name,key=key)
            elif CoordDF is not None:
                self._CoordDF = CoordDF
            else:
                msg = "If CoordDF datastruct is not present in HDF file, a datastruct must be provided"
                raise KeyError(msg)
        else:
            self._CoordDF = coordstruct(CoordDF._data,copy=True)

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

    @property
    def data_shape(self):
        return self._data[self.index[0]].shape

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

class flowstruct2D(flowstruct_base):
    def _set_coords(self,CoordDF,from_hdf,*args,**kwargs):
        super()._set_coords(CoordDF,from_hdf,*args,**kwargs)
        if not len(self._CoordDF.keys()) != 3:
            msg = "for a 2D flowstruct the number of keys in the coordstruct should be 3"
            raise ValueError(msg)
    def _check_coord(self,time,comp):
        flow_slice = self[time,comp]

        comps = self.CoordDF.keys()
        coord_shape = tuple(x.size for _,x in self.CoordDF)
        if flow_slice.shape == coord_shape:
            x_coord = self.CoordDF[comps[0]]
            y_coord = self.CoordDF[comps[1]]
        elif flow_slice.shape == coord_shape[::-1]:
            x_coord = self.CoordDF[comps[1]]
            y_coord = self.CoordDF[comps[0]]
        else:
            msg = "The coordinate arrays are invalid for this flowstruct"
            raise ValueError(msg)
        return x_coord, y_coord, flow_slice

    def plot_contour(self,comp,time=None,fig=None,ax=None,pcolor_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        self.check_times(time)
        self.check_comp(comp)


        x_coord, y_coord, flow_slice = self._check_coord(time,comp)

        X,Y = np.meshgrid(x_coord,y_coord)
        ax = ax.pcolormesh(X,Y,flow_slice,**pcolor_kw)

        return fig, ax

    def plot_vector(self,comps,time=None,spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        quiver_kw = cplt.update_quiver_kw(quiver_kw)

        self.check_times(time)

        for comp in comps:
            self.check_comp(comp)

        U = self[time,comps[0]][::spacing[0],::spacing[1]]
        V = self[time,comps[1]][::spacing[0],::spacing[1]]

        coord_comps  = self.CoordDF.keys()
        coord_1 = self.CoordDF[coord_comps[0]]
        coord_2 = self.CoordDF[coord_comps[1]]

        if coord_1.size == U.shape[0]:
            coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        else:
            coord_1_mesh, coord_2_mesh = np.meshgrid(coord_2,coord_1)
        if not coord_1_mesh.shape == U.shape:
            msg = "The size of the coords is wrong"
            raise ValueError(msg)

        
        scale = np.amax(U)*coord_1.size/np.amax(coord_1)/scaling
        ax = ax.quiver(coord_1_mesh, coord_2_mesh,U,V,angles='uv',scale_units='xy', scale=scale,**quiver_kw)

        return fig, ax

    def plot_line(self,time,comp,axis_dir,axis_vals,fig=None,ax=None,line_kw=None,**kwargs):
        axis = set(axis_dir).difference(self.CoordDF.keys())
        axis_index = self.CoordDF.index_calc(axis,axis_vals)

        flow_slice = self[time,comp]

        
        coord = self.CoordDF[axis_dir]
        if coord.size == flow_slice.shape[0]:
            flow_lines = flow_slice[:,axis_index]
        else:
            flow_lines = flow_slice[axis_index]


        fig,ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        for flow_line in flow_lines:
            ax.cplot(coord,flow_line)

        return fig, ax

    def plot_surf(self,comp,plane,time=None,fig=None,ax=None,surf_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,projection='3d',**kwargs)
        surf_kw = cplt.update_surf_kw(surf_kw)

        self.check_times(time)
        self.check_comp(comp)

        x_coord, y_coord, flow_slice = self._check_coord(time,comp)

        X,Y = np.meshgrid(x_coord,y_coord)

        ax = ax.plot_surface( Y,X,flow_slice.T,**surf_kw)

        return fig, ax

class flowstruct3D(datastruct):
    def __init__(self,CoordDF,*args,from_hdf=False,**kwargs):
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        self._set_coords(CoordDF,from_hdf,*args,**kwargs)

    def _set_coords(self,CoordDF,from_hdf,*args,**kwargs):
        super()._set_coords(CoordDF,from_hdf,*args,**kwargs)
        if not len(self._CoordDF.keys()) != 3:
            msg = "for a 3D flowstruct the number of keys in the coordstruct should be 3"
        raise ValueError(msg)

    def to_vtk(self,file_name):
        
        for i,time in enumerate(self.times):
            file_base = os.path.basename(file_name)
            file_name = os.path.join(file_base,".vtk")
            grid = self.CoordDF.vtkStructuredGrid()
            if len(self.times) > 1:
                num_zeros = int(np.log10(len(self.times)))+1
                ext = str(num_zeros).zfill(num_zeros)
                file_name = os.path.join(file_name,".%s"%ext)

            for comp in self.comp:
                grid.point_arrays[comp] = self[time,comp].flatten()
            pyvista.save_meshio(file_name,grid,file_format="vtk")


    def plot_contour(self,comp,plane,axis_val,time=None,fig=None,ax=None,pcolor_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)

        self.check_times(time)
        self.check_comp(comp)

        plane, coord = self.CoordDF.check_plane(plane)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = indexing.contour_indexer(self[time,comp],axis_index,coord)

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        X,Y = np.meshgrid(x_coord,y_coord)
        ax = ax.pcolormesh(X,Y,flow_slice,**pcolor_kw)

        return fig, ax

    def plot_vector(self,plane,axis_val,time=None,spacing=(1,1),scaling=1,fig=None,ax=None,quiver_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        quiver_kw = cplt.update_quiver_kw(quiver_kw)

        self.check_times(time)

        plane, coord = self.CoordDF.check_plane(plane)

        coord_1 = self.CoordDF[slice[0]][::spacing[0]]
        coord_2 = self.CoordDF[slice[1]][::spacing[1]]
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

    def plot_surf(self,comp,plane,axis_val,time=None,fig=None,ax=None,surf_kw=None,**kwargs):
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,projection='3d',**kwargs)
        surf_kw = cplt.update_surf_kw(surf_kw)

        self.check_times(time)
        self.check_comp(comp)

        plane, coord = self.CoordDF.check_plane(plane)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = indexing.contour_indexer(self[time,comp],axis_index,coord)

        x_coord = self.CoordDF[plane[0]]
        y_coord = self.CoordDF[plane[1]]

        X,Y = np.meshgrid(x_coord,y_coord)

        axis_index = self.CoordDF.index_calc(coord,axis_val)
        flow_slice = indexing.contour_indexer(self[time,comp],axis_index,coord)
        ax = ax.plot_surface( Y,X,flow_slice.T,**surf_kw)

        return fig, ax


class flowstruct_time_base(flowstruct_base):
    def __init__(self,CoordDF,*args,from_hdf=False,**kwargs):
        super().__init__(*args,from_hdf=from_hdf,**kwargs)
        self._set_coords(CoordDF,from_hdf,*args,**kwargs)
        self._times = []

    def add_time(self,time,data,comps):
        if not len(comps) == len(data):
            msg = "The number of components must be equal to the length of the data"
            raise ValueError(msg)

        if not self._times:
            self._data = dict(zip(comps,data))
        else:
            for comp, dat in zip(comps,data):
                self._data[comp] =dat
        self._times.append(time)

    def __getitem__(self,key):
        if key[0] is not None:
            return super().__getitem__(key)
        else:
            shape = (len(self.times),*self.data_shape)
            array = np.zeros(shape[::-1])
            for i,val in enumerate(self._data.values()):
                array[:,i] = val

            return array

class flowstruct1D_time(flowstruct_time_base):
    def plot_line(self,time,comp,axis_dir,axis_vals,fig=None,ax=None,line_kw=None,**kwargs):
        
        fig, ax = cplt.create_fig_ax_without_squeeze(fig=fig,ax=ax,**kwargs)
        if axis_dir in self.CoordDF.index:
            flow = self._data[None,comp]
            axis_index = self.CoordDF.index_calc(axis_dir,axis_vals)
            flow = [flow[index] for index in axis_index]
            coord = self.CoordDF[axis_dir]
        else:
            self.check_times(axis_vals)
            flow = [self._data[val,comp] for val in axis_vals]
            coord = [float(time) for time in self.times]

        line_kw = cplt.update_line_kw(line_kw)
        for line in flow:
            ax.cplot(coord,line,**line_kw)

        return fig, ax

class flowstruct2D_time(flowstruct_time_base,flowstruct2D):
    pass