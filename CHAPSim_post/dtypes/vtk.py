
from pyvista import StructuredGrid
import numpy as np
from abc import abstractproperty
import os
import copy
import vtk

class VTKstruct_base:
    def __init__(self,flowstruct_obj,use_celldata=True):
        self._flowstruct = flowstruct_obj
        self._use_cell_data = use_celldata
        
        if self._check_polar():
            self._use_pipe_rep = True
        else:
            self._use_pipe_rep = False
            
    def _check_polar(self):
        return self._flowstruct._polar_plane is not None
    
    def __call__(self,*,use_pipe=True):
        if self._flowstruct.Domain.is_polar:
            self._use_pipe_rep = use_pipe
        
        return self
    
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
                key = tuple([k] for k in key)

        new_flowstruct = self._flowstruct[key]

        for k, array in new_flowstruct:

            if len(new_flowstruct.outer_index) < 2:
                k = k[1]
            return_grid.cell_data[np.str_(k)] = array.flatten()
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

        super().__init__(flowstruct_obj,use_celldata=cell_data)

    @property
    def _grid(self):
        plane = self._flowstruct._data_layout        
        
        if self._use_cell_data:
            coord_1 = self._flowstruct.Coord_ND_DF[plane[0]]
            coord_2 = self._flowstruct.Coord_ND_DF[plane[1]]
            coord_3 = self._flowstruct.Coord_ND_DF[plane[2]]
        else:
            coord_1 = self._flowstruct.CoordDF[plane[0]]
            coord_2 = self._flowstruct.CoordDF[plane[1]]
            coord_3 = self._flowstruct.CoordDF[plane[2]]
        
        Y,X,Z = np.meshgrid(coord_2,coord_3,coord_1)


        if self._use_pipe_rep:
            data_list = [Z,Y,X]
            
            polar_plane = self._flowstruct._polar_plane.copy()
            wall_line = self._flowstruct._wall_normal_line

            r_loc = self._flowstruct._data_layout.index(wall_line)
            polar_plane.remove(wall_line)
            theta_loc = self._flowstruct._data_layout.index(polar_plane[0])

            r_array = data_list[r_loc]
            theta_array = data_list[theta_loc]
            
            y_cart = r_array*np.sin(theta_array)
            x_cart = r_array*np.cos(theta_array)
            
            data_list[r_loc] = x_cart
            data_list[theta_loc] = y_cart
            
            Z,Y,X = data_list        
            
                

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


class VTKstruct2D(VTKstruct_base):
    def __init__(self,flowstruct_obj,cell_data=True):
        if flowstruct_obj._dim != 2 :
            msg = "This class can only be used on objects of type FlowStruct2D"
            raise TypeError(msg)

        super().__init__(flowstruct_obj,use_celldata=cell_data)
        
        
    @property
    def _grid(self):
        plane = self._flowstruct._data_layout
            
        if self._check_polar():
            r = self._flowstruct._wall_normal_line
            theta = self._flowstruct._polar_plane
            theta.remove(r)
            
            if self._use_cell_data:
                r_array = self._flowstruct.Coord_ND_DF[r]
                theta_array =  self._flowstruct.Coord_ND_DF[theta]
            else:
                r_array = self._flowstruct.CoordDF[r]
                theta_array =  self._flowstruct.CoordDF[theta]
            
            x = r_array*np.sin(theta_array)
            y = r_array*np.cos(theta_array)
            
            if plane[0] == r:
                coord_1 = x
                coord_2 = y
            else:
                coord_1 = y
                coord_2 = x
        else:
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
