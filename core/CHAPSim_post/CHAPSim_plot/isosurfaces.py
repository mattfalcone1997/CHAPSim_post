from shutil import register_archive_format
import matplotlib as mpl
from matplotlib.pyplot import title, xlabel
import numpy as np

import warnings

Has_pyvista = True
try:
    import pyvista
    import vtk.util.numpy_support as nps
except ImportError:
    msg = "pyvista is not available, 3D plots cannot be created"
    warnings.warn(msg)
    Has_pyvista = False
    


if Has_pyvista:
    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme('report')
    pyvista.rcParams['font']['family'] = 'times'

    class vtkFigure(pyvista.Plotter):

        def __init__(self,nrow=1,ncol=1,**kwargs):
            if 'notebook' not in kwargs.keys():
                kwargs['notebook'] = False
            
            self.shape = (nrow,ncol)
            
            super().__init__(shape=self.shape, **kwargs)
            self._grids = []
            self._no_iso_plots = 0 
            self.show_bounds()

            self[0,0].show_bounds(show_xaxis=False,
                                    show_yaxis=False,
                                    show_zaxis=False,
                                    show_xlabels=False,
                                    show_ylabels=False,
                                    show_zlabels=False)

        def __getitem__(self,key):

            if isinstance(key,int):
                if self.shape[1] == 1 and self.shape[0] > 1:
                    key = (self.shape[0],0)
                else:
                    msg = "integers can be provided to this method only if there is a single column"
                    raise IndexError(msg)
            elif isinstance(key,tuple):
                check_index = all(ind < size for ind, size in zip(key,self.shape))
                if len(key) >2:
                    msg = "Less than two values must be provided to this indexing function"
                    raise IndexError(msg)

                if not check_index:
                    msg = f"The indices provided {key} must be less than the number of subplots {self.shape}"
                    raise IndexError(msg)

            self.subplot(*key)
            return self

        
        def __iter__(self):
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    yield self[i,j]
            
        def _check_grids(self,grid):
            existing_point_arr = [arr for arr in self._grids]

            if any([np.array_equal(grid.points,arr.points) for arr in existing_point_arr]):
                for i, arr in enumerate(existing_point_arr):
                    if np.array_equal(grid.points,arr.points):
                        grid = arr; break
            else:
                self._grids.append(grid)


            return grid

        def _check_coords(self,scalar_array,*coords,use_cells=True):
            if use_cells:
                shape = tuple(x.size-1 for x in coords)
            else:
                shape = tuple(x.size for x in coords)

            if shape != scalar_array.shape:
                if not all(coord.ndim ==1 for coord in coords):
                    msg = "Each coordinate array must have dimension one"
                else:
                    msg = ("The size of the each input array must be 1 more"
                            " than the corresponding scalar array shape")
                raise ValueError(msg)   

            if len(coords) ==2:
                z = np.array([0.])
                coords = (*coords,z)

            return np.meshgrid(*coords) 


        def plot_surface(self,x,y,V,label=None,**mesh_kw):
            
            X,Y,Z = self._check_coords(V,x,y,use_cells=False)

            grid = pyvista.StructuredGrid(X,Y,Z)

            grid = self._check_grids(grid)

            grid.point_arrays[label] = V.flatten()
            # pgrid = grid.cell_data_to_point_data()

            surf = grid.warp_by_scalar(scalars=label)
            
            mesh_kw = update_mesh_kw(mesh_kw,interpolate_before_map=True)
            self.add_mesh(surf,**mesh_kw)



        def plot_isosurface(self,x,y,z,V,isovalue,label=None,**mesh_kw):
            
            X,Y,Z = self._check_coords(V,x,y,z)

            grid = pyvista.StructuredGrid(X,Y,Z)
            
            grid = self._check_grids(grid)

            if label is None:
                self._no_iso_plots += 1
                label = 'iso_%d'% self._no_iso_plots
            
            grid.cell_arrays[label] = V.flatten()
            # pgrid = grid.cell_data_to_point_data()


            contour = grid.contour(isosurfaces=1,scalars=label,
                                    preference='point',rng=(isovalue,isovalue))

            mesh_kw = update_mesh_kw(mesh_kw,interpolate_before_map=True)

            
            self.add_mesh(contour,**mesh_kw)
            # self.remove_scalarbar()

        def set_title(self,str):
            self.show_bounds(title=str)

        def set_xlabel(self,str):
            self.show_bounds(xlabel=str)

        def set_ylabel(self,str):
            self.show_bounds(ylabel=str)

        def set_zlabel(self,str):
            self.show_bounds(zlabel=str)

        def set_xlim(self,min_max):
            bounds = self.bounds
            bounds[0], bounds[1] = min_max
            self.show_bounds(bounds=bounds)

        def set_ylim(self,min_max):
            bounds = self.bounds
            bounds[2], bounds[3] = min_max
            self.show_bounds(bounds=bounds)

        def set_zlim(self,min_max):
            bounds = self.bounds
            bounds[4], bounds[5] = min_max
            self.show_bounds(bounds=bounds)

        def set_clim(self,clim):
            self.mapper.scalar_range = clim[0], clim[1]

        def savefig(self,filename):
            self.show(screenshot=filename)

        def __del__(self):
            self.close()
    
    def update_mesh_kw(mesh_kw,**kwargs):
        if mesh_kw is None:
            mesh_kw = {}

        for key, val in kwargs.items():
            if key not in mesh_kw.keys():
                mesh_kw[key] = val
        
        return mesh_kw


        

class Figure3D:
    def __new__(cls,*args,**kwargs):
        if Has_pyvista:
            return vtkFigure(*args,**kwargs)
        else:
            msg = ("There was an issue importing the pyvista module so"
                    " 3D plotting is disabled. Consult pyvista documentation "
                    "on remedies")
            raise ImportError(msg)