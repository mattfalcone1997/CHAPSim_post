from shutil import register_archive_format
import matplotlib as mpl
from matplotlib.pyplot import title, xlabel
import numpy as np

import warnings
from .mpl_utils import subplots
from CHAPSim_post import rcParams
Has_pyvista = True
usePyvista = True
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
            
            # self.shape = (nrow,ncol)
            
            super().__init__(shape=(nrow,ncol), **kwargs)

            self._axes = np.empty(self.shape,dtype=object).flatten()
            
            if len(self._render_idxs) == 1:
                indexer = self._render_idxs[0]
            else:
                indexer = self._render_idxs

            for i, index in enumerate(indexer):
                self._axes[i] = vtkRenderWindow(self,index)

        @property
        def axes(self):
            return self._axes

        def __getitem__(self,key):
            print(key)
            if isinstance(key,(int,np.int64)):
                if self.shape[1] == 1 and self.shape[0] > 1:
                    key = (key,0)
                elif self.shape[0] == 1 and self.shape[1] > 1:
                    key = (0,key)
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

        
        # def __iter__(self):
        #     for i in range(self.shape[0]):
        #         for j in range(self.shape[1]):
        #             yield self[i,j]
        


        # def _get_grids_from_active_renderer(self):
        #     grid_list = []
        #     for grid, ind,_ in self._grids:
        #         if ind == self._active_renderer_index:
        #             grid_list.append(grid)
        #     return grid_list

        # def set_title(self,str):
        #     self.show_bounds(title=str)

        # def set_xlabel(self,str):
        #     self.show_bounds(xlabel=str)

        # def set_ylabel(self,str):
        #     self.show_bounds(ylabel=str)

        # def set_zlabel(self,str):
        #     self.show_bounds(zlabel=str)

        # def set_xlim(self,min_max):
        #     bounds = self.show_bounds()
        #     bounds[0], bounds[1] = min_max
        #     self.show_bounds(bounds=bounds)
        #     for i,(grid, ind,_) in enumerate(self._grids):
        #         if ind == self._active_renderer_index:
        #             self._grids[i][0] = grid.clip_box(bounds=bounds,invert=False)

        # def set_ylim(self,min_max):
        #     bounds = self.bounds
        #     print(bounds)
        #     # bounds[0], bounds[1] = [-bounds[1],bounds[1]]
        #     bounds[2], bounds[3] = min_max
        #     self.show_bounds(bounds=bounds)
        #     # print(bounds)
        #     for i,(grid, ind,_) in enumerate(self._grids):
        #         if ind == self._active_renderer_index:
        #             self._grids[i][0] = grid.clip_box(bounds=bounds,invert=False)

        # def set_zlim(self,min_max):
        #     bounds = self.bounds
        #     bounds[4], bounds[5] = min_max
        #     self.show_bounds(bounds=bounds)
        #     for i,(grid, ind,_) in enumerate(self._grids):
        #         if ind == self._active_renderer_index:
        #             self._grids[i][0] = grid.clip_box(bounds=bounds,invert=False)

        # def set_clim(self,clim):
        #     self.mapper.scalar_range = clim[0], clim[1]

        # def _plot_grids(self):
        #     for grid, rend_index, mesh_kw in self._grids:
        #         print(rend_index)
        #         self[rend_index].add_mesh(grid,**mesh_kw)
        
        def _render(self):
            for a in self._axes:
                for actor, mesh_kw in a._actors:
                    actor1 = actor.clip_box(a.bounds,invert=False)
                    self[a._render_index].add_mesh(actor1,**mesh_kw)
                    
                    if a._clims is not None:
                        self[a._render_index].mapper.scalar_range = a._clims[0], a._clims[1]

        def figure_show(self,*args,**kwargs):
            self._render()
            return self.show(*args,**kwargs)

        def savefig(self,filename,cpos=None,**kwargs):
            self._render()
            if cpos is None:
                self._on_first_render_request(cpos)
                cpos = self.camera_position.to_list()
                cpos[0] = (cpos[0][0],-cpos[0][1],cpos[0][2])
            # for renderer in self.renderers:
            #     print(renderer.bounds)
            # cpos = pyvista.CameraPosition(*cpos)
            for f in self:
                f.camera_position = pyvista.CameraPosition(*cpos)
            return self.show(screenshot=filename,cpos=cpos,**kwargs)

        # def __del__(self):
        #     self.close()
    class vtkRenderWindow:
        def __init__(self,Plotter,render_index):
            self._figure = Plotter
            self._render_index = render_index

            self._lims=[None]*6
            self._clims = None
            self._actors = []

            self._no_iso_plots = 0 
            self._no_surf_plots = 0 

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

        def set_xlim(self,lims):
            self._lims=self._figure[self._render_index].bounds
            self._lims[0], self._lims[1] = lims


        def calc_grid_extent(self):
            x_min = np.float("inf"); x_max = -np.float("inf")
            y_min = np.float("inf"); y_max = -np.float("inf")
            z_min = np.float("inf"); z_max = -np.float("inf")
            for actor,_ in self._actors:
                x_min = min(x_min,np.amin(actor.x))
                x_max = max(x_max,np.amax(actor.x))
                y_min = min(y_min,np.amin(actor.y))
                y_max = max(y_max,np.amax(actor.y))
                z_min = min(z_min,np.amin(actor.z))
                z_max = max(z_max,np.amax(actor.z))

            return [x_min,x_max,y_min,y_max,z_min,z_max]
        def set_ylim(self,lims):
            self._lims[2], self._lims[3] = lims

        def set_zlim(self,lims):
            self._lims[4], self._lims[5] = lims

        @property
        def bounds(self):
            lims = self.calc_grid_extent()
            for i, lim in enumerate(self._lims):
                if lim is None:
                    self._lims[i] = lims[i]

            return self._lims

        def set_clim(self,lims):
            self._clims = lims
        def add_actor(self,actor,**mesh_kw):
            self._actors.append([actor,mesh_kw])

        def set_title(self,str):
            self[self._render_index].show_bounds(title=str)



        def plot_surface(self,x,y,V,label=None,**mesh_kw):
            
            X,Y,Z = self._check_coords(V,x,y,use_cells=False)

            grid = pyvista.StructuredGrid(X,Y,Z)

            if label is None:
                self._no_surf_plots += 1
                label = 'surf_%d'% self._no_surf_plots
            
            grid.point_arrays[label] = V.flatten()

            surf = grid.warp_by_scalar(scalars=label)
            
            mesh_kw = update_mesh_kw(mesh_kw,interpolate_before_map=True)
            self.add_actor(surf,label=label,**mesh_kw)



        def plot_isosurface(self,x,y,z,V,isovalue,label=None,**mesh_kw):
            
            X,Y,Z = self._check_coords(V,x,y,z)

            grid = pyvista.StructuredGrid(X,Y,Z)
            
            

            if label is None:
                self._no_iso_plots += 1
                label = 'iso_%d'% self._no_iso_plots
            
            grid.cell_arrays[label] = V.flatten()

            contour = grid.contour(isosurfaces=1,scalars=label,
                                    preference='point',rng=(isovalue,isovalue))

            mesh_kw = update_mesh_kw(mesh_kw,interpolate_before_map=True)

            self.add_actor(contour,**mesh_kw)

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

def subplots3D(*args,**kwargs):
    if rcParams['UsePyVista']:
        fig = Figure3D(*args,**kwargs)
        ax = fig.axes
    else:
        if 'subplot_kw' not in kwargs.keys():
            kwargs['subplot_kw'] = {}
        kwargs['subplot_kw']['projection'] = '3d'
        print(kwargs['subplot_kw'])
        fig, ax = subplots(*args,**kwargs)
        
    return fig, ax