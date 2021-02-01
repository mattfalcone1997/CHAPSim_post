"""
## _common.py
A module to create base level visualisation classes 
providing functionality common to several classes
"""
import numpy as np
import matplotlib as mpl
import warnings
from abc import abstractmethod, ABC

import CHAPSim_post.CHAPSim_plot as cplt
from CHAPSim_post.utils import indexing, misc_utils

class common3D(ABC):
    @abstractmethod
    def _check_outer_index(self,processDF,outer,warn_msg,err_msg):

        # warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        # err_msg = "PhyTime provided is not in the CHAPSim_AVG datastruct, recovery impossible"
        with warnings.catch_warnings(record=True) as w:
            key = processDF.check_index(outer,err_msg=err_msg,warn_msg=warn_msg,outer=True)
            a = w
        if outer is not None and len(a)>0:
            for warn in a:
                warnings.warn(a.message)
        return key[0]

    def _check_attr(self):
        attr_list = ["CoordDF","meta_data"]
        for attr in attr_list:
            if not hasattr(self,attr):
                msg = (f"The class {self.__class__.__name__}, requires the attribute"
                        f" {attr} to use base class Common3D's methods")
                raise AttributeError(msg)

    def plot_contour(self,ProcessDF,avg_data,comp,axis_vals,plane='xz',PhyTime=None,x_split_list=None,y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        self._check_attr()
        
        PhyTime = self._check_outer_index(ProcessDF,PhyTime)        
        
        axis_vals = misc_utils.check_list_vals(axis_vals)
            
        plane , coord, axis_index = indexing.contour_plane(plane,axis_vals,avg_data,y_mode,PhyTime)


        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = ProcessDF[PhyTime,comp]
                
        if x_split_list is None:
            x_split_list=[np.amin(x_coords),np.amax(x_coords)]
        
        x_size = np.amax(x_coords) - np.amin(x_coords)
        z_size = 1.2*(np.amax(z_coords) - np.amin(z_coords))

        ax_layout = (len(x_split_list)-1,len(axis_vals))
        figsize=[x_size*len(axis_vals),z_size*(len(x_split_list)-1)]

        
        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(*ax_layout,fig=fig,ax=ax,**kwargs)

        ax=ax.flatten()

        # x_coords_split=indexing.coord_index_calc(self.CoordDF,'x',x_split_list)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)
        
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        X, Z = np.meshgrid(x_coords, z_coords)

        ax=ax.flatten()
        max_val = np.amax(fluct); min_val=np.amin(fluct)
        for j,_ in enumerate(x_split_list[:-1]):
            for i,_ in enumerate(axis_vals):
                fluct_slice = indexing.contour_indexer(fluct,axis_index[i],coord)
                # print(fluct_slice.shape)
                ax1 = ax[j*len(axis_vals)+i].pcolormesh(X,Z,fluct_slice,**pcolor_kw)
                ax1.set_clim(min_val,max_val)

                ax1.axes.set_xlabel(r"$%s/\delta$" % plane[0])
                ax1.axes.set_ylabel(r"$%s/\delta$" % plane[0])
                ax1.axes.set_xlim([x_split_list[j],x_split_list[j+1]])
                ax1.axes.set_title(r"$%s/\delta=%.3g$"%(title_symbol,axis_vals[i]),loc='right')
                ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                
                cbar=fig.colorbar(ax1,ax=ax[j*len(axis_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp)

                ax[j*len(axis_vals)+i]=ax1
                ax[j*len(axis_vals)+i].axes.set_aspect('equal')
        fig.tight_layout()

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
            
    def plot_streaks(self,ProcessDF,avg_data,comp,vals_list,x_split_list=None,PhyTime=None,ylim='',Y_plus=True,*args,colors='',fig=None,ax=None,**kwargs):

        self._check_attr()
        PhyTime = self._check_outer_index(ProcessDF,PhyTime)   

        fluct = ProcessDF[PhyTime,comp]
        
        if ylim:
            if Y_plus:
                y_index= indexing.Y_plus_index_calc(avg_data,self.CoordDF,ylim)
            else:
                y_index=indexing.coord_index_calc(self.CoordDF,'y',ylim)
            # y_coords=y_coords[:(y_index+1)]
            fluct=fluct[:,:y_index,:]
        else:
            y_index = self.shape[1]
        # x_coords, y_coords , z_coords = self.meta_data.return_edge_data()
        # y_coords=y_coords[:(y_index+1)]

        # Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)
        
        # if not fig:
        #     fig = cplt.vtkFigure()
        # for val in vals_list:
        #     fig.plot_isosurface(X,Z,Y,fluct,val)
        X = self.meta_data.CoordDF['x']
        Y = self.meta_data.CoordDF['y'][:y_index]
        Z = self.meta_data.CoordDF['z']
        # print(X.shape,Y.shape,Z.shape,fluct.shape)
        if fig is None:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [9,5.5*(len(x_split_list)-1)]
            fig = cplt.mCHAPSimFigure(visible='off',**kwargs)
        else:
            if not isinstance(fig, cplt.matlabFigure):
                raise TypeError("fig must be of type %s not %s"\
                                %(cplt.matlabFigure,type(fig)))
        if ax is None:
            ax = fig.subplots(len(x_split_list)-1,squeeze=False)
        else:
            if not isinstance(ax, cplt.matlabAxes) and not isinstance(ax,np.ndarray):
                raise TypeError("fig must be of type %s not %s"\
                                %(cplt.matlabAxes,type(ax)))
        for j in range(len(x_split_list)-1):
            x_start = indexing.coord_index_calc(self.CoordDF,'x',x_split_list[j])
            x_end = indexing.coord_index_calc(self.CoordDF,'x',x_split_list[j+1])
            for val,i in zip(vals_list,range(len(vals_list))):
                
                
                color = colors[i%len(colors)] if colors else ''
                patch = ax[j].plot_isosurface(Y,Z,X[x_start:x_end],fluct[:,:,x_start:x_end],val,color)
                ax[j].add_lighting()
                # patch.set_color(colors[i%len(colors)])

        return fig, ax

    def plot3D_xz(self,ProcessDF,avg_data,comp,y_vals,y_mode='half-channel',PhyTime=None,x_split_list=None,fig=None,ax=None,surf_kw=None,**kwargs):

        self._check_attr()
        PhyTime = self._check_outer_index(ProcessDF,PhyTime)    

        y_vals = misc_utils.check_list_vals(y_vals)    

        axis_index = indexing.y_coord_index_norm(avg_data,y_vals,0,y_mode)
        axis_index = np.diag(axis_index)

        CoordDF = self.meta_data.CoordDF

        x_coords, z_coords = (CoordDF['x'],CoordDF['z'])

        Z, X = np.meshgrid(z_coords,x_coords)

        fluct = ProcessDF[PhyTime,comp][:,axis_index,:]
                              
        surf_kw = cplt.update_mesh_kw(surf_kw,rcount=100,ccount=150)

        single_input=False
        if not isinstance(ax,np.ndarray):
            single_input=True

        if x_split_list is None:
            x_split_list = [np.amin(x_coords),np.amax(x_coords)]

        x_index_list = indexing.coord_index_calc(CoordDF,'x',x_split_list)

        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'},antialiased=True)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(x_index_list)-1,len(y_vals),fig=fig,ax=ax,**kwargs)
        ax = ax.flatten()
        
        max_val = -np.float('inf'); min_val = np.float('inf')
        for j,_ in enumerate(x_index_list[:-1]):
            for i,_ in enumerate(y_vals):
                max_val = np.amax(fluct[:,i,:]); min_val=np.amin(fluct[:,i,:])
                
                z_vals = Z[x_index_list[j]:x_index_list[j+1]]
                x_vals = X[x_index_list[j]:x_index_list[j+1]]
                surf_vals = fluct[:,i,x_index_list[j]:x_index_list[j+1]].T

                ax[j*len(x_index_list[:-1])+i] = ax[j*len(x_index_list[:-1])+i].plot_surface( z_vals,x_vals,surf_vals,**surf_kw)

                ax[j*len(x_index_list[:-1])+i].axes.set_ylabel(r'$x/\delta$')
                ax[j*len(x_index_list[:-1])+i].axes.set_xlabel(r'$z/\delta$')
                ax[j*len(x_index_list[:-1])+i].axes.set_zlabel(r'$%s^\prime$'%comp)
                ax[j*len(x_index_list[:-1])+i].axes.invert_xaxis()
                # ax[j*len(x_split_list[:-1])+i].axes.set_ylim([x_split_list[j],x_split_list[j+1]])

        for a in ax.flatten():
            a.set_clim([min_val,max_val])
            a.axes.set_zlim([min_val,max_val])
        if single_input:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_vector(self,ProcessDF,avg_data,slice,ax_val,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        self._check_attr()
        PhyTime = self._check_outer_index(ProcessDF,PhyTime)   
        
        ax_val = misc_utils.check_list_vals(ax_val)

        slice, coord, axis_index = indexing.contour_plane(slice,ax_val,avg_data,y_mode,PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[8,4*len(ax_val)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(ax_val),fig=fig,ax=ax,**kwargs)

        ax = ax.flatten()


        coord_1 = self.CoordDF[slice[0]][::spacing[0]]
        coord_2 = self.CoordDF[slice[1]][::spacing[1]]
        UV_slice = [chr(ord(x)-ord('x')+ord('u')) for x in slice]
        U = ProcessDF[PhyTime,UV_slice[0]]
        V = ProcessDF[PhyTime,UV_slice[1]]

        U_space, V_space = indexing.vector_indexer(U,V,axis_index,coord,spacing[0],spacing[1])
        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        ax_out=[]
        quiver_kw = cplt.update_quiver_kw(quiver_kw)
        for i in range(len(ax_val)):
            scale = np.amax(U_space[:,:,i])*coord_1.size/np.amax(coord_1)/scaling
            extent_array = (np.amin(coord_1),np.amax(coord_1),np.amin(coord_2),np.amax(coord_1))

            Q = ax[i].quiver(coord_1_mesh, coord_2_mesh,U_space[:,:,i].T,V_space[:,:,i].T,angles='uv',scale_units='xy', scale=scale,**quiver_kw)
            ax[i].set_xlabel(r"$%s/\delta$"%slice[0])
            ax[i].set_ylabel(r"$%s/\delta$"%slice[1])
            ax[i].set_title(r"$%s = %.3g$"%(coord,ax_val[i]),loc='right')
            ax[i].set_title(r"$t^*=%s$"%PhyTime,loc='left')
            ax_out.append(Q)
        if len(ax_out) ==1:
            return fig, ax_out[0]
        else:
            return fig, np.array(ax_out)