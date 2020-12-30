"""
# _fluct.py
File contains the implementation of the classes to visualise the 
fluctuations of the velocity and pressure fields. Also is used 
to process the additional statistics including autocovarianace
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from scipy import integrate

import sys
import os
import warnings
import gc
import itertools
from abc import ABC, abstractmethod

from .. import CHAPSim_plot as cplt
from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd


import CHAPSim_post as cp
import CHAPSim_post.CHAPSim_post._utils as utils

from ._meta import CHAPSim_meta
_meta_class=CHAPSim_meta

from ._average import CHAPSim_AVG,CHAPSim_AVG_io,CHAPSim_AVG_tg_base
_avg_io_class = CHAPSim_AVG_io
_avg_class = CHAPSim_AVG
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._instant import CHAPSim_Inst
_instant_class = CHAPSim_Inst

class CHAPSim_fluct_base(ABC):
    _module = sys.modules[__name__]
    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key= 'CHAPSim_fluct'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(key)
        group.create_dataset("NCL",data=np.array(self.shape))
        hdf_file.close()
        self.meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self.fluctDF.to_hdf(file_name,key=key+'/fluctDF',mode='a')#,format='fixed',data_columns=True)

    def check_PhyTime(self,PhyTime):
        warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        err_msg = "PhyTime provided is not in the CHAPSim_AVG datastruct, recovery impossible"
        with warnings.catch_warnings(record=True) as w:
            key = self.fluctDF.check_index(PhyTime,err_msg=err_msg,warn_msg=warn_msg,outer=True)
            a = w
        if PhyTime is not None and len(a)>0:
            for warn in a:
                warnings.warn(a.message)
        return key[0]
    
    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime=None,x_split_list=None,y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
                
        PhyTime = self.check_PhyTime(PhyTime)        
        
        axis_vals = utils.check_list_vals(axis_vals)
            
        plane , coord, axis_index = CT.contour_plane(plane,axis_vals,self.avg_data,y_mode,PhyTime)


        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF[PhyTime,comp]
                
        if x_split_list is None:
            x_split_list=[np.amin(x_coords),np.amax(x_coords)]
        
        ax_layout = (len(x_split_list)-1,len(axis_vals))
        figsize=[10*len(axis_vals),3*(len(x_split_list)-1)]
        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(*ax_layout,fig=fig,ax=ax,**kwargs)

        ax=ax.flatten()

        x_coords_split=CT.coord_index_calc(self.CoordDF,'x',x_split_list)

        title_symbol = CT.get_title_symbol(coord,y_mode,False)
        
        pcolor_kw = cplt.update_pcolor_kw(pcolor_kw)
        X, Z = np.meshgrid(x_coords, z_coords)

        ax=ax.flatten()
        max_val = np.amax(fluct); min_val=np.amin(fluct)
        for j,_ in enumerate(x_split_list[:-1]):
            for i,_ in enumerate(axis_vals):
                fluct_slice = CT.contour_indexer(fluct,axis_index[i],coord)

                ax1 = ax[j*len(axis_vals)+i].pcolormesh(X,Z,fluct_slice,**pcolor_kw)
                ax1.set_clim(min_val,max_val)

                ax1.axes.set_xlabel(r"$%s/\delta$" % 'x')
                ax1.axes.set_ylabel(r"$%s/\delta$" % 'z')
                ax1.axes.set_xlim([x_split_list[j],x_split_list[j+1]])
                ax1.axes.set_title(r"$%s=%.3g$"%(title_symbol,axis_vals[i]),loc='right')
                ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                
                cbar=fig.colorbar(ax1,ax=ax[j*len(axis_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp)

                ax[j*len(axis_vals)+i]=ax1
                ax[j*len(axis_vals)+i].axes.set_aspect('equal')
        fig.tight_layout()
        if ax.size == 1:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_streaks(self,comp,vals_list,x_split_list=None,PhyTime=None,ylim='',Y_plus=True,*args,colors='',fig=None,ax=None,**kwargs):

        PhyTime = self.check_PhyTime(PhyTime)
        fluct = self.fluctDF[PhyTime,comp]
        
        if ylim:
            if Y_plus:
                y_index= CT.Y_plus_index_calc(self.avg_data,self.CoordDF,ylim)
            else:
                y_index=CT.coord_index_calc(self.CoordDF,'y',ylim)
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
            x_start = CT.coord_index_calc(self.CoordDF,'x',x_split_list[j])
            x_end = CT.coord_index_calc(self.CoordDF,'x',x_split_list[j+1])
            for val,i in zip(vals_list,range(len(vals_list))):
                
                
                color = colors[i%len(colors)] if colors else ''
                patch = ax[j].plot_isosurface(Y,Z,X[x_start:x_end],fluct[:,:,x_start:x_end],val,color)
                ax[j].add_lighting()
                # patch.set_color(colors[i%len(colors)])

        return fig, ax
    def plot_fluct3D_xz(self,comp,y_vals,PhyTime=None,x_split_list=None,fig=None,ax=None,**kwargs):

        PhyTime = self.check_PhyTime(PhyTime)    

        y_vals = utils.check_list_vals(y_vals)    
                
        x_coords = self.CoordDF['x']
        z_coords = self.CoordDF['z']
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF[PhyTime,comp][:,y_vals,:]
        
        if x_split_list is None:
            x_split_list=[np.min(x_coords),np.max(x_coords)]
            
        x_coords_split=CT.coord_index_calc(self.CoordDF,'x',x_split_list)
          
        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'})
        
        if fig is None:
            kwargs = cplt.update_subplots_kw(kwargs,override=True,squeeze=False)
            kwargs = cplt.update_subplots_kw(kwargs,figsize=[10*len(y_vals),5*(len(x_split_list)-1)])
            fig, ax = plt.subplots((len(x_split_list)-1),len(y_vals),**kwargs)
        elif ax is None:
            kwargs = cplt.update_subplots_kw(kwargs,override=True,subplot_kw={"squeeze":False})
            ax = fig.subplots((len(x_split_list)-1),len(y_vals),**kwargs)

        ax=ax.flatten()
        max_val = -np.float('inf'); min_val = np.float('inf')
        for j,_ in enumerate(x_split_list[:-1]):
            for i,_ in enumerate(y_vals):
                max_val = np.amax(fluct[:,i,:]); min_val=np.amin(fluct[:,i,:])
                
                surf=ax[j*len(y_vals)+i].plot_surface(Z, X, fluct, rstride=1, cstride=1, cmap='jet',
                                            linewidth=0, antialiased=False)

                ax[j*len(y_vals)+i].set_ylabel(r'$x/\delta$')
                ax[j*len(y_vals)+i].set_xlabel(r'$z/\delta$')
                ax[j*len(y_vals)+i].set_zlabel(r'$%s^\prime$'%comp)
                
                ax[j*len(y_vals)+i].set_xlim(x_split_list[j],x_split_list[j+1])

                surf.set_clim(min_val,max_val)
                cbar=fig.colorbar(surf,ax=ax[j*len(y_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp)

                ax[j*len(y_vals)+i]=surf

        for a in ax:
            a.axes.set_zlim(min_val,max_val)
        fig.tight_layout()
        return fig, ax

    def plot_vector(self,slice,ax_val,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        
        ax_val = utils.check_list_vals(ax_val)

        slice, coord, axis_index = CT.contour_plane(slice,ax_val,self.avg_data,y_mode,PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[8,4*len(ax_val)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(ax_val),fig=fig,ax=ax,**kwargs)

        ax = ax.flatten()


        coord_1 = self.CoordDF[slice[0]][::spacing[0]]
        coord_2 = self.CoordDF[slice[1]][::spacing[1]]
        UV_slice = [chr(ord(x)-ord('x')+ord('u')) for x in slice]
        U = self.fluctDF[PhyTime,UV_slice[0]]
        V = self.fluctDF[PhyTime,UV_slice[1]]

        U_space, V_space = CT.vector_indexer(U,V,axis_index,coord,spacing[0],spacing[1])
        coord_1_mesh, coord_2_mesh = np.meshgrid(coord_1,coord_2)
        ax_out=[]
        quiver_kw = cplt.update_quiver_kw(quiver_kw)
        for i in range(len(ax_val)):
            scale = np.amax(U_space[:,:,i])*coord_1.size/np.amax(coord_1)/scaling
            extent_array = (np.amin(coord_1),np.amax(coord_1),np.amin(coord_2),np.amax(coord_1))

            Q = ax[i].quiver(coord_1_mesh, coord_2_mesh,U_space[:,:,i].T,V_space[:,:,i].T,angles='uv',scale_units='xy', scale=scale,**quiver_kw)
            ax[i].set_xlabel(r"$%s^*$"%slice[0])
            ax[i].set_ylabel(r"$%s$"%slice[1])
            ax[i].set_title(r"$%s = %.3g$"%(coord,ax_val[i]),loc='right')
            ax[i].set_title(r"$t^*=%s$"%PhyTime,loc='left')
            ax_out.append(Q)
        if len(ax_out) ==1:
            return fig, ax_out[0]
        else:
            return fig, np.array(ax_out)


    @classmethod
    def create_video(cls,axis_vals,comp,avg_data=None,contour=True,plane='xz',meta_data=None,path_to_folder='.',time_range=None,
                            abs_path=True,tgpost=False,x_split_list=None,plot_kw=None,lim_min=None,lim_max=None,
                            ax_func=None,fluct_func=None,fluct_args=(),fluct_kw={}, fig=None,ax=None,**kwargs):

        times= CT.time_extract(path_to_folder,abs_path)
        max_time = np.amax(times) if time_range is None else time_range[1]

        if avg_data is None and not tgpost:
            time0 = time_range[0] if time_range is not None else ""
            avg_data=cls._module._avg_class(max_time,meta_data,path_to_folder,time0,abs_path,tgpost=cls.tgpost)
        

        axis_vals = utils.check_list_vals(axis_vals)        
        
        if x_split_list is None:
            if meta_data is None:
                meta_data = cls._module._meta_class(path_to_folder,abs_path,tgpost=tgpost)
            x_coords = meta_data.CoordDF['x']
            x_split_list=[np.min(x_coords),np.max(x_coords)]

        if fig is None:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7*len(axis_vals),3*(len(x_split_list)-1)]
            fig = cplt.figure(**kwargs)

        def func(fig,time):
            axes = fig.axes
            for ax in axes:
                ax.remove()

            fluct_data = cls(time,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            if contour:
                plot_kw = cplt.update_pcolor_kw(plot_kw)
                fig, ax = fluct_data.plot_contour(comp,axis_vals,plane=plane,PhyTime=time,x_split_list=x_split_list,fig=fig,**plot_kw)
            else:
                fig,ax = fluct_data.plot_fluct3D_xz(axis_vals,comp,time,x_split_list,fig,**plot_kw)
            ax[0].axes.set_title(r"$t^*=%.3g$"%time,loc='left')
            if fluct_func is not None:
                fluct_func(fig,ax,time,*fluct_args,**fluct_kw)
            if ax_func is not None:
                ax = ax_func(ax)
            for im in ax:
                im.set_clim(vmin=lim_min)
                im.set_clim(vmax=lim_max)

            fig.tight_layout()
            return ax

        return cplt.create_general_video(fig,path_to_folder,
                                        abs_path,func,(),{})
        
    def __str__(self):
        return self.fluctDF.__str__()

class CHAPSim_fluct_io(CHAPSim_fluct_base):
    tgpost = False
    def __init__(self,time_inst_data_list,avg_data=None,path_to_folder='.',abs_path=True,*args,**kwargs):
        
        if avg_data is None:
            time = CT.max_time_calc(path_to_folder,abs_path)
            avg_data = self._module._avg_io_class(time,path_to_folder=path_to_folder,abs_path=abs_path,*args,**kwargs)
        if not isinstance(time_inst_data_list,(list,tuple)):
            time_inst_data_list = [time_inst_data_list]
        for time_inst_data in time_inst_data_list:
            if isinstance(time_inst_data,self._module._instant_class):
                if 'inst_data' not in locals():
                    inst_data = time_inst_data
                else:
                    inst_data += time_inst_data
            else:
                if 'inst_data' not in locals():
                    inst_data = self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=False)
                else:
                    inst_data += self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=False)

        self.fluctDF = self._fluctDF_calc(inst_data,avg_data)

        self.avg_data = avg_data
        self.meta_data = avg_data._meta_data
        self.NCL = self.meta_data.NCL
        self.CoordDF = self.meta_data.CoordDF
        self.shape = inst_data.shape

        
    def _fluctDF_calc(self, inst_data, avg_data):
        
        fluct = np.zeros((len(inst_data.InstDF.index),*inst_data.shape[:]))
        avg_time = list(set([x[0] for x in avg_data.flow_AVGDF.index]))
        
        assert len(avg_time) == 1, "In this context therecan only be one time in avg_data"
        fluct = np.zeros((len(inst_data.InstDF.index),*inst_data.shape[:]))
        j=0
        
        for j, (time, comp) in enumerate(inst_data.InstDF.index):
            avg_values = avg_data.flow_AVGDF[avg_time[0],comp]
            inst_values = inst_data.InstDF[time,comp]

            for i in range(inst_data.shape[0]):
                fluct[j,i] = inst_values[i] -avg_values
            del inst_values

        return cd.datastruct(fluct,index=inst_data.InstDF.index.copy())
    
class CHAPSim_fluct_tg(CHAPSim_fluct_base):
    tgpost = True
    def __init__(self,time_inst_data_list,avg_data=None,path_to_folder='.',abs_path=True,*args,**kwargs):
        if not hasattr(time_inst_data_list,'__iter__'):
            time_inst_data_list = [time_inst_data_list]
        for time_inst_data in time_inst_data_list:
            if isinstance(time_inst_data,self._module._instant_class):
                if 'inst_data' not in locals():
                    inst_data = time_inst_data
                else:
                    inst_data += time_inst_data
            else:
                if 'inst_data' not in locals():
                    inst_data = self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=True)
                else:
                    inst_data += self._module._instant_class(time_inst_data,path_to_folder=path_to_folder,abs_path=abs_path,tgpost=True)
        
        if avg_data is None:
            times = [float(x[0]) for x in inst_data.InstDF.index]
            avg_data = self._module._avg_tg_base_class(times,path_to_folder=path_to_folder,abs_path=abs_path)

        self.fluctDF = self._fluctDF_calc(inst_data,avg_data)
        self.avg_data = avg_data
        self.meta_data = avg_data._meta_data
        self.NCL = self.meta_data.NCL
        self.CoordDF = self.meta_data.CoordDF
        self.shape = inst_data.shape
    
    def _fluctDF_calc(self, inst_data, avg_data):
        avg_times = avg_data.get_times()
        inst_times = list(set([x[0] for x in inst_data.InstDF.index]))
        u_comp = [x[1] for x in avg_data.flow_AVGDF.index]
        indices = inst_data.InstDF.index
        fluct = np.zeros((len(indices),*inst_data.shape))
        if len(avg_times) == 1:
            j=0
            for time in inst_times:
                avg_index=avg_data._return_index(time)
                for comp in u_comp:
                    avg_values = avg_data.flow_AVGDF[None,comp]
                    inst_values = inst_data.InstDF[time,comp]

                    for i in range(inst_data.shape[0]):
                        for k in range(inst_data.shape[2]):
                            fluct[j,i,:,k] = inst_values[i,:,k] -avg_values[:,avg_index]
                    j+=1
        elif all(time in avg_times for time in inst_times):
            for j,index in enumerate(indices):
                avg_index=avg_data._return_index(index[0])
                avg_values = avg_data.flow_AVGDF[None,index[1]]
                inst_values = inst_data.InstDF[index]
                for i in range(inst_data.shape[0]):
                    for k in range(inst_data.shape[2]):
                        fluct[j,i,:,k] = inst_values[i,:,k] -avg_values[:,avg_index]
        else:
            raise ValueError("avg_data must either be length 1 or the same length as inst_data")
        # DF_shape = (len(indices),np.prod(inst_data.shape))
        return cd.datastruct(fluct,index=inst_data.InstDF.index)#.data(inst_data.shape)
    
    # def plot_vector(self,*args,**kwargs):
    #     PhyTime = None
    #     return super().plot_vector(*args,PhyTime=PhyTime,**kwargs)

    @classmethod
    def create_video(cls,*args,**kwargs):
        kwargs["tgpost"] = True
        return super().create_video(*args,**kwargs)
