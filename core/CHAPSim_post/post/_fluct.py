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

from CHAPSim_post.utils import docstring, gradient, indexing, misc_utils

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd


import CHAPSim_post as cp
# import CHAPSim_post.CHAPSim_post._utils as utils

from ._meta import CHAPSim_meta
_meta_class=CHAPSim_meta

from ._average import CHAPSim_AVG,CHAPSim_AVG_io,CHAPSim_AVG_tg_base
from ._common import Common

_avg_io_class = CHAPSim_AVG_io
_avg_class = CHAPSim_AVG
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._instant import CHAPSim_Inst
_instant_class = CHAPSim_Inst

class CHAPSim_fluct_base(Common):

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key= self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        self._meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self.fluctDF.to_hdf(file_name,key=key+'/fluctDF',mode='a')

    @property
    def shape(self):
        fluct_index = self.fluctDF.index[0]
        return self.fluctDF[fluct_index].shape
    
    def check_PhyTime(self,PhyTime):
        warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        err_msg = f"PhyTime provided ({PhyTime}) is not in the {self.__class__.__name__} datastruct, recovery impossible"
        
        err = ValueError(err_msg)
        warn = UserWarning(warn_msg)
        return self.fluctDF.check_times(PhyTime,err_msg,warn_msg)
    
    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime=None,x_split_list=None,y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        plane = self.Domain.out_to_in(plane)
        axis_vals = misc_utils.check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.fluctDF.CoordDF.check_plane(plane)

        if coord == 'y':
            axis_vals = indexing.ycoords_from_coords(self.avg_data,axis_vals,mode=y_mode)[0]
            int_vals = indexing.ycoords_from_norm_coords(self.avg_data,axis_vals,mode=y_mode)[0]
        else:
            axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)
            int_vals = axis_vals.copy()

        x_size, z_size = self.fluctDF.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

        for i,val in enumerate(int_vals):
            fig, ax1 = self.fluctDF.plot_contour(comp,plane,val,time=PhyTime,fig=fig,ax=ax[i],pcolor_kw=pcolor_kw)
            ax1.axes.set_xlabel(r"$%s/\delta$" % plane[0])
            ax1.axes.set_ylabel(r"$%s/\delta$" % plane[1])
            ax1.axes.set_title(r"$%s=%.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax1.axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')
            
            cbar=fig.colorbar(ax1,ax=ax[i])
            cbar.set_label(r"$%s^\prime$"%comp)

            ax[i]=ax1
            ax[i].axes.set_aspect('equal')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
            
    def plot_streaks(self,comp,vals_list,x_split_pair=None,PhyTime=None,y_limit=None,y_mode='wall',colors=None,surf_kw=None,fig=None,ax=None,**kwargs):
        
        vals_list = misc_utils.check_list_vals(vals_list)
        PhyTime = self.check_PhyTime(PhyTime)
        
        if y_limit is not None:
            y_lim_int = indexing.ycoords_from_norm_coords(self.avg_data,[y_limit],mode=y_mode)[0][0]
        else:
            y_lim_int = None

        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'})
        fig, ax = cplt.create_fig_ax_with_squeeze(fig=fig,ax=ax,**kwargs)

        
        for i,val in enumerate(vals_list):
            if colors is not None:
                color = colors[i%len(colors)]
                surf_kw['facecolor'] = color
            fig, ax1 = self.fluctDF.plot_isosurface(comp,val,time=PhyTime,y_limit=y_lim_int,
                                            x_split_pair=x_split_pair,fig=fig,ax=ax,
                                            surf_kw=surf_kw)
            ax.axes.set_ylabel(r'$x/\delta$')
            ax.axes.set_xlabel(r'$z/\delta$')
            ax.axes.invert_xaxis()

        return fig, ax1

        
    def plot_fluct3D_xz(self,comp,y_vals,y_mode='half-channel',PhyTime=None,x_split_pair=None,fig=None,ax=None,surf_kw=None,**kwargs):
        
        y_vals = misc_utils.check_list_vals(y_vals)
        PhyTime = self.check_PhyTime(PhyTime)
        y_int_vals  = indexing.ycoords_from_norm_coords(self.avg_data,y_vals,mode=y_mode)[0]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False
        kwargs = cplt.update_subplots_kw(kwargs,subplot_kw={'projection':'3d'},antialiased=True)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(y_int_vals),fig=fig,ax=ax,**kwargs)


        for i, val in enumerate(y_int_vals):
            fig, ax[i] = self.fluctDF.plot_surf(comp,'xz',val,time=PhyTime,x_split_pair=x_split_pair,fig=fig,ax=ax[i],surf_kw=surf_kw)
            ax[i].axes.set_ylabel(r'$x/\delta$')
            ax[i].axes.set_xlabel(r'$z/\delta$')
            ax[i].axes.set_zlabel(r'$%s^\prime$'%comp)
            ax[i].axes.invert_xaxis()

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
        

    def plot_vector(self,plane,axis_vals,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        plane = self.Domain.out_to_in(plane)
        axis_vals = misc_utils.check_list_vals(axis_vals)
        PhyTime = self.check_PhyTime(PhyTime)

        plane, coord = self.fluctDF.CoordDF.check_plane(plane)

        if coord == 'y':
            axis_vals = indexing.ycoords_from_coords(self.avg_data,axis_vals,mode=y_mode)[0]
            int_vals = indexing.ycoords_from_norm_coords(self.avg_data,axis_vals,mode=y_mode)[0]
        else:
            int_vals = axis_vals = indexing.true_coords_from_coords(self.CoordDF,coord,axis_vals)

        x_size, z_size = self.fluctDF.get_unit_figsize(plane)
        figsize=[x_size,z_size*len(axis_vals)]

        axes_output = True if isinstance(ax,mpl.axes.Axes) else False

        kwargs = cplt.update_subplots_kw(kwargs,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(len(axis_vals),fig=fig,ax=ax,**kwargs)

        title_symbol = misc_utils.get_title_symbol(coord,y_mode,False)

        for i, val in enumerate(int_vals):
            fig, ax[i] = self.fluctDF.plot_vector(plane,val,time=PhyTime,spacing=spacing,scaling=scaling,
                                                    fig=fig,ax=ax[i],quiver_kw=quiver_kw)
            ax[i].axes.set_xlabel(r"$%s/\delta$"%plane[0])
            ax[i].axes.set_ylabel(r"$%s/\delta$"%plane[1])
            ax[i].axes.set_title(r"$%s = %.2g$"%(title_symbol,axis_vals[i]),loc='right')
            ax[i].axes.set_title(r"$t^*=%s$"%PhyTime,loc='left')

        if axes_output:
            return fig, ax[0]
        else:
            return fig, ax
       
    @classmethod
    def create_video(cls,axis_vals,comp,avg_data=None,contour=True,plane='xz',meta_data=None,path_to_folder='.',time_range=None,
                            abs_path=True,tgpost=False,x_split_list=None,plot_kw=None,lim_min=None,lim_max=None,
                            ax_func=None,fluct_func=None,fluct_args=(),fluct_kw={}, fig=None,ax=None,**kwargs):

        times= misc_utils.time_extract(path_to_folder,abs_path)
        max_time = np.amax(times) if time_range is None else time_range[1]

        if avg_data is None and not tgpost:
            time0 = time_range[0] if time_range is not None else None
            avg_data=cls._module._avg_class(max_time,meta_data,path_to_folder,time0,abs_path,tgpost=cls.tgpost)
        

        axis_vals = misc_utils.check_list_vals(axis_vals)        
        
        if x_split_list is None:
            if meta_data is None:
                meta_data = cls._module._meta_class(path_to_folder,abs_path,tgpost=tgpost)
            x_coords = meta_data.CoordDF['x']
            x_split_list=[np.min(x_coords),np.max(x_coords)]

        if fig is None:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7*len(axis_vals),3*(len(x_split_list)-1)]
            fig = cplt.figure(**kwargs)
        if contour:
            plot_kw = cplt.update_pcolor_kw(plot_kw)
        def func(fig,time):
            axes = fig.axes
            for ax in axes:
                ax.remove()

            fluct_data = cls(time,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            
            if contour:
                fig, ax = fluct_data.plot_contour(comp,axis_vals,plane=plane,PhyTime=time,x_split_list=x_split_list,fig=fig,pcolor_kw=plot_kw)
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
                                        abs_path,func,time_range=time_range)
        
    def __str__(self):
        return self.fluctDF.__str__()

class CHAPSim_fluct_io(CHAPSim_fluct_base):
    tgpost = False
    def __init__(self,time_inst_data_list,avg_data=None,path_to_folder='.',abs_path=True,*args,**kwargs):
        
        if avg_data is None:
            time = misc_utils.max_time_calc(path_to_folder,abs_path)
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

        self.avg_data = avg_data
        self._meta_data = avg_data._meta_data

        self.fluctDF = self._fluctDF_calc(inst_data,avg_data)


    def _fluctDF_calc(self, inst_data, avg_data):
        
        avg_time = list(set([x[0] for x in avg_data.flow_AVGDF.index]))
        
        assert len(avg_time) == 1, "In this context therecan only be one time in avg_data"
        fluct = np.zeros((len(inst_data.InstDF.index),*inst_data.shape[:]),dtype=cp.rcParams['dtype'])
        j=0
        
        for j, (time, comp) in enumerate(inst_data.InstDF.index):
            avg_values = avg_data.flow_AVGDF[avg_time[0],comp]
            inst_values = inst_data.InstDF[time,comp]

            for i in range(inst_data.shape[0]):
                fluct[j,i] = inst_values[i] -avg_values
            del inst_values

        return cd.flowstruct3D(self._coorddata,fluct,index=inst_data.InstDF.index.copy())
    
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
            times = list(set([float(x[0]) for x in inst_data.InstDF.index]))
            avg_data = self._module._avg_tg_base_class(times,path_to_folder=path_to_folder,abs_path=abs_path)
        
        self.avg_data = avg_data
        self._meta_data = avg_data._meta_data

        self.fluctDF = self._fluctDF_calc(inst_data,avg_data)

    
    def _fluctDF_calc(self, inst_data, avg_data):
        avg_times = avg_data.times
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
        return cd.flowstruct3D(self._coorddata,fluct,index=inst_data.InstDF.index)#.data(inst_data.shape)
    
    # def plot_vector(self,*args,**kwargs):
    #     PhyTime = None
    #     return super().plot_vector(*args,PhyTime=PhyTime,**kwargs)

    @classmethod
    def create_video(cls,*args,**kwargs):
        kwargs["tgpost"] = True
        return super().create_video(*args,**kwargs)
