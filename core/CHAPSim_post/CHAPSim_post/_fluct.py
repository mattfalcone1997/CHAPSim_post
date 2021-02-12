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

import CHAPSim_post.CHAPSim_plot as cplt
import CHAPSim_post.CHAPSim_Tools as CT
import CHAPSim_post.CHAPSim_dtypes as cd


import CHAPSim_post as cp
# import CHAPSim_post.CHAPSim_post._utils as utils

from ._meta import CHAPSim_meta
_meta_class=CHAPSim_meta

from ._average import CHAPSim_AVG,CHAPSim_AVG_io,CHAPSim_AVG_tg_base
from ._common import common3D

_avg_io_class = CHAPSim_AVG_io
_avg_class = CHAPSim_AVG
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._instant import CHAPSim_Inst
_instant_class = CHAPSim_Inst

class CHAPSim_fluct_base(common3D):
    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key= 'CHAPSim_fluct'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(key)
        group.create_dataset("NCL",data=np.array(self.shape))
        hdf_file.close()
        self.meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self.fluctDF.to_hdf(file_name,key=key+'/fluctDF',mode='a')#,format='fixed',data_columns=True)

    def _check_outer(self,ProcessDF,PhyTime):
        warn_msg = f"PhyTime invalid ({PhyTime}), varaible being set to only PhyTime present in datastruct"
        err_msg = "PhyTime provided is not in the CHAPSim_AVG datastruct, recovery impossible"

        return super()._check_outer(self.fluctDF,PhyTime,err_msg,warn_msg)
    
    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime=None,x_split_list=None,y_mode='wall',fig=None,ax=None,pcolor_kw=None,**kwargs):
        
        return super().plot_contour(self.fluctDF,self.avg_data,
                                    comp,axis_vals,plane=plane,PhyTime=PhyTime,
                                    x_split_list=x_split_list,y_mode=y_mode,fig=fig,ax=ax,
                                    pcolor_kw=pcolor_kw,**kwargs)
            
    def plot_streaks(self,comp,vals_list,x_split_list=None,PhyTime=None,ylim='',Y_plus=True,*args,colors='',fig=None,ax=None,**kwargs):
        return super().plot_streaks(self.fluctDF,self.avg_data,
                                    comp,vals_list,x_split_list=x_split_list,
                                    PhyTime=PhyTime,ylim=ylim,Y_plus=Y_plus,
                                    *args,colors=colors,fig=fig,ax=ax,
                                    **kwargs)
        
    def plot_fluct3D_xz(self,comp,y_vals,y_mode='half-channel',PhyTime=None,x_split_list=None,fig=None,ax=None,surf_kw=None,**kwargs):
        
        return super().plot3D_xz(self.fluctDF,self.avg_data,
                                comp,y_vals,y_mode=y_mode,
                                PhyTime=PhyTime,x_split_list=x_split_list,
                                fig=fig, ax=ax,surf_kw=surf_kw,**kwargs)
        

    def plot_vector(self,slice,ax_val,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        return super().plot_vector(self.fluctDF,self.avg_data,
                                    slice,ax_val,PhyTime=PhyTime,y_mode=y_mode,
                                    spacing=spacing,scaling=scaling,x_split_list=x_split_list,
                                    fig=fig,ax=ax,quiver_kw=quiver_kw,**kwargs)
       
    @classmethod
    def create_video(cls,axis_vals,comp,avg_data=None,contour=True,plane='xz',meta_data=None,path_to_folder='.',time_range=None,
                            abs_path=True,tgpost=False,x_split_list=None,plot_kw=None,lim_min=None,lim_max=None,
                            ax_func=None,fluct_func=None,fluct_args=(),fluct_kw={}, fig=None,ax=None,**kwargs):

        times= CT.time_extract(path_to_folder,abs_path)
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

        super().__init__(self.meta_data)

        
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

        super().__init__(self.meta_data)
    
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
