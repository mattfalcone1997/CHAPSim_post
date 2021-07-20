"""
# _joint_pdf.py
Module for calculating joint probability distributions from
instantaneous results from CHAPSim DNS solver
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import seaborn

import sys
import warnings
import gc
import time
from abc import ABC, abstractmethod

import CHAPSim_post as cp
from CHAPSim_post.utils import docstring, gradient, indexing, misc_utils

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd

from ._average import CHAPSim_AVG_io, CHAPSim_AVG_tg_base
_avg_io_class = CHAPSim_AVG_io
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._fluct import CHAPSim_fluct_tg, CHAPSim_fluct_io
_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg

from ._meta import CHAPSim_meta
_meta_class = CHAPSim_meta

from ._common import Common

class CHAPSim_joint_PDF_base(Common,ABC):

    def __init__(self,*args,fromfile=False,**kwargs):
        if fromfile:
            self._hdf_extract(*args,**kwargs)
        else:
            self._extract_fluct(*args,**kwargs)

    @property
    def shape(self):
        pdf_index = self.pdf_arrayDF.index[0]
        return self.pdf_arrayDF[pdf_index].shape

    @abstractmethod
    def _hdf_extract(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _extract_fluct(self,*args,**kwargs):
        raise NotImplementedError


    def save_hdf(self,file_name,mode='a',key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        hdf_obj.attrs["y_mode"] = self._y_mode.encode('utf-8')
        hdf_obj.create_dataset("x_loc_norm",data=self._x_loc_norm)

        self.pdf_arrayDF.to_hdf(file_name,key=key+'/pdf_arrayDF',mode='a')#,format='fixed',data_columns=True)
        self.u_arrayDF.to_hdf(file_name,key=key+'/u_arrayDF',mode='a')#,format='fixed',data_columns=True)
        self.v_arrayDF.to_hdf(file_name,key=key+'/v_arrayDF',mode='a')#,format='fixed',data_columns=True)     
        
        self._meta_data.save_hdf(file_name,'a',key+"/meta_data")
        self.avg_data.save_hdf(file_name,'a',key+"/avg_data")

    @classmethod
    def from_hdf(cls,file_name,key=None):
        return cls(file_name,key=key,fromfile=True)

    def plot_joint_PDF(self, xy_list,contour_kw=None,fig=None, ax=None,**kwargs):
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[8,4*len(xy_list)])
        fig, ax = cplt.create_fig_ax_without_squeeze(len(xy_list),fig=fig,ax=ax,**kwargs)   
        ax = ax.flatten()

        contour_kw = cplt.update_contour_kw(contour_kw)
        i=0
        y_unit = r"y/\delta" if self._y_mode=='half_channel' \
                else r"\delta_u" if self._y_mode=='disp_thickness' \
                else r"\theta" if self._y_mode=='mom_thickness' else r"y^+" \
                if self._y_mode=='wall' and any(self._x_loc_norm!=0) else r"y^{+0}"
        
        for xy in xy_list:
            u_array = self.u_arrayDF[xy]
            v_array = self.v_arrayDF[xy]
            pdf_array = self.pdf_arrayDF[xy]
            U_mesh,V_mesh = np.meshgrid(u_array,v_array) 
            C = ax[i].contour(U_mesh,V_mesh,pdf_array,**contour_kw)#seaborn.kdeplot(x=x_vals,y=y_vals,ax=ax[i],**pdf_kwargs)
            ax[i].set_xlabel(r"$u'$")
            ax[i].set_ylabel(r"$v'$")
            ax[i].set_title(r"$x/\delta=%g$, $%s=%g$"%(xy[0],y_unit,xy[1]),loc='right')
            i+=1
        fig.tight_layout()
        return fig, ax


class CHAPSim_joint_PDF_io(CHAPSim_joint_PDF_base):

    def _extract_fluct(self,x,y,path_to_folder=None,time0=None,gridsize=200,y_mode='half-channel',use_ini=True,xy_inner=True,tgpost=False,abs_path=True):
        
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if cp.rcParams['TEST']:
            times.sort(); times= times[-5:]
        self._meta_data = self._module._meta_class(path_to_folder,abs_path)

        try:
            self.avg_data = self._module._avg_io_class(max(times),self._meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            self.avg_data = self._module._avg_io_class(max(times),self._meta_data,path_to_folder,time0)

        
        if xy_inner:
            if len(x) != len(y):
                msg = "length of x coordinate array must be same"+\
                        " as the y coord array. Lengths provided %d (x),"%len(x)+\
                            " %d (y)"%len(y)
                raise ValueError(msg)
            
            x_coord_list = x; y_coord_list = y
        else:
            x_coord_list = []; y_coord_list=[]
            for x_val in x:
                for y_val in y:
                    x_coord_list.append(x_val)
                    y_coord_list.append(y_val)

        x_index = indexing.coord_index_calc(self.avg_data.CoordDF,'x',x_coord_list)

        self._x_loc_norm = x_coord_list if not use_ini else [0]*len(y_coord_list)
        y_index = indexing.y_coord_index_norm(self.avg_data,y_coord_list,self._x_loc_norm,y_mode)
        
        y_index = np.diag(np.array(y_index))
        u_prime_array = [ [] for _ in range(len(y_index)) ]
        v_prime_array = [ [] for _ in range(len(y_index)) ]

        for time in times:
            fluct_data = self._module._fluct_io_class(time,self.avg_data,path_to_folder,abs_path)
            u_prime_data = fluct_data.fluctDF[time,'u']
            v_prime_data = fluct_data.fluctDF[time,'v']
            for i in range(len(y_index)):
                u_prime_array[i].extend(u_prime_data[:,y_index[i],x_index[i]])
                v_prime_array[i].extend(v_prime_data[:,y_index[i],x_index[i]])
                if cp.rcParams['SymmetryAVG'] and self.metaDF['iCase'] ==1:
                    y_size = u_prime_data.shape[1]
                    u_prime_array[i].extend(u_prime_data[:,-1-y_index[i],x_index[i]])
                    v_prime_array[i].extend(-1*v_prime_data[:,-1-y_index[i],x_index[i]])
            # del fluct_data#; del u_prime_data; del v_prime_data
            gc.collect()

        pdf_array = [ [] for _ in range(len(y_index)) ]
        u_array = [ [] for _ in range(len(y_index)) ]
        v_array = [ [] for _ in range(len(y_index)) ]
        estimator = seaborn._statistics.KDE(gridsize=gridsize)
        for i, y in enumerate(y_index):
            pdf_array[i],(u_array[i],v_array[i]) = estimator(np.array(u_prime_array[i]),
                                                            np.array(v_prime_array[i]))

            # ax = seaborn.kdeplot(u_prime_array[i],v_prime_array[i],gridsize=gridsize)
            # for artist in ax.get_children():
            #     if isinstance(artist,mpl.contour.QuadContourSet):


        index = list(zip(x_coord_list,y_coord_list))

        pdf_array = np.array(pdf_array)
        u_array = np.array(u_array)
        v_array = np.array(v_array)

        self._y_mode=y_mode
        self.pdf_arrayDF = cd.datastruct(pdf_array,index= index)
        self.u_arrayDF = cd.datastruct(u_array,index= index)
        self.v_arrayDF = cd.datastruct(v_array,index= index)

    def _hdf_extract(self, file_name, key=None):
        if key is None:
            key = 'CHAPSim_joint_PDF_io' 

        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._y_mode = hdf_obj.attrs['y_mode'].decode('utf-8')
        self._x_loc_norm = hdf_obj['x_loc_norm'][:]
        
        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')
        self.avg_data = self._module.CHAPSim_AVG_io.from_hdf(file_name,key+'/avg_data')

        self.pdf_arrayDF = cd.datastruct.from_hdf(file_name,key=key+'/pdf_arrayDF')
        self.u_arrayDF = cd.datastruct.from_hdf(file_name,key=key+'/u_arrayDF')
        self.v_arrayDF = cd.datastruct.from_hdf(file_name,key=key+'/v_arrayDF')
