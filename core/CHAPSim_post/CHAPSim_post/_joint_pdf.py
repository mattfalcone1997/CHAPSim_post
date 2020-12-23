"""
# _joint_pdf.py
Module for calculating joint probability distributions from
instantaneous results from CHAPSim DNS solver
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import numba
import seaborn

import sys
import warnings
import gc
import time

import CHAPSim_post as cp
from .. import CHAPSim_plot as cplt
from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd

from ._average import CHAPSim_AVG_io, CHAPSim_AVG_tg_base
_avg_io_class = CHAPSim_AVG_io
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._fluct import CHAPSim_fluct_tg, CHAPSim_fluct_io
_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg

from ._meta import CHAPSim_meta
_meta_class = CHAPSim_meta


class CHAPSim_joint_PDF_base():
    _module = sys.modules[__name__]
    def __init__(self,*args,fromfile=False,**kwargs):
        if fromfile:
            self.pdf_arrayDF, self.u_arrayDF, self.v_arrayDF, self.avg_data,\
            self.meta_data, self.NCL, self._y_mode, self._x_loc_norm = self._hdf_extract(*args,**kwargs)
        else:
            self.pdf_arrayDF, self.u_arrayDF, self.v_arrayDF, self.avg_data,\
            self.meta_data, self.NCL, self._y_mode,self._x_loc_norm = self._extract_fluct(*args,**kwargs)

    def save_hdf(self,file_name,mode='a',base_name='CHAPSim_joint_PDF_base'):
        hdf_file = h5py.File(file_name,mode)
        group = hdf_file.create_group(base_name)
        group.attrs["y_mode"] = self._y_mode.encode('utf-8')
        group.create_dataset("x_loc_norm",data=self._x_loc_norm)
        hdf_file.close()
        self.pdf_arrayDF.to_hdf(file_name,key=base_name+'/pdf_arrayDF',mode='a')#,format='fixed',data_columns=True)
        self.u_arrayDF.to_hdf(file_name,key=base_name+'/u_arrayDF',mode='a')#,format='fixed',data_columns=True)
        self.v_arrayDF.to_hdf(file_name,key=base_name+'/v_arrayDF',mode='a')#,format='fixed',data_columns=True)     
        
        self.meta_data.save_hdf(file_name,'a',base_name+"/meta_data")
        self.avg_data.save_hdf(file_name,'a',base_name+"/avg_data")

    @classmethod
    def from_hdf(cls,file_name,base_name=None):
        return cls(file_name,base_name=base_name,fromfile=True)

    def plot_joint_PDF(self, xy_list,contour_kw=None,fig=None, ax=None,**kwargs):
        
        if fig is None:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [8,4*len(xy_list)]
            else:
                warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(xy_list),**kwargs)
        elif ax is None:
            ax=fig.subplots(len(xy_list),squeeze=False)      
        ax = ax.flatten()

        if contour_kw is not None:
            if not isinstance(contour_kw,dict):
                raise TypeError("pdf_kwargs must be of type dict")
        else:
            contour_kw={}
        i=0
        y_unit = "y/\delta" if self._y_mode=='half_channel' \
                else "\delta_u" if self._y_mode=='disp_thickness' \
                else "\theta" if self._y_mode=='mom_thickness' else "y^+" \
                if self._y_mode=='wall' and any(self._x_loc_norm!=0) else "y^{+0}"
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
    _module = sys.modules[__name__]

    def _extract_fluct(self,x,y,path_to_folder=None,time0=None,gridsize=200,y_mode='half-channel',use_ini=True,xy_inner=True,tgpost=False,abs_path=True):
        
        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if cp.Params['TEST']:
            times.sort(); times= times[-5:]
        meta_data = self._module._meta_class(path_to_folder,abs_path)
        NCL = meta_data.NCL

        try:
            avg_data = self._module._avg_io_class(max(times),meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            avg_data = self._module._avg_io_class(max(times),meta_data,path_to_folder,time0)

        
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

        x_index = CT.coord_index_calc(avg_data.CoordDF,'x',x_coord_list)

        x_loc_norm = x_coord_list if not use_ini else [0]*len(y_coord_list)
        y_index = CT.y_coord_index_norm(avg_data,avg_data.CoordDF,y_coord_list,x_loc_norm,y_mode)
        
        y_index = np.diag(np.array(y_index))
        u_prime_array = [ [] for _ in range(len(y_index)) ]
        v_prime_array = [ [] for _ in range(len(y_index)) ]

        for time in times:
            fluct_data = self._module._fluct_io_class(time,avg_data,path_to_folder,abs_path)
            u_prime_data = fluct_data.fluctDF[time,'u']
            v_prime_data = fluct_data.fluctDF[time,'v']
            for i in range(len(y_index)):
                u_prime_array[i].extend(u_prime_data[:,y_index[i],x_index[i]])
                v_prime_array[i].extend(v_prime_data[:,y_index[i],x_index[i]])
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

        pdf_arrayDF = cd.datastruct(pdf_array,index= index)
        u_arrayDF = cd.datastruct(u_array,index= index)
        v_arrayDF = cd.datastruct(v_array,index= index)
        return pdf_arrayDF, u_arrayDF, v_arrayDF, avg_data,meta_data,NCL,y_mode,x_loc_norm

    def save_hdf(self,file_name,mode='a',base_name='CHAPSim_joint_PDF_io'):
        super().save_hdf(file_name,mode,base_name)

    def _hdf_extract(self, file_name, base_name):
        if base_name is None:
            base_name = 'CHAPSim_joint_PDF_io' 

        hdf_file = h5py.File(file_name,'r')
        y_mode = hdf_file[base_name].attrs['y_mode'].decode('utf-8')
        x_loc_norm = hdf_file[base_name+'/x_loc_norm'][:]
        hdf_file.close()
        meta_data = self._module.CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL=meta_data.NCL
        avg_data = self._module.CHAPSim_AVG_io.from_hdf(file_name,base_name+'/avg_data')
        # uv_primeDF = pd.read_hdf(file_name,key=base_name+'/uv_primeDF')
        pdf_arrayDF = cd.datastruct.from_hdf(file_name,key=base_name+'/pdf_arrayDF')
        u_arrayDF = cd.datastruct.from_hdf(file_name,key=base_name+'/u_arrayDF')
        v_arrayDF = cd.datastruct.from_hdf(file_name,key=base_name+'/v_arrayDF')

        return pdf_arrayDF,u_arrayDF,v_arrayDF,avg_data, meta_data, NCL, y_mode,x_loc_norm