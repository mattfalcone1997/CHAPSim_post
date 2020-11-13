"""
# CHAPSim_post_base
This is a module that provides the base functionality for the CHAPSim_post
library
"""
from os.path import abspath
import numpy as np
import pandas as pd; from pandas.errors import PerformanceWarning 
import matplotlib as mpl
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate, fft
import os
import itertools
import time
import warnings
import sys
import seaborn

import CHAPSim_Tools as CT
import CHAPSim_plot as cplt
import CHAPSim_post_v2 as cp
import CHAPSim_dtypes as cd
#import plotly.graph_objects as go

#import loop_accel as la
import numba
import h5py
#on iceberg HPC vtkmodules not found allows it to work as this resource isn't need on iceberg
try: 
    import pyvista as pv
    import pyvistaqt as pvqt
except ImportError:
    warnings.warn("\033[1;33module `pyvista' has missing modules will not work correctly", stacklevel=2)

module = sys.modules[__name__]
module.TEST = False

class CHAPSim_AVG_base():
    module = sys.modules[__name__]
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        copy = kwargs.pop('copy',False)
        if fromfile:
            self._meta_data, self.CoordDF,self._metaDF,\
            self.NCL,self.shape,self._times,self.flow_AVGDF,self.PU_vectorDF,\
            self.UU_tensorDF,self.UUU_tensorDF,\
            self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
            self.DUDX2_tensorDF = self._hdf_extract(*args,**kwargs)
        elif copy:
            self._meta_data, self.CoordDF,self._metaDF,\
            self.NCL,self.shape,self._times,self.flow_AVGDF,self.PU_vectorDF,\
            self.UU_tensorDF,self.UUU_tensorDF,\
            self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
            self.DUDX2_tensorDF = self._copy_extract(*args,**kwargs)
        else:
            self._meta_data, self.CoordDF,self._metaDF,\
            self.NCL,self.shape,self._times,self.flow_AVGDF,self.PU_vectorDF,\
            self.UU_tensorDF,self.UUU_tensorDF,\
            self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
            self.DUDX2_tensorDF = self._extract_avg(*args,**kwargs)

    def _extract_avg(self,*args,**kwargs):
        raise NotImplementedError
    
    @classmethod
    def copy(cls,avg_data):
        return cls(avg_data,copy=True)

    def _copy_extract(self, avg_data):
        try:
            meta_data = self.module.CHAPSim_meta.copy(avg_data._meta_data)
        except AttributeError:
            meta_data = avg_data._meta_data
        CoordDF = avg_data.CoordDF
        metaDF = avg_data._metaDF
        NCL = avg_data.NCL
        shape = avg_data.shape
        times = avg_data._times
        flow_AVGDF = avg_data.flow_AVGDF
        PU_vectorDF = avg_data.PU_vectorDF
        UU_tensorDF = avg_data.UU_tensorDF
        UUU_tensorDF = avg_data.UUU_tensorDF
        Velo_grad_tensorDF = avg_data.Velo_grad_tensorDF
        PR_Velo_grad_tensorDF = avg_data.PR_Velo_grad_tensorDF
        DUDX2_tensorDF = avg_data.DUDX2_tensorDF

        return_list = [meta_data, CoordDF, metaDF, NCL,shape,times,flow_AVGDF,PU_vectorDF,\
                        UU_tensorDF,UUU_tensorDF, Velo_grad_tensorDF, PR_Velo_grad_tensorDF,\
                        DUDX2_tensorDF]
        return itertools.chain(return_list)
    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)
    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_AVG'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(base_name)
        group.create_dataset("NCL",data=self.NCL)
        hdf_file.close()
        self._meta_data.save_hdf(file_name,'a',group_name=base_name+'/meta_data')
        self.flow_AVGDF.to_hdf(file_name,key=base_name+'/flow_AVGDF',mode='a')#,format='fixed',data_columns=True)
        self.PU_vectorDF.to_hdf(file_name,key=base_name+'/PU_vectorDF',mode='a')#,format='fixed',data_columns=True)
        self.UU_tensorDF.to_hdf(file_name,key=base_name+'/UU_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.UUU_tensorDF.to_hdf(file_name,key=base_name+'/UUU_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.Velo_grad_tensorDF.to_hdf(file_name,key=base_name+'/Velo_grad_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.PR_Velo_grad_tensorDF.to_hdf(file_name,key=base_name+'/PR_Velo_grad_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.DUDX2_tensorDF.to_hdf(file_name,key=base_name+'/DUDX2_tensorDF',mode='a')#,format='fixed',data_columns=True)

    def _hdf_extract(self,file_name,shape,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_AVG'
        flow_AVGDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/flow_AVGDF')#pd.read_hdf(file_name,key=base_name+'/flow_AVGDF')
        PU_vectorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/PU_vectorDF')
        UU_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/UU_tensorDF')
        UUU_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/UUU_tensorDF')
        Velo_grad_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/Velo_grad_tensorDF')
        PR_Velo_grad_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/PR_Velo_grad_tensorDF')
        DUDX2_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/DUDX2_tensorDF')

        return_list = [flow_AVGDF,PU_vectorDF,UU_tensorDF,
                        UUU_tensorDF,Velo_grad_tensorDF,
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF]
        return itertools.chain(return_list)
    def _Reverse_decomp(self,flow_AVGDF,PU_vectorDF,UU_tensorDF,
                        UUU_tensorDF,Velo_grad_tensorDF,
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF):

        pu_vector_index = PU_vectorDF.index
        for index in pu_vector_index:
            P_mean = flow_AVGDF[index[0],'P']
            u_mean = flow_AVGDF[index]
            PU_vectorDF[index] -= P_mean*u_mean

        uu_tensor_index = UU_tensorDF.index
        for index in uu_tensor_index:
            u1_mean = flow_AVGDF[index[0],index[1][0]]
            u2_mean = flow_AVGDF[index[0],index[1][1]]
            UU_tensorDF[index] -= u1_mean*u2_mean


        uuu_tensor_index = UUU_tensorDF.index
        for index in uuu_tensor_index:
            u1u2 = UU_tensorDF[index[0],index[1][:2]]
            u2u3 = UU_tensorDF[index[0],index[1][1:]]
            comp13 = index[1][0] + index[1][2]
            u1u3 = UU_tensorDF[index[0],comp13]
            u1_mean = flow_AVGDF[index[0],index[1][0]]
            u2_mean = flow_AVGDF[index[0],index[1][1]]
            u3_mean = flow_AVGDF[index[0],index[1][2]]
            UUU_tensorDF[index] -= (u1_mean*u2_mean*u3_mean + u1_mean*u2u3 \
                        + u2_mean*u1u3 + u3_mean*u1u2)

        PR_velo_grad_index = Velo_grad_tensorDF.index
        for index in PR_velo_grad_index:
            p_mean = flow_AVGDF[index[0],'P']
            u_grad = Velo_grad_tensorDF[index]
            PR_Velo_grad_tensorDF[index] -= p_mean*u_grad

        dudx2_tensor_index = DUDX2_tensorDF.index
        for index in dudx2_tensor_index:
            comp1 = index[1][1] + index[1][3]
            comp2 = index[1][5] + index[1][7]
            u1x_grad = Velo_grad_tensorDF[index[0],comp1]
            u2x_grad = Velo_grad_tensorDF[index[0],comp2]
            DUDX2_tensorDF[index] -= u1x_grad*u2x_grad

        return flow_AVGDF,PU_vectorDF,UU_tensorDF,\
                        UUU_tensorDF,Velo_grad_tensorDF,\
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF

    def _AVG_extract(self, *args,**kwargs):
        raise NotImplementedError

    def _return_index(self,*args,**kwargs):
        raise NotImplementedError
    
    def get_times(self):
        return self._times

    def _wall_unit_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        
        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star

        REN = self._metaDF['REN']
        
        tau_w = self._tau_calc(PhyTime)
    
        u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
        delta_v_star = (nu_star/u_tau_star)/REN
        return u_tau_star, delta_v_star

    def _y_plus_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)

        u_tau_star, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus_shape=(self.shape[1],int(self.NCL[1]*0.5))
        y_plus = np.zeros(y_plus_shape)
        y_coord = self.CoordDF['y'][:int(self.NCL[1]*0.5)]
        for i in range(len(delta_v_star)):
            y_plus[i] = (1-abs(y_coord))/delta_v_star[i]
        return y_plus

    def _int_thickness_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        U0_index = int(np.floor(self.NCL[1]*0.5))
        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']

        U0 = U_mean[U0_index]
        theta_integrand = np.zeros((U0_index,self.shape[1]))
        delta_integrand = np.zeros((U0_index,self.shape[1]))
        mom_thickness = np.zeros(self.shape[1])
        disp_thickness = np.zeros(self.shape[1])
        for i in range(U0_index):
            theta_integrand[i] = (np.divide(U_mean[i],U0))*(1 - np.divide(U_mean[i],U0))
            delta_integrand[i] = (1 - np.divide(U_mean[i],U0))
        for j in range(self.shape[1]):
            mom_thickness[j] = integrate.simps(theta_integrand[:,j],y_coords[:U0_index])
            disp_thickness[j] = integrate.simps(delta_integrand[:,j],y_coords[:U0_index])
        shape_factor = np.divide(disp_thickness,mom_thickness)
        
        return disp_thickness, mom_thickness, shape_factor
    def _plot_shape_factor(self,*arg,fig='',ax='',**kwargs):
        _, _, shape_factor = self._int_thickness_calc(*arg)
        # x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)
        ax.cplot(shape_factor,label=r"$H$")
        # ax.set_xlabel(r"$x/\delta$ ")# ,fontsize=18)
        ax.set_ylabel(r"$H$")# ,fontsize=18)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

    def plot_mom_thickness(self,*arg,fig='',ax='',**kwargs):
        _, theta, _ = self.int_thickness_calc(*arg)
        # x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)
        ax.cplot(theta,label=r"$\theta$")
        # ax.set_xlabel(r"$x/\delta$ ")# ,fontsize=18)
        ax.set_ylabel(r"$\theta$")# ,fontsize=18)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

    def plot_disp_thickness(self,*arg,fig='',ax='',**kwargs):
        delta, _, _ = self.int_thickness_calc(*arg)
        # x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)
        ax.cplot(delta,label=r"$\delta^*$")
        # ax.set_xlabel(r"$x/\delta$ ")# ,fontsize=18)
        ax.set_ylabel(r"$\delta^*$")# ,fontsize=18)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

    def _avg_line_plot(self,x_vals, PhyTime,comp,fig='',ax='',**kwargs):
        # if not isinstance(PhyTime,str) and not np.isnan(PhyTime):
        #     PhyTime = "{:.9g}".format(PhyTime)

        if not fig:
            if "figsize" not in kwargs.keys():
                kwargs['figsize'] = [8,6]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.subplots(**kwargs)
        # y_size = int(self.shape[0]*0.5)
        for x in x_vals:
            try:
                index = self._return_index(x)
            except ValueError:
                warnings.warn("Wrong time input", stacklevel=2)
                continue

            velo_mean=self.flow_AVGDF[PhyTime,comp][:,index]
            ax.cplot(velo_mean)
        return fig, ax        

    def avg_line_plot(self,x_vals, *args,**kwargs):

        fig, ax = self._avg_line_plot(x_vals,*args,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        y_coords = self.CoordDF['y']
        for line in lines:
            line.set_xdata(y_coords)
        ax.set_xlabel(r"$y^*$")
        ax.set_ylabel(r"$\bar{u}$")
        ax.relim()
        ax.autoscale_view()

        return fig, ax
    
    def plot_near_wall(self,x_vals,PhyTime,fig='',ax='',**kwargs):
        fig, ax = self._avg_line_plot(x_vals,PhyTime,'u',fig=fig,ax=ax,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        u_tau_star, _ = self.wall_unit_calc(PhyTime)
        y_plus = self._y_plus_calc(PhyTime)
        Y_extent = int(self.shape[0]/2)

        for line,x_val in zip(lines,x_vals):
            x_loc = self._return_index(x_val)
            y_data = line.get_ydata()
            y_data = y_data[:Y_extent]/u_tau_star[x_loc]
            line.set_ydata(y_data)
            line.set_xdata(y_plus[x_loc])
            ylim = ax.get_ylim()[1]
            ax.set_ylim(top=1.1*max(ylim,np.amax(y_data)))

        ax.cplot(y_plus[x_loc],y_plus[x_loc],color='red',linestyle='--',
                            label=r"$\bar{u}^+=y^+$")
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")
        ax.set_xscale('log')
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_Reynolds(self,comp1,comp2,x_val,PhyTime,norm=None,Y_plus=True,fig='',ax='',**kwargs):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        comp_uu =comp1 + comp2
        if comp1 == 'w' and (comp2=='v' or comp2 =='u'):
            comp_uu = comp_uu[::-1]
        elif comp1 == 'v' and comp2 =='u':
            comp_uu = comp_uu[::-1]     
        
        x_loc = [self._return_index(x) for x in x_val]
        uu = self.UU_tensorDF[PhyTime,comp_uu].copy()
        
        if comp_uu == 'uv':
            uu *= -1.0

        uu = uu[:int(self.NCL[1]/2)]
        y_coord = self.CoordDF['y'][:int(self.NCL[1]/2)]
        #Reynolds_wall_units = np.zeros_like(uu)
        u_tau_star, delta_v_star = self._wall_unit_calc(PhyTime)
        if norm=='wall':
            for i in range(self.shape[1]):
                uu[:,i] = uu[:,i]/(u_tau_star[i]*u_tau_star[i])

        elif norm=='local-bulk':
            velo_bulk=self._bulk_velo_calc(PhyTime)
            for i in range(self.shape[1]):
                uu[:,i] = uu[:,i]/(velo_bulk[i]*velo_bulk[i])

        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)

        if isinstance(x_loc,int):
            if Y_plus:
                y_coord_local = (1-np.abs(y_coord))/delta_v_star[x_loc]
            else:
                y_coord_local = y_coord
            ax.cplot(y_coord_local,uu[:,x_loc])

        elif isinstance(x_loc,list):
            for j,x in enumerate(x_loc):
                if Y_plus:
                    y_coord_local = (1-np.abs(y_coord))/delta_v_star[x]
                else:
                    y_coord_local = y_coord
                label=r"$x/\delta =%.3g$" % self.CoordDF['x'][x]
                ax.cplot(y_coord_local,uu[:,x],label=label)

            # axes_items_num = len(ax.get_lines())
            # ncol = 4 if axes_items_num>3 else axes_items_num
            # ax.clegend(vertical=False,ncol=ncol)
        else:
            raise TypeError("\033[1;32 x_loc must be of type list or int")

        y_label = comp_uu[0] +'\'' + comp_uu[1] +'\''
        if norm=='wall':
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/u_\tau^2$"% y_label)# ,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/u_\tau^2$"% y_label)# ,fontsize=20)
        elif norm=='local-bulk':
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/U_b^2$"% y_label)# ,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/U_b^2$"% y_label)# ,fontsize=20)
        else:
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/U_{b0}^2$"% y_label)# ,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"% y_label)# ,fontsize=20)
        
        if Y_plus:
            ax.set_xlabel(r"$y^+$")# ,fontsize=20)
        else:
            ax.set_xlabel(r"$y/\delta$")# ,fontsize=20)
        #ax.grid()
        
        fig.tight_layout()
        
        return fig, ax

    def plot_Reynolds_x(self,comp1,comp2,y_vals_list,Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        if PhyTime:
            if not isinstance(PhyTime,str) and PhyTime is not None:
                PhyTime = "{:.9g}".format(PhyTime)
        
        # PUT IN IO CLASS
        # if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
        #     avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
        #     if PhyTime and PhyTime != avg_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
        #     PhyTime = avg_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        
        comp_uu =comp1 + comp2
        if comp1 == 'w' and (comp2=='v' or comp2 =='u'):
            comp_uu = comp_uu[::-1]
        elif comp1 == 'v' and comp2 =='u':
            comp_uu = comp_uu[::-1]  
        
        if y_vals_list != 'max':
            if Y_plus:
                y_index = CT.Y_plus_index_calc(self, self.CoordDF, y_vals_list)
            else:
                y_index = CT.coord_index_calc(self.CoordDF,'y',y_vals_list)
            rms_vals = self.UU_tensorDF[PhyTime,comp_uu].copy()[y_index]
            # U1_mean = self.flow_AVGDF.loc[PhyTime,comp1].copy().values.reshape(self.shape)[y_index]
            # U2_mean = self.flow_AVGDF.loc[PhyTime,comp2].copy().values.reshape(self.shape)[y_index]
            # rms_vals = UU-U1_mean*U2_mean
            
        else:
            y_index = [None]
            rms_vals = self.UU_tensorDF[PhyTime,comp_uu].copy()
            # U1_mean = self.flow_AVGDF.loc[PhyTime,comp1].copy().values.reshape(self.shape)
            # U2_mean = self.flow_AVGDF.loc[PhyTime,comp2].copy().values.reshape(self.shape)
            # rms_vals = UU-U1_mean*U2_mean
            rms_vals = np.amax(rms_vals,axis=0)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)

        # x_coords = self.CoordDF['x'].dropna().values            
        y_label = comp_uu[0] +'\'' + comp_uu[1] +'\''
        if y_vals_list != 'max':
        
            for i in range(len(y_index)):
                ax.cplot(rms_vals[i],label=r"$y^+=%.3g$"% y_vals_list[i])
            axes_items_num = len(ax.get_lines())
            ncol = 4 if axes_items_num>3 else axes_items_num
            ax.clegend(vertical=False,ncol=ncol, fontsize=16)
            # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
            ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)$"%y_label)# ,fontsize=20)#)# ,fontsize=22)
            
        else:
            ax.cplot(rms_vals,label=r"$(\langle %s\rangle/U_{b0}^2)_{max}$"%y_label)
            # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
            ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"%y_label)# ,fontsize=20)#)# ,fontsize=22)
        
        # ax.set_xlabel(r"$x/\delta$",text_kwargs)
        # ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)_{max}$"%comp_uu,text_kwargs)#)# ,fontsize=22)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

        # def _avg_line_plot(self,x_vals,PhyTime,comp,)

    def _bulk_velo_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
            
        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        # wall_velo = self._meta_data.moving_wall_calc()
        
        bulk_velo=np.zeros(self.shape[1])
        # if relative:
        #     for i in range(self.NCL[1]):
        #         u_velo[i,:]=u_velo[i,:] - wall_velo
        for i in range(self.shape[1]):
            bulk_velo[i] = 0.5*integrate.simps(u_velo[:,i],ycoords)
            
        return bulk_velo

    def plot_bulk_velocity(self,PhyTime,fig='',ax='',**kwargs):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
    
        bulk_velo = self._bulk_velo_calc(PhyTime)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax=cplt.subplots(**kwargs)
        elif not ax:
            ax =fig.add_subplot(1,1,1)
        ax.cplot(bulk_velo)
        ax.set_ylabel(r"$U_b^*$")# ,fontsize=20)
        # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
        #ax.grid()
        return fig, ax

    def _tau_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)

        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        # wall_velo = self._meta_data.moving_wall_calc()
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        #time0=time.time()
        #sol, h_list = CT.Stencil_calc([0,1,2,3], 1)
        #print(time.time()-time0)
        #a,b,c,d = CT.Stencil_coeffs_eval(sol,h_list,[ycoords[0]--1.0,ycoords[1]-ycoords[0],ycoords[2]-ycoords[1]])
        for i in range(self.shape[1]):
            #tau_star[i] = mu_star*(a*wall_velo[i] + b*u_velo[0,i] + c*u_velo[1,i] + d*u_velo[2,i])
            tau_star[i] = mu_star*(u_velo[0,i]-0.0)/(ycoords[0]--1.0)#*(-1*u_velo[1,i] + 4*u_velo[0,i] - 3*wall_velo[i])/(0.5*ycoords[1]-1.5*(-1.0)+y_coords[0])
    
        return tau_star

    def plot_skin_friction(self,PhyTime,fig='',ax='',**kwargs):
        rho_star = 1.0
        REN = self._metaDF['REN']
        tau_star = self._tau_calc(PhyTime)
        bulk_velo = self._bulk_velo_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/REN)*tau_star
        # xcoords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        
        ax.cplot(skin_friction,label=r"$C_f$")
        # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
        ax.set_ylabel(r"$C_f$")# ,fontsize=20)
        fig.tight_layout()
        #ax.grid()
        return fig, ax

    def plot_eddy_visc(self,x_val,PhyTime,Y_plus=True,Y_plus_max=100,fig='',ax='',**kwargs):
        
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
                
        x_loc = [self._return_index(x) for x in x_val]
    
        # PUT IN IO CLASS
        # if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
        #     avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
        #     if PhyTime and PhyTime != avg_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
        #     PhyTime = avg_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        # if type(PhyTime) == float:
        #     PhyTime = "{:.9g}".format(PhyTime)
        if isinstance(x_loc,int):
            x_loc_local = [x_loc]
        elif isinstance(x_loc,list):
            x_loc_local=x_loc
        else:
            raise TypeError("\033[1;32 variable x_loc must be of type int of list of int")
        uv = self.UU_tensorDF[PhyTime,'uv']
        # U = self.flow_AVGDF.loc[PhyTime,'u'].values.reshape(self.shape)
        # V = self.flow_AVGDF.loc[PhyTime,'v'].values.reshape(self.shape)
        # uv = UV-U*V
        dUdy = self.Velo_grad_tensorDF[PhyTime,'uy']
        dVdx = self.Velo_grad_tensorDF[PhyTime,'vx']
        REN = self._metaDF['REN']
        mu_t = -uv*REN/(dUdy + dVdx)
        mu_t = mu_t[:,x_loc_local]
        y_coord = self._meta_data.CoordDF['y']
        
        
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        linestyle_list=['-','--','-.']
        # x_coord = self._meta_data.CoordDF['x'].dropna().values
        
        for i in range(len(x_loc_local)):
            if Y_plus:
                avg_time = self.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                # u_tau_star, delta_v_star = self._wall_unit_calc(avg_time)
                y_coord_local = self._y_plus_calc(PhyTime)[i]
                #y_coord_local = y_coord_local[y_coord_local<Y_plus_max]
                
                mu_t_local = mu_t[:int(y_coord.size/2),i]
                
            else:
                y_coord_local = y_coord
                mu_t_local = mu_t[:,i]
            
            # label = r"$x/\delta = %.3g$" %x_coord[x_loc_local[i]]
            
            ax.cplot(y_coord_local,mu_t_local)#label=label)
            if Y_plus:
                ax.set_xlabel(r"$y^+$")# ,fontsize=18)
                ax.set_xlim([0,Y_plus_max])
                ax.set_ylim([-0.5,max(mu_t_local)*1.2])
            else:
                ax.set_xlabel(r"$y/\delta$")# ,fontsize=18)
                ax.set_xlim([-1,-0.1])
                ax.set_ylim([-0.5,max(mu_t_local)*1.2])
            ax.set_ylabel(r"$\mu_t/\mu$")# ,fontsize=16)

        return fig, ax
    def __iadd__(self,other_avg):
        pass
class CHAPSim_budget_base():
    def __init__(self,comp1,comp2,avg_data='',PhyTime=None,*args,**kwargs):

        if avg_data:
            self.avg_data = avg_data
        elif PhyTime:
            self.avg_data = CHAPSim_AVG_io(PhyTime,*args,**kwargs)
        else:
            raise Exception
        if PhyTime is None:
            PhyTime = list(set([x[0] for x in self.avg_data.flow_AVGDF.index]))[0]
        self.comp = comp1+comp2
        self.budgetDF = self._budget_extract(PhyTime,comp1,comp2)
        self.shape = self.avg_data.shape

    def _budget_extract(self,PhyTime,comp1,comp2):
        if type(PhyTime) == float and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
            
        production = self._production_extract(PhyTime,comp1,comp2)
        advection = self._advection_extract(PhyTime,comp1,comp2)
        turb_transport = self._turb_transport(PhyTime,comp1,comp2)
        pressure_diffusion = self._pressure_diffusion(PhyTime,comp1,comp2)
        pressure_strain = self._pressure_strain(PhyTime,comp1,comp2)
        viscous_diff = self._viscous_diff(PhyTime,comp1,comp2)
        dissipation = self._dissipation_extract(PhyTime,comp1,comp2)
        array_concat = [production,advection,turb_transport,pressure_diffusion,\
                        pressure_strain,viscous_diff,dissipation]

        budget_array = np.stack(array_concat,axis=0)
        
        budget_index = ['production','advection','turbulent transport','pressure diffusion',\
                     'pressure strain','viscous diffusion','dissipation']  
        phystring_index = [PhyTime]*7
    
        budgetDF = cd.datastruct(budget_array,index =[phystring_index,budget_index])
        
        return budgetDF

    def _budget_plot(self,PhyTime, x_list,wall_units=True, fig='', ax ='',**kwargs):
        if type(PhyTime) == float and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        u_tau_star, delta_v_star = self.avg_data._wall_unit_calc(PhyTime)
        budget_scale = u_tau_star**3/delta_v_star
        
        Y_extent= int(self.avg_data.shape[0]/2)
        if wall_units:
            Y_coords = self.avg_data._y_plus_calc(PhyTime)
        else:
            Y = np.zeros(self.avg_data.shape[::-1])
            for i in range(self.avg_data.shape[1]):
                Y[i] = self.avg_data.CoordDF['y']
            Y_coords = (1-np.abs(Y[:,:Y_extent]))
        
        if isinstance(x_list,list):
            ax_size = len(x_list)
        else:
            ax_size = 1

        ax_size=(int(np.ceil(ax_size/2)),2) if ax_size >2 else (ax_size,1)

        kwargs['squeeze'] = False
        lower_extent= 0.2
        kwargs['gridspec_kw'] = {'bottom': lower_extent}
        kwargs['constrained_layout'] = False
        # gridspec_kw={'bottom': lower_extent}
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize']=[7*ax_size[1],5*ax_size[0]+1]
            else:
                warnings.warn("Figure size calculator overidden: Figure quality may be degraded", stacklevel=2)
            fig, ax = cplt.subplots(*ax_size,**kwargs)
        elif not ax:
            ax=fig.subplots(*ax_size,**kwargs)
        ax=ax.flatten()
        comp_list = tuple([x[1] for x in self.budgetDF.index])
        #print(comp_list)
        i=0

        def ax_convert(j):
            return (int(j/2),j%2)
        
        
        if not hasattr(x_list,'__iter__'):
            x_list=[x_list]

        for i,x_loc in enumerate(x_list):
            for comp in comp_list:
                budget_values = self.budgetDF[PhyTime,comp].copy()
                x = self.avg_data._return_index(x_loc)
                if wall_units:
                    budget = budget_values[:Y_extent,x]/budget_scale[x]
                else:
                    budget = budget_values[:Y_extent,x]
                
                if self.comp == 'uv':
                    budget= budget * -1.0

                ax[i].cplot(Y_coords[x,:],budget,label=comp.title())
    
                
                if wall_units:
                    ax[i].set_xscale('log')
                    ax[i].set_xlim(left=1.0)
                    ax[i].set_xlabel(r"$y^+$")# ,fontsize=18)
                else:
                    ax[i].set_xlabel(r"$y/\delta$")# ,fontsize=18)
                
                # ax[i].set_title(r"$x/\delta=%.3g$" %x_coords[x],loc='right',pad=-20)
                # if i == 0:
                if mpl.rcParams['text.usetex'] == True:
                    ax[i].set_ylabel("Loss\ \ \ \ \ \ \ \ Gain")# ,fontsize=18)
                else:
                    ax[i].set_ylabel("Loss        Gain")# ,fontsize=18)
                
                #ax[i].grid()

                # handles, labels = ax[0,0].get_legend_handles_labels()
                # handles = cplt.flip_leg_col(handles,4)
                # labels = cplt.flip_leg_col(labels,4)

                # fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)# ,fontsize=17)
                # #ax[0,0].clegend(vertical=False,loc = 'lower left',ncol=4, bbox_to_anchor=(0,1.02))# ,fontsize=13)
                
            
        # if type(ax_size)==tuple:
        #     while k <ax_size[0]*ax_size[1]:
        #         i=ax_convert(k)
        #         ax[i].remove()exit
        #         k+=1   
        # gs = ax[0,0].get_gridspec()
        # gs.ax[0,0].get_gridspec()tight_layout(fig,rect=(0,0.1,1,1))
        return fig, ax
    def _plot_integral_budget(self,comp=None,PhyTime='',wall_units=True,fig='',ax='',**kwargs):
        if type(PhyTime) == float and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        budget_terms = tuple([x[1] for x in self.budgetDF.index])
        y_coords = self.avg_data.CoordDF['y']

        if comp is None:
            comp_list = tuple([x[1] for x in self.budgetDF.index])
        elif hasattr(comp,"__iter__") and not isinstance(comp,str) :
            comp_list = comp
        elif isinstance(comp,str):
            comp_list = [comp]
        else:
            raise TypeError("incorrect time")
        # print(budget_terms)
        if not all([comp in budget_terms for comp in comp_list]):
            raise KeyError("Invalid budget term provided")

        
        u_tau_star, delta_v_star = self.avg_data._wall_unit_calc(PhyTime)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplots(1,1,1)
        xaxis_vals = self.avg_data._return_xaxis()

        for comp in comp_list:
            integral_budget = np.zeros(self.avg_data.shape[1])
            budget_term = self.budgetDF[PhyTime,comp].copy()
            for i in range(self.avg_data.shape[1]):
                integral_budget[i] = 0.5*integrate.simps(budget_term[:,i],y_coords)
                if wall_units:
                    delta_star=1.0
                    integral_budget[i] /=(delta_star*u_tau_star[i]**3/delta_v_star[i])
            label = r"$\int^{\delta}_{-\delta}$ %s $dy$"%comp.title()
            ax.cplot(xaxis_vals,integral_budget,label=label)
        budget_symbol = {}

        # if wall_units:
        #     ax.set_ylabel(r"$\int_{-\delta}^{\delta} %s$ budget $dy\times u_{\tau}^4\delta/\nu $" %self.comp)# ,fontsize=16)
        # else:
        #     ax.set_ylabel(r"$\int_{-\delta}^{\delta} %s$ budget $dy\times 1/U_{b0}^3 $"%self.comp)# ,fontsize=16)
        ax.set_xlabel(r"$x/\delta$")# ,fontsize=18)
        ax.clegend(ncol=4,vertical=False)
        #ax.grid()
        return fig, ax
    def _plot_budget_x(self,comp,y_vals_list,Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        if PhyTime:
            if not isinstance(PhyTime,str) and PhyTime is not None:
                PhyTime = "{:.9g}".format(PhyTime)
        
        comp_index = [x[1] for x in self.budgetDF.index]
        if not comp.lower() in comp_index:
            msg = "comp must be a component of the"+\
                        " Reynolds stress budget: %s" %comp_index
            raise KeyError(msg) 
        if y_vals_list != 'max':
            if Y_plus:
                y_index = CT.Y_plus_index_calc(self, self.CoordDF, y_vals_list)
            else:
                y_index = CT.coord_index_calc(self.CoordDF,'y',y_vals_list)
            budget_term = self.budgetDF[PhyTime,comp]
            
        else:
            y_index = [None]
            budget_term = self.budgetDF[PhyTime,comp]
            budget_term = np.amax(budget_term,axis=0)

        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)
        xaxis_vals = self.avg_data._return_xaxis()
        if y_vals_list != 'max':
            for i in range(len(y_index)):
                ax.cplot(budget_term[i],label=r"$y^+=%.3g$"% y_vals_list[i])
            axes_items_num = len(ax.get_lines())
            ncol = 4 if axes_items_num>3 else axes_items_num
            ax.clegend(vertical=False,ncol=ncol, fontsize=16)
            # ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)$"%y_label)# ,fontsize=20)#)# ,fontsize=22)
            
        else:
            ax.cplot(xaxis_vals,budget_term,label=r"maximum %s"%comp)
            # ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"%y_label)# ,fontsize=20)#)# ,fontsize=22)

        fig.tight_layout()
        return fig, ax

    def __str__(self):
        return self.budgetDF.__str__()
        
class CHAPSim_fluct_base():
    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_fluct'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(base_name)
        group.create_dataset("NCL",data=np.array(self.shape))
        hdf_file.close()
        self.meta_data.save_hdf(file_name,'a',base_name+'/meta_data')
        self.fluctDF.to_hdf(file_name,key=base_name+'/fluctDF',mode='a')#,format='fixed',data_columns=True)

    
    def plot_contour(self,comp,axis_vals,plane='xz',PhyTime='',x_split_list='',y_mode='wall',fig='',ax='',**kwargs):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        
        
        if not hasattr(axis_vals,'__iter__'):
            if isinstance(axis_vals,float) or isinstance(axis_vals,int):
                axis_vals = [axis_vals] 
            else:
                msg = "axis_vals must be an interable or a float or an int not a %s"%type(axis_vals)
                raise TypeError(msg)
        
        plane , coord, axis_index = CT.contour_plane(plane,axis_vals,self.avg_data,y_mode,PhyTime)


        # y_index = CT.y_coord_index_norm(self.avg_data,self.avg_data.CoordDF,
        #                                 y_vals,x_vals=0,mode='wall')
        # y_index = list(itertools.chain(*y_index))
        x_coords = self.CoordDF[plane[0]]
        z_coords = self.CoordDF[plane[1]]
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF[PhyTime,comp]
        
        #fluct=fluct.reshape((self.NCL[2],self.NCL[0]))
        # if isinstance(y_vals,int):
        #     y_vals=[y_vals]
        # elif not isinstance(y_vals,list):
        #     raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
        
        if not x_split_list:
            x_split_list=[np.amin(x_coords),np.amax(x_coords)]
        
        kwargs['squeeze'] = False
        single_input=False
        if not fig:
            if  'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10*len(axis_vals),3*(len(x_split_list)-1)]
            else:
                warnings.warn('Figure size algorithm overridden', stacklevel=2)
            fig, ax = cplt.subplots(len(x_split_list)-1,len(axis_vals),**kwargs)
        elif not ax:
            ax=fig.subplots(len(x_split_list)-1,len(axis_vals),squeeze=False)      
        else:
            if isinstance(ax,mpl.axes.Axes):
                single_input=True
                ax = np.array([ax]).reshape((1,1))
            elif not isinstance(ax,np.ndarray):
                msg = "Input axes must be an instance %s or %s"%(mpl.axes,np.ndarray)
                raise TypeError(msg)
        ax=ax.flatten()

        x_coords_split=CT.coord_index_calc(self.CoordDF,'x',x_split_list)

        title_symbol = CT.get_title_symbol(coord,y_mode,False)
        
        X, Z = np.meshgrid(x_coords, z_coords)
        max_val = np.amax(fluct); min_val=np.amin(fluct)
        if len(x_split_list)==2:
            for i in range(len(axis_vals)):
                # print(X.shape,Z.shape,fluct.shape)
                fluct_slice = CT.contour_indexer(fluct,axis_index[i],coord)
                ax1 = ax[i].pcolormesh(X,Z,fluct_slice,cmap='jet',shading='auto')
                ax2 = ax1.axes
                ax1.set_clim(min_val,max_val)
                cbar=fig.colorbar(ax1,ax=ax[i])
                cbar.set_label(r"$%s^\prime$"%comp)# ,fontsize=12)
                ax2.set_xlim([x_split_list[0],x_split_list[1]])
                ax2.set_xlabel(r"$%s/\delta$" % plane[0])# ,fontsize=20)
                ax2.set_ylabel(r"$%s/\delta$" % plane[1])# ,fontsize=20)
                ax2.set_title(r"$%s=%.3g$"%(title_symbol,axis_vals[i]),loc='right')
                ax2.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                ax[i]=ax1
        else:
            ax=ax.flatten()
            max_val = np.amax(fluct); min_val=np.amin(fluct)
            for j in range(len(x_split_list)-1):
                for i in range(len(axis_vals)):
                    fluct_slice = CT.contour_indexer(fluct,axis_index[i],coord)
                    # print(X.shape,Z.shape,fluct.shape)
                    ax1 = ax[j*len(axis_vals)+i].pcolormesh(X,Z,fluct_slice,cmap='jet',shading='auto')
                    ax2 = ax1.axes
                    ax1.set_clim(min_val,max_val)
                    cbar=fig.colorbar(ax1,ax=ax[j*len(axis_vals)+i])
                    cbar.set_label(r"$%s^\prime$"%comp)# ,fontsize=12)
                    ax2.set_xlabel(r"$%s/\delta$" % 'x')# ,fontsize=20)
                    ax2.set_ylabel(r"$%s/\delta$" % 'z')# ,fontsize=20)
                    ax2.set_xlim([x_split_list[j],x_split_list[j+1]])
                    ax2.set_title(r"$%s=%.3g$"%(title_symbol,axis_vals[i]),loc='right')
                    ax2.set_title(r"$t^*=%s$"%PhyTime,loc='left')
                    ax[j*len(axis_vals)+i]=ax1
                    ax[j*len(axis_vals)+i].axes.set_aspect('equal')
        fig.tight_layout()
        if single_input:
            return fig, ax[0]
        else:
            return fig, ax

    def plot_streaks(self,comp,vals_list,x_split_list='',PhyTime='',ylim='',Y_plus=True,*args,colors='',fig=None,ax=None,**kwargs):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
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
    def plot_fluct3D_xz(self,comp,y_vals,PhyTime='',x_split_list='',fig='',ax=''):
        if PhyTime:
            if type(PhyTime) == float:
                PhyTime = "{:.9g}".format(PhyTime)
                
        if len(set([x[0] for x in self.fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
            if PhyTime and PhyTime != fluct_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
            PhyTime = fluct_time
        else:
            assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        if isinstance(y_vals,int):
            y_vals=[y_vals]
        elif not isinstance(y_vals,list):
            raise TypeError("\033[1;32 y_vals must be type int or list but is type %s"%type(y_vals))
            
        x_coords = self.CoordDF['x']
        z_coords = self.CoordDF['z']
        X,Z = np.meshgrid(x_coords,z_coords)
        fluct = self.fluctDF[PhyTime,comp][:,y_vals,:]
        
        if not x_split_list:
            x_split_list=[np.min(x_coords),np.max(x_coords)]
            
        x_coords_split=CT.coord_index_calc(self.CoordDF,'x',x_split_list)
          
        
        if not fig:
            subplot_kw={'projection':'3d'}
            fig, ax = plt.subplots((len(x_split_list)-1),len(y_vals),figsize=[10*len(y_vals),5*(len(x_split_list)-1)],subplot_kw=subplot_kw,squeeze=False)
        elif not ax:
            ax_list=[]
            for i in range(len(x_split_list)-1):
                for j in range(len(y_vals)):
                    ax_list.append(fig.add_subplot(len(x_split_list)-1,len(y_vals),i*len(y_vals)+j+1,projection='3d'))
            ax=np.array(ax_list)
        
        ax=ax.flatten()
        max_val = -np.float('inf'); min_val = np.float('inf')
        for j in range(len(x_split_list)-1):
            for i in range(len(y_vals)):
                max_val = np.amax(fluct[:,i,:]); min_val=np.amin(fluct[:,i,:])
                
                #ax[j*len(y_vals)+i] = fig.add_subplot(j+1,i+1,j*len(y_vals)+i+1,projection='3d')
                #axisEqual3D(ax)
                surf=ax[j*len(y_vals)+i].plot_surface(Z[:,x_coords_split[j]:x_coords_split[j+1]],
                                     X[:,x_coords_split[j]:x_coords_split[j+1]],
                                     fluct[:,i,x_coords_split[j]:x_coords_split[j+1]],
                                     rstride=1, cstride=1, cmap='jet',
                                            linewidth=0, antialiased=False)
                ax[j*len(y_vals)+i].set_ylabel(r'$x/\delta$')# ,fontsize=20)
                ax[j*len(y_vals)+i].set_xlabel(r'$z/\delta$')# ,fontsize=20)
                ax[j*len(y_vals)+i].set_zlabel(r'$%s^\prime$'%comp)# ,fontsize=20)
                ax[j*len(y_vals)+i].set_zlim(min_val,max_val)
                surf.set_clim(min_val,max_val)
                cbar=fig.colorbar(surf,ax=ax[j*len(y_vals)+i])
                cbar.set_label(r"$%s^\prime$"%comp)# ,fontsize=12)
                ax[j*len(y_vals)+i]=surf
        fig.tight_layout()
        return fig, ax

    def plot_vector(self,slice,ax_val,PhyTime=None,y_mode='half_channel',spacing=(1,1),scaling=1,x_split_list=None,fig=None,ax=None,quiver_kw=None,**kwargs):
        
        # if PhyTime is not None:
        #     if type(PhyTime) == float:
        #         PhyTime = "{:.9g}".format(PhyTime)
                
        # if len(set([x[0] for x in self.fluctDF.index])) == 1:
        #     fluct_time = list(set([x[0] for x in self.fluctDF.index]))[0]
        #     if PhyTime is not None and PhyTime != fluct_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_fluct class" %float(fluct_time), stacklevel=2)
        #     PhyTime = fluct_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.fluctDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        # if slice not in ['xy','zy','xz']:
        #     slice = slice[::-1]
        #     if slice not in ['xy','yz','xz']:
        #         msg = "The contour slice must be either %s"%['xy','yz','xz']
        #         raise KeyError(msg)
        
        # slice = list(slice)
        # slice_set = set(slice)
        # coord_set = set(list('xyz'))
        # coord = "".join(coord_set.difference(slice_set))

        if not hasattr(ax_val,'__iter__'):
            if isinstance(ax_val,float) or isinstance(ax_val,int):
                ax_val = [ax_val] 
            else:
                msg = "ax_val must be an interable or a float or an int not a %s"%type(ax_val)
                raise TypeError(msg)

        # if not x_split_list:
        #     x_split_list=[np.amin(x_coords),np.amax(x_coords)]

        slice, coord, axis_index = CT.contour_plane(slice,ax_val,self.avg_data,y_mode,PhyTime)
        # if coord == 'y':
        #     norm_vals = [0]*len(ax_val)
        #     axis_index = CT.y_coord_index_norm(self.avg_data,self.CoordDF,ax_val,norm_vals,y_mode)
        # else:
        #     axis_index = CT.coord_index_calc(self.CoordDF,coord,ax_val)
        #     if not hasattr(axis_index,'__iter__'):
        #         axis_index = [axis_index]

        if fig is None:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [8,4*len(ax_val)]
            else:
                warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(ax_val),**kwargs)
        elif ax is None:
            ax=fig.subplots(len(ax_val),squeeze=False)   
        else:
            if isinstance(ax,mpl.axes.Axes):
                ax = np.array([ax]).reshape((1,1))
            elif not isinstance(ax,np.ndarray):
                msg = "Input axes must be an instance %s or %s"%(mpl.axes,np.ndarray)
                raise TypeError(msg)
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
            # im=ax[i].imshow(U_space[:,:,i].astype('f4').T,cmap='jet',extent=)
            # ax[i].pcolormesh(coord_1_mesh,coord_2_mesh,U_space[:,:,i].T,cmap='jet')
            # im.axes.set_xticks(coord_1)
            # im.axes.set_yticks(coord_2)
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
    def create_video(cls,axis_vals,comp,contour=True,plane='xz',meta_data='',path_to_folder='',time0='',
                            abs_path=True,tgpost=False,x_split_list='',lim_min=None,lim_max=None,
                            ax_func=None,fluct_func=None,fluct_args=(),fluct_kw={}, fig='',ax='',**kwargs):

        module = sys.modules[cls.__module__]
        times= CT.time_extract(path_to_folder,abs_path)
        
        if time0:
            times = sorted(list(filter(lambda x: x > time0, times)))

        if TEST:
            times = times[-10:]

        max_time = np.amax(times)
        avg_data=module.CHAPSim_AVG(max_time,meta_data,path_to_folder,time0,abs_path,tgpost=cls.tgpost)
        if isinstance(axis_vals,(int,float)):
            axis_vals=[axis_vals]
        elif not isinstance(axis_vals,list):
            raise TypeError("\033[1;32 axis_vals must be type int or list but is type %s"%type(axis_vals))
        
        x_coords = avg_data.CoordDF['x']
        if not x_split_list:
            x_split_list=[np.min(x_coords),np.max(x_coords)]
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7*len(axis_vals),3*(len(x_split_list)-1)]
            fig = cplt.figure(**kwargs)
        fig_list = [fig]*len(times)

        frames=list(zip(times,fig_list))
        def animate(frames):
            time=frames[0]; fig=frames[1]
            ax_list=fig.axes
            for ax in ax_list:
                ax.remove()
            fluct_data = cls(time,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)

            if contour:
                fig, ax = fluct_data.plot_contour(comp,axis_vals,plane=plane,PhyTime=time,x_split_list=x_split_list,fig=fig)
            else:
                fig,ax = fluct_data.plot_fluct3D_xz(axis_vals,comp,time,x_split_list,fig)
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
        anim = mpl.animation.FuncAnimation(fig,animate,frames=frames)
        
        # print("\033[1;32m The colorbar limits are set to (%.3g,%.3g) the maximum limits are (%.3g,%.3g)\033[0;37m\n"%(lim_min,lim_max,lims[0],lims[1]))
        # del lims
        return anim

        
    def __str__(self):
        return self.fluctDF.__str__()

class CHAPSim_autocov_base():
    def __init__(self,*args,**kwargs):#self,comp1,comp2,max_x_sep=None,max_z_sep=None,path_to_folder='',time0='',abs_path=True):
        fromfile=kwargs.pop('fromfile',False)
        if not fromfile:
            self._meta_data, self.comp, self.NCL,\
            self._avg_data, self.autocorrDF, self.shape_x,\
            self.shape_z = self._autocov_extract(*args,**kwargs)
        else:
            self._meta_data, self.comp, self.NCL,\
            self._avg_data, self.autocorrDF, self.shape_x,\
            self.shape_z = self._hdf_extract(*args,**kwargs)

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_autocov'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(base_name)
        group.attrs['shape_x'] = np.array(self.shape_x)
        group.attrs['shape_z'] = np.array(self.shape_z)
        group.attrs['comp'] = np.array([np.string_(x) for x in self.comp])
        hdf_file.close()

        self._meta_data.save_hdf(file_name,'a',base_name+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',base_name+'/avg_data')
        self.autocorrDF.to_hdf(file_name,key=base_name+'/autocorrDF',mode='a')#,format='fixed',data_columns=True)

    def plot_autocorr_line(self,comp,axis_vals,y_vals,y_mode='half_channel',norm_xval=None,norm=True,fig='',ax='',**kwargs):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals = [axis_vals]
        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)


        if norm_xval is not None:
            if norm_xval ==0:
                norm_xval = np.amin(self._avg_data._return_xaxis())
            x_axis_vals=[norm_xval]*len(axis_vals)
        else:
            x_axis_vals=axis_vals

        coord = self._meta_data.CoordDF[comp][:shape[0]]
        if not hasattr(y_vals,'__iter__'):
            y_vals = [y_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,x_axis_vals,y_mode)
        Ruu = self.autocorrDF[comp]
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        if not fig:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [10,5*len(y_vals)]
                if len(y_vals) >1:
                    warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(y_vals),**kwargs)
        elif not ax:
            kwargs['subplot_kw'] = {'squeeze',False}
            ax = fig.subplots(len(y_vals),**kwargs)

        ax=ax.flatten()
        coord = self._meta_data.CoordDF[comp][:shape[0]]
        for j in range(len(y_vals)):
            for i in range(len(axis_index)):
                ax[j].cplot(coord,Ruu[:,y_index_axis_vals[i][j],axis_index[i]])
                #creating title label
                y_unit="y" if y_mode=='half_channel' \
                        else "\delta_u" if y_mode=='disp_thickness' \
                        else "\theta" if y_mode=='mom_thickness' \
                        else "y^+" if norm_xval !=0 else "y^{+0}"
 

                ax[j].set_title(r"$%s=%.3g$"%(y_unit,y_vals[j]),loc='left')
            ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp)# ,fontsize=20)
            ax[j].set_xlabel(r"$%s/\delta$"%comp)# ,fontsize=20)
            
        

        return fig, ax
    def plot_spectra(self,comp,axis_vals,y_vals,y_mode='half_channel',norm_xval=None,fig='',ax='',**kwargs):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals = [axis_vals]
        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)
        
            #raise TypeError("Variable `axis_vals' must be an int or iterable\n")

        if norm_xval is not None:
            if norm_xval ==0:
                x_axis_vals = [np.amin(self._avg_data._return_xaxis())]*len(axis_vals)
            else:
                x_axis_vals=[norm_xval]*len(axis_vals)
        else:
            x_axis_vals=axis_vals

        coord = self._meta_data.CoordDF[comp][:shape[0]]
        if not hasattr(y_vals,'__iter__'):
            y_vals = [y_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    y_vals,x_axis_vals,y_mode)
        Ruu = self.autocorrDF[comp]#[:,axis_vals,:]

        if not fig:
            if 'figsize' not in kwargs:
                kwargs['figsize'] = [10,5*len(y_vals)]
                if len(y_vals) >1:
                    warnings.warn("figure size algorithm overrided: may result in poor quality graphs", stacklevel=2)
            kwargs['squeeze'] = False
            fig,ax = cplt.subplots(len(y_vals),**kwargs)
        elif not ax:
            kwargs['subplot_kw'] = {'squeeze',False}
            ax = fig.subplots(len(y_vals),**kwargs)
        ax=ax.flatten()

        for j in range(len(y_vals)):
            for i in range(len(axis_index)):
                wavenumber_spectra = fft.rfft(Ruu[:,y_index_axis_vals[i][j],axis_index[i]])
                delta_comp = coord[1]-coord[0]
                Fs = (2.0*np.pi)/delta_comp
                comp_size= Ruu[:,y_index_axis_vals[i][j],axis_index[i]].size
                wavenumber_comp = 2*np.pi*fft.rfftfreq(comp_size,coord[1]-coord[0])
                y_unit="y" if y_mode=='half_channel' \
                        else "\delta_u" if y_mode=='disp_thickness' \
                        else "\theta" if y_mode=='mom_thickness' \
                        else "y^+" if norm_xval !=0 else "y^{+0}"
                ax[j].cplot(wavenumber_comp,2*np.abs(wavenumber_spectra))
                ax[j].set_title(r"$%s=%.3g$"%(y_unit,y_vals[j]),loc='left')
            string= (ord(self.comp[0])-ord('u')+1,ord(self.comp[1])-ord('u')+1,comp)
            ax[j].set_ylabel(r"$E_{%d%d}(\kappa_%s)$"%string)# ,fontsize=20)
            ax[j].set_xlabel(r"$\kappa_%s$"%comp)# ,fontsize=20)
        
        return fig, ax

    def autocorr_contour_y(self,comp,axis_vals,Y_plus=False,Y_plus_0=False,
                                Y_plus_max ='',norm=True,
                                show_positive=True,fig=None,ax=None):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals = [axis_vals]
        axis_index = [self._avg_data._return_index(x) for x in axis_vals]#CT.coord_index_calc(self._meta_data.CoordDF,'x',axis_vals)
 
        Ruu = self.autocorrDF[comp][:,:,axis_index]
       
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0
        if fig is None:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif ax is None:
            subplot_kw = {'squeeze':'False'}
            ax = fig.subplots(len(axis_vals),subplot_kw)
        ax=ax.flatten()
        # x_coord = self._meta_data.CoordDF['x'].copy().dropna()\
        #             .values
        max_val = -np.float('inf'); min_val = np.float('inf')

        for i in range(len(axis_vals)):
            y_coord = self._meta_data.CoordDF['y'].copy()
            coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]
            if Y_plus:
                avg_time = self._avg_data.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                _, delta_v_star = self._avg_data.wall_unit_calc(avg_time)
                y_coord = y_coord[:int(y_coord.size/2)]
                if i==0:
                    Ruu = Ruu[:,:y_coord.size]
                if Y_plus_0:
                    y_coord = (1-np.abs(y_coord))/delta_v_star[0]
                else:   
                    y_coord = (1-np.abs(y_coord))/delta_v_star[axis_index[i]]
            min_val = min(min_val,np.amin(np.squeeze(Ruu[:,:,i])))
            max_val = max(max_val,np.amax(np.squeeze(Ruu[:,:,i])))

            X,Y = np.meshgrid(coord,y_coord)
            ax[i] = ax[i].pcolormesh(X,Y,np.squeeze(Ruu[:,:,i]).T,cmap='jet',shading='auto')
            ax[i].axes.set_xlabel(r"$\Delta %s/\delta$" %comp)
            if Y_plus and Y_plus_0:
                ax[i].axes.set_ylabel(r"$Y^{+0}$")
            elif Y_plus and not Y_plus_0:
                ax[i].axes.set_ylabel(r"$Y^{+}$")
            else:
                ax[i].axes.set_ylabel(r"$y/\delta$")
            if Y_plus_max:
                ax[i].axes.set_ylim(top=Y_plus_max)
            fig.colorbar(ax[i],ax=ax[i].axes)
            fig.tight_layout()

        for a in ax:     
            a.set_clim(min_val,max_val)

        if not show_positive:
            for a in ax:   
                cmap = a.get_cmap()
                min_val,max_val = a.get_clim()
                new_color = cmap(np.linspace(0,1,256))[::-1]
                new_color[-1] = np.array([1,1,1,1])
                a.set_cmap(mpl.colors.ListedColormap(new_color))
                a.set_clim(min_val,0)
        return fig, ax

    def autocorr_contour_x(self,comp,axis_vals,axis_mode='half_channel',norm=True,fig='',ax=''):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")
        
        if not hasattr(axis_vals,'__iter__'):
            axis_vals=[axis_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        
        Ruu_all = self.autocorrDF[comp]
        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i]
        
        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(shape[0]):
                Ruu[i]/=Ruu_0

        if not fig:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif not ax:
            subplot_kw = {'squeeze':False}
            ax = fig.subplots(len(axis_vals),subplot_kw=subplot_kw)
        ax=ax.flatten()
        
        x_axis =self._avg_data._return_xaxis()
        # y_coord = self._meta_data.CoordDF['y'].copy().dropna().values
        coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]

        ax_out=[]
        for i in range(len(axis_vals)):

            X,Y = np.meshgrid(x_axis,coord)
            ax1 = ax[i].pcolormesh(X,Y,Ruu[:,i],cmap='jet',shading='auto')            
            ax1.axes.set_ylabel(r"$\Delta %s/\delta$" %comp)
            title = r"$%s=%.3g$"%("y" if axis_mode=='half_channel' \
                        else "\delta_u" if axis_mode=='disp_thickness' \
                        else "\theta" if axis_mode=='mom_thickness' else "y^+", axis_vals[i] )
            ax1.axes.set_title(title,loc='left')# ,fontsize=15,loc='left')
            fig.colorbar(ax1,ax=ax1.axes)
            ax_out.append(ax1)
        fig.tight_layout()
        
        return fig, np.array(ax_out)

    def spectrum_contour(self,comp,axis_vals,axis_mode='half_channel',fig='',ax=''):
        if comp == 'x':
            shape=self.shape_x
        elif comp=='z':
            shape=self.shape_z
        else:
            raise ValueError(" Variable `comp' must eiher be 'x' or 'z'\n")

        if not hasattr(axis_vals,'__iter__'):
            axis_vals=[axis_vals]
        y_index_axis_vals = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                    axis_vals,None,axis_mode)
        Ruu_all = self.autocorrDF[comp]#[:,axis_vals,:]
        Ruu=np.zeros((shape[0],len(axis_vals),shape[2]))
        for i,vals in zip(range(shape[2]),y_index_axis_vals):
            Ruu[:,:,i] = Ruu_all[:,vals,i]

        if not fig:
            fig, ax = plt.subplots(len(axis_vals),figsize=[10,4*len(axis_vals)],squeeze=False)
        elif not ax:
            subplot_kw = {'squeeze':False}
            ax = fig.subplots(len(axis_vals),subplot_kw=subplot_kw)
        ax=ax.flatten()
        # x_coord = self._meta_data.CoordDF['x'].copy().dropna()\
        #         .values[:shape[2]]
        coord = self._meta_data.CoordDF[comp].copy()[:shape[0]]
        x_axis =self._avg_data._return_xaxis()

        ax_out=[]
        for i in range(len(axis_vals)):
            wavenumber_spectra = np.zeros((int(0.5*shape[0])+1,shape[2]))
            for j in range(shape[2]):
                wavenumber_spectra[:,j]=fft.rfft(Ruu[:,i,j])
            delta_comp = coord[1]-coord[0]
            Fs = (2.0*np.pi)/delta_comp
            comp_size= shape[0]
            wavenumber_comp = 2*np.pi*fft.rfftfreq(comp_size,coord[1]-coord[0])
            X, Y = np.meshgrid(x_axis,wavenumber_comp)
            ax1 = ax[i].pcolormesh(X,Y,np.abs(wavenumber_spectra),cmap='jet',shading='auto')
            ax1.axes.set_ylabel(r"$\kappa_%s$"%comp)
            title = r"$%s=%.3g$"%("y" if axis_mode=='half_channel' \
                        else "\delta_u" if axis_mode=='disp_thickness' \
                        else "\theta" if axis_mode=='mom_thickness' else "y^+", axis_vals[i] )
            ax1.axes.set_ylim([np.amin(wavenumber_comp[1:]),np.amax(wavenumber_comp)])
            ax1.axes.set_title(title)# ,fontsize=15,loc='left')
            fig.colorbar(ax1,ax=ax1.axes)
            ax_out.append(ax1)
        fig.tight_layout()
        return fig, np.array(ax_out)

    def __str__(self):
        return self.autocorrDF.__str__()

class CHAPSim_Quad_Anl_base():
    def __init__(self,*args,**kwargs):
        fromfile=kwargs.pop('fromfile',False)
        if not fromfile:
            self._meta_data, self.NCL, self._avg_data,\
            self.QuadAnalDF,self.shape = self._quad_extract(*args,**kwargs)
        else:
            self._meta_data, self.NCL, self._avg_data,\
            self.QuadAnalDF,self.shape = self._hdf_extract(*args,**kwargs)
    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Quad_Anal'

        self._meta_data.save_hdf(file_name,write_mode,base_name+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',base_name+'/avg_data')
        self.QuadAnalDF.to_hdf(file_name,key=base_name+'/QuadAnalDF',mode='a')

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    @staticmethod
    def _quadrant_extract(fluctDF,PhyTime,NCL):
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)
        if len(set([x[0] for x in fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in fluctDF.index]))[0]

        u_array=fluctDF[PhyTime,'u']
        v_array=fluctDF[PhyTime,'v']

        u_array_isneg=u_array<0
        v_array_isneg=v_array<0

        quadrant_array = np.zeros_like(v_array_isneg,dtype='i4')

        for i in range(1,5): #determining quadrant
            if i ==1:
                quadrant_array_temp = np.logical_and(~u_array_isneg,~v_array_isneg)#not fluct_u_isneg and not fluct_v_isneg
                quadrant_array += quadrant_array_temp*1
            elif i==2:
                quadrant_array_temp = np.logical_and(u_array_isneg,~v_array_isneg)#not fluct_u_isneg and fluct_v_isneg
                quadrant_array += quadrant_array_temp*2
            elif i==3:
                quadrant_array_temp =  np.logical_and(u_array_isneg,v_array_isneg)
                quadrant_array += quadrant_array_temp*3
            elif i==4:
                quadrant_array_temp =  np.logical_and(~u_array_isneg,v_array_isneg)#fluct_u_isneg and not fluct_v_isneg
                quadrant_array += quadrant_array_temp*4

        assert(quadrant_array.all()<=4 and quadrant_array.all()>=1)  
        fluct_uv=u_array*v_array 

        return fluct_uv, quadrant_array 

    def line_plot(self,h_list,coord_list,prop_dir,x_vals=0,y_mode='half_channel',norm=False,fig='',ax='',**kwargs):
        assert x_vals is None or not hasattr(x_vals,'__iter__')
        kwargs['squeeze'] = False
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [12,5*len(coord_list)]
            else:
                warnings.warn('Figure size algorithm overidden', stacklevel=2)
            fig, ax = cplt.subplots(4,len(coord_list),**kwargs)
        elif not ax:
            ax = fig.subplots(4,len(coord_list),**kwargs)
        if prop_dir =='y':
            index = [self._avg_data._return_index(x) for x in coord_list]

        elif prop_dir == 'x':
            index = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                        coord_list,x_vals,y_mode)
            if x_vals is not None:
                index=list(itertools.chain(*index))
        else:
            raise ValueError("The propagation direction of the quadrant analysis must be `x' or `y'")
        if norm: 
            avg_time = list(set([x[0] for x in self._avg_data.UU_tensorDF.index]))[0]
            uv=self._avg_data.UU_tensorDF[avg_time,'uv']

        coords = self._meta_data.CoordDF[prop_dir]

        unit="x/\delta"if prop_dir =='y' else "y/\delta" if y_mode=='half_channel' \
                else "\delta_u" if y_mode=='disp_thickness' \
                else "\theta" if y_mode=='mom_thickness' else "y^+" \
                if x_vals is None or x_vals!=0 else "y^{+0}"

        for i in range(1,5):
            for h in h_list:
                quad_anal = self.QuadAnalDF[h,i]
                if norm:
                    quad_anal/=uv
                for j in range(len(coord_list)):
                    if x_vals is None and prop_dir=='x':
                        quad_anal_index= np.zeros(self.shape[1])
                        for k in range(self.shape[1]):
                            quad_anal_index[k]=quad_anal[index[k][j],k]
                    else:
                        quad_anal_index=quad_anal[index[j],:] if prop_dir == 'x' else quad_anal[:,index[j]].T
                    ax[i-1,j].cplot(coords,quad_anal_index,label=r"$h=%.5g$"%h)
                    ax[i-1,j].set_xlabel(r"$%s/\delta$"%prop_dir)# ,fontsize=20)
                    ax[i-1,j].set_ylabel(r"$Q%d$"%i)# ,fontsize=20)
                    ax[i-1,j].set_title(r"$%s=%.5g$"%(unit,coord_list[j]),loc='left')# ,fontsize=16)
                    ax[i-1,j].toggle_default_line_markers()

        ncol = 4 if len(h_list)>3 else len(h_list)
        ax[0,0].clegend(vertical=False,ncol=ncol)
        ax[0,0].get_gridspec().tight_layout(fig)

        fig.tight_layout()
        return fig, ax

class CHAPSim_joint_PDF_base():
    _module = sys.modules[__module__]
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
