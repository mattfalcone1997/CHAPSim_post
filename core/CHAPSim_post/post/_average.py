"""
# _averaged.py
File contains the implementation of the classes to process the 
averaged results from the CHAPSim DNS solver 
"""

import numpy as np

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

import os
import gc
import itertools
import copy
from abc import ABC, abstractmethod, abstractproperty

import CHAPSim_post as cp

from CHAPSim_post.utils import docstring, indexing, misc_utils

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd


from ._meta import CHAPSim_meta
from ._common import Common

_meta_class = CHAPSim_meta

class CHAPSim_AVG_base(Common,ABC):

    @docstring.copy_fromattr("_extract_avg")
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        if fromfile:
            self._hdf_extract(*args,**kwargs)
        else:
            self._extract_avg(*args,**kwargs)

    def _extract_file_io(self,PhyTime,path_to_folder,abs_path):
        instant = "%0.9E" % PhyTime

        full_path = misc_utils.check_paths(path_to_folder,'2_averaged_rawdata',
                                                            '2_averagd_D')

        file_string = "DNS_perioz_AVERAGD_T" + instant + "_FLOW.D"
        
        if not abs_path:
            file_path = os.path.abspath(os.path.join(full_path, file_string))
        else:
            file_path = os.path.join(full_path, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)

        int_info = np.fromfile(file,dtype='int32',count=4)    
        
        NCL1 = int_info[0]
        NCL2 = int_info[1]
        NSTATIS = int_info[3]
        
        dummy_size = NCL1*NCL2*50*21
        r_info = np.fromfile(file,dtype='float64',count=3)
        PhyTime = r_info[0]

        AVG_info = np.zeros(dummy_size)
        AVG_info = np.fromfile(file,dtype='float64',count=dummy_size)

        file.close()
        return AVG_info, NSTATIS, PhyTime, [NCL1,NCL2]


    def _extract_file_tg(self,PhyTime,path_to_folder,abs_path):
        instant = "%0.9E" % PhyTime
        
        full_path = misc_utils.check_paths(path_to_folder,'2_averaged_rawdata',
                                                            '2_averagd_D')


        file_string = "DNS_perixz_AVERAGD_T" + instant + "_FLOW.D"
        
        if not abs_path:
            file_path = os.path.abspath(os.path.join(full_path, file_string))
        else:
            file_path = os.path.join(full_path, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)
        int_info = np.fromfile(file,dtype='int32',count=4)    
        
        NCL2 = int_info[0]
        NSZ = int_info[1]
        ITERG = int_info[2]
        NSTATIS = int_info[3]
        dummy_size = NCL2*NSZ
        r_info = np.fromfile(file,dtype='float64',count=3)
        
        PhyTime = r_info[0]
        AVG_info = np.zeros(dummy_size)
        AVG_info = np.fromfile(file,dtype='float64',count=dummy_size)

        AVG_info = AVG_info.reshape(NSZ,NCL2)

        file.close()
        
        return AVG_info, NSTATIS, PhyTime, NCL2
    @abstractmethod
    def _extract_avg(self,*args,**kwargs):
        raise NotImplementedError
    
    @property
    def shape(self):
        avg_index = self.flow_AVGDF.index[0]
        return self.flow_AVGDF[avg_index].shape

    @abstractproperty
    def times(self):
        pass

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    @docstring.copy_fromattr("_hdf_extract")
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)
        
    def save_hdf(self,file_name,write_mode,key=None):
        """
        Saving the CHAPSim_AVG classes to a file in hdf5 file format

        Parameters
        ----------
        file_name : str, path-like
            File path of the resulting hdf5 file
        write_mode : str
            Mode of file opening the file must be able to modify the file. Passed to the h5py.File method
        key : str (path-like), optional
            Location in the hdf file, by default it is the name of the class
        """
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,mode=write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        hdf_obj.create_dataset("NCL",data=self.NCL)

        self._meta_data.save_hdf(file_name,'a',key=key+'/meta_data')
        self.flow_AVGDF.to_hdf(file_name,key=key+'/flow_AVGDF',mode='a')
        self.PU_vectorDF.to_hdf(file_name,key=key+'/PU_vectorDF',mode='a')
        self.UU_tensorDF.to_hdf(file_name,key=key+'/UU_tensorDF',mode='a')
        self.UUU_tensorDF.to_hdf(file_name,key=key+'/UUU_tensorDF',mode='a')
        self.Velo_grad_tensorDF.to_hdf(file_name,key=key+'/Velo_grad_tensorDF',mode='a')
        self.PR_Velo_grad_tensorDF.to_hdf(file_name,key=key+'/PR_Velo_grad_tensorDF',mode='a')
        self.DUDX2_tensorDF.to_hdf(file_name,key=key+'/DUDX2_tensorDF',mode='a')

    def _hdf_extract(self,file_name,key=None):
        """
        Creates class by extracting data from the hdf5 file

        Parameters
        ----------
        file_name : str, path-like
            File path of the resulting hdf5 file
        write_mode : str
            Mode of file opening the file must be able to modify the file. Passed to the h5py.File method
        key : str (path-like), optional
            Location in the hdf file, by default it is the name of the class
        """
        if key is None:
            key = self.__class__.__name__

        misc_utils.check_path_exists(file_name,True)

        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module._meta_class.from_hdf(file_name,key+'/meta_data')

        self.flow_AVGDF = cd.datastruct.from_hdf(file_name,key=key+'/flow_AVGDF')
        self.PU_vectorDF = cd.datastruct.from_hdf(file_name,key=key+'/PU_vectorDF')
        self.UU_tensorDF = cd.datastruct.from_hdf(file_name,key=key+'/UU_tensorDF')
        self.UUU_tensorDF = cd.datastruct.from_hdf(file_name,key=key+'/UUU_tensorDF')
        self.Velo_grad_tensorDF = cd.datastruct.from_hdf(file_name,key=key+'/Velo_grad_tensorDF')
        self.PR_Velo_grad_tensorDF = cd.datastruct.from_hdf(file_name,key=key+'/PR_Velo_grad_tensorDF')
        self.DUDX2_tensorDF = cd.datastruct.from_hdf(file_name,key=key+'/DUDX2_tensorDF')

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

    @abstractmethod
    def _AVG_extract(self, *args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _return_index(self,*args,**kwargs):
        raise NotImplementedError
    
    @abstractmethod
    def _return_xaxis(self,*args,**kwargs):
        raise NotImplementedError


    def _wall_unit_calc(self,PhyTime):
        
        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star

        REN = self.metaDF['REN']
        
        tau_w = self._tau_calc(PhyTime)
    
        u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
        delta_v_star = (nu_star/u_tau_star)/REN
        return u_tau_star, delta_v_star

    def _y_plus_calc(self,PhyTime):

        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus_shape=(self.shape[1],int(self.NCL[1]*0.5))
        y_plus = np.zeros(y_plus_shape)
        y_coord = self.CoordDF['y'][:int(self.NCL[1]*0.5)]
        for i in range(len(delta_v_star)):
            y_plus[i] = (1-abs(y_coord))/delta_v_star[i]
        return y_plus

    def _int_thickness_calc(self,PhyTime):

        U0_index = int(np.floor(self.NCL[1]*0.5))
        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']

        U0 = U_mean[U0_index]
        theta_integrand = np.zeros_like(U_mean)
        delta_integrand = np.zeros_like(U_mean)
        mom_thickness = np.zeros(self.shape[1])
        disp_thickness = np.zeros(self.shape[1])

        for i, _ in enumerate(theta_integrand):
            theta_integrand[i] = (U_mean[i]/U0)*(1 - U_mean[i]/U0)
            delta_integrand[i] = 1 - U_mean[i]/U0

        for j in range(self.shape[1]):
            mom_thickness[j] = 0.5*integrate_simps(theta_integrand[:,j],y_coords)
            disp_thickness[j] = 0.5*integrate_simps(delta_integrand[:,j],y_coords)
        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor
    
    def _plot_shape_factor(self,*arg,fig=None,ax=None,line_kw=None,**kwargs):
        
        _, _, shape_factor = self._int_thickness_calc(*arg)
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        xaxis = self._return_xaxis()
        line_kw = cplt.update_line_kw(line_kw,label = r"$H$")
        ax.cplot(xaxis,shape_factor,**line_kw)
        ax.set_ylabel(r"$H$")
        fig.tight_layout()

        return fig, ax

    def plot_mom_thickness(self,*arg,fig=None,ax=None,line_kw=None,**kwargs):


        _, theta, _ = self._int_thickness_calc(*arg)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        xaxis = self._return_xaxis()

        line_kw = cplt.update_line_kw(line_kw,label=r"$\theta$")
        ax.cplot(xaxis,theta,**line_kw)
        ax.set_ylabel(r"$\theta$")
        fig.tight_layout()

        return fig, ax

    @docstring.sub
    def plot_disp_thickness(self,*arg,fig=None,ax=None,line_kw=None,**kwargs):

        delta, _, _ = self._int_thickness_calc(*arg)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        xaxis = self._return_xaxis()

        line_kw = cplt.update_line_kw(line_kw,label=r"$\delta$")
        ax.cplot(xaxis,delta,**line_kw)
        ax.set_ylabel(r"$\delta^*$")
        fig.tight_layout()

        return fig, ax

    def _avg_line_plot(self,x_vals, PhyTime,comp,fig=None,ax=None,line_kw=None,**kwargs):

        x_vals = misc_utils.check_list_vals(x_vals)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw = cplt.update_line_kw(line_kw)
        y_coords = self.CoordDF['y']
        for x in x_vals:
            index = self._return_index(x)
            velo_mean=self.flow_AVGDF[PhyTime,comp][:,index]
            ax.cplot(y_coords,velo_mean,**line_kw)

        return fig, ax        

    def plot_mean_flow(self,x_vals,comp,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        fig, ax = self._avg_line_plot(x_vals, PhyTime,comp,fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        # ax.set_xlabel(r"$y^*$")
        x_label = self.Domain.in_to_out(r"$y^*$")
        y_label = self.Domain.in_to_out(r"$\bar{u}$")
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.relim()
        ax.autoscale_view()

        return fig, ax
    
    def plot_near_wall(self,x_vals,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        fig, ax = self._avg_line_plot(x_vals,PhyTime,'u',fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        lines = ax.get_lines()[-len(x_vals):]
        u_tau_star, _ = self._wall_unit_calc(PhyTime)
        y_plus = self._y_plus_calc(PhyTime)
        
        Y_extent = int(self.shape[0]*0.5) if self.metaDF['iCase'] != 5\
                        else self.shape[0]

        for line,x_val in zip(lines,x_vals):
            x_loc = self._return_index(x_val)
            y_data = line.get_ydata()
            y_data = y_data[:Y_extent]/u_tau_star[x_loc]

            line.set_ydata(y_data)
            line.set_xdata(y_plus[x_loc])
            ylim = ax.get_ylim()[1]
            ax.set_ylim(top=1.1*max(ylim,np.amax(y_data)))
        
        line_kw = cplt.update_line_kw(line_kw,color='red',linestyle='--',
                                      label=r"$\bar{u}^+=y^+$")
        ax.cplot(y_plus[0],y_plus[0],**line_kw)
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")
        ax.set_xscale('log')

        return fig, ax

    def plot_Reynolds(self,comp_uu,x_val,PhyTime,norm=None,Y_plus=True,fig=None,ax=None,line_kw=None,**kwargs):

        if comp_uu not in self.UU_tensorDF.inner_index:
            comp_uu = comp_uu[::-1]
            if not comp_uu not in self.UU_tensorDF.inner_index:
                msg = "Reynolds stress component %s not found"%comp_uu
                raise ValueError(msg)   
        
        x_val = misc_utils.check_list_vals(x_val)

        x_loc = [self._return_index(x) for x in x_val]
        uu = self.UU_tensorDF[PhyTime,comp_uu][:int(self.NCL[1]/2)].copy()
        
        if comp_uu == 'uv':
            uu *= -1.0

        y_coord = self.CoordDF['y'][:int(self.NCL[1]/2)]

        u_tau_star, delta_v_star = self._wall_unit_calc(PhyTime)
        if norm=='wall':
            for i in range(self.shape[1]):
                uu[:,i] = uu[:,i]/(u_tau_star[i]*u_tau_star[i])

        elif norm=='local-bulk':
            velo_bulk=self._bulk_velo_calc(PhyTime)
            for i in range(self.shape[1]):
                uu[:,i] = uu[:,i]/(velo_bulk[i]*velo_bulk[i])

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw = cplt.update_line_kw(line_kw)

        for x in x_loc:
            if Y_plus:
                y_coord_local = (1-np.abs(y_coord))/delta_v_star[x]
            else:
                y_coord_local = y_coord

            label=r"$x/\delta =%.2g$" % self.CoordDF['x'][x]
            label = self.Domain.in_to_out(label)
            ax.cplot(y_coord_local,uu[:,x],label=label,**line_kw)

        uu_label = self.Domain.in_to_out(r"%s'%s'"%tuple(comp_uu))

        denom = r"u_\tau" if norm == 'wall' else r"U_b" if norm == 'local-bulk' else r'U_{b0}'
        sign = "-" if comp_uu == 'uv' else ""

        AVG_style = misc_utils.GetStyle_func()
        y_label = r"$%s %s/%s^2$"%(sign,AVG_style(uu_label),denom)

        ax.set_ylabel(y_label)# ,fontsize=20)
                    
        if Y_plus:
            ax.set_xlabel(r"$y^+$")
        else:
            x_label = self.Domain.in_to_out(r"$y/\delta$")
            ax.set_xlabel(x_label)
        
        fig.tight_layout()
        
        return fig, ax

    def plot_Reynolds_x(self,comp_uu,y_vals_list,Y_plus=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        if comp_uu not in self.UU_tensorDF.inner_index:
            comp_uu = comp_uu[::-1]
            if not comp_uu not in self.UU_tensorDF.inner_index:
                msg = "Reynolds stress component %s not found"%comp_uu
                raise ValueError(msg)   
        
        if y_vals_list != 'max':
            if Y_plus:
                y_index = indexing.Y_plus_index_calc(self, self.CoordDF, y_vals_list)
            else:
                y_index = indexing.coord_index_calc(self.CoordDF,'y',y_vals_list)
            rms_vals = self.UU_tensorDF[PhyTime,comp_uu].copy()[y_index]
        else:
            y_index = [None]
            rms_vals = self.UU_tensorDF[PhyTime,comp_uu].copy()
            rms_vals = np.amax(rms_vals,axis=0)

        xaxis = self._return_xaxis()


        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        uu_label = self.Domain.in_to_out(r"%s'%s'"%tuple(comp_uu))

        line_kw = cplt.update_line_kw(line_kw)

        AVG_style = misc_utils.GetStyle_func()

        if y_vals_list != 'max':

            y_vals_list = indexing.ycoords_from_coords(self,y_vals_list,mode='wall')[0]
            for i,y_val in enumerate(y_vals_list):
                ax.cplot(xaxis,rms_vals[i],label=r"$y^+=%.2g$"% y_val,**line_kw)

            ncol = cplt.get_legend_ncols(len(y_index))
            ax.clegend(vertical=False,ncol=ncol)            
        else:
            ax.cplot(xaxis,rms_vals,label=r"$(%s/U_{b0}^2)_{max}$"%AVG_style(uu_label),**line_kw)
        
        ax.set_ylabel(r"$%s/U_{b0}^2$"%AVG_style(uu_label))
        
        fig.tight_layout()
        return fig, ax

    def _bulk_velo_calc(self,PhyTime):
            
        u_velo = self.flow_AVGDF[PhyTime,'u'].squeeze()
        ycoords = self.CoordDF['y']

        bulk_velo = 0.5*integrate_simps(u_velo,ycoords,axis=0)
            
        return bulk_velo

    def plot_bulk_velocity(self,PhyTime,fig=None,ax=None,line_kw=None,**kwargs):

        bulk_velo = self._bulk_velo_calc(PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        xaxis = self._return_xaxis()
        line_kw = cplt.update_line_kw(line_kw,label=r"$U_{b0}$")

        ax.cplot(xaxis,bulk_velo,**line_kw)
        ax.set_ylabel(r"$U_b^*$")
        return fig, ax

    def _tau_calc(self,PhyTime):
        
        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        for i in range(self.shape[1]):
            tau_star[i] = mu_star*(u_velo[0,i]-0.0)/(ycoords[0]--1.0)
    
        return tau_star

    def plot_skin_friction(self,PhyTime,fig=None,ax=None,line_kw=None,**kwargs):

        rho_star = 1.0
        REN = self.metaDF['REN']
        tau_star = self._tau_calc(PhyTime)
        bulk_velo = self._bulk_velo_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/REN)*tau_star
        xaxis = self._return_xaxis()

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw,label=r"$C_f$")
        ax.cplot(xaxis,skin_friction,**line_kw)
        ax.set_ylabel(r"$C_f$")
        
        fig.tight_layout()
        return fig, ax

    def plot_eddy_visc(self,x_val,PhyTime,Y_plus=True,Y_plus_max=100,fig=None,ax=None,line_kw=None,**kwargs):
        
        x_val = misc_utils.check_list_vals(x_val)
        
        x_loc = [self._return_index(x) for x in x_val]
    
        uv = self.UU_tensorDF[PhyTime,'uv']
        dUdy = self.Velo_grad_tensorDF[PhyTime,'uy']
        dVdx = self.Velo_grad_tensorDF[PhyTime,'vx']
        REN = self.metaDF['REN']

        mu_t = -uv*REN/(dUdy + dVdx)
        y_coord = self._meta_data.CoordDF['y']
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)

        for x in x_loc:
            if Y_plus:
                y_coord_local = self._y_plus_calc(PhyTime)[x].squeeze()                
                mu_t_local = mu_t[:int(y_coord.size*0.5),x]
            else:
                y_coord_local = y_coord.squeeze()
                mu_t_local = mu_t[:,x]

            ax.cplot(y_coord_local,mu_t_local,**line_kw)
            if Y_plus:
                ax.set_xlabel(r"$y^+$")
                ax.set_xlim([0,Y_plus_max])
            else:
                x_label = self.Domain.in_to_out(r"$y/\delta$")
                ax.set_xlabel(x_label)
                ax.set_xlim([-1,-0.1])

            ax.set_ylim([-0.5,max(mu_t_local)*1.2])
            ax.set_ylabel(r"$\mu_t/\mu_0$")

        return fig, ax

    def __iadd__(self,other_avg):
        pass

class CHAPSim_AVG_io(CHAPSim_AVG_base):
    tgpost = False
    def _extract_avg(self,time,meta_data=None,path_to_folder=".",time0=None,abs_path=True):
        """
        Instantiates an instance of the CHPSim_AVG_io class from the result data

        Parameters
        ----------
        time : float or list or floats
            Physcial time to be extracted
        meta_data : CHAPSim_meta, optional
            metadata class provided to the class, by default None
        path_to_folder : str, optional
            path to the results folder, by default "."
        time0 : float, optional
            A previous time to average between, by default None
        abs_path : bool, optional
            Whether the path provided is absolute or relative, by default True
        """

        if meta_data is None:
            meta_data = self._module._meta_class(path_to_folder,abs_path,False)

        self._meta_data = meta_data.copy()
        time = misc_utils.check_list_vals(time)

        for PhyTime in time:
            if 'DF_list' not in locals():
                DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
            else:
                DF_temp=[]
                local_DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                for i, local_DF in enumerate(DF_list):
                    DF_list[i].concat(local_DF_list[i])

        DF_list=self._Reverse_decomp(*DF_list)


        self.flow_AVGDF,self.PU_vectorDF,\
        self.UU_tensorDF,self.UUU_tensorDF,\
        self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
        self.DUDX2_tensorDF = DF_list
        

        self._times = list(set([x[0] for x in self.flow_AVGDF.index]))
    
    @property
    def times(self):
        return list(set([x[0] for x in self.flow_AVGDF.index]))

    def _AVG_extract(self,Time_input,time0,path_to_folder,abs_path):

        AVG_info, NSTATIS1, PhyTime, NCL = self._extract_file_io(Time_input,path_to_folder,abs_path)

        if time0 is not None:
            AVG_info0, NSTATIS0,_,_ = self._extract_file_io(time0,path_to_folder,abs_path)
            AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)
            del AVG_info0

        (NCL1, NCL2) = NCL
        AVG_info = AVG_info.reshape(21,50,NCL2,NCL1)
            
        #Velo_AVG = np.zeros((3,NCL2,NCL1))
        Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        Pr_Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        DUDX2_tensor = np.zeros((81,NCL2,NCL1))
        
        for i in range(3):
            for j in range(3):
                Velo_grad_tensor[i*3+j,:,:] = AVG_info[6+j,i,:,:]
                Pr_Velo_grad_tensor[i*3+j,:,:] = AVG_info[9+j,i,:,:]
                
        for i in range(9):
            for j in range(9):
                DUDX2_tensor[i*9+j] = AVG_info[12+j,i,:,:] 
        
        flow_AVG = AVG_info[0,:4,:,:].copy()
        PU_vector = AVG_info[2,:3,:,:].copy()
        UU_tensor = AVG_info[3,:6,:,:].copy()
        UUU_tensor = AVG_info[5,:10,:,:].copy()

        del AVG_info; gc.collect()



        Phy_string = '%.9g' % PhyTime
        flow_index = [[Phy_string]*4,['u','v','w','P']]
        vector_index = [[Phy_string]*3,['u','v','w']]
        sym_2_tensor_index = [[Phy_string]*6,['uu','uv','uw','vv','vw','ww']]
        sym_3_tensor_index = [[Phy_string]*10,['uuu','uuv','uuw','uvv',\
                                'uvw','uww','vvv','vvw','vww','www']]
        tensor_2_index = [[Phy_string]*9,['ux','uy','uz','vx','vy','vz',\
                                         'wx','wy','wz']]
        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']

        dudx_list = list(itertools.product(du_list,dx_list))
        dudx_list = ["".join(dudx) for dudx in dudx_list]
        comp_string_list = list(itertools.product(dudx_list,dudx_list))
        comp_string_list = ["".join(comp_string) for comp_string in comp_string_list]

        shape = [NCL2,NCL1]
        tensor_4_index = [[Phy_string]*81,comp_string_list]

        flow_AVGDF = cd.datastruct(flow_AVG,index=flow_index,copy=False) 
        PU_vectorDF = cd.datastruct(PU_vector,index=vector_index,copy=False) 
        UU_tensorDF = cd.datastruct(UU_tensor,index=sym_2_tensor_index,copy=False) 
        UUU_tensorDF = cd.datastruct(UUU_tensor,index=sym_3_tensor_index,copy=False) 
        Velo_grad_tensorDF = cd.datastruct(Velo_grad_tensor,index=tensor_2_index,copy=False) 
        PR_Velo_grad_tensorDF = cd.datastruct(Pr_Velo_grad_tensor,index=tensor_2_index,copy=False) 
        DUDX2_tensorDF = cd.datastruct(DUDX2_tensor,index=tensor_4_index,copy=False) 

        if cp.rcParams["SymmetryAVG"] and self.metaDF['iCase'] == 1:
            flow_AVGDF = 0.5*(flow_AVGDF + \
                            flow_AVGDF.symmetrify(dim=0))
            PU_vectorDF = 0.5*(PU_vectorDF +\
                            PU_vectorDF.symmetrify(dim=0))
            UU_tensorDF = 0.5*(UU_tensorDF + \
                            UU_tensorDF.symmetrify(dim=0))
            UUU_tensorDF = 0.5*(UUU_tensorDF + \
                            UUU_tensorDF.symmetrify(dim=0))
            Velo_grad_tensorDF = 0.5*(Velo_grad_tensorDF + \
                            Velo_grad_tensorDF.symmetrify(dim=0))
            PR_Velo_grad_tensorDF = 0.5*(PR_Velo_grad_tensorDF + \
                            PR_Velo_grad_tensorDF.symmetrify(dim=0))
            DUDX2_tensorDF = 0.5*(DUDX2_tensorDF + \
                            DUDX2_tensorDF.symmetrify(dim=0))
        
        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]

    def _return_index(self,x_val):
        return indexing.coord_index_calc(self.CoordDF,'x',x_val)

    def _return_xaxis(self):
        return self.CoordDF['x']

    def check_PhyTime(self,PhyTime):
        """
        Checks whether the physical time provided is valid
        and if not whether it can be recovered.

        Parameters
        ----------
        PhyTime : float or int
            Input Physical time to be checked

        Returns
        -------
        float or int
            Correct or corrected physical time
        """

        warn_msg = f"PhyTime invalid ({PhyTime}), variable being set to only PhyTime present in datastruct"
        err_msg = "PhyTime provided is not in the CHAPSim_AVG datastruct, recovery impossible"
        
        return self.flow_AVGDF.check_outer(PhyTime,err_msg,warn_msg)

    @docstring.sub
    def int_thickness_calc(self, PhyTime=None):
        """
        Calculates the integral thicknesses and shape factor 

        Parameters
        ----------
        PhyTime : float or int, optional
            Physical ime, by default None

        Returns
        -------
        %(ndarray)s:
            Displacement thickness
        %(ndarray)s:
            Momentum thickness
        %(ndarray)s:
            Shape factor
        """

        PhyTime = self.check_PhyTime(PhyTime)
        return super()._int_thickness_calc(PhyTime)

    @docstring.sub
    def wall_unit_calc(self,PhyTime=None):
        """
        returns arrays for the friction velocity and viscous lengthscale

        Parameters
        ----------
        PhyTime : float or int, optional
            Physical time, by default None

        Returns
        -------
        %(ndarray)s:
            Friction velocity array
        %(ndarray)s:
            Viscous lengthscale array
            
        """
        PhyTime = self.check_PhyTime(PhyTime)
        return self._wall_unit_calc(PhyTime)

    @docstring.sub
    def plot_shape_factor(self, PhyTime=None,fig=None,ax=None,**kwargs):
        """
        Plots the shape factor from the class against the streamwise coordinate

        Parameters
        ----------
        PhyTime : float, optional
            Physical time, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """
        PhyTime = self.check_PhyTime(PhyTime)        
        fig, ax = self._plot_shape_factor(PhyTime,fig=fig,ax=ax,**kwargs)

        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        ax.relim()
        ax.autoscale_view()
        return fig, ax

    @docstring.sub
    def plot_mom_thickness(self, PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the momentum thickness from the class against the streamwise coordinate

        Parameters
        ----------
        PhyTime : float, optional
            Physical time, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        PhyTime = self.check_PhyTime(PhyTime)
        fig, ax = super().plot_mom_thickness(PhyTime,fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        ax.relim()
        ax.autoscale_view()
        return fig, ax

    @docstring.sub
    def plot_disp_thickness(self, PhyTime=None,fig=None,ax=None,**kwargs):
        """
        Plots the displacement thickness from the class against the streamwise coordinate

        Parameters
        ----------
        PhyTime : float, optional
            Physical time, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        PhyTime = self.check_PhyTime(PhyTime)        
        fig, ax = super().plot_disp_thickness(PhyTime,fig=fig,ax=ax,**kwargs)

        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        ax.relim()
        ax.autoscale_view()
        return fig, ax

    @docstring.sub
    def plot_Reynolds(self,comp,x_vals,PhyTime=None,norm=None,Y_plus=True,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots the wall-normal distribution of the Reynolds stresses

        Parameters
        ----------
        comp : str
            Component of the Reynolds stress 
            tensor to be process
        x_vals : list or int/float
            streamwise locations to be plotted
        PhyTime : int or float, optional
            Physical time, by default None
        norm : str, optional
            how the Reynolds stresses are normalised. if 'wall' they are 
            local inner scaled; 'local-bulk' by the local bulk velocity
            None then default normalisation , by default None
        Y_plus : bool, optional
            Whether y coordinate is local wall normalised or not, by default True
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """ 
        
        PhyTime = self.check_PhyTime(PhyTime)
        fig, ax = super().plot_Reynolds(comp,x_vals,PhyTime,
                                        norm=norm,Y_plus=Y_plus,
                                        fig=fig,ax=ax,line_kw=line_kw,**kwargs)

        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)


        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            x_label = self.Domain.in_to_out(r"$x/\delta=%.2g$"%float(x))
            line.set_label(x_label)

        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax        
    
    @docstring.sub
    def plot_Reynolds_x(self,comp,y_vals_list,Y_plus=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Plots a Reynolds stress component against the streamwise coordinate or time

        Parameters
        ----------
        comp : str
            Component of the Reynolds stress 
            tensor to be process

        y_vals_list : list or "max"
            contains y coordinates to be processed or max if the wall
            normal maximum is desired
        Y_plus : bool, optional
            Whether the coordinates in y_vals list are local inner-scaled,
            not relevant if y_vals_list is `max'. by default True
        PhyTime : (int or float), optional
            Physical time, by default None
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        
        PhyTime = self.check_PhyTime(PhyTime)        
        fig, ax = super().plot_Reynolds_x(comp,y_vals_list,Y_plus=Y_plus,
                                            PhyTime=PhyTime,fig=fig,ax=ax,line_kw=line_kw,**kwargs)
        
        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    @docstring.sub
    def bulk_velo_calc(self,PhyTime=None):
        """
        Method to calculate the bulk velocity against the streamwise coordinate

        Parameters
        ----------
        PhyTime : (int,float), optional
            Physical time, by default None

        Returns
        -------
        %(ndarray)s
            array containing the bulk velocity
        """

        PhyTime = self.check_PhyTime(PhyTime)
        return self._bulk_velo_calc(PhyTime)

    @docstring.sub
    def plot_bulk_velocity(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        """
        Method to plot the bulk velocity variation in the streamwise 
        direction

        Parameters
        ----------
        PhyTime : (float,int), optional
            Physcial time, by default None
        fig : %(fig)s, optional
            Pre-existing figure instance, by default None
        ax : %(ax)s, optional
            Pre-existing axes instance, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot function, by default None
        
        **kwargs : 
            keyword arguments to be passed to the figure/subplot creation function
        Returns
        -------
        %(fig)s, %(ax)s
            returns the figure and axes instances associated with the bulk velocity
        """

        PhyTime = self.check_PhyTime(PhyTime)        
        fig, ax = super().plot_bulk_velocity(PhyTime,fig,ax,line_kw=line_kw,**kwargs)

        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        ax.relim()
        ax.autoscale_view()
        return fig, ax

    @docstring.sub
    def tau_calc(self,PhyTime=None):
        """
        method to return the wall shear stress array

        Parameters
        ----------
        PhyTime : (float,int), optional
            Physical time, if value is invalid or None the routine will
            attempt recovery, by default None

        Returns
        -------
        %(ndarray)s
            Wall shear stress array 
        """
        PhyTime = self.check_PhyTime(PhyTime)            
        return self._tau_calc(PhyTime)

    @docstring.sub
    def plot_skin_friction(self,PhyTime=None,fig=None,ax=None,**kwargs):
        """
        Plots the skin friction coefficient

        Parameters
        ----------
        PhyTime : [type], optional
            [description], by default None
        fig : %(fig)s, optional
            Pre-existing figure instance, by default None
        ax : %(ax)s, optional
            Pre-existing axes instance, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot function, by default None
        
        **kwargs : 
            keyword arguments to be passed to the figure/subplot creation function
        Returns
        -------
        %(fig)s, %(ax)s
            returns the figure and axes instances associated with the bulk velocity
        """
        
        PhyTime = self.check_PhyTime(PhyTime)        
        fig, ax = super().plot_skin_friction(PhyTime,fig,ax,**kwargs)

        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        ax.relim()
        ax.autoscale_view()
        return fig, ax

    @docstring.sub
    def plot_eddy_visc(self,x_vals,PhyTime=None,Y_plus=True,Y_plus_max=100,fig=None,ax=None,**kwargs):
        """
        Plot the eddy viscosity against the wall-normal coordinate

        Parameters
        ----------
        x_vals : list or int/float
            streamwise locations to be plotted
        PhyTime : int or float, optional
            Physical time, by default None
        norm : str, optional
            how the Reynolds stresses are normalised. if 'wall' they are 
            local inner scaled; 'local-bulk' by the local bulk velocity
            None then default normalisation , by default None
        Y_plus : bool, optional
            Whether y coordinate is local wall normalised or not, by default True
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        PhyTime = self.check_PhyTime(PhyTime)

        fig, ax = super().plot_eddy_visc(x_vals,PhyTime,Y_plus,Y_plus_max,fig,ax,**kwargs)
        
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        
        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines,x_vals):
            x_label = self.Domain.in_to_out(r"$x/\delta=%.1f$"%float(x))
            line.set_label(x_label)

        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)

        return fig, ax

    @docstring.sub
    def plot_mean_flow(self,x_vals,comp='u',PhyTime=None,fig=None,ax=None,*args,**kwargs):
        """
        Plot the mean flow field against the wall-normal coordinate

        Parameters
        ----------
        x_vals : float ot list of floats
            Streamwise locations for plotting
        comp : str
            mean flow component to be plotted
        PhyTime : float, optional
            Physical time, by default None
        fig : %(fig)s, optional
            pre-existing figure, by default None
        ax : %(ax)s, optional
            pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        PhyTime = self.check_PhyTime(PhyTime)

        fig, ax = super().plot_mean_flow(x_vals,comp,PhyTime,fig=fig,ax=ax,**kwargs)
        
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)

        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines,x_vals):
            x_label = self.Domain.in_to_out(r"$x/\delta=%.1f$"%float(x))
            line.set_label(x_label)

        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_near_wall(self,x_vals,PhyTime=None,fig=None,ax=None,**kwargs):
        """
        Plots $\bar{u}^+$ vs $y^+$

        Parameters
        ----------
        x_vals : float ot list of floats
            Streamwise locations for plotting
        PhyTime : float, optional
            Physical time, by default None
        fig : %(fig)s, optional
            pre-existing figure, by default None
        ax : %(ax), optional
            pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """
        
        PhyTime = self.check_PhyTime(PhyTime)
        fig, ax = super().plot_near_wall(x_vals,PhyTime,fig=fig,ax=ax,**kwargs)

        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)

        lines = ax.get_lines()[-len(x_vals)-1:-1]
        for line, x in zip(lines,x_vals):
            x_label = self.Domain.in_to_out(r"$x/\delta=%.2g$"%float(x))
            line.set_label(x_label)

        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_AVG_tg_base(CHAPSim_AVG_base):
    tgpost = True

    @classmethod
    def with_phase_average(cls,*args,**kwargs):
        
        if 'path_to_folder' not in kwargs.keys():
            msg = "keyword argument `path_to_folder' must be present to use this method"
            raise ValueError(msg)
        
        abs_path = kwargs.get('abs_path',True)
        path_to_folder = kwargs['path_to_folder']
        if not isinstance(path_to_folder,(tuple,list)):
            msg = f"To use this method, path_to_folder must be a tuple or a list not a {type(path_to_folder)}"
            raise TypeError(msg)
        
        shift_vals = kwargs.pop("shift_vals",[0]*len(path_to_folder))
        time0 = kwargs.get("time0",None)
        for (i,path),val in zip(enumerate(path_to_folder),shift_vals):
            if i ==0:
                PhyTimes = [x-val for x in misc_utils.time_extract(path,abs_path)]
                if time0 is not None:
                    PhyTimes = list(filter(lambda x: x > (time0-val), PhyTimes))
            else:
                times = [x-val for x in misc_utils.time_extract(path,abs_path)]
                PhyTimes = sorted(set(PhyTimes).intersection(times))
        
        for (i,path),val in zip(enumerate(path_to_folder),shift_vals):

            coe1 = i/(i+1)
            coe2 = 1/(i+1)
            
            kwargs['path_to_folder'] = path
            if path == path_to_folder[0]:
                avg_data_tg_base = cls(*args,**kwargs)
                avg_data_tg_base.shift_times(-val)
                avg_data_tg_base.filter_times(PhyTimes)
            else:
                avg_data_tg_base_new = cls(*args,**kwargs)
                avg_data_tg_base_new.shift_times(-val)
                avg_data_tg_base_new.filter_times(PhyTimes)
                avg_data_tg_base = coe1*avg_data_tg_base + coe2*avg_data_tg_base_new
        
        return avg_data_tg_base

    def _extract_avg(self,PhyTimes,*,meta_data=None,path_to_folder='.',time0=None,abs_path=True):
        """
        Instantiates an instance of the CHPSim_AVG_tg class from the result data
       
        Parameters
        ----------
        time : float or list or floats
            Physcial time to be extracted
        meta_data : CHAPSim_meta, optional
            metadata class provided to the class, by default None
        path_to_folder : str, optional
            path to the results folder, by default "."
        time0 : float, optional
            A previous time to average between, by default None
        abs_path : bool, optional
            Whether the path provided is absolute or relative, by default True
        """

        PhyTimes = misc_utils.check_list_vals(PhyTimes)

        if cp.rcParams['TEST']:
            PhyTimes=PhyTimes[-3:]
        PhyTimes.sort()

        if meta_data is None:
            meta_data = self._module._meta_class(path_to_folder,abs_path,tgpost=True)
        self._meta_data = meta_data
    
        times = ['%.9g' % time for time in PhyTimes]
        for PhyTime in PhyTimes:
            if 'DF_list' not in locals():
                DF_list = self._AVG_extract(PhyTime,path_to_folder,abs_path,self.metaDF,time0)
                
            else:
                local_DF_list = self._AVG_extract(PhyTime,path_to_folder,abs_path,self.metaDF,time0)
                for i, _ in enumerate(DF_list):
                    DF_list[i].append(local_DF_list[i],axis=0)            

        self._times=times

        for i,_ in enumerate(DF_list):
            for key, _ in DF_list[i]:
                DF_list[i][key] = DF_list[i][key].T


        DF_list=self._Reverse_decomp(*DF_list)

        self.flow_AVGDF,self.PU_vectorDF,\
        self.UU_tensorDF,self.UUU_tensorDF,\
        self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
        self.DUDX2_tensorDF = DF_list

    def shift_times(self,val):
        times = sorted([float(x) for x in self._times])
        self._times = ["%.9g"%(x+float(val)) for x in times]

    def filter_times(self,times):
        filter_times = ["%.9g"% time for time in times]
        time_list = list(set(self._times).intersection(set(filter_times)))
        time_list = sorted(float(x) for x in time_list)
        # time_list = ["%.9g"%x for x in time_list]
        index_list = [self._return_index(x) for x in time_list]
        
        self._times = ["%.9g"% time for time in time_list]

        for index in self.flow_AVGDF.index:
            self.flow_AVGDF[index] = self.flow_AVGDF[index][:,index_list]

        for index in self.UU_tensorDF.index:
            self.UU_tensorDF[index] = self.UU_tensorDF[index][:,index_list]
        
        for index in self.UUU_tensorDF.index:
            self.UUU_tensorDF[index] = self.UUU_tensorDF[index][:,index_list]

        for index in self.PU_vectorDF.index:
            self.PU_vectorDF[index] = self.PU_vectorDF[index][:,index_list]

        for index in self.Velo_grad_tensorDF.index:
            self.Velo_grad_tensorDF[index] = self.Velo_grad_tensorDF[index][:,index_list]
        
        for index in self.PR_Velo_grad_tensorDF.index:
            self.PR_Velo_grad_tensorDF[index] = self.PR_Velo_grad_tensorDF[index][:,index_list]

        for index in self.DUDX2_tensorDF.index:
            self.DUDX2_tensorDF[index] = self.DUDX2_tensorDF[index][:,index_list]

    @property
    def times(self):
        print(self._times)
        times = [float(time) for time in self._times]
        return np.array(times)

    def get_times(self):
        self._times = ["%.9g"%float(time) for time in self._times]
        return self._times

    def __mul__(self,val):
        if isinstance(val,(float,int)):
            copy_self = self.copy()

            copy_self.flow_AVGDF = copy_self.flow_AVGDF*val
            copy_self.UU_tensorDF = copy_self.UU_tensorDF*val
            copy_self.UUU_tensorDF = copy_self.UUU_tensorDF*val
            copy_self.PU_vectorDF = copy_self.PU_vectorDF*val
            copy_self.Velo_grad_tensorDF = copy_self.Velo_grad_tensorDF*val
            copy_self.PR_Velo_grad_tensorDF = copy_self.PR_Velo_grad_tensorDF*val
            copy_self.DUDX2_tensorDF = copy_self.DUDX2_tensorDF*val

            return copy_self
        
        else:
            msg = f"This class can only be multiplies by real scalar not {type(val)}"
            raise TypeError(msg)

    def __rmul__(self,val):
        return self.__mul__(val)
    
    def __add__(self,other_avg_tg):
        if isinstance(other_avg_tg,CHAPSim_AVG_tg_base):
            copy_self = self.copy()

            copy_self.flow_AVGDF = copy_self.flow_AVGDF + other_avg_tg.flow_AVGDF
            copy_self.UU_tensorDF = copy_self.UU_tensorDF + other_avg_tg.UU_tensorDF
            copy_self.UUU_tensorDF = copy_self.UUU_tensorDF + other_avg_tg.UUU_tensorDF
            copy_self.PU_vectorDF = copy_self.PU_vectorDF + other_avg_tg.PU_vectorDF
            copy_self.Velo_grad_tensorDF = copy_self.Velo_grad_tensorDF + other_avg_tg.Velo_grad_tensorDF
            copy_self.PR_Velo_grad_tensorDF = copy_self.PR_Velo_grad_tensorDF + other_avg_tg.PR_Velo_grad_tensorDF
            copy_self.DUDX2_tensorDF = copy_self.DUDX2_tensorDF + other_avg_tg.DUDX2_tensorDF

            return copy_self
        
        else:
            msg = f"This class can only be added to other {self.__class__} not {type(other_avg_tg)}"
            raise TypeError(msg)

    def _AVG_extract(self,PhyTime,path_to_folder,abs_path,metaDF,time0):

        AVG_info, NSTATIS1, _, _ = self._extract_file_tg(PhyTime,path_to_folder,abs_path)
        
        factor = metaDF['NCL1_tg']*metaDF['NCL3'] if cp.rcParams["dissipation_correction"] else 1.0
        ioflowflg = self.metaDF['iDomain'] in [2,3]

        if ioflowflg and time0:
            AVG_info0, NSTATIS0, _, _ = self._extract_file_tg(time0,path_to_folder,abs_path)
            AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)

        flow_AVG = AVG_info[:4]
        PU_vector = AVG_info[4:7]
        UU_tensor = AVG_info[7:13]
        UUU_tensor = AVG_info[13:23]
        Velo_grad_tensor = AVG_info[23:32]
        Pr_Velo_grad_tensor = AVG_info[32:41]
        DUDX2_tensor = AVG_info[41:]*factor

        flow_index = [[None]*4,['u','v','w','P']]
        vector_index = [[None]*3,['u','v','w']]
        sym_2_tensor_index = [[None]*6,['uu','uv','uw','vv','vw','ww']]
        sym_3_tensor_index = [[None]*10,['uuu','uuv','uuw','uvv',\
                                'uvw','uww','vvv','vvw','vww','www']]
        tensor_2_index = [[None]*9,['ux','uy','uz','vx','vy','vz',\
                                         'wx','wy','wz']]
        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']

        dudx_list = list(itertools.product(du_list,dx_list))
        dudx_list = ["".join(dudx) for dudx in dudx_list]
        comp_string_list = list(itertools.product(dudx_list,dudx_list))
        comp_string_list = ["".join(comp_string) for comp_string in comp_string_list]
        
        tensor_4_index=[[None]*81,comp_string_list]

        flow_AVGDF = cd.datastruct(flow_AVG,index=flow_index)
        PU_vectorDF = cd.datastruct(PU_vector,index=vector_index)
        UU_tensorDF = cd.datastruct(UU_tensor,index=sym_2_tensor_index)
        UUU_tensorDF = cd.datastruct(UUU_tensor,index=sym_3_tensor_index)
        Velo_grad_tensorDF = cd.datastruct(Velo_grad_tensor,index=tensor_2_index)
        PR_Velo_grad_tensorDF = cd.datastruct(Pr_Velo_grad_tensor,index=tensor_2_index)
        DUDX2_tensorDF = cd.datastruct(DUDX2_tensor,index=tensor_4_index)

        if cp.rcParams["SymmetryAVG"] and self.metaDF['iCase'] == 1:
            flow_AVGDF = 0.5*(flow_AVGDF + \
                            flow_AVGDF.symmetrify(dim=0))
            PU_vectorDF = 0.5*(PU_vectorDF +\
                            PU_vectorDF.symmetrify(dim=0))
            UU_tensorDF = 0.5*(UU_tensorDF + \
                            UU_tensorDF.symmetrify(dim=0))
            UUU_tensorDF = 0.5*(UUU_tensorDF + \
                            UUU_tensorDF.symmetrify(dim=0))
            Velo_grad_tensorDF = 0.5*(Velo_grad_tensorDF + \
                            Velo_grad_tensorDF.symmetrify(dim=0))
            PR_Velo_grad_tensorDF = 0.5*(PR_Velo_grad_tensorDF + \
                            PR_Velo_grad_tensorDF.symmetrify(dim=0))
            DUDX2_tensorDF = 0.5*(DUDX2_tensorDF + \
                            DUDX2_tensorDF.symmetrify(dim=0))

        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]
    
    @docstring.inherit
    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__
        
        misc_utils.check_path_exists(file_name,True)

        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)
        self._times = list(np.char.decode(hdf_obj.attrs["times"]))

        super()._hdf_extract(file_name,key=key)




    @docstring.inherit
    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key =  self.__class__.__name__

        super().save_hdf(file_name,write_mode,key)

        hdf_obj = cd.hdfHandler(file_name,'a',key=key)
        hdf_obj.attrs['times'] = np.array([np.string_(x) for x in self._times])

    def _return_index(self,PhyTime):
        if not isinstance(PhyTime,str):
            PhyTime = "{:.9g}".format(PhyTime)

        if PhyTime not in self.get_times():
            raise ValueError("time %s must be in times"% PhyTime)
        for i in range(len(self.get_times())):
            if PhyTime==self.get_times()[i]:
                return i

    def _return_xaxis(self):
        return self.times

    @docstring.sub
    def wall_unit_calc(self):
        """
        returns arrays for the friction velocity and viscous lengthscale


        Returns
        -------
        %(ndarray)s
            friction velocity array
        %(ndarray)s
            viscous lengthscale array
        """
        return self._wall_unit_calc(None)

    @docstring.sub
    def int_thickness_calc(self):
        """
        Calculates the integral thicknesses and shape factor 

        Returns
        -------
        %(ndarray)s:
            Displacement thickness
        %(ndarray)s:
            Momentum thickness
        %(ndarray)s:
            Shape factor
        """

        PhyTime = None
        return super()._int_thickness_calc(PhyTime)

    @docstring.sub
    def plot_shape_factor(self,fig=None,ax=None,**kwargs):
        """
        Plots the shape factor from the class against the streamwise coordinate

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        PhyTime = None
        fig, ax = self._plot_shape_factor(PhyTime,fig=fig,ax=ax,**kwargs)

        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    @docstring.sub
    def plot_Reynolds(self,comp,PhyTime,norm=None,Y_plus=True,fig=None,ax=None,**kwargs):
        """
        Plots the wall-normal distribution of the Reynolds stresses

        Parameters
        ----------
        comp : str
            Component of the Reynolds stress 
            tensor to be process
        x_vals : list or int/float
            streamwise locations to be plotted
        norm : str, optional
            how the Reynolds stresses are normalised. if 'wall' they are 
            local inner scaled; 'local-bulk' by the local bulk velocity
            None then default normalisation , by default None
        Y_plus : bool, optional
            Whether y coordinate is local wall normalised or not, by default True
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """ 
        

        fig, ax = super().plot_Reynolds(comp,PhyTime,None,
                                        norm=norm,Y_plus=Y_plus,
                                        fig=fig,ax=ax,**kwargs)

        lines = ax.get_lines()[-len(PhyTime):]
        for line,time in zip(lines,PhyTime):
            line.set_label(r"$t^*=%.3g$"%float(time))

        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.relim()
        ax.autoscale_view()

        fig.tight_layout()
        return fig, ax
    @docstring.sub
    def plot_Reynolds_x(self,comp,y_vals_list,Y_plus=True,fig=None,ax=None,**kwargs):
        """
        Plots a Reynolds stress component against the streamwise coordinate or time

        Parameters
        ----------
        comp : str
            Component of the Reynolds stress 
            tensor to be process

        y_vals_list : list or "max"
            contains y coordinates to be processed or max if the wall
            normal maximum is desired
        Y_plus : bool, optional
            Whether the coordinates in y_vals list are local inner-scaled,
            not relevant if y_vals_list is `max'. by default True
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        fig, ax = super().plot_Reynolds_x(comp,y_vals_list,Y_plus=True,
                                            PhyTime=None,fig=fig,ax=ax,**kwargs)
        
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax    

    @docstring.sub
    def bulk_velo_calc(self):
        """
        Method to calculate the bulk velocity against the streamwise coordinate
        
        Returns
        -------
        %(ndarray)s
            array containing the bulk velocity
        """

        return super()._bulk_velo_calc(None)

    @docstring.sub
    def plot_bulk_velocity(self,fig=None,ax=None,**kwargs):
        """
        Method to plot the bulk velocity variation in the streamwise 
        direction

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure instance, by default None
        ax : %(ax)s, optional
            Pre-existing axes instance, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot function, by default None
        
        **kwargs : 
            keyword arguments to be passed to the figure/subplot creation function
        Returns
        -------
        %(fig)s, %(ax)s
            returns the figure and axes instances associated with the bulk velocity
        """

        fig, ax = super().plot_bulk_velocity(None,fig,ax,**kwargs)

        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    @docstring.sub
    def tau_calc(self):
        """
        method to return the wall shear stress array

        Returns
        -------
        %(ndarray)s
            Wall shear stress array 
        """
        return self._tau_calc(None)

    @docstring.sub
    def plot_skin_friction(self,fig=None,ax=None,**kwargs):
        """
        Plots the skin friction coefficient

        Parameters
        ----------
        fig : %(fig)s, optional
            Pre-existing figure instance, by default None
        ax : %(ax)s, optional
            Pre-existing axes instance, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot function, by default None
        
        **kwargs : 
            keyword arguments to be passed to the figure/subplot creation function
        Returns
        -------
        %(fig)s, %(ax)s
            returns the figure and axes instances associated with the bulk velocity
        """

        fig, ax = super().plot_skin_friction(None,fig=fig,ax=ax,**kwargs)

        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self,times,Y_plus=True,Y_plus_max=100,fig=None,ax=None,**kwargs):
        """
        Plot the eddy viscosity against the wall-normal coordinate

        Parameters
        ----------
        x_vals : list or int/float
            streamwise locations to be plotted
        norm : str, optional
            how the Reynolds stresses are normalised. if 'wall' they are 
            local inner scaled; 'local-bulk' by the local bulk velocity
            None then default normalisation , by default None
        Y_plus : bool, optional
            Whether y coordinate is local wall normalised or not, by default True
        fig : %(fig)s, optional
            Pre-existing figure, by default None
        ax : %(ax)s, optional
            Pre-existing axes, by default None
        line_kw : dict, optional
            keyword arguments to be passed to the plot method, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        fig, ax = super().plot_eddy_visc(times,None,Y_plus,Y_plus_max,fig,ax,**kwargs)
        lines = ax.get_lines()[-len(times):]

        for line, time in zip(lines,times):
            line.set_label(r"$t^*=%.3g$" % float(time))


        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)

        return fig, ax
    @docstring.sub
    def plot_mean_flow(self,times,comp='u',fig=None,ax=None,**kwargs):
        """
        Plot the mean flow field against the wall-normal coordinate

        Parameters
        ----------
        x_vals : float ot list of floats
            Streamwise locations for plotting
        comp : str
            mean flow component to be plotted
        fig : %(fig)s, optional
            pre-existing figure, by default None
        ax : %(ax)s, optional
            pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        fig, ax = super().plot_mean_flow(times,comp,PhyTime=None,fig=fig,ax=ax,**kwargs)

        lines = ax.get_lines()[-len(times):]
        for line, time in zip(lines,times):
            line.set_label(r"$t^*=%g$"% float(time))
            
        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        
        return fig, ax

    @docstring.sub
    def plot_near_wall(self,times,fig=None,ax=None,**kwargs):
        """
        Plots $\bar{u}^+$ vs $y^+$

        Parameters
        ----------
        x_vals : float ot list of floats
            Streamwise locations for plotting
        fig : %(fig)s, optional
            pre-existing figure, by default None
        ax : %(ax)s, optional
            pre-existing axes, by default None

        Returns
        -------
        %(fig)s, %(ax)s
            output figure and axes objects
        """

        fig, ax = super().plot_near_wall(times,None,fig=fig,ax=ax,**kwargs)
        line_no=len(times)
        lines = ax.get_lines()[-line_no:]
        for line, time in zip(lines,times):
            line.set_label(r"$t^*=%.3g$"% time)

        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax
        
class CHAPSim_AVG_tg(CHAPSim_AVG_tg_base):
    def _extract_avg(self,path_to_folder='.',time0=None,abs_path=True,*args,**kwargs):
        """
        Instantiates an instance of the CHPSim_AVG_tg class from the result data
       
        Parameters
        ----------
        path_to_folder : str, optional
            path to the results folder, by default "."
        time0 : float, optional
            A previous time to average between, by default None
        abs_path : bool, optional
            Whether the path provided is absolute or relative, by default True
        """

        if isinstance(path_to_folder,list):
            times = misc_utils.time_extract(path_to_folder[0],abs_path)
        else:
            times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        return super()._extract_avg(times,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path,*args,**kwargs)

class CHAPSim_AVG():
    def __new__(cls,*args,tgpost=False,**kwargs):
        if tgpost:
            return CHAPSim_AVG_tg(*args,**kwargs)
        else:
            return CHAPSim_AVG_io(*args,**kwargs)

    @classmethod
    def from_hdf(cls,*args,tgpost=False,**kwargs):
        if tgpost:
            return cls(tgpost=tgpost,fromfile=True,*args,**kwargs)
        else:
            return cls(fromfile=True,*args,**kwargs)

# class CHAPSim_AVG_tg_new(CHAPSim_AVG_base):
#     def _extract_avg(self,time,meta_data=None,path_to_folder=".",time0=None,abs_path=True):
#         if meta_data is None:
#             meta_data = self._module._meta_class(path_to_folder,abs_path,False)

#         self._meta_data = meta_data.copy()
#         time = misc_utils.check_list_vals(time)

#         for PhyTime in time:
#             if 'DF_list' not in locals():
#                 DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
#             else:
#                 DF_temp=[]
#                 local_DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
#                 for i, local_DF in enumerate(DF_list):
#                     DF_list[i].concat(local_DF_list[i])

#         DF_list=self._Reverse_decomp(*DF_list)


#         self.flow_AVGDF,self.PU_vectorDF,\
#         self.UU_tensorDF,self.UUU_tensorDF,\
#         self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
#         self.DUDX2_tensorDF = DF_list
        

#         self._times = list(set([x[0] for x in self.flow_AVGDF.index]))

#         def _AVG_extract(self,Time_input,time0,path_to_folder,abs_path):

#             AVG_info, NSTATIS1, PhyTime, NCL2 = self._extract_file_tg(Time_input,path_to_folder,abs_path)

#             if time0 is not None:
#                 AVG_info0, NSTATIS0,_,_ = self._extract_file_io(time0,path_to_folder,abs_path)
#                 AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)
#                 del AVG_info0