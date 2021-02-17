"""
# CHAPSim_moving_wall
Module extension for CHAPSim_post to handle flows with streamwise moving wall.
Additional classes for this module include CHAPSim_perturb to analyse the
developing flow from the acceleration.
"""

import h5py
import numpy as np
import os
from scipy import integrate
import sys

from . import CHAPSim_post as cp

from . import CHAPSim_plot as cplt

from CHAPSim_post.utils import misc_utils,indexing
from CHAPSim_post import POD



class CHAPSim_Inst(cp.CHAPSim_Inst):
    pass
_instant_class = CHAPSim_Inst

class CHAPSim_AVG_io(cp.CHAPSim_AVG_io):

    def _tau_calc(self,PhyTime):

        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        wall_velo = self._meta_data.moving_wall_calc()
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        for i in range(self.shape[1]):
            tau_star[i] = mu_star*(u_velo[0,i]-wall_velo[i])/(ycoords[0]--1.0)#*(-1*u_velo[1,i] + 4*u_velo[0,i] - 3*wall_velo[i])/(0.5*ycoords[1]-1.5*(-1.0)+y_coords[0])
    
        return tau_star

    def _bulk_velo_calc(self,PhyTime):

        u_velo = self.flow_AVGDF[PhyTime,'u'].copy()
        ycoords = self.CoordDF['y']
        wall_velo = self._meta_data.moving_wall_calc()
        
        bulk_velo=np.zeros(self.shape[1])
        for i in range(self.NCL[1]):
            u_velo[i,:]=u_velo[i,:] - wall_velo
        for i in range(self.shape[1]):
            bulk_velo[i] = 0.5*integrate.simps(u_velo[:,i],ycoords)
            
        return bulk_velo

    def plot_bulk_velocity(self,PhyTime=None,relative=False,fig=None,ax=None,line_kw=None,**kwargs):
        
        PhyTime = self.check_PhyTime(PhyTime)
        if relative:
            bulk_velo = self._bulk_velo_calc(PhyTime)
        else:
            bulk_velo = super()._bulk_velo_calc(PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        xaxis = self._return_xaxis()

        line_kw = cplt.update_line_kw(line_kw,label=r"$U_{b0}$")

        ax.cplot(xaxis,bulk_velo,**line_kw)
        ax.set_ylabel(r"$U_b^*$")
        ax.set_xlabel(r"$x/\delta$")
        return fig, ax

    def accel_param_calc(self,PhyTime=None):

        PhyTime = self.check_PhyTime(PhyTime)

        U_mean = self.flow_AVGDF[PhyTime,'u']
        U0_index = int(self.NCL[1]*0.5)
        U0 = U_mean[U0_index]
        wall_velo = self._meta_data.moving_wall_calc()
        x_coords = self.CoordDF['x']
        
        U_infty_grad = np.zeros(self.NCL[0])
        U_infty = U0 - wall_velo
        REN = self._metaDF['REN']
        for i in range(self.NCL[0]):
            if i ==0:
                U_infty_grad[i] = (U_infty[i+1] - U_infty[i])/\
                                (x_coords[i+1] - x_coords[i])
            elif i == self.NCL[0]-1:
                 U_infty_grad[i] = (U_infty[i] - U_infty[i-1])/\
                                (x_coords[i] - x_coords[i-1])
            else:
                U_infty_grad[i] = (U_infty[i+1] - U_infty[i-1])/\
                                (x_coords[i+1] - x_coords[i-1])
        accel_param = (1/(REN*U_infty**2))*U_infty_grad
        
        return accel_param

    def plot_accel_param(self,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):
        
        accel_param = self.accel_param_calc(PhyTime)
        x_coords = self.CoordDF['x']

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw,label = r"$K$")
        
        ax.cplot(x_coords,accel_param,**line_kw)
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$K$")
        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        fig.tight_layout()
        return fig,ax

    def _int_thickness_calc(self,PhyTime):

        U0_index = int(np.floor(self.NCL[1]*0.5))
        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        
        wall_velo = self._meta_data.moving_wall_calc()
        for i in range(wall_velo.size):
            U_mean[:,i] -= wall_velo[i]

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
        
    def plot_near_wall(self,x_vals,PhyTime=None,*args,**kwargs):
        
        fig, ax = super().plot_near_wall(x_vals,*args,**kwargs)
        wall_velo = self._meta_data.moving_wall_calc()
        u_tau_star, _ = self.wall_unit_calc(PhyTime)
        lines = ax.get_lines()[-len(x_vals)-1:-1]
        for line, x_val in zip(lines,x_vals):
            y_data = line.get_ydata()
            x = self._return_index(x_val)
            y_data = y_data - wall_velo[x]/u_tau_star[x]
            line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_mean_flow(self,x_vals,*args,relative=False,fig=None,ax=None,**kwargs):
        
        if not relative:
            return super().plot_mean_flow(x_vals,*args,fig=fig,ax=ax,**kwargs)
        else:
            fig, ax = super().plot_mean_flow(x_vals,*args,fig=fig,ax=ax,**kwargs)
            x_indices = indexing.coord_index_calc(self.CoordDF,'x',x_vals)
            moving_wall = self._meta_data.moving_wall_calc()[x_indices]
            for line, val in zip(ax.get_lines(),moving_wall):
                ydata = line.get_ydata().copy()
                ydata-=val
                line.set_ydata(ydata)
            ax.relim()
            ax.autoscale_view()
            return fig, ax
_avg_io_class = CHAPSim_AVG_io

class CHAPSim_AVG():
    def __new__(cls,*args,**kwargs):
        if kwargs.pop("tgpost",False):
            msg = "This module has not class to postprocess the turbulence generator"
            raise ValueError(msg)
        return CHAPSim_AVG_io(*args,**kwargs)
_avg_class = CHAPSim_AVG

class CHAPSim_perturb():
    def __init__(self,time=None,avg_data=None, meta_data=None,path_to_folder='.',time0=None,abs_path=True):
        if avg_data is None:
            self.__avg_data = avg_data
            times = list(set([x[0] for x in avg_data.UU_tensorDF.index]))
            if len(times)>1 and not times:
                msg="If there is more than one time present in"+\
                            " CHAPSim_AVG_io, a time must be provided"
                raise ValueError(msg)
        else:
            assert time is not None, "In the absence of input of class CHAPSim_AVG a time must be provided"
            self.__avg_data = CHAPSim_AVG_io(time,meta_data,path_to_folder,time0,abs_path,False)
        if meta_data is None:
            meta_data = self.__avg_data._meta_data
        self._meta_data = meta_data

    def tau_du_calc(self,PhyTime=None):
        PhyTime = self.__avg_data.check_PhyTime(PhyTime)
        tau_w = self.__avg_data.tau_calc(PhyTime)
        return tau_w - tau_w[0]

    def mean_velo_peturb_calc(self,comp,PhyTime):
        U_velo_mean = self.__avg_data.flow_AVGDF[PhyTime,comp].copy()
        wall_velo = self._meta_data.moving_wall_calc()
        for i in range(self.__avg_data.shape[0]):
            U_velo_mean[i] -= wall_velo

        start = self._meta_data.metaDF['location_start_end'][0]
        x_loc = indexing.coord_index_calc(self.__avg_data.CoordDF,'x',start)
        
        centre_index =int(0.5*self.__avg_data.shape[0])
        U_c0 = U_velo_mean[centre_index,0]
        mean_velo_peturb = np.zeros((self.__avg_data.shape[0],self.__avg_data.shape[1]-x_loc))
        for i in range(x_loc,self.__avg_data.shape[1]):
            mean_velo_peturb[:,i-x_loc] = (U_velo_mean[:,i]-U_velo_mean[:,0])/(U_velo_mean[centre_index,i]-U_c0)
        return mean_velo_peturb

    def plot_perturb_velo(self,x_vals,PhyTime=None,comp='u',Y_plus=False,Y_plus_max=100,fig=None,ax =None,**kwargs):
        velo_peturb = self.mean_velo_peturb_calc(comp,PhyTime)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig,ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        y_coord = self._meta_data.CoordDF['y']
        if not PhyTime:
            PhyTime = self.__avg_data.flow_AVGDF.index[0][0]

        _, delta_v_star = self.__avg_data.wall_unit_calc(PhyTime)
        if Y_plus:
            y_coord = y_coord[:int(y_coord.size/2)]
            y_coord = (1-np.abs(y_coord))/delta_v_star[0]
            velo_peturb = velo_peturb[:int(y_coord.size)]
        else:
            y_max= Y_plus_max*delta_v_star[0]-1.0

        start = self._meta_data.metaDF['location_start_end'][0]
        x_vals = [x-start for x in x_vals]
        x_loc = indexing.coord_index_calc(self.__avg_data.CoordDF,'x',x_vals)
        for x, x_val in zip(x_loc,x_vals):
            label=r"$x/\delta = %.3g$" % x_val
            ax.cplot(velo_peturb[:,x],y_coord,label=label)
        ax.set_xlabel(r"$\bar{U}^{\wedge}$")
        if Y_plus:
            ax.set_ylabel(r"$y^+$")# ,fontsize=16)
            ax.set_ylim([0,Y_plus_max])
        else:
            ax.set_ylabel(r"$y/\delta$")# ,fontsize=16)
            ax.set_ylim([-1,y_max])

        ncol = cplt.get_legend_ncols(len(ax.get_lines()))
        ax.clegend(vertical=False,ncol=ncol, fontsize=16)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_peturb_cf(self,PhyTime=None,wall_units=False,fig=None,ax=None,**kwargs):
        PhyTime = self.__avg_data.check_PhyTime(PhyTime)
        tau_du = self.tau_du_calc(PhyTime)
        bulkvelo = self.__avg_data._bulk_velo_calc(PhyTime)

        start = self._meta_data.metaDF['location_start_end'][0]
        x_loc = indexing.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        REN = self._meta_data.metaDF['REN']
        rho_star = 1.0
        Cf_du = tau_du[x_loc:]/(0.5*REN*rho_star*(bulkvelo[x_loc:]-bulkvelo[0])**2)
        
        x_coords = self._meta_data.CoordDF['x'][x_loc:] 
        x_coords -= start

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig,ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
            
        ax.cplot(x_coords, Cf_du)
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$C_{f,du}$")
        ax.set_ylim([0,2*Cf_du[-1]])
        return fig, ax
    def int_thickness_calc(self,PhyTime=None):
        
        PhyTime  = self.__avg_data.check_PhyTime(PhyTime)

        mean_velo = self.mean_velo_peturb_calc('u',PhyTime)

        start = self._meta_data.metaDF['location_start_end'][0]
        x_loc = indexing.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        y_coords = self.__avg_data.CoordDF['y']

        U0_index = int(self.__avg_data.shape[0]*0.5)
        theta_integrand = np.zeros((U0_index,self.__avg_data.shape[1]))
        delta_integrand = np.zeros((U0_index,self.__avg_data.shape[1]))
        mom_thickness = np.zeros(self.__avg_data.shape[1]-x_loc)
        disp_thickness = np.zeros(self.__avg_data.shape[1]-x_loc)
        theta_integrand = mean_velo[:U0_index]*(1-mean_velo[:U0_index])
        delta_integrand = 1-mean_velo[:U0_index]
        for j in range(self.__avg_data.shape[1]-x_loc):
            mom_thickness[j] = integrate.simps(theta_integrand[:,j],y_coords[:U0_index])
            disp_thickness[j] = integrate.simps(delta_integrand[:,j],y_coords[:U0_index])
        shape_factor = np.divide(disp_thickness,mom_thickness)

        

        return disp_thickness, mom_thickness, shape_factor

    def plot_shape_factor(self,PhyTime=None,fig=None,ax=None,**kwargs):
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig,ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        start = self._meta_data.metaDF['location_start_end'][0]
        x_loc = indexing.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        x_coords = self._meta_data.CoordDF['x'][x_loc:] 
        x_coords -= start
        delta, theta, H = self.int_thickness_calc(PhyTime)

        ax.cplot(x_coords, H,label=r"$H$")
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$H$")
        ax.set_ylim([0,2*H[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_mom_thickness(self,PhyTime=None,fig=None,ax=None,**kwargs):

        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        start = self._meta_data.metaDF['location_start_end'][0]
        x_loc = indexing.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        x_coords = self._meta_data.CoordDF['x'][x_loc:] 
        x_coords -= start
        delta, theta, H = self.int_thickness_calc(PhyTime)

        ax.cplot(x_coords, theta,label=r"$\theta$")
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$\theta$")
        ax.set_ylim([0,2*theta[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_disp_thickness(self,PhyTime=None,fig=None,ax=None,**kwargs):

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        start = self._meta_data.metaDF['location_start_end'][0]
        x_loc = indexing.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        x_coords = self._meta_data.CoordDF['x'][x_loc:] 
        x_coords -= start
        delta, theta, H = self.int_thickness_calc(PhyTime)

        ax.cplot(x_coords, delta,label=r"$\delta^*$")
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$\delta^*$")
        ax.set_ylim([0,2*delta[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_meta(cp.CHAPSim_meta):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        fromfile = kwargs.pop('fromfile',False)
        if fromfile:
            self.__moving_wall = self._extract_moving_wall(*args,**kwargs)
        else:
            tgpost = kwargs.get('tgpost',False)
            self.__moving_wall = self.__moving_wall_setup(self.NCL,self.path_to_folder,
                                    self._abs_path,self.metaDF,tgpost)
    
    def _extract_moving_wall(self,file_name,key=None):
        if key is None:
            key = 'CHAPSim_meta'
        hdf_file = h5py.File(file_name,'r')
        moving_wall = hdf_file[key+"/moving_wall"][:]
        hdf_file.close()
        return moving_wall

    def __moving_wall_setup(self,NCL,path_to_folder,abs_path,metaDF,tgpost):
        wall_velo = np.zeros(NCL[0])
        if int(metaDF['moving_wallflg']) == 1 and not tgpost:
            full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
                                                            '.')
            if not abs_path:
                file_path = os.path.abspath(os.path.join(full_path,'CHK_MOVING_WALL.dat'))
            else:
                file_path = os.path.join(full_path,'CHK_MOVING_WALL.dat')
            
            mw_file = open(file_path)
            wall_velo_ND = np.loadtxt(mw_file,comments='#',usecols=1)
            for i in range(1,NCL[0]+1):
                wall_velo[i-1] = 0.5*(wall_velo_ND[i+1] + wall_velo_ND[i])
            mw_file.close()
        return wall_velo

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = 'CHAPSim_meta'
        super().save_hdf(file_name,write_mode,key)
        hdf_file = h5py.File(file_name,'a')
        hdf_file[key].create_dataset("moving_wall",data=self.__moving_wall)
        hdf_file.close()

    @property
    def wall_velocity(self):
        return self.__moving_wall

    def moving_wall_calc(self):
        return self.__moving_wall
_meta_class = CHAPSim_meta

class CHAPSim_fluct_io(cp.CHAPSim_fluct_io):
    pass
_fluct_io_class = CHAPSim_fluct_io
_fluct_class = CHAPSim_fluct_io

class CHAPSim_budget_io(cp.CHAPSim_budget_io):
    pass

class CHAPSim_autocov_io(cp.CHAPSim_autocov_io):
   pass

class CHAPSim_Quad_Anl_io(cp.CHAPSim_Quad_Anl_io):
   pass

class CHAPSim_joint_PDF_io(cp.CHAPSim_joint_PDF_io):
    pass

class POD2D(POD.POD2D):
    pass
_POD2D_class = POD2D
class POD3D(POD.POD3D):
    pass
_POD3D_class = POD3D

class flowReconstruct2D(POD.flowReconstruct2D):
    pass

class flowReconstruct3D(POD.flowReconstruct3D):
    pass
