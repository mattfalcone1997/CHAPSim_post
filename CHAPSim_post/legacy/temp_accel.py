"""
# CHAPSim_temp_accel 
Module for processing temporal acceleration

"""

from . import post as cp
from . import dtypes as cd
from . import plot as cplt
from .utils import misc_utils, indexing
import sys
import warnings
import numpy as np

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

import h5py
import os

class CHAPSim_Inst(cp.CHAPSim_Inst_tg):
    pass
_instant_class = CHAPSim_Inst

class CHAPSim_AVG_custom_t(cp.CHAPSim_AVG_tg_base):

    @classmethod
    def with_phase_average(cls,*args,**kwargs):

        if 'path_to_folder' not in kwargs.keys():
            msg = "keyword argument `path_to_folder' must be present to use this method"
            raise ValueError(msg)

        path_to_folder = kwargs['path_to_folder']
        if not isinstance(path_to_folder,(tuple,list)):
            msg = f"To use this method, path_to_folder must be a tuple or a list not a {type(path_to_folder)}"
            raise TypeError(msg)
        abs_path = kwargs.get('abs_path',True)

        metaDF_list = []
        for path in path_to_folder:
            metaDF_list.append(CHAPSim_meta(path,abs_path).metaDF)
        kwargs['shift_vals'] = [float(metaDF['temp_start_end'][0]) for metaDF in metaDF_list]

        obj = super().with_phase_average(*args,**kwargs)

        start = 0.0; end = obj.metaDF['temp_start_end'][1] - obj.metaDF['temp_start_end'][0]
        obj.metaDF['temp_start_end'] = [start,end]
        return obj

    def shift_times(self,val):
        super().shift_times(val)
        self.metaDF['temp_start_end'][0] -= val
        self.metaDF['temp_start_end'][1] -= val
    
    
    def conv_distance_calc(self):
        
        bulk_velo = self.bulk_velo_calc()

        time0 = float(self._times[0])
        times = [float(x)-time0 for x in self._times]

        accel_start = self.metaDF['temp_start_end'][0]
        start_distance = bulk_velo[0]*(accel_start - time0)
        conv_distance = np.zeros_like(bulk_velo)
        for i , _ in enumerate(bulk_velo):
            conv_distance[i] = integrate_simps(bulk_velo[:(i+1)],times[:(i+1)])
        return conv_distance - start_distance

    # @staticmethod
    # def _conv_distance_calc(bulk_velo,times):

    def accel_param_calc(self):
        U_mean = self.flow_AVGDF[None,'u']
        U_infty = U_mean[int(self.NCL[1]*0.5)]

        times = self._return_xaxis()
        dudt = np.gradient(U_infty,times,edge_order=2)
        REN = self.metaDF['REN']

        accel_param = (1/(REN*U_infty**3))*dudt
        return accel_param

    def plot_accel_param(self,fig=None,ax=None,**kwargs):
        accel_param = self.accel_param_calc()

        xaxis = self._return_xaxis()

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        ax.cplot(xaxis,accel_param,label=r"$K$")

        ax.set_xlabel(r"$t^*$")# ,fontsize=18)
        ax.set_ylabel(r"$K$")# ,fontsize=18)

        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        #ax.grid()
        fig.tight_layout()
        return fig,ax

_avg_tg_base_class = CHAPSim_AVG_custom_t

class CHAPSim_AVG_tg(CHAPSim_AVG_custom_t):
    def _extract_avg(self,path_to_folder=".",time0=None,abs_path=True,*args,**kwargs):

        if isinstance(path_to_folder,list):
            times = misc_utils.time_extract(path_to_folder[0],abs_path)
        else:
            times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        return super()._extract_avg(times,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path,*args,**kwargs)
        
_avg_tg_class = CHAPSim_AVG_tg

class CHAPSim_AVG_tg_conv(CHAPSim_AVG_tg):

    @classmethod
    def with_phase_average(cls,*args,**kwargs):
        obj = super().with_phase_average(*args, **kwargs)
        obj.CoordDF['x'] = obj.conv_distance_calc()
        return obj

    # @property
    # def CoordDF(self):
    #     CoordDF = super().CoordDF
    #     CoordDF['x'] = self.conv_distance_calc()
    #     return CoordDF

    def _extract_avg(self,*args,**kwargs):
        super()._extract_avg(*args,**kwargs)

        self.CoordDF['x'] = self.conv_distance_calc()

    def _return_index(self,x_val):
        return indexing.coord_index_calc(self.CoordDF,'x',x_val)

    def _return_time_index(self,time):
        return super()._return_index(time)
    def _return_xaxis(self):
        return self.CoordDF['x']

    def filter_times(self,times):
        filter_times = ["%.9g"% time for time in times]
        time_list = list(set(self._times).intersection(set(filter_times)))
        time_list = sorted(float(x) for x in time_list)
        # time_list = ["%.9g"%x for x in time_list]
        index_list = [self._return_time_index(x) for x in time_list]
        
        self._times = time_list

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

    def plot_shape_factor(self,*args,**kwargs):
        fig, ax = super().plot_shape_factor(*args,**kwargs)    
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax

    def plot_Reynolds(self,comp,x_vals,*args,**kwargs):
        fig, ax = super().plot_Reynolds(comp,x_vals,*args,**kwargs)
        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
        
        ax.get_legend().remove()
        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)

        return fig, ax

    def plot_Reynolds_x(self,*args,**kwargs):
        fig, ax = super().plot_Reynolds_x(*args,**kwargs)
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax  

    def plot_bulk_velocity(self,*args,**kwargs):
        fig, ax = super().plot_bulk_velocity(*args,**kwargs)
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_skin_friction(self,*args,**kwargs):
        fig, ax = super().plot_skin_friction(*args,**kwargs)
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self,x_vals, *args, **kwargs):
        fig, ax =  super().plot_eddy_visc(x_vals,*args, **kwargs)

        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
        ax.get_legend().remove()
        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_mean_flow(self,x_vals,*args,**kwargs):
        fig, ax = super().plot_mean_flow(x_vals,*args,**kwargs)

        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
        ax.get_legend().remove()
        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_near_wall(self,x_vals,*args,**kwargs):
        fig, ax = super().plot_near_wall(x_vals,*args,**kwargs)

        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
        ax.get_legend().remove()
        ncol = cplt.get_legend_ncols(len(lines))
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

from CHAPSim_post.post._common import Common
class CHAPSim_perturb(Common):
    def __init__(self,avg_data=None, meta_data=None,path_to_folder='.',time0=None,abs_path=True):
        if avg_data is not None:
            if not isinstance(avg_data,cp.CHAPSim_AVG_tg_base):
                msg = f"avg_data must be a subclass of {cp.CHAPSim_AVG_tg_base}"
                raise TypeError(msg)
            self.__avg_data = avg_data
        else:
            self.__avg_data = CHAPSim_AVG_tg(path_to_folder,time0,abs_path,meta_data=meta_data)
        if meta_data is None:
            meta_data = self.__avg_data._meta_data
        self._meta_data = meta_data
        self.start  = self.metaDF['temp_start_end'][0]

    @property
    def start_index(self):
        accel_start = self.metaDF['temp_start_end'][0]
        return self.__avg_data._return_index(accel_start)+ 1

    @property
    def shape(self):
        avg_shape = self.__avg_data.shape
        return (avg_shape[0],avg_shape[1]- self.start_index)

    def tau_du_calc(self):
        
        tau_w = self.__avg_data.tau_calc()
        start_index = self.start_index
        return tau_w[start_index:] - tau_w[start_index]

    def mean_velo_peturb_calc(self,comp):
        U_velo_mean = self.__avg_data.flow_AVGDF[None,comp].copy()
        time_0_index = self.start_index

        centre_index = self.__avg_data.shape[0])
        U_c0 = U_velo_mean[centre_index,time_0_index]
        mean_velo_peturb = np.zeros(self.shape)

        for i in range(time_0_index,self.__avg_data.shape[1]):
            mean_velo_peturb[:,i-time_0_index] = (U_velo_mean[:,i]-U_velo_mean[:,time_0_index])/(U_velo_mean[centre_index,i]-U_c0)
        return mean_velo_peturb

    def plot_perturb_velo(self,times,comp='u',Y_plus=False,Y_plus_max=100,fig=None,ax =None,**kwargs):
        velo_peturb = self.mean_velo_peturb_calc(comp)
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])   
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)     

        y_coord = self._meta_data.CoordDF['y']
        
        u_tau_star, delta_v_star = self.__avg_data.wall_unit_calc()
        if Y_plus:
            y_coord = y_coord[:int(y_coord.size/2)]
            y_coord = (1-np.abs(y_coord))/delta_v_star[0]
            velo_peturb = velo_peturb[:int(y_coord.size)]
        else:
            y_max= Y_plus_max*delta_v_star[0]-1.0

        time_0_index = self.start_index
        time_loc = np.array([self.__avg_data._return_index(x) for x in times]) - time_0_index

        for x, x_val in zip(time_loc,times):
            label=r"$x/\delta = %.3g$" % x_val
            ax.cplot(velo_peturb[:,x],y_coord,label=label)
        ax.set_xlabel(r"$\bar{U}^{\wedge}$")
        if Y_plus:
            ax.set_ylabel(r"$y^+$")# ,fontsize=16)
            ax.set_ylim([0,Y_plus_max])
        else:
            ax.set_ylabel(r"$y/\delta$")# ,fontsize=16)
            ax.set_ylim([-1,y_max])

        axes_items_num = len(ax.get_lines())
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol, fontsize=16)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_perturb_cf(self,wall_units=False,fig=None,ax=None,**kwargs):

        tau_du = self.tau_du_calc()
        bulkvelo = self.__avg_data._bulk_velo_calc(None)
        x_loc = self.start_index

        REN = self.metaDF['REN']
        rho_star = 1.0

        Cf_du = tau_du/(0.5*REN*rho_star*(bulkvelo[x_loc:]-bulkvelo[0])**2)
        
        
        times = self.__avg_data._return_xaxis()[x_loc:] - self.start

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
            
        ax.cplot(times, Cf_du)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$C_{f,du}$")
        ax.set_ylim([0,10*Cf_du[-1]])
        return fig, ax

    def int_thickness_calc(self):

        mean_velo = self.mean_velo_peturb_calc('u')

        x_loc = self.__avg_data._return_index(self.start)+1
        y_coords = self.__avg_data.CoordDF['y']

        U0_index = int(self.__avg_data.shape[0]*0.5)
        mom_thickness = np.zeros(self.__avg_data.shape[1]-x_loc)
        disp_thickness = np.zeros(self.__avg_data.shape[1]-x_loc)
        theta_integrand = mean_velo[:U0_index]*(1-mean_velo[:U0_index])
        delta_integrand = 1-mean_velo[:U0_index]

        for j in range(self.__avg_data.shape[1]-x_loc):
            mom_thickness[j] = integrate_simps(theta_integrand[:,j],y_coords[:U0_index])
            disp_thickness[j] = integrate_simps(delta_integrand[:,j],y_coords[:U0_index])
        shape_factor = np.divide(disp_thickness,mom_thickness)

        

        return disp_thickness, mom_thickness, shape_factor

    def plot_shape_factor(self,fig=None,ax=None,line_kw=None,**kwargs):

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_loc = self.__avg_data._return_index(self.start)+1

        times = self.__avg_data._return_xaxis()[x_loc:] - self.start

        _, _, H = self.int_thickness_calc()

        line_kw = cplt.update_line_kw(line_kw,label = r"$H$")

        ax.cplot(times, H,**line_kw)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$H$")
        ax.set_ylim([0,2*H[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_mom_thickness(self,fig=None,ax=None,**kwargs):

        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_loc = self.__avg_data._return_index(self.start)+1

        times = self.__avg_data._return_xaxis()[x_loc:] - self.start

        _, theta, _ = self.int_thickness_calc()

        ax.cplot(times, theta,label=r"$\theta$")
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$\theta$")
        ax.set_ylim([0,2*theta[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_disp_thickness(self,fig=None,ax=None,**kwargs):

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_loc = self.__avg_data._return_index(self.start)+1

        times = self.__avg_data._return_xaxis()[x_loc:] - self.start

        delta, _, _ = self.int_thickness_calc()

        ax.cplot(times, delta,label=r"$\delta^*$")
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$\delta^*$")
        ax.set_ylim([0,2*delta[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_meta(cp.CHAPSim_meta):
    def __init__(self,*args,**kwargs):
        if len(args) < 3:
            kwargs['tgpost'] = True
        super().__init__(*args,**kwargs)
_meta_class = CHAPSim_meta

# class CHAPSim_meta_moving_wall(CHAPSim_meta):
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#         fromfile = kwargs.pop('fromfile',False)
#         if fromfile:
#             self.__moving_wall = self._extract_moving_wall(*args,**kwargs)
#         else:
#             self.__moving_wall = self.__moving_wall_setup(self.NCL,self.path_to_folder,
#                                     self._abs_path,self.metaDF)
    
#     def _extract_moving_wall(self,file_name,key=None):
#         if key is None:
#             key = self.__class__.__name__
#         hdf_file = h5py.File(file_name,'r')
#         moving_wall = hdf_file[key+"/moving_wall"][:]
#         hdf_file.close()
#         return moving_wall

#     def __moving_wall_setup(self,NCL,path_to_folder,abs_path,metaDF):
#         wall_velo = np.zeros(NCL[0])
#         if int(metaDF['temp_accelflg']) == 2:
#             full_path = misc_utils.check_paths(path_to_folder,'0_log_monitors',
#                                                             '.')
#             if not abs_path:
#                 file_path = os.path.abspath(os.path.join(full_path,'CHK_TEMP_ACCEL.dat'))
#             else:
#                 file_path = os.path.join(full_path,'CHK_TEMP_ACCEL.dat')
            
#             mw_file = open(file_path)
#             wall_velo_ND = np.loadtxt(mw_file,comments='#',usecols=1)
#             times = np.loadtxt(mw_file,comments='#',usecols=0)
#             file_times = misc_utils.time_extract(path_to_folder,abs_path)
#             print(wall_velo_ND)
#             def _interp(velo,times, file_times):
#                 wall_velo = np.zeros_like(file_times)
#                 for j,timef in enumerate(file_times):
#                     for i, time in enumerate(times):
#                         if time > timef:
#                             grad = (velo[i]-velo[i-1])/(times[i] - times[i-1])
#                             wall_velo[j] = velo[i-1] + grad*(timef-times[i-1])
#                             break
                
#                 return wall_velo

#             wall_velo = _interp(wall_velo_ND,times,file_times)

#             mw_file.close()
#         return wall_velo

    # def save_hdf(self,file_name,write_mode,key=None):
    #     if key is None:
    #         key = self.__class__.__name__

    #     super().save_hdf(file_name,write_mode,key)
    #     hdf_file = h5py.File(file_name,'a')
    #     hdf_file[key].create_dataset("moving_wall",data=self.__moving_wall)
    #     hdf_file.close()

    # @property
    # def wall_velocity(self):
    #     return self.__moving_wall

# class CHAPSim_AVG_moving_wall(CHAPSim_AVG_tg):
#     def __init__(self,*args,**kwargs):
#         if self._module._meta_class != CHAPSim_meta_moving_wall:
#             msg = ("This module can only be used if _meta_class is"
#                     " set to CHAPSim_meta_moving_wall")
#             raise ValueError(msg)

#         super().__init__(*args,**kwargs)
        
#     def accel_param_calc(self):
#         U_mean = self.flow_AVGDF[None,'u']
#         U_infty = U_mean[int(self.NCL[1]*0.5)]
#         wall_velo = self._meta_data.wall_velocity
#         times = self._return_xaxis()
#         dudt = np.gradient(U_infty-wall_velo,times,edge_order=2)
#         REN = self.metaDF['REN']

#         accel_param = (1/(REN*U_infty**3))*dudindext
#         return accel_param
#     def _tau_calc(self,PhyTime):

#         u_velo = self.flow_AVGDF[PhyTime,'u']
#         ycoords = self.CoordDF['y']
        
#         wall_velo = self._meta_data.wall_velocity
        
#         tau_star = np.zeros_like(u_velo[1])
#         mu_star = 1.0
#         for i in range(self.shape[1]):
#             tau_star[i] = mu_star*(u_velo[0,i]-wall_velo[i])/(ycoords[0]--1.0)#*(-1*u_velo[1,i] + 4*u_velo[0,i] - 3*wall_velo[i])/(0.5*ycoords[1]-1.5*(-1.0)+y_coords[0])
    
#         return tau_star
#     def _bulk_velo_calc(self,PhyTime):

#         u_velo = self.flow_AVGDF[PhyTime,'u'].copy()
#         ycoords = self.CoordDF['y']
#         wall_velo = self._meta_data.wall_velocity
        
#         u_velo = u_velo - wall_velo
#         bulk_velo = 0.5*integrate_simps(u_velo,ycoords,axis=0)
            
#         return bulk_velo

#     def plot_bulk_velocity(self,relative=False,fig=None,ax=None,line_kw=None,**kwargs):
        
#         if relative:
#             bulk_velo = self._bulk_velo_calc(None)
#         else:
#             bulk_velo = super()._bulk_velo_calc(None)

#         kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
#         fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

#         xaxis = self._return_xaxis()

#         line_kw = cplt.update_line_kw(line_kw,label=r"$U_{b0}$")

#         ax.cplot(xaxis,bulk_velo,**line_kw)
#         ax.set_ylabel(r"$U_b^*$")
#         ax.set_xlabel(r"$t^*$")
#         return fig, ax

#     def _int_thickness_calc(self,PhyTime):

#         U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
#         y_coords = self.CoordDF['y']

#         wall_velo = self._meta_data.wall_velocity
#         U_mean = U_mean - wall_velo
#         U0 = U_mean[-1]
#         theta_integrand = np.zeros_like(U_mean)
#         delta_integrand = np.zeros_like(U_mean)

#         for i, _ in enumerate(theta_integrand):
#             theta_integrand[i] = (U_mean[i]/U0)*(1 - U_mean[i]/U0)
#             delta_integrand[i] = 1 - U_mean[i]/U0

#         mom_thickness = integrate_simps(theta_integrand,y_coords,axis=0)
#         disp_thickness = integrate_simps(delta_integrand,y_coords,axis=0)
#         shape_factor = disp_thickness/mom_thickness
        
#         return disp_thickness, mom_thickness, shape_factor

#     def plot_mean_flow(self,times,*args,relative=False,fig=None,ax=None,**kwargs):
#         if not relative:
#             return super().plot_mean_flow(times,*args,fig=fig,ax=ax,**kwargs)
#         else:
#             fig, ax = super().plot_mean_flow(times,*args,fig=fig,ax=ax,**kwargs)
#             t_indices = self._return_index(times)
#             moving_wall = self._meta_data.wall_velocity[t_indices]
#             for line, val in zip(ax.get_lines(),moving_wall):
#                 ydata = line.get_ydata().copy()
#                 ydata-=val
#                 line.set_ydata(ydata)
#             ax.relim()
#             ax.autoscale_view()
#             return fig, ax
class CHAPSim_fluct_tg(cp.CHAPSim_fluct_tg):
    pass
_fluct_tg_class = CHAPSim_fluct_tg

class CHAPSim_budget_tg(cp.CHAPSim_budget_tg):
    pass

class CHAPSim_momentum_budget_tg(cp.CHAPSim_Momentum_budget_tg):
    def __init__(self,comp,avg_data=None,PhyTime=None,apparent_Re=False,*args,**kwargs):
        
        cp.CHAPSim_Momentum_budget_io.__init__(self,comp,avg_data,PhyTime,*args,**kwargs)
        
        if apparent_Re:
            self._create_uniform(PhyTime)
    def _budget_extract(self, PhyTime, comp):
        transient  = np.stack([self._transient_extract(None,comp)],axis=0)
        budgetDF = cd.datastruct(transient,index=[(None,'transient')])

        otherBudgetDF =  super()._budget_extract(PhyTime, comp)
        budgetDF.concat(otherBudgetDF)
        return budgetDF
        # array_concat.insert(0,transient)
        # budget_index.insert(0,'transient')
        # phystring_index = [PhyTime]*5

    def _create_uniform(self,PhyTime):
        advection = self.budgetDF[PhyTime,'transient']
        centre_index =  advection.shape[0])
        advection_centre = advection[centre_index]

        uniform_bf = self.budgetDF[PhyTime,'pressure gradient'] + advection_centre
        non_uniform_bf = advection - advection_centre

        times = self.budgetDF.outer_index
        for time in times:
            key1 = (time,'transient')
            key2 = (time,'pressure gradient')
            del self.budgetDF[key1]; del self.budgetDF[key2]

        index = [[PhyTime]*2,['uniform','non-uniform']]
        dstruct = cd.datastruct(np.array([uniform_bf,non_uniform_bf]),
                            index=index)
        self.budgetDF.concat(dstruct)
    def _transient_extract(self,PhyTime,comp):
        U = self.avg_data.flow_AVGDF[None,comp]

        times = self.avg_data._return_xaxis()
        return -np.gradient(U,times,axis=1)

    def _pressure_grad(self, PhyTime, comp):
        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        times = self.avg_data._return_xaxis()

        REN = self.avg_data.metaDF['REN']
        d2u_dy2 = self.Domain.Grad_calc(self.avg_data.CoordDF,
                    self.Domain.Grad_calc(self.avg_data.CoordDF,U_mean,'y'),'y')
        
        uv = self.avg_data.UU_tensorDF[PhyTime,'uv']
        duv_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uv,'y')
        dudt = np.gradient(U_mean,times,axis=-1,edge_order=2)

        return -( (1/REN)*d2u_dy2 - duv_dy  - dudt)
class CHAPSim_autocov_tg(cp.CHAPSim_autocov_tg):
    pass

class CHAPSim_Quad_Anl_tg(cp.CHAPSim_Quad_Anl_tg):
    pass

class CHAPSim_FIK_tg_conv(cp.CHAPSim_FIK_tg):
    def plot(self,*args,**kwargs):
        fig, ax = super().plot(*args,**kwargs)
        conv_distance = self.avg_data.conv_distance_calc()
        for line in ax.set_lines()[-3:]:
            line.set_xdata(conv_distance)

        ax.set_xlabel(r"$x_{conv}$")
        return fig, ax

class CHAPSim_FIK_tg(cp.CHAPSim_FIK_tg):
    pass

class CHAPSim_FIK_perturb(CHAPSim_FIK_tg):
    @property
    def start_index(self):
        accel_start = self.metaDF['temp_start_end'][0] + 1
        return self.avg_data._return_index(accel_start)

    @property
    def shape(self):
        avg_shape = self.avg_data.shape
        return (avg_shape[0],avg_shape[1]- self.start_index)

    def _scale_vel(self,PhyTime):
        
        index = self.start_index
        bulk_velo = self.avg_data.bulk_velo_calc() 
        return bulk_velo[index:] - bulk_velo[0]

    def _laminar_extract(self,PhyTime):
        bulk = self._scale_vel(PhyTime)
        REN = self.avg_data.metaDF['REN']
        const = 4.0 if self.Domain.is_polar else 6.0
        return const/(REN*bulk)

    def _turbulent_extract(self,PhyTime):
        bulk = self._scale_vel(PhyTime)
        index = self.start_index

        y_coords = self.avg_data.CoordDF['y']
        uv = self.avg_data.UU_tensorDF[PhyTime,'uv']

        turbulent = np.zeros(self.shape)
        for i,y in enumerate(y_coords):
            turbulent[i] =    6*y*(uv[i,index:] - uv[i,index])

        return self.Domain.Integrate_tot(self.CoordDF,turbulent)/bulk**2

    def _inertia_extract(self,PhyTime):
        PhyTime=None
        y_coords = self.avg_data.CoordDF['y']

        bulk = self._scale_vel(PhyTime)

        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        times = self.avg_data._return_xaxis()

        REN = self.avg_data.metaDF['REN']
        d2u_dy2 = self.Domain.Grad_calc(self.avg_data.CoordDF,
                    self.Domain.Grad_calc(self.avg_data.CoordDF,U_mean,'y'),'y')
        
        uv = self.avg_data.UU_tensorDF[PhyTime,'uv']
        duv_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uv,'y')
        dudt = np.gradient(U_mean,times,axis=-1,edge_order=2)

        dpdx = (1/REN)*d2u_dy2 - duv_dy  - dudt

        dp_prime_dx = dpdx - self.Domain.Integrate_tot(self.CoordDF,
                                            (1/REN)*d2u_dy2 - duv_dy) 

        UV = self.avg_data.flow_AVGDF[PhyTime,'u']*self.avg_data.flow_AVGDF[PhyTime,'v']
        I_x = self.Domain.Grad_calc(self.avg_data.CoordDF,UV,'y')

        I_x_prime = I_x - self.Domain.Integrate_tot(self.CoordDF,I_x)

        index = self.start_index

        temp = dp_prime_dx + I_x_prime + dudt
        out = np.zeros(self.shape)
        for i,y in enumerate(y_coords):
            out[i] = (temp[i,index:] - temp[i,index])*y**2

        return -3.0*self.Domain.Integrate_tot(self.CoordDF,out)/(bulk**2)

    def plot(self,budget_terms=None,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        budget_terms = self._check_terms(budget_terms)
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= cplt.update_line_kw(line_kw)
        xaxis_vals = self.avg_data.times[self.start_index:] - self.avg_data.times[self.start_index]


        for comp in budget_terms:
            budget_term = self.budgetDF[PhyTime,comp].copy()
                
            label = r"%s"%comp.title()
            ax.cplot(xaxis_vals,budget_term,label=label,**line_kw)
        ax.cplot(xaxis_vals,np.sum(self.budgetDF.values,axis=0),label="Total",**line_kw)

        ax.set_xlabel(r"$t^*$")

        ncol = cplt.get_legend_ncols(len(budget_terms))
        ax.clegend(ncol=ncol,vertical=False)
        return fig, ax