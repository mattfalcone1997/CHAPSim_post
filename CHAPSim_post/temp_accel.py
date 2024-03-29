"""
# CHAPSim_temp_accel 
Module for processing temporal acceleration

"""

from . import post as cp
from . import dtypes as cd
from . import plot as cplt
from .utils import indexing
import types
import numpy as np

import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

class temp_accel_base(cp.temporal_base):
    @classmethod
    def phase_average(cls, *objects_temp, items=None):
        temp_object = super().phase_average(*objects_temp, items=items)
        
        accel_info = temp_object.metaDF['temp_start_end'].copy()        
        temp_object.metaDF['temp_start_end'] = [x + \
            temp_object._time_shift for x in accel_info]
        
        return temp_object
        
    @property
    def _time_shift(self):
        return -self.metaDF['temp_start_end'][0]

    @classmethod
    def _get_time_shift(cls,path):
        meta_data = cls._module._meta_class(path,tgpost=True)
        return - meta_data.metaDF['temp_start_end'][0]
            
    def _shift_times(self,time):
        super()._shift_times(time)
        
        self.metaDF['temp_start_end'] = [ x + time \
                                    for x in self.metaDF['temp_start_end']]
    
class CHAPSim_Inst_temp(cp.CHAPSim_Inst_temp):
    pass
_inst_temp_class = CHAPSim_Inst_temp

class CHAPSim_AVG_temp(temp_accel_base,cp.CHAPSim_AVG_temp):
    
            
    def conv_distance_calc(self):
        
        bulk_velo = self.bulk_velo_calc()

        time0 = self.times[0]
        times = [x-time0 for x in self.times]

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
        U_infty = U_mean[self.NCL[1] // 2 ]

        times = self.times
        dudt = np.gradient(U_infty,times,edge_order=2)
        REN = self.metaDF['REN']

        accel_param = (1/(REN*U_infty**3))*dudt
        return accel_param

    def plot_accel_param(self,fig=None,ax=None,line_kw=None,**kwargs):
        accel_param = self.accel_param_calc()

        xaxis = self._return_xaxis()

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw = cplt.update_line_kw(line_kw,label = r"$K$")


        ax.cplot(xaxis,accel_param,**line_kw)

        ax.set_xlabel(r"$t^*$")# ,fontsize=18)
        ax.set_ylabel(r"$K$")# ,fontsize=18)

        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        #ax.grid()
        fig.tight_layout()
        return fig,ax
    
    def _get_data_attr(self):
        data_dict__ = {x : self.__dict__[x] for x in self.__dict__ \
                        if not isinstance(x,types.MethodType)}
        return data_dict__

_avg_temp_class = CHAPSim_AVG_temp
_avg_tg_class = cp.CHAPSim_AVG_tg

class CHAPSim_AVG_temp_conv(CHAPSim_AVG_temp):
        
    def __init__(self,other_avg):
        
        self.__dict__.update(other_avg._get_data_attr())
        
        self.CoordDF['x'] = self.conv_distance_calc()

    def _return_index(self,x_val):
        return indexing.coord_index_calc(self.CoordDF,'x',x_val)

    def _return_time_index(self,time):
        return super()._return_index(time)
    def _return_xaxis(self):
        return self.CoordDF['x']

    def plot_accel_param(self,*args,**kwargs):
        fig, ax = super().plot_accel_param(*args,**kwargs)    
        line = ax.get_lines()[-1]
        xdata = self.conv_distance_calc()
        line.set_xdata(xdata)

        ax.set_xlabel(r"$x_{conv}$")

        return fig, ax
    
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
            line.set_xdata(self.CoordDF['x'])
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
        
        return fig, ax

    def plot_Reynolds_x(self,*args,**kwargs):
        fig, ax = super().plot_Reynolds_x(*args,**kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax  

    def plot_bulk_velocity(self,*args,**kwargs):
        fig, ax = super().plot_bulk_velocity(*args,**kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
        ax.set_xlabel(r"$x_{conv}$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_skin_friction(self,*args,**kwargs):
        fig, ax = super().plot_skin_friction(*args,**kwargs)
        ax.get_lines()[-1].set_xdata(self.CoordDF['x'])
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

        return fig, ax

    def plot_mean_flow(self,x_vals,*args,**kwargs):
        fig, ax = super().plot_mean_flow(x_vals,*args,**kwargs)

        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))
            
        return fig, ax

    def plot_near_wall(self,x_vals,*args,**kwargs):
        fig, ax = super().plot_near_wall(x_vals,*args,**kwargs)

        x_vals = indexing.true_coords_from_coords(self.CoordDF,'x',x_vals)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x_{conv}=%.3g$"%float(x))

        return fig, ax

from CHAPSim_post.post._common import Common
class CHAPSim_perturb(Common):
    def __init__(self,avg_data=None, meta_data=None,path_to_folder='.',time0=None,abs_path=True):
        if avg_data is not None:
            if not isinstance(avg_data,cp.CHAPSim_AVG_temp):
                msg = f"avg_data must be a subclass of {cp.CHAPSim_AVG_temp}"
                raise TypeError(msg)
            self.__avg_data = avg_data
        else:
            self.__avg_data = CHAPSim_AVG_temp(path_to_folder,time0,abs_path,meta_data=meta_data)
        if meta_data is None:
            meta_data = self.__avg_data._meta_data
        self._meta_data = meta_data
        self.start  = self.metaDF['temp_start_end'][0]

    @property
    def shape(self):
        time_0_index = self.__avg_data._return_index(self.start) -1
        shape = (self.__avg_data._shape_devel[0],self.__avg_data._shape_devel[1]-time_0_index)
        return shape

    def tau_du_calc(self):
        
        tau_w = self.__avg_data.tau_calc()
        start_index = self.__avg_data._return_index(self.start)-1
        return tau_w - tau_w[start_index]

    def mean_velo_peturb_calc(self,comp):
        U_velo_mean = self.__avg_data.flow_AVGDF[None,comp].copy()
    
        time_0_index = self.__avg_data._return_index(self.start) -1

        centre_index =self.__avg_data.shape[0] // 2
        U_c0 = U_velo_mean[centre_index,time_0_index]
        mean_velo_peturb = np.zeros((self.__avg_data.shape[0],self.__avg_data._shape_devel[1]-time_0_index))

        for i in range(time_0_index,self.__avg_data._shape_devel[1]):
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

        time_0_index = self.__avg_data._return_index(self.start)
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

        return fig, ax

    def plot_perturb_cf(self,wall_units=False,fig=None,ax=None,**kwargs):

        tau_du = self.tau_du_calc()
        bulkvelo = self.__avg_data._bulk_velo_calc(None)
        x_loc = self.__avg_data._return_index(self.start)+1

        REN = self.metaDF['REN']
        rho_star = 1.0
        Cf_du = tau_du[x_loc:]/(0.5*REN*rho_star*(bulkvelo[x_loc:]-bulkvelo[0])**2)
        
        
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
        mom_thickness = np.zeros(self.__avg_data._shape_devel[1]-x_loc)
        disp_thickness = np.zeros(self.__avg_data._shape_devel[1]-x_loc)
        theta_integrand = mean_velo[:U0_index]*(1-mean_velo[:U0_index])
        delta_integrand = 1-mean_velo[:U0_index]

        for j in range(self.__avg_data._shape_devel[1]-x_loc):
            mom_thickness[j] = integrate_simps(theta_integrand[:,j],y_coords[:U0_index])
            disp_thickness[j] = integrate_simps(delta_integrand[:,j],y_coords[:U0_index])
        shape_factor = np.divide(disp_thickness,mom_thickness)

        

        return disp_thickness, mom_thickness, shape_factor

    def plot_shape_factor(self,fig=None,ax=None,line_kw=None,**kwargs):

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_loc = self.__avg_data._return_index(self.start)+1

        times = np.array(self.__avg_data._return_xaxis())[x_loc:] - self.start

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

        times = np.array(self.__avg_data._return_xaxis())[x_loc:] - self.start

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

        times = np.array(self.__avg_data.times[x_loc:]) - self.start

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
class CHAPSim_fluct_temp(cp.CHAPSim_fluct_temp):
    pass
_fluct_temp_class = CHAPSim_fluct_temp

class CHAPSim_budget_temp(cp.CHAPSim_budget_temp):
    pass

class CHAPSim_k_budget_temp(cp.CHAPSim_k_budget_temp):
    pass

class CHAPSim_momentum_budget_temp(cp.CHAPSim_momentum_budget_temp):
    def __init__(self,comp,avg_data,PhyTime=None,apparent_Re=False):
        
        super().__init__(comp,avg_data,PhyTime)
        
        if apparent_Re:
            self._create_uniform(PhyTime)
    def _budget_extract(self, PhyTime, comp):
        transient  = {(PhyTime,'transient') : self._transient_extract(PhyTime,comp)}
        budgetDF = self._flowstruct_class(self._coorddata,transient)

        otherBudgetDF =  super()._budget_extract(PhyTime, comp)
        budgetDF.concat(otherBudgetDF)
        return budgetDF
        # array_concat.insert(0,transient)
        # budget_index.insert(0,'transient')
        # phystring_index = [PhyTime]*5

    def _create_uniform(self,PhyTime):
        advection = self.budgetDF[PhyTime,'transient']
        centre_index = advection.shape[0] // 2
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
        dudt = np.gradient(U,times,axis=1,edge_order=2)

        index = self.avg_data._return_index(PhyTime)
        return -dudt[:,index]

    def _pressure_grad(self, PhyTime, comp):
        if comp != 'u':
            return np.zeros(self.shape)

        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        times = self.avg_data._return_xaxis()

        REN = self.avg_data.metaDF['REN']
        d2u_dy2 = self.Domain.Grad_calc(self.avg_data.CoordDF,
                    self.Domain.Grad_calc(self.avg_data.CoordDF,U_mean,'y'),'y')
        
        uv = self.avg_data.UU_tensorDF[PhyTime,'uv']
        duv_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uv,'y')
        dudt = -self._transient_extract(PhyTime,comp)

        return -( (1/REN)*d2u_dy2 - duv_dy  - dudt)
class CHAPSim_autocov_tg(cp.CHAPSim_autocov_tg):
    pass

class CHAPSim_Quad_Anl_tg(cp.CHAPSim_Quad_Anl_tg):
    pass

class CHAPSim_FIK_temp_conv(cp.CHAPSim_FIK_temp):
    def plot(self,*args,**kwargs):
        fig, ax = super().plot(*args,**kwargs)
        conv_distance = self.avg_data.conv_distance_calc()

        for line in ax.set_lines()[-3:]:
            line.set_xdata(conv_distance)

        ax.set_xlabel(r"$x_{conv}$")
        return fig, ax

class CHAPSim_FIK_temp(cp.CHAPSim_FIK_temp):
    pass

class velocitySpectra1D_temp(temp_accel_base,cp.velocitySpectra1D_temp):
    pass