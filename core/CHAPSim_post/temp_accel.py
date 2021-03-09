"""
# CHAPSim_temp_accel 
Module for processing temporal acceleration

"""

from . import post as cp
from . import plot as cplt
from .utils import misc_utils
import sys
import warnings
import numpy as np
from scipy import integrate



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
        kwargs['shift_vals'] = [float(metaDF['accel_start_end'][0]) for metaDF in metaDF_list]

        return super().with_phase_average(*args,**kwargs)

    def shift_times(self,val):
        super().shift_times(val)
        self._metaDF['accel_start_end'][0] -= val
        self._metaDF['accel_start_end'][1] -= val

    def conv_distance_calc(self):
        
        bulk_velo = self.bulk_velo_calc()
        times = self._return_xaxis()

        conv_distance = np.zeros_like(bulk_velo)
        for i , _ in enumerate(bulk_velo):
            conv_distance[i] = integrate.simps(bulk_velo[:(i+1)],times[:(i+1)])
        return conv_distance

    def accel_param_calc(self):
        U_mean = self.flow_AVGDF[None,'u']
        U_infty = U_mean[int(self.NCL[1]*0.5)]

        times = self._return_xaxis()
        dudt = np.gradient(U_infty,times,edge_order=2)
        REN = self._metaDF['REN']

        accel_param = (1/(REN*U_infty**3))*dudt
        return accel_param

    def plot_accel_param(self,convective=False,fig=None,ax=None,**kwargs):
        accel_param = self.accel_param_calc()
        if convective:
            xaxis = self.conv_distance_calc()
        else:
            xaxis = self._return_xaxis()

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        ax.cplot(xaxis,accel_param,label=r"$K$")

        if convective:
            ax.set_xlabel(r"$x^*_{conv}$")# ,fontsize=18)
        else:
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

class CHAPSim_perturb():
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
        self.start  = self._meta_data.metaDF['accel_start_end'][0]
    def tau_du_calc(self):
        
        tau_w = self.__avg_data.tau_calc()
        return tau_w - tau_w[0]

    def mean_velo_peturb_calc(self,comp):
        U_velo_mean = self.__avg_data.flow_AVGDF[None,comp].copy()

        self.start
        time_0_index = self.__avg_data._return_index(self.start)
        
        centre_index =int(0.5*self.__avg_data.shape[0])
        U_c0 = U_velo_mean[centre_index,0]
        mean_velo_peturb = np.zeros((self.__avg_data.shape[0],self.__avg_data.shape[1]-time_0_index))
        for i in range(time_0_index,self.__avg_data.shape[1]):
            mean_velo_peturb[:,i-time_0_index] = (U_velo_mean[:,i]-U_velo_mean[:,0])/(U_velo_mean[centre_index,i]-U_c0)
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

        axes_items_num = len(ax.get_lines())
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol, fontsize=16)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_peturb_cf(self,wall_units=False,fig=None,ax=None,**kwargs):

        tau_du = self.tau_du_calc()
        bulkvelo = self.__avg_data._bulk_velo_calc(None)

        x_loc = self.__avg_data._return_index(self.start)+1

        REN = self._meta_data.metaDF['REN']
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

        print(self.start)
        x_loc = self.__avg_data._return_index(self.start)+1
        print(x_loc)
        y_coords = self.__avg_data.CoordDF['y']

        U0_index = int(self.__avg_data.shape[0]*0.5)
        mom_thickness = np.zeros(self.__avg_data.shape[1]-x_loc)
        disp_thickness = np.zeros(self.__avg_data.shape[1]-x_loc)
        theta_integrand = mean_velo[:U0_index]*(1-mean_velo[:U0_index])
        delta_integrand = 1-mean_velo[:U0_index]
        print(delta_integrand.shape,disp_thickness.shape)
        for j in range(self.__avg_data.shape[1]-x_loc):
            mom_thickness[j] = integrate.simps(theta_integrand[:,j],y_coords[:U0_index])
            disp_thickness[j] = integrate.simps(delta_integrand[:,j],y_coords[:U0_index])
        shape_factor = np.divide(disp_thickness,mom_thickness)

        

        return disp_thickness, mom_thickness, shape_factor

    def plot_shape_factor(self,fig=None,ax=None,**kwargs):

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        x_loc = self.__avg_data._return_index(self.start)+1

        times = self.__avg_data._return_xaxis()[x_loc:] - self.start

        _, _, H = self.int_thickness_calc()

        ax.cplot(times, H,label=r"$H$")
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

class CHAPSim_fluct_tg(cp.CHAPSim_fluct_tg):
    pass
_fluct_tg_class = CHAPSim_fluct_tg

class CHAPSim_budget_tg(cp.CHAPSim_budget_tg):
    pass
class CHAPSim_autocov_tg(cp.CHAPSim_autocov_tg):
    pass

class CHAPSim_Quad_Anl_tg(cp.CHAPSim_Quad_Anl_tg):
    pass

