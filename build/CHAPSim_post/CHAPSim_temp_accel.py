"""
# CHAPSim_temp_accel 
Module for processing temporal acceleration

"""

from . import CHAPSim_post as cp
from . import CHAPSim_Tools as CT
from . import CHAPSim_plot as cplt

import sys
import warnings
import numpy as np
from scipy import integrate


class CHAPSim_Inst(cp.CHAPSim_Inst_tg):
    _module = sys.modules[__name__]
_instant_class = CHAPSim_Inst

class CHAPSim_AVG_custom_t(cp.CHAPSim_AVG_tg_base):
    _module = sys.modules[__name__]

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

        if fig is None:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig,ax=cplt.subplots(**kwargs)
        elif ax is None:
            ax = fig.add_subplot(1,1,1)

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
    _module = sys.modules[__name__]
    def _extract_avg(self,path_to_folder='',time0='',abs_path=True,*args,**kwargs):

        if isinstance(path_to_folder,list):
            times = CT.time_extract(path_to_folder[0],abs_path)
        else:
            times = CT.time_extract(path_to_folder,abs_path)
        if time0:
            times = list(filter(lambda x: x > time0, times))
        return super()._extract_avg(times,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path,*args,**kwargs)
        
_avg_tg_class = CHAPSim_AVG_tg

class CHAPSim_perturb():
    def __init__(self,avg_data='', meta_data='',path_to_folder='',time0='',abs_path=True):
        if avg_data:
            if not isinstance(avg_data,cp.CHAPSim_AVG_tg_base):
                msg = f"avg_data must be a subclass of {cp.CHAPSim_AVG_tg_base}"
                raise TypeError(msg)
            self.__avg_data = avg_data
        else:
            self.__avg_data = CHAPSim_AVG_tg(path_to_folder,time0,abs_path,meta_data=meta_data)
        if not meta_data:
            meta_data = self.__avg_data._meta_data
        self._meta_data = meta_data
        self.start  = self._meta_data.metaDF['accel_start_end'][0]
    def tau_du_calc(self):
        
        tau_w = self.__avg_data.tau_calc()
        return tau_w - tau_w[0]

    def mean_velo_peturb_calc(self,comp):
        U_velo_mean = self.__avg_data.flow_AVGDF[None,comp].copy()
        # wall_velo = self._meta_data.moving_wall_calc()
        # for i in range(self.__avg_data.shape[0]):
        #     U_velo_mean[i] -= wall_velo

        # start = self._meta_data.metaDF['loc_start_end'][0]*self._meta_data.metaDF['HX_tg_io'][1]
        self.start
        time_0_index = self.__avg_data._return_index(self.start)
        # x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)
        
        centre_index =int(0.5*self.__avg_data.shape[0])
        U_c0 = U_velo_mean[centre_index,0]
        mean_velo_peturb = np.zeros((self.__avg_data.shape[0],self.__avg_data.shape[1]-time_0_index))
        for i in range(time_0_index,self.__avg_data.shape[1]):
            mean_velo_peturb[:,i-time_0_index] = (U_velo_mean[:,i]-U_velo_mean[:,0])/(U_velo_mean[centre_index,i]-U_c0)
        return mean_velo_peturb
    def plot_perturb_velo(self,times,comp='u',Y_plus=False,Y_plus_max=100,fig='',ax ='',**kwargs):
        velo_peturb = self.mean_velo_peturb_calc(comp)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        y_coord = self._meta_data.CoordDF['y']
        
        # print(self.__avg_data.shape)
        u_tau_star, delta_v_star = self.__avg_data.wall_unit_calc()
        if Y_plus:
            y_coord = y_coord[:int(y_coord.size/2)]
            y_coord = (1-np.abs(y_coord))/delta_v_star[0]
            velo_peturb = velo_peturb[:int(y_coord.size)]
        else:
            y_max= Y_plus_max*delta_v_star[0]-1.0

        # start = self._meta_data.metaDF['loc_start_end'][0]*self._meta_data.metaDF['HX_tg_io'][1]
        time_0_index = self.__avg_data._return_index(self.start)
        time_loc = np.array([self.__avg_data._return_index(x) for x in times]) - time_0_index
        # x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',x_vals)
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

    def plot_peturb_cf(self,wall_units=False,fig='',ax='',**kwargs):

        tau_du = self.tau_du_calc()
        bulkvelo = self.__avg_data._bulk_velo_calc(None)

        # start = self._meta_data.metaDF['loc_start_end'][0]*self._meta_data.metaDF['HX_tg_io'][1]
        x_loc = self.__avg_data._return_index(self.start)+1

        REN = self._meta_data.metaDF['REN']
        rho_star = 1.0
        Cf_du = tau_du[x_loc:]/(0.5*REN*rho_star*(bulkvelo[x_loc:]-bulkvelo[0])**2)
        
        
        times = self.__avg_data._return_xaxis()[x_loc:] - self.start
        
        # print(bulkvelo[x_loc:]-bulkvelo[0])
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
            
        ax.cplot(times, Cf_du)
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$C_{f,du}$")
        ax.set_ylim([0,10*Cf_du[-1]])
        return fig, ax

    def int_thickness_calc(self):

        # if len(set([x[0] for x in self.__avg_data.UU_tensorDF.index])) == 1:
        #     avg_time = list(set([x[0] for x in self.__avg_data.UU_tensorDF.index]))[0]
        #     if PhyTime and PhyTime != avg_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
        #     PhyTime = avg_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.__avg_data.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        mean_velo = self.mean_velo_peturb_calc('u')

        # start = self._meta_data.metaDF['loc_start_end'][0]*self._meta_data.metaDF['HX_tg_io'][1]
        print(self.start)
        x_loc = self.__avg_data._return_index(self.start)+1
        print(x_loc)
        y_coords = self.__avg_data.CoordDF['y']

        U0_index = int(self.__avg_data.shape[0]*0.5)
        # theta_integrand = np.zeros((U0_index,self.__avg_data.shape[1]))
        # delta_integrand = np.zeros((U0_index,self.__avg_data.shape[1]))
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

    def plot_shape_factor(self,fig='',ax='',**kwargs):
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)

        # start = self._meta_data.metaDF['loc_start_end'][0]*self._meta_data.metaDF['HX_tg_io'][1]
        x_loc = self.__avg_data._return_index(self.start)+1

        times = self.__avg_data._return_xaxis()[x_loc:] 
        times -= self.start
        _, _, H = self.int_thickness_calc()

        ax.cplot(times, H,label=r"$H$")
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$H$")
        ax.set_ylim([0,2*H[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_mom_thickness(self,fig='',ax='',**kwargs):

        
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)

        # start = self._meta_data.metaDF['loc_start_end'][0]*self._meta_data.metaDF['HX_tg_io'][1]
        # x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1
        x_loc = self.__avg_data._return_index(self.start)+1

        times = self.__avg_data._return_xaxis()[x_loc:] 
        times -= self.start
        _, theta, _ = self.int_thickness_calc()

        ax.cplot(times, theta,label=r"$\theta$")
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$\theta$")
        ax.set_ylim([0,2*theta[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_disp_thickness(self,fig='',ax='',**kwargs):
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)

        # start = self._meta_data.metaDF['loc_start_end'][0]*self._meta_data.metaDF['HX_tg_io'][1]
        # x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        x_loc = self.__avg_data._return_index(self.start)+1

        times = self.__avg_data._return_xaxis()[x_loc:] 
        times -= self.start
        # _, theta, _ = self.int_thickness_calc()
        delta, _, _ = self.int_thickness_calc()

        ax.cplot(times, delta,label=r"$\delta^*$")
        ax.set_xlabel(r"$t^*$")
        ax.set_ylabel(r"$\delta^*$")
        ax.set_ylim([0,2*delta[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_meta(cp.CHAPSim_meta):
    _module = sys.modules[__name__]
    def __init__(self,*args,**kwargs):
        if len(args) < 3:
            kwargs['tgpost'] = True
        super().__init__(*args,**kwargs)
_meta_class = CHAPSim_meta

class CHAPSim_fluct_tg(cp.CHAPSim_fluct_tg):
    _module = sys.modules[__name__]
_fluct_tg_class = CHAPSim_fluct_tg

class CHAPSim_budget_tg(cp.CHAPSim_budget_tg):
    _module = sys.modules[__name__]

class CHAPSim_autocov_tg(cp.CHAPSim_autocov_tg):
    _module = sys.modules[__name__]

class CHAPSim_Quad_Anl_tg(cp.CHAPSim_Quad_Anl_tg):
    _module = sys.modules[__name__]

