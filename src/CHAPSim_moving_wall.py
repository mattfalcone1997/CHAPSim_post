import h5py
import numpy as np
import os
import itertools
import time
import warnings
from scipy import integrate, fft
import sys

import pandas as pd
import CHAPSim_post_v2 as cp
import CHAPSim_Tools as CT
import CHAPSim_plot as cplt

# TEST = cp.TEST
class CHAPSim_Inst(cp.CHAPSim_Inst):
    pass

class CHAPSim_AVG_io(cp.CHAPSim_AVG_io):
    module = sys.modules[__name__]
    def _extract_avg(self,time,meta_data='',path_to_folder='',time0='',abs_path=True):
        if not meta_data:
            meta_data = CHAPSim_meta(path_to_folder,abs_path,False)

        return super()._extract_avg(time,meta_data=meta_data,path_to_folder=path_to_folder,
                                    time0=time0,abs_path=abs_path)

    def _hdf_extract(self,file_name,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_io'
        return_list = list(super()._hdf_extract(file_name,group_name))
        return_list[0] = CHAPSim_meta.from_hdf(file_name,group_name+'/meta_data')
        
        return itertools.chain(return_list)

    # def _copy_extract(self, avg_data):
    #      return_list = list(super()._copy_extract(avg_data))
    #      return_list[0] = CHAPSim_meta.copy(avg_data._meta_data)
    #      return itertools.chain(return_list)
    def _tau_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and not np.isnan(PhyTime):
            PhyTime = "{:.9g}".format(PhyTime)

        u_velo = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape(self.shape)
        ycoords = self.CoordDF['y'].dropna().values
        
        wall_velo = self._meta_data.moving_wall_calc()
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        for i in range(self.shape[1]):
            tau_star[i] = mu_star*(u_velo[0,i]-wall_velo[i])/(ycoords[0]--1.0)#*(-1*u_velo[1,i] + 4*u_velo[0,i] - 3*wall_velo[i])/(0.5*ycoords[1]-1.5*(-1.0)+y_coords[0])
    
        return tau_star

    def _bulk_velo_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and not np.isnan(PhyTime):
            PhyTime = "{:.9g}".format(PhyTime)
            
        u_velo = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape(self.shape)
        ycoords = self.CoordDF['y'].dropna().values
        wall_velo = self._meta_data.moving_wall_calc()
        
        bulk_velo=np.zeros(self.shape[1])
        for i in range(self.NCL[1]):
            u_velo[i,:]=u_velo[i,:] - wall_velo
        for i in range(self.shape[1]):
            bulk_velo[i] = 0.5*integrate.simps(u_velo[:,i],ycoords)
            
        return bulk_velo

    def accel_param_calc(self,PhyTime=''):
        if type(PhyTime) == float:
            PhyTime = "{:.9g}".format(PhyTime)

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"


        U_mean = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape((self.NCL[1],self.NCL[0]))
        U0_index = int(np.floor(self.NCL[1]*0.5))
        U0 = U_mean[U0_index]
        wall_velo = self._meta_data.moving_wall_calc()
        x_coords = self.CoordDF['x'].dropna().values
        
        U_infty_grad = np.zeros(self.NCL[0])
        U_infty = U0 - wall_velo
        REN = self._metaDF.loc['REN'].values[0]
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

    def plot_accel_param(self,PhyTime='',fig='',ax='',**kwargs):
        accel_param = self.accel_param_calc(PhyTime)
        x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig,ax=cplt.subplots(**kwargs)
        elif not ax:
            ax =fig.add_subplot(1,1,1)
        
        ax.cplot(x_coords,accel_param)
        ax.set_xlabel(r"$x/\delta$")# ,fontsize=18)
        ax.set_ylabel(r"$K$")# ,fontsize=18)
        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        #ax.grid()
        fig.tight_layout()
        return fig,ax

    def _int_thickness_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and not np.isnan(PhyTime):
            PhyTime = "{:.9g}".format(PhyTime)
        U0_index = int(np.floor(self.NCL[1]*0.5))
        U_mean = self.flow_AVGDF.loc[PhyTime,'u'].copy().values.reshape(self.shape)
        
        wall_velo = self._meta_data.moving_wall_calc()
        for i in range(wall_velo.size):
            U_mean[:,i] -= wall_velo[i]

        y_coords = self.CoordDF['y'].dropna().values
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
        
    def plot_near_wall(self,x_vals,PhyTime='',*args,**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        fig, ax = super().plot_near_wall(x_vals,*args,**kwargs)
        wall_velo = self._meta_data.moving_wall_calc()
        u_tau_star, delta_v_star = self.wall_unit_calc(PhyTime)
        lines = ax.get_lines()[-len(x_vals)-1:-1]
        for line, x_val in zip(lines,x_vals):
            y_data = line.get_ydata()
            x = self._return_index(x_val)
            y_data = y_data - wall_velo[x]/u_tau_star[x]
            line.set_ydata(y_data)
        ax.relim()
        ax.autoscale_view()
        return fig, ax

class CHAPSim_AVG():
    def __new__(cls,*args,tgpost=False,**kwargs):
        if not tgpost:
            return CHAPSim_AVG_io(*args,**kwargs)
        else:
            return cp.CHAPSim_AVG_tg(*args, **kwargs)

class CHAPSim_perturb():
    def __init__(self,time='',avg_data='', meta_data='',path_to_folder='',time0='',abs_path=True):
        if avg_data:
            self.__avg_data = avg_data
            times = list(set([x[0] for x in avg_data.UU_tensorDF.index]))
            if len(times)>1 and not times:
                msg="If there is more than one time present in"+\
                            " CHAPSim_AVG_io, a time must be provided"
                raise ValueError(msg)
        else:
            assert time, "In the absence of input of class CHAPSim_AVG a time must be provided"
            if not path_to_folder:
                warnings.warn("\033[1;33No path_to_folder selected in the absence of an CHAPSim_AVG input class")
            if not time0:
                warnings.warn("\033[1;33No time0 input selected in the absence of an CHAPSim_AVG input class")
            self.__avg_data = CHAPSim_AVG_io(time,meta_data,path_to_folder,time0,abs_path,False)
        if not meta_data:
            meta_data = self.__avg_data._meta_data
        self._meta_data = meta_data

    def tau_du_calc(self,PhyTime=''):
        if not PhyTime:
            PhyTime = self.__avg_data.flow_AVGDF.index[0][0]
        tau_w = self.__avg_data.tau_calc(PhyTime)
        return tau_w - tau_w[0]

    def mean_velo_peturb_calc(self,comp):
        U_velo_mean = self.__avg_data.flow_AVGDF.loc[self.__avg_data.flow_AVGDF.index[0][0],comp]\
                        .values.copy().reshape((self.__avg_data.NCL[1],self.__avg_data.NCL[0]))
        wall_velo = self._meta_data.moving_wall_calc()
        for i in range(self.__avg_data.shape[0]):
            U_velo_mean[i] -= wall_velo

        start = self._meta_data.metaDF.loc['loc_start_end'].values[0]*self._meta_data.metaDF.loc['HX_tg_io'].values[1]
        x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)
        
        centre_index =int(0.5*self.__avg_data.shape[0])
        U_c0 = U_velo_mean[centre_index,0]
        mean_velo_peturb = np.zeros((self.__avg_data.shape[0],self.__avg_data.shape[1]-x_loc))
        for i in range(x_loc,self.__avg_data.shape[1]):
            mean_velo_peturb[:,i-x_loc] = (U_velo_mean[:,i]-U_velo_mean[:,0])/(U_velo_mean[centre_index,i]-U_c0)
        return mean_velo_peturb

    def plot_perturb_velo(self,x_vals,PhyTime='',comp='u',Y_plus=False,Y_plus_max=100,fig='',ax ='',**kwargs):
        velo_peturb = self.mean_velo_peturb_calc(comp)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        y_coord = self._meta_data.CoordDF['y'].dropna().values
        if not PhyTime:
            PhyTime = self.__avg_data.flow_AVGDF.index[0][0]
        print(self.__avg_data.shape)
        u_tau_star, delta_v_star = self.__avg_data.wall_unit_calc(PhyTime)
        if Y_plus:
            y_coord = y_coord[:int(y_coord.size/2)]
            y_coord = (1-np.abs(y_coord))/delta_v_star[0]
            velo_peturb = velo_peturb[:int(y_coord.size)]
        else:
            y_max= Y_plus_max*delta_v_star[0]-1.0

        start = self._meta_data.metaDF.loc['loc_start_end'].values[0]*self._meta_data.metaDF.loc['HX_tg_io'].values[1]
        x_vals = [x-start for x in x_vals]
        x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',x_vals)
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

        axes_items_num = len(ax.get_lines())
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol, fontsize=16)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_peturb_cf(self,PhyTime='',wall_units=False,fig='',ax='',**kwargs):
        if not PhyTime:
            PhyTime = self.__avg_data.flow_AVGDF.index[0][0]
        tau_du = self.tau_du_calc(PhyTime)
        bulkvelo = self.__avg_data._bulk_velo_calc(PhyTime)

        start = self._meta_data.metaDF.loc['loc_start_end'].values[0]*self._meta_data.metaDF.loc['HX_tg_io'].values[1]
        x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        REN = self._meta_data.metaDF.loc['REN'].values[0]
        rho_star = 1.0
        Cf_du = tau_du[x_loc:]/(0.5*REN*rho_star*(bulkvelo[x_loc:]-bulkvelo[0])**2)
        
        x_coords = self._meta_data.CoordDF['x'].dropna().values[x_loc:] 
        x_coords -= start
        # print(bulkvelo[x_loc:]-bulkvelo[0])
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
            
        ax.cplot(x_coords, Cf_du)
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$C_{f,du}$")
        ax.set_ylim([0,2*Cf_du[-1]])
        return fig, ax
    def int_thickness_calc(self,PhyTime=''):
        if not PhyTime:
            PhyTime = self.__avg_data.flow_AVGDF.index[0][0]

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        mean_velo = self.mean_velo_peturb_calc('u')

        start = self._meta_data.metaDF.loc['loc_start_end'].values[0]*self._meta_data.metaDF.loc['HX_tg_io'].values[1]
        x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        y_coords = self.__avg_data.CoordDF['y'].dropna().values

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

    def plot_shape_factor(self,PhyTime='',fig='',ax='',**kwargs):
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)

        start = self._meta_data.metaDF.loc['loc_start_end'].values[0]*self._meta_data.metaDF.loc['HX_tg_io'].values[1]
        x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        x_coords = self._meta_data.CoordDF['x'].dropna().values[x_loc:] 
        x_coords -= start
        delta, theta, H = self.int_thickness_calc(PhyTime)

        ax.cplot(x_coords, H)
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$H$")
        ax.set_ylim([0,2*H[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_mom_thickness(self,PhyTime='',fig='',ax='',**kwargs):

        
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)

        start = self._meta_data.metaDF.loc['loc_start_end'].values[0]*self._meta_data.metaDF.loc['HX_tg_io'].values[1]
        x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        x_coords = self._meta_data.CoordDF['x'].dropna().values[x_loc:] 
        x_coords -= start
        delta, theta, H = self.int_thickness_calc(PhyTime)

        ax.cplot(x_coords, theta,label=r"$\theta$")
        ax.set_xlabel(r"$x/\delta$")
        ax.set_ylabel(r"$\theta$")
        ax.set_ylim([0,2*theta[-1]])
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_disp_thickness(self,PhyTime='',fig='',ax='',**kwargs):
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)

        start = self._meta_data.metaDF.loc['loc_start_end'].values[0]*self._meta_data.metaDF.loc['HX_tg_io'].values[1]
        x_loc = CT.coord_index_calc(self.__avg_data.CoordDF,'x',start)+1

        x_coords = self._meta_data.CoordDF['x'].dropna().values[x_loc:] 
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
    
    def _extract_moving_wall(self,file_name,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_meta'
        hdf_file = h5py.File(file_name,'r')
        moving_wall = hdf_file[base_name+"/moving_wall"][:]
        hdf_file.close()
        return moving_wall

    def __moving_wall_setup(self,NCL,path_to_folder,abs_path,metaDF,tgpost):
        wall_velo = np.zeros(NCL[0])
        if int(metaDF.loc['moving_wallflg',0]) == 1 and not tgpost:
            if not abs_path:
                file_path = os.path.abspath(os.path.join(path_to_folder,'CHK_MOVING_WALL.dat'))
            else:
                file_path = os.path.join(path_to_folder,'CHK_MOVING_WALL.dat')
            
            mw_file = open(file_path)
            wall_velo_ND = np.loadtxt(mw_file,comments='#',usecols=1)
            for i in range(1,NCL[0]+1):
                wall_velo[i-1] = 0.5*(wall_velo_ND[i+1] + wall_velo_ND[i])

        return wall_velo

    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_meta'
        super().save_hdf(file_name,write_mode,group_name)
        hdf_file = h5py.File(file_name,'a')
        hdf_file[base_name].create_dataset("moving_wall",data=self.__moving_wall)
        hdf_file.close()

    def moving_wall_calc(self):
        return self.__moving_wall

class CHAPSim_fluct_io(cp.CHAPSim_fluct_io):
    pass

class CHAPSim_budget_io(cp.CHAPSim_budget_io):
    def __init__(self,comp1,comp2,avg_data='',PhyTime='',*args,**kwargs):
 
        if avg_data:
            self.avg_data = avg_data
        elif PhyTime:
            self.avg_data = CHAPSim_AVG_io(PhyTime,*args,**kwargs)
        else:
            raise Exception
        if not PhyTime:
            PhyTime = list(set([x[0] for x in self.avg_data.flow_AVGDF.index]))[0]
        self.comp = comp1+comp2
        self.budgetDF = self._budget_extract(PhyTime,comp1,comp2)

    def _advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 

        uu = self.avg_data.UU_tensorDF.loc[PhyTime,uu_comp]\
                        .values.reshape(self.avg_data.shape)
        U_mean = self.avg_data.flow_AVGDF.loc[PhyTime,'u'].copy()\
                        .values.reshape(self.avg_data.shape)
        V_mean = self.avg_data.flow_AVGDF.loc[PhyTime,'v']\
                        .values.reshape(self.avg_data.shape)

        wall_velo = self.avg_data._meta_data.moving_wall_calc()
        for i in range(self.avg_data.shape[0]):
            U_mean[i] -= wall_velo
        uu_dx = cp.Grad_calc(self.avg_data.CoordDF,uu,'x')
        uu_dy = cp.Grad_calc(self.avg_data.CoordDF,uu,'y')

        advection = -(U_mean*uu_dx + V_mean*uu_dy)
        return advection.flatten()

class CHAPSim_autocov_io(cp.CHAPSim_autocov_io):
    def _autocov_extract(self,*args,path_to_folder='',time0='',abs_path=True,**kwargs):
        meta_data, comp, NCL, avg_data, autocorrDF,\
            shape_x,shape_z = super()._autocov_extract(*args,path_to_folder=path_to_folder,**kwargs)
        meta_data = CHAPSim_meta.copy(meta_data)
        avg_data = CHAPSim_AVG_io.copy(avg_data)
        return meta_data, comp, NCL, avg_data, autocorrDF,\
                  shape_x,shape_z

    def _hdf_extract(self,file_name, group_name=''):
        base_name=group_name if group_name else 'CHAPSim_autocov_io'
        hdf_file = h5py.File(file_name,'r')
        shape_x = tuple(hdf_file[base_name].attrs["shape_x"][:])
        shape_z = tuple(hdf_file[base_name].attrs["shape_z"][:])
        comp = tuple(np.char.decode(hdf_file[base_name].attrs["comp"][:]))
        hdf_file.close()        
        autocorrDF = pd.read_hdf(file_name,key=base_name+'/autocorrDF')
        meta_data = CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL=meta_data.NCL
        avg_data = CHAPSim_AVG_io.from_hdf(file_name,base_name+'/avg_data')
        return meta_data, comp, NCL, avg_data, autocorrDF, shape_x, shape_z

class CHAPSim_Quad_Anl_io(cp.CHAPSim_Quad_Anl_io):
    def _quad_extract(self,h_list,path_to_folder='',time0='',abs_path=True):
        meta_data, NCL, avg_data, QuadAnalDF, shape = super()._quad_extract(h_list,
                                        path_to_folder=path_to_folder,time0=time0,
                                        abs_path=abs_path)
        meta_data = CHAPSim_meta.copy(meta_data)
        avg_data = CHAPSim_AVG_io.copy(avg_data)
        return meta_data, NCL, avg_data, QuadAnalDF, shape

    def _hdf_extract(self,file_name,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Quad_Anal'
        meta_data = CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL= meta_data.NCL
        avg_data = CHAPSim_AVG_io.from_hdf(file_name,base_name+'/avg_data')
        QuadAnalDF = pd.read_hdf(file_name,key=base_name+'/QuadAnalDF')
        shape = (NCL[1],NCL[0])
        return meta_data, NCL, avg_data, QuadAnalDF, shape

class CHAPSim_joint_PDF_io(cp.CHAPSim_joint_PDF_io):
    _module = sys.modules[__module__]