
import numpy as np
import pandas as pd
import os
from abc import ABC
import scipy

if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

from scipy.interpolate import interp1d

from . import post as cp

from . import plot as cplt

from CHAPSim_post import POD
import CHAPSim_post.dtypes as cd

class CHAPSim_Inst_io(cp.CHAPSim_Inst_io):
    pass
_inst_io_class = CHAPSim_Inst_io

class CHAPSim_fluct_io(cp.CHAPSim_fluct_io):
    pass
_fluct_io_class = CHAPSim_fluct_io

class CHAPSim_AVG_io(cp.CHAPSim_AVG_io):
    def _int_thickness_calc(self,PhyTime):

        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']

        U0 = self.flow_AVGDF[PhyTime,'u'][-1,:]
            
        theta_integrand = np.zeros_like(U_mean)
        delta_integrand = np.zeros_like(U_mean)
        mom_thickness = np.zeros(self.shape[1])
        disp_thickness = np.zeros(self.shape[1])

        for i, _ in enumerate(theta_integrand):
            theta_integrand[i] = (U_mean[i]/U0)*(1 - U_mean[i]/U0)
            delta_integrand[i] = 1 - U_mean[i]/U0

        for j in range(self.shape[1]):
            mom_thickness[j] = integrate_simps(theta_integrand[:,j],y_coords)
            disp_thickness[j] = integrate_simps(delta_integrand[:,j],y_coords)
        shape_factor = disp_thickness/mom_thickness
        
        return disp_thickness, mom_thickness, shape_factor
    
    def blayer_thickness_calc(self,PhyTime=None,method='99'):
        PhyTime= self.check_PhyTime(PhyTime)
        if method == '99':
            return self._delta99_calc(PhyTime)
        else:
            msg = ("Invalid boundary layer thickness"
                   " calculation method")
            raise Exception(msg)
        
    def _delta99_calc(self,PhyTime):
        u_mean = self.flow_AVGDF[PhyTime,'u']
        y_coords = self.CoordDF['y']
        
        delta99 = np.zeros(u_mean.shape[-1])
        for i in range(u_mean.shape[-1]):
            u_99 = 0.99*u_mean[-1,i]
            int = interp1d(u_mean[:,i], y_coords)
            
            delta99[i] = int(u_99)
            
        return delta99
                
        
    def _tau_calc(self,PhyTime):
        
        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        for i in range(self.shape[1]):
            tau_star[i] = mu_star*(u_velo[0,i]-0.0)/(ycoords[0]-0.0)
    
        return tau_star
    def _bulk_velo_calc(self, PhyTime):
        return self.flow_AVGDF[PhyTime,'u'][-1,:].copy()

    def _y_plus_calc(self,PhyTime):

        y_coord = self.CoordDF['y']
        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus = y_coord[:,np.newaxis]*delta_v_star
        return y_plus
    
    def _get_uplus_yplus_transforms(self,PhyTime,x_val):
        u_tau, delta_v = self.wall_unit_calc(PhyTime)
        x_index = self.CoordDF.index_calc('x',x_val)[0]
        x_transform = lambda y:  y/delta_v[x_index]
        y_transform = lambda u: u/u_tau[x_index]
        
        return x_transform, y_transform
    
    def accel_param_calc(self,PhyTime=None):

        PhyTime = self.check_PhyTime(PhyTime)

        U_infty = self._bulk_velo_calc(PhyTime)
        U_infty_grad = np.gradient(U_infty,self.CoordDF['x'])

        REN = self.metaDF['REN']

        accel_param = (1/(REN*U_infty**2))*U_infty_grad
        
        return accel_param

    def plot_accel_param(self,PhyTime=None,desired=False,fig=None,ax=None,line_kw=None,**kwargs):
        
        accel_param = self.accel_param_calc(PhyTime)
        x_coords = self.CoordDF['x']

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw,label = r"$K$")
        
        ax.cplot(x_coords,accel_param,**line_kw)
        if desired:
            REN = self.metaDF['REN']
            U = self._meta_data.U_infty
            k_des = (1/(REN*U**2))*np.gradient(U,x_coords)

            ax.cplot(x_coords,k_des,label=r'$K_{des}$')
        xlabel = self.Domain.create_label(r"$x$")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$K$")
        
        ax.ticklabel_format(style='sci',axis='y',scilimits=(-5,5))
        return fig,ax

_avg_io_class = CHAPSim_AVG_io

class CHAPSim_meta(cp.CHAPSim_meta):
    def __extract_meta(self,path_to_folder='.',abs_path=True,tgpost=False):
        super().__extract_meta(path_to_folder,abs_path,False)
        
        file = os.path.join(path_to_folder,
                            '0_log_monitors',
                            'CHK_U_INFTY.dat')
        if os.path.isfile(file):
            self.U_infty = np.loadtxt(file)[:,-1]
            
        
    def _hdf_extract(self, file_name, key=None):
        if key is None:
            key = self.__class__.__name__
            
        super()._hdf_extract(file_name, key)
        hdf_obj = cd.hdfHandler(file_name,mode='r',key=key)
        
        if 'U_infty' in hdf_obj.keys():
            self.U_infty = hdf_obj['U_infty'][:]
        
    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = self.__class__.__name__
            
        super().save_hdf(file_name,write_mode,key=key)
        hdf_obj = cd.hdfHandler(file_name,mode='a',key=key)
        
        if hasattr(self,'U_infty'):
            hdf_obj.create_dataset('U_infty',data=self.U_infty)

_meta_class = CHAPSim_meta

class CHAPSim_budget_io(cp.CHAPSim_budget_io):
    pass
        
class CHAPSim_momentum_budget_io(cp.CHAPSim_momentum_budget_io):
    def __init__(self,comp,avg_data,PhyTime=None,advection_split=False):
        
        super().__init__(comp,avg_data,PhyTime)

        if advection_split:
            PhyTime = self.avg_data.check_PhyTime(PhyTime)
            self._advection_split(comp,PhyTime)
        

    def _advection_split(self,comp,PhyTime):
        advection = self.budgetDF[PhyTime,'advection']

        U = self.avg_data.flow_AVGDF[PhyTime,'u']
        V = self.avg_data.flow_AVGDF[PhyTime,'v']
        U_comp = self.avg_data.flow_AVGDF[PhyTime,comp] 

        U_dU_dx = -1*U* self.Domain.Grad_calc(self.avg_data.CoordDF,U,'x')
        V_dU_dy = -1*V* self.Domain.Grad_calc(self.avg_data.CoordDF,U,'y')
        self.budgetDF[PhyTime,'advection (term 1)'] = U_dU_dx
        self.budgetDF[PhyTime,'advection (term 2)'] = V_dU_dy

        del self.budgetDF[PhyTime,'advection']
class blayer_TEST_base(ABC):
    blayer_folder = "8_BLAYER_TESTS/"
    blayer_avg_test = "7_BLAYER_DATA"
    def __init__(self,path_to_folder):
        
        self._debug_folder = os.path.join(path_to_folder,self.blayer_folder)
        if not os.path.isdir(self._debug_folder):
            self._debug_folder = os.path.join(path_to_folder,'7_BLAYER_TESTS')
            self._blayer_folder = None
        else:
            self._blayer_folder = os.path.join(path_to_folder,self.blayer_avg_test)
        self._meta_data = CHAPSim_meta(path_to_folder)
        
class TEST_flow_quant(blayer_TEST_base):
    def __init__(self,time,path_to_folder,iter=None):
        super().__init__(path_to_folder)
        
        self._get_ini_thickesses(iter)
        self._get_Cf(iter)
        self._get_interp_test()
        self._get_eps_calc()
        if iter is not None:
            self._get_avg_data(time)
    
    def _get_ini_thickesses(self,iter=None):
        
        if iter is None:
            file_int = os.path.join(self._debug_folder,'CHK_INI_DISP_THICK.dat')
        else:
            file_int = os.path.join(self._debug_folder,'CHK_DISP_THICK.%d.dat'%iter)
            assert os.path.isfile(file_int)
        if iter is None:
            file_blayer = os.path.join(self._debug_folder,'CHK_BLAYER_THICK.dat')
        else:
            file_blayer = os.path.join(self._debug_folder,'CHK_BLAYER_THICK.%d.dat'%iter)
            assert os.path.isfile(file_blayer)
            
        _, _, disp, mom, H,U_infty = np.loadtxt(file_int).T
        
        _, _, blayer = np.loadtxt(file_blayer).T
        
        
        columns = ['displacement thickness',
                 'momentum thickness',
                 'H',
                 'blayer thickness',
                 'U_infty']
        
        index = self._meta_data.Coord_ND_DF['x'][:-1]
        array = np.array([disp,mom, H, blayer,U_infty]).T
        self.IntThickIniDF = pd.DataFrame(array,
                                        columns=columns,
                                        index=index)
        
    def _get_Cf(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_SKIN_FRICTION.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_SKIN_FRICTION.%d.dat'%iter)
            assert os.path.isfile(file)

        _, _, Cf = np.loadtxt(file).T
        
        file = os.path.join(self._debug_folder,'CHK_WALL_UNIT.dat')
        _, _, u_tau, delta_v = np.loadtxt(file).T
        
        array = np.array([Cf,u_tau, delta_v]).T
        
        columns = ['skin friction', 'u tau', 'delta v']
        index = self._meta_data.Coord_ND_DF['x'][:-1]
        
        self.SkinFrictionDF = pd.DataFrame(array,
                                        columns=columns,
                                        index=index)
        
    # def _get_outlet_conv_test(self):
    #     file = os.path.join(self._debug_folder,'CHK_outlet_conv_u.csv')
    #     self.outlet_convDF = pd.read_csv(file).dropna(axis=1).pivot_table(index='YCC')

    def _get_interp_test(self):
              
        file = os.path.join(self._debug_folder,'CHK_MEAN_INTERP.dat')
        _, y_in, y_out, array_in, array_out = np.loadtxt(file).T
        
        columns = ['y_in', 'y_out', 'array_in', 'array_out']
        array = np.array([y_in, y_out, array_in, array_out]).T
        index = self._meta_data.CoordDF['y']
        
        self.meanInterpDF = pd.DataFrame(array,
                                        columns=columns,
                                        index=index)
        
        file = os.path.join(self._debug_folder,'CHK_FLUCT_INTERP.dat')
        _, y_in, y_out, array_in1, array_out1, array_in10, array_out10,array_in40, array_out40 = np.loadtxt(file).T
        
        columns = ['y_in', 'y_out', 'array_in1', 'array_out1',
                'array_in10', 'array_out10','array_in40',
                'array_out40']
        
        array = np.array([y_in, y_out,array_in1, array_out1,
                        array_in10, array_out10,array_in40,
                        array_out40]).T
        
        self.fluctInterpDF = pd.DataFrame(array,
                                        columns=columns,
                                        index=index)
    def _get_eps_calc(self):
        file = os.path.join(self._debug_folder,'CHK_quant_eps_calc.csv')
        
        self.epsDF = pd.read_csv(file).dropna(axis=1).pivot_table(index='Iteration').squeeze()
        
        
    def _get_avg_data(self,time):
        if self._blayer_folder is None:
            self.AVGdata = None
            return
        
        file_name = 'DNS_QuantAVG_T%0.9E.D'%time
        file = os.path.join(self._blayer_folder,file_name)
        
        int_info = np.fromfile(file,count=2, dtype=np.int32)
        size = self._meta_data.NCL[0]*self._meta_data.NCL[1]*3
        avg_info = np.fromfile(file,count=size, dtype=np.float64)
        avg_info = avg_info.reshape((3,
                                    self._meta_data.NCL[1],
                                    self._meta_data.NCL[0]))

        index = [[time]*3,['u','v','w']]
        self.AVGdata = cd.FlowStruct2D(self._meta_data._coorddata,
                                       avg_info,index=index)
        

    def plot_mom_thickness(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        x_coords = self._meta_data.Coord_ND_DF['x'][:-1]
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        
        theta =  self.IntThickIniDF['momentum thickness']
        
        fig, ax = cplt.subplots()
        
        ax.cplot(x_coords,theta,label=r'$\theta$ (CHAPSim)',**line_kw)
        
        return fig, ax
    
    def plot_freestream_velocity(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        x_coords = self._meta_data.Coord_ND_DF['x'][:-1]
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        
        U_infty =  self.IntThickIniDF['U_infty']
        
        fig, ax = cplt.subplots()
        
        ax.cplot(x_coords,U_infty,label=r'$U_\infty$ (CHAPSim)',**line_kw)
        
        return fig, ax
    
    def plot_disp_thickness(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        x_coords = self._meta_data.Coord_ND_DF['x'][:-1]
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        
        theta =  self.IntThickIniDF['displacement thickness']
        
        fig, ax = cplt.subplots()
        
        ax.cplot(x_coords,theta,label=r'$\delta^*$ (CHAPSim)',**line_kw)
        
        return fig, ax
    
    def plot_shape_factor(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        x_coords = self._meta_data.Coord_ND_DF['x'][:-1]
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        
        theta =  self.IntThickIniDF['H']
        
        fig, ax = cplt.subplots()
        
        ax.cplot(x_coords,theta,label=r'$H$ (CHAPSim)',**line_kw)
        
        return fig, ax
    
    def plot_blayer_thickness(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        x_coords = self._meta_data.Coord_ND_DF['x'][:-1]
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        
        theta =  self.IntThickIniDF['blayer thickness']
        
        fig, ax = cplt.subplots()
        
        ax.cplot(x_coords,theta,label=r'$\delta$ (CHAPSim)',**line_kw)
        
        return fig, ax
    
    
        
    def plot_Cf_Re_delta(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        Cf = self.SkinFrictionDF['skin friction']
        delta =  self.IntThickIniDF['blayer thickness']
        REN = self._meta_data.metaDF['REN']
        
        Re_delta = REN*delta
        Cf_corr = 0.01947*Re_delta**-0.1785
        
        fig, ax = cplt.subplots()
        
        ax.cplot(Re_delta,Cf,label='Actual',**line_kw)
        ax.cplot(Re_delta,Cf_corr,label=r'$C_f = 0.01947Re_\delta^{-0.1785}$',**line_kw)
        
        return fig, ax
    
    def plot_Cf_Re_theta(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
                
        Cf = self.SkinFrictionDF['skin friction']
        theta =  self.IntThickIniDF['momentum thickness']
        REN = self._meta_data.metaDF['REN']
        
        Re_theta = REN*theta
        Cf_corr = 0.024*Re_theta**-0.25
        
        fig, ax = cplt.subplots()
        
        ax.cplot(Re_theta,Cf,label='Actual',**line_kw)
        ax.cplot(Re_theta,Cf_corr,label=r'$C_f = 0.024Re_\theta^{-1/4}$',**line_kw)
        
        return fig, ax
        
        
        
    
    
class TEST_initialiseBlayer(blayer_TEST_base):
    def __init__(self,path_to_folder):
        super().__init__(path_to_folder)
        self._get_initialise_test()
        self._get_initialise_corr()
        self._get_fluct_ini()
    def _get_initialise_test(self):
        file = os.path.join(self._debug_folder,'CHK_INIL_VELO_PROF.dat')
        file_array = np.loadtxt(file)
        
        columns = ['y','y plus','eta']
        index = self._meta_data.CoordDF['y']
        
        self.y_scalesDF = pd.DataFrame(file_array[:,1:4],
                                        columns=columns,
                                        index=index)
        columns = ['spaldings','velo defect']
        self.velo_compsDF = pd.DataFrame(file_array[:,4:6],
                                        columns=columns,
                                        index=index)
        
        columns = ['u plus','u','u_check']
        self.veloDF = pd.DataFrame(file_array[:,6:9],
                                        columns=columns,
                                        index=index)
        columns = ['delta v', 'u tau', 'Cf']
        self.wall_unitDF = pd.DataFrame(file_array[:,9:12],
                                        columns=columns,
                                        index=index)
    def _get_fluct_ini(self):
        file = os.path.join(self._debug_folder,'CHK_fluct_ini.csv')
        self.fluct_iniDF = pd.read_csv(file).dropna(axis=1).pivot_table(index='YCC')

        file = os.path.join(self._debug_folder,'CHK_fluct_avg.csv')
        self.fluct_avgDF = pd.read_csv(file).dropna(axis=1).pivot_table(index='YCC')

    def _get_initialise_corr(self):
        file = os.path.join(self._debug_folder,'CHK_INI_CORR.dat')
        file_array = np.loadtxt(file)
        
        x_coords = self._meta_data.Coord_ND_DF['x'][:-1]
        columns = ['Cf', 'u tau', 'delta v', 'blayer thick']
        self.correlationsDF = pd.DataFrame(file_array[:,2:],
                                           columns=columns,
                                           index = x_coords)
        
    def plot_Cf_Re_delta(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        Cf = self.correlationsDF['Cf']
        delta =  self.correlationsDF['blayer thick']
        REN = self._meta_data.metaDF['REN']
        
        Re_delta = REN*delta
        Cf_corr = 0.01947*Re_delta**-0.1785
        
        fig, ax = cplt.subplots()
        
        ax.cplot(Re_delta,Cf,label='Actual',**line_kw)
        ax.cplot(Re_delta,Cf_corr,label=r'$C_f = 0.01947Re_\delta^{-0.1785}$',**line_kw)
        
        return fig, ax
        
class TEST_FreestreamWall(blayer_TEST_base):
    def __init__(self,path_to_folder,iter=None):
        super().__init__(path_to_folder)
        self._get_Q_freestream(iter)
        try:
            self._get_G_freestream(iter)
        except Exception as e:
            pass
        
    def _get_Q_freestream(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_VELO.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_VELO.%d.dat'%iter)
            assert os.path.isfile(file)
            
        file_array = np.loadtxt(file)
        index = self._meta_data.Coord_ND_DF['x'][:-1]
        columns = ['u','v','w']
        self.VeloDF = pd.DataFrame(file_array[:,[2,3,5]],
                                        columns=columns,
                                        index=index)
        
        columns = ['disp grad', 'delta','dx']
        self.DispGradDF = pd.DataFrame(file_array[:,6:9],
                                        columns=columns,
                                        index=index)
        
        columns = ['U vel _grad', 'W velo grad']
        self.VeloGrad = pd.DataFrame(file_array[:,9:11],
                                        columns=columns,
                                        index=index)
                
    def _get_G_freestream(self,iter):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_G.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_G.%d.dat'%iter)
            assert os.path.isfile(file)
            
        file_array = np.loadtxt(file)
        index = self._meta_data.Coord_ND_DF['x'][:-1]
        columns = ['u','v','w']
        self.GVeloDF = pd.DataFrame(file_array[:,2:5],
                                        columns=columns,
                                        index=index)
        
        columns = ['ddelta_dx', 'delta','dx']
        self.GDispGradDF = pd.DataFrame(file_array[:,5:8],
                                        columns=columns,
                                        index=index)
        
        columns = ['U_velo_grad', 'W_velo_grad']
        self.GVeloGrad = pd.DataFrame(file_array[:,8:],
                                        columns=columns,
                                        index=index)
        

class TEST_outlet(blayer_TEST_base):
    def __init__(self,path_to_folder,iter=None):
        super().__init__(path_to_folder)
        
        self._get_tests(iter)
    def _get_tests(self,iter):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_conv_outlet.csv')
            file1 = os.path.join(self._debug_folder,'CHK_outlet_extract.csv')
        else:
            file = os.path.join(self._debug_folder,'CHK_conv_outlet.%d.csv'%iter)
            file1 = os.path.join(self._debug_folder,'CHK_outlet_extract.%d.csv'%iter)

        self.conv_outletDF = pd.read_csv(file).dropna(axis=1).pivot_table(index='YCC')

        self.outlet_velDF = pd.read_csv(file1).dropna(axis=1).pivot_table(index='YCC')


class TEST_recycle_rescale(blayer_TEST_base):
    def __init__(self,path_to_folder,iter=None):
        super().__init__(path_to_folder)
        
        if iter is None:
            self._get_gather_tests()
            self._get_rescaling()
            self._get_velo_prof()
            self._get_gather_tests()
            self._get_scalings()
        else:
            self._get_velo_prof(iter)
            self._get_gather_tests(iter)
            
    def _get_rescaling(self):
        file = os.path.join(self._debug_folder,'CHK_rescale_params.csv')
        self.paramsDF = pd.read_csv(file).dropna(axis=1).drop_duplicates(subset='Iteration').pivot_table(index='Iteration')
        
    def _get_velo_prof(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_PROF.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_PROF.%d.dat'%iter)
            assert os.path.isfile(file)
        
        file_array = np.loadtxt(file)
        
        columns = [1,6,20,40]
        index = self._meta_data.CoordDF['y']
        self.Q_inletDF = pd.DataFrame(file_array[:,2:6],
                                        columns=columns,
                                        index=index)
        
        self.G_inletDF = pd.DataFrame(file_array[:,6:10],
                                        columns=columns,
                                        index=index)
        columns = ['Q plane', 'Q initial mean']
        self.Q_planeDF =  pd.DataFrame(file_array[:,-2:],
                                       columns=columns,
                                       index=index)
        
    def _get_gather_tests(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP.%d.dat'%iter)
            assert os.path.isfile(file)
            
        file_array = np.loadtxt(file)
        index = self._meta_data.CoordDF['y']
        columns = ['inner','outer','plane']
        
        self.UmeanDF = pd.DataFrame(file_array[:,-3:],
                                        columns=columns,
                                        index=index)
        
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP_FLUCT.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP_FLUCT.%d.dat'%iter)
            assert os.path.isfile(file)
        
    
        file_array = np.loadtxt(file)
        
        columns = ['inner','outer','plane']
        self.UfluctDF = pd.DataFrame(file_array[:,-3:],
                                        columns=columns,
                                        index=index)
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_V_VELO_INTERP.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_V_VELO_INTERP.%d.dat'%iter)
            assert os.path.isfile(file)
            
        file_array = np.loadtxt(file)
        index = self._meta_data.CoordDF['y'][1:]
        columns = ['inner','outer','plane']
        
        self.VmeanDF = pd.DataFrame(file_array[:,-3:],
                                        columns=columns,
                                        index=index)
        
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_V_VELO_INTERP_FLUCT.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_V_VELO_INTERP_FLUCT.%d.dat'%iter)
            assert os.path.isfile(file)
        
    
        file_array = np.loadtxt(file)
        
        columns = ['inner','outer','plane']
        self.VfluctDF = pd.DataFrame(file_array[:,-3:],
                                        columns=columns,
                                        index=index)
            
    def _get_scalings(self):
        file = os.path.join(self._debug_folder,'CHK_Y_SCALE_U.dat')
        file_array = np.loadtxt(file)
        index = np.array([0,
                    *self._meta_data.CoordDF['y'],
                    self._meta_data.Coord_ND_DF['y'][-1]])
        columns = ['inlet','recy']
        self.yplusDF = pd.DataFrame(file_array[:,2:4],
                                        columns=columns,
                                        index=index)
        
        self.etaDF = pd.DataFrame(file_array[:,4:],
                                        columns=columns,
                                        index=index)
        
    def _get_weights(self):
        file = os.path.join(self._debug_folder,'CHK_RESCALE_WEIGHT.dat')
        file_array = np.loadtxt(file)
        index = np.array([0,
                          *self._meta_data.CoordDF['y'],
                          self._meta_data.Coord_ND_DF['y'][-1]])
        
        columns = ['eta','W']
        self.Weight = pd.DataFrame(file_array[:,1:3],
                                        columns=columns,
                                        index=index)

        
