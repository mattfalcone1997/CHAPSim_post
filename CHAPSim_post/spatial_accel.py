
import h5py
import numpy as np
import pandas as pd
import os
from abc import ABC
import scipy
if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

import sys

from . import post as cp

from . import plot as cplt

from CHAPSim_post.utils import misc_utils,indexing
from CHAPSim_post import POD

class CHAPSim_Inst_io(cp.CHAPSim_Inst_io):
    pass
_instant_class = CHAPSim_Inst_io

class CHAPSim_AVG_io(cp.CHAPSim_AVG_io):
    def _int_thickness_calc(self,PhyTime):

        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']

        U0 = U_mean[-1]
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

        _, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus_shape=(delta_v_star.size,self.NCL[1])
        y_plus = np.zeros(y_plus_shape)
        y_coord = self.CoordDF['y']
        for i in range(len(delta_v_star)):
            y_plus[i] = y_coord/delta_v_star[i]
        return y_plus
_avg_io_class = CHAPSim_AVG_io

class CHAPSim_meta(cp.CHAPSim_meta):
    pass
_meta_class = CHAPSim_meta

class blayer_TEST_base(ABC):
    blayer_folder = "7_BLAYER_TESTS/"
    def __init__(self,path_to_folder):
        self._debug_folder = os.path.join(path_to_folder,self.blayer_folder)
        self._meta_data = CHAPSim_meta(path_to_folder)
        
class TEST_flow_quant(blayer_TEST_base):
    def __init__(self,path_to_folder,iter=None):
        super().__init__(path_to_folder)
        
        self._get_ini_thickesses(iter)
        self._get_Cf(iter)
        self._get_interp_test()
    
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
            
        _, _, disp, mom, H = np.loadtxt(file_int).T
        
        _, _, blayer = np.loadtxt(file_blayer).T
        
        
        index = ['displacement thickness',
                 'momentum thickness',
                 'H',
                 'blayer thickness']
        
        self.IntThickIniDF = pd.DataFrame([disp,mom, H, blayer],index = index )
        
    def _get_Cf(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_SKIN_FRICTION.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_SKIN_FRICTION.%d.dat'%iter)
            assert os.path.isfile(file)

        _, _, Cf = np.loadtxt(file).T
        
        file = os.path.join(self._debug_folder,'CHK_WALL_UNIT.dat')
        _, _, u_tau, delta_v = np.loadtxt(file).T
        
        
        
        index = ['skin friction', 'u_tau', 'delta_v']
        self.SkinFrictionDF = pd.DataFrame([Cf,u_tau, delta_v],index = index )
        
    
    def _get_interp_test(self):
              
        file = os.path.join(self._debug_folder,'CHK_MEAN_INTERP.dat')
        _, y_in, y_out, array_in, array_out = np.loadtxt(file).T
        
        index = ['y_in', 'y_out', 'array_in', 'array_out']
        self.meanInterpDF = pd.DataFrame([y_in, y_out, array_in, array_out],index = index )
        
        file = os.path.join(self._debug_folder,'CHK_FLUCT_INTERP.dat')
        _, y_in, y_out, array_in1, array_out1, array_in10, array_out10,array_in40, array_out40 = np.loadtxt(file).T
        
        index = ['y_in', 'y_out', 'array_in1', 'array_out1',
                'array_in10', 'array_out10','array_in40',
                'array_out40']
        
        self.fluctInterpDF = pd.DataFrame([y_in, y_out,array_in1, array_out1,
                                            array_in10, array_out10,array_in40,
                                            array_out40],index = index )

    def plot_mom_thickness(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        x_coords = self._meta_data.Coord_ND_DF['x']
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        
        theta =  self.IntThickIniDF['momentum thickness']
        
        fig, ax = cplt.subplots()
        
        ax.cplot(x_coords,theta,label=r'\theta',**line_kw)
        
        return fig, ax
    
    def plot_disp_thickness(self,fig=None, ax = None, line_kw=None,**kwargs):
        
        x_coords = self._meta_data.Coord_ND_DF['x']
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[7,5])
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        
        line_kw = cplt.update_line_kw(line_kw)
        
        theta =  self.IntThickIniDF['displacement thickness']
        
        fig, ax = cplt.subplots()
        
        ax.cplot(x_coords,theta,label=r'\delta^*',**line_kw)
        
        return fig, ax
    
    
        
    def plot_ini_Cf_Re_delta(self,fig=None, ax = None, line_kw=None,**kwargs):
        
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
    
    def plot_ini_Cf_Re_theta(self,fig=None, ax = None, line_kw=None,**kwargs):
        
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
        
    def _get_initialise_test(self):
        file = os.path.join(self._debug_folder,'CHK_INIL_VELO_PROF.dat')
        file_array = np.loadtxt(file).T
        index = ['y','y^+','eta']
        self.y_scalesDF = pd.DataFrame(file_array[1:4],index=index)
        index = ['logistic','spaldings','velo_defect']
        self.velo_compsDF = pd.DataFrame(file_array[4:7],index=index)
        
        index = ['u^+','u']
        self.veloDF = pd.DataFrame(file_array[7:9],index=index)
        index = ['delta_v', 'u_tau']
        self.wall_unitDF = pd.DataFrame(file_array[9:11],index=index)
        
class TEST_FreestreamWall(blayer_TEST_base):
    def __init__(self,path_to_folder,iter=None):
        super().__init__(path_to_folder)
        self._get_Q_freestream(iter)
        self._get_G_freestream(iter)
        
    def _get_Q_freestream(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_VELO.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_VELO.%d.dat'%iter)
            assert os.path.isfile(file)
            
        file_array = np.loadtxt(file).T
        
        self.VeloDF = pd.DataFrame(file_array[[2,3,5]],index= ['u','v','w'])
        self.DispGradDF = pd.DataFrame(file_array[6:10],index= ['ddelta_dx_l', 'ddelta_dx_g','delta','dx'])
        
        self.VeloGrad = pd.DataFrame(file_array[10:12],index=['U_velo_grad', 'W_velo_grad'])
        
    def _get_G_freestream(self,iter):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_G.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_FREESTREAM_G.%d.dat'%iter)
            assert os.path.isfile(file)
            
        file_array = np.loadtxt(file).T
        
        self.GVeloDF = pd.DataFrame(file_array[2:5],index= ['u','v','w'])
        self.GDispGradDF = pd.DataFrame(file_array[5:9],index= ['ddelta_dx_l', 'ddelta_dx_g','delta','dx'])
        
        self.GVeloGrad = pd.DataFrame(file_array[9:],index=['U_velo_grad', 'W_velo_grad'])

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
        self.paramsDF = pd.read_csv(file)
        
    def _get_velo_prof(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_PROF.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_PROF.%d.dat'%iter)
            assert os.path.isfile(file)
        
        file_array = np.loadtxt(file).T
        
        index = [1,6,20,40]
        self.Q_inletDF = pd.DataFrame(file_array[2:6],index = index)
        self.G_inletDF = pd.DataFrame(file_array[6:10],index = index)
        self.Q_plane = file_array[-1]
        
    def _get_gather_tests(self,iter=None):
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP.%d.dat'%iter)
            assert os.path.isfile(file)
            
        file_array = np.loadtxt(file).T
        
        self.UmeanDF = pd.DataFrame(file_array[3:],index=['inner','outer','plane'])
        
        if iter is None:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP_FLUCT.dat')
        else:
            file = os.path.join(self._debug_folder,'CHK_U_VELO_INTERP_FLUCT.%d.dat'%iter)
            assert os.path.isfile(file)
        
    
        file_array = np.loadtxt(file).T
        
        self.UfluctDF = pd.DataFrame(file_array[2:],index=['inner','outer','plane'])
            
    def _get_scalings(self):
        file = os.path.join(self._debug_folder,'CHK_Y_SCALE_U.dat')
        file_array = np.loadtxt(file).T
        
        self.yplusDF = pd.DataFrame(file_array[2:4],index=['inlet','recy'])
        self.etaDF = pd.DataFrame(file_array[4:],index=['inlet','recy'])
        
    def _get_weights(self):
        file = os.path.join(self._debug_folder,'CHK_RESCALE_WEIGHT.dat')
        file_array = np.loadtxt(file).T
        
        
        self.Weight = pd.DataFrame(file_array[1:3],index=['eta','W'])

        
