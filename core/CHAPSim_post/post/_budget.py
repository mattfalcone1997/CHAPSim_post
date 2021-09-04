"""
# _budget.py
File contains the implementation of the classes to process the 
Reynolds stress and turbulent kinetic energy budgets from the 
CHAPSim_AVG classes 
"""

import numpy as np
import matplotlib as mpl
import scipy

if scipy.__version__ >= '1.6':
    from scipy.integrate import simpson as integrate_simps
else:
    from scipy.integrate import simps as integrate_simps

import itertools
from abc import ABC, abstractmethod

from CHAPSim_post.utils import indexing, misc_utils

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd

from ._common import Common

from ._average import CHAPSim_AVG
_avg_class = CHAPSim_AVG

__all__ = ['CHAPSim_budget_io','CHAPSim_budget_tg','CHAPSim_k_budget_io',
            'CHAPSim_k_budget_tg','CHAPSim_Momentum_budget_io','CHAPSim_Momentum_budget_tg',
            'CHAPSim_FIK_io','CHAPSim_FIK_tg']

class CHAPSim_budget_base(Common,ABC):

    def __init__(self,comp1,comp2,avg_data=None,PhyTime=None,*args,**kwargs):

        if avg_data is not None:
            self.avg_data = avg_data
        elif PhyTime is not None:
            self.avg_data = self._module._avg_class(PhyTime,*args,**kwargs)
        else:
            raise Exception

        if PhyTime is None:
            PhyTime = list(set([x[0] for x in self.avg_data.flow_AVGDF.index]))[0]
        
        
        self.comp = comp1+comp2
        self.budgetDF = self._budget_extract(PhyTime,comp1,comp2)

    @property
    def _meta_data(self):
        return self.avg_data._meta_data

    @property
    def shape(self):
        return self.avg_data.shape
    # @property
    # def vtk(self):
    #     grid = self._create_vtk2D_grid()
    #     self._set_vtk_cell_array(grid,"flow_AVGDF")
    #     self._set_vtk_cell_array(grid,"PU_vectorDF")
    #     self._set_vtk_cell_array(grid,"UU_tensorDF")
    #     self._set_vtk_cell_array(grid,"UUU_tensorDF")
    #     self._set_vtk_cell_array(grid,"Velo_grad_tensorDF")
    #     self._set_vtk_cell_array(grid,"PR_Velo_grad_tensorDF")
    #     self._set_vtk_cell_array(grid,"DUDX2_tensorDF")
    
    def _check_terms(self,comp):
        
        budget_terms = tuple([x[1] for x in self.budgetDF.index])

        if comp is None:
            comp_list = budget_terms
        elif isinstance(comp,(tuple,list)):
            comp_list = comp
        elif isinstance(comp,str):
            comp_list = [comp]
        else:
            raise TypeError("incorrect time")
        
        # print(budget_terms)
        if not all([comp in budget_terms for comp in comp_list]):
            raise KeyError("Invalid budget term provided")

        return comp_list

    def _budget_plot(self,PhyTime, x_list,budget_terms=None,wall_units=True, fig=None, ax =None,line_kw=None,**kwargs):

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

        lower_extent= 0.2
        gridspec_kw = {'bottom': lower_extent}
        figsize= [7*ax_size[1],5*ax_size[0]+1]

        kwargs = cplt.update_subplots_kw(kwargs,gridspec_kw=gridspec_kw,figsize=figsize)
        fig, ax = cplt.create_fig_ax_without_squeeze(*ax_size,fig=fig,ax=ax,**kwargs)
        ax=ax.flatten()

        budget_terms = self._check_terms(budget_terms)


        x_list = misc_utils.check_list_vals(x_list)
        
        line_kw= cplt.update_line_kw(line_kw)
        for i,x_loc in enumerate(x_list):
            for comp in budget_terms:
                budget_values = self.budgetDF[PhyTime,comp].copy()
                x = self.avg_data._return_index(x_loc)
                if wall_units:
                    budget = budget_values[:Y_extent,x]/budget_scale[x]
                else:
                    budget = budget_values[:Y_extent,x]
                
                if self.comp == 'uv':
                    budget= budget * -1.0

                ax[i].cplot(Y_coords[x,:].squeeze(),budget,label=comp.title(),**line_kw)
    
                
                if wall_units:
                    ax[i].set_xscale('log')
                    # ax[i].set_xlim(left=1.0)
                    ax[i].set_xlabel(r"$y^+$")
                else:
                    x_label = self.Domain.in_to_out(r"$y/\delta$")
                    ax[i].set_xlabel(x_label)

                if mpl.rcParams['text.usetex'] == True:
                    ax[i].set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
                else:
                    ax[i].set_ylabel(r"Loss        Gain")
        return fig, ax

    def _plot_integral_budget(self,budget_terms=None,PhyTime=None,wall_units=True,fig=None,ax=None,line_kw=None,**kwargs):

        y_coords = self.avg_data.CoordDF['y']

        budget_terms = self._check_terms(budget_terms)

        
        u_tau_star, delta_v_star = self.avg_data._wall_unit_calc(PhyTime)
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= cplt.update_line_kw(line_kw)
        xaxis_vals = self.avg_data._return_xaxis()

        for comp in budget_terms:
            integral_budget = np.zeros(self.avg_data.shape[1])
            budget_term = self.budgetDF[PhyTime,comp].copy()
            for i in range(self.avg_data.shape[1]):
                integral_budget[i] = 0.5*integrate_simps(budget_term[:,i],y_coords)
                if wall_units:
                    delta_star=1.0
                    integral_budget[i] /=(delta_star*u_tau_star[i]**3/delta_v_star[i])
            label = r"$\int^{\delta}_{-\delta}$ %s $dy$"%comp.title()
            ax.cplot(xaxis_vals,integral_budget,label=label,**line_kw)
        
        ncol = cplt.get_legend_ncols(len(budget_terms))
        ax.clegend(ncol=ncol,vertical=False)
        #ax.grid()
        return fig, ax

    def _plot_budget_x(self,budget_terms,y_vals_list,Y_plus=True,PhyTime=None,fig=None,ax=None,**kwargs):
        
        budget_terms = self._check_terms(budget_terms)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig,ax = cplt.create_fig_ax_with_squeeze(fig,ax **kwargs)

        xaxis_vals = self.avg_data._return_xaxis()
        for comp in budget_terms:
            if y_vals_list != 'max':
                if Y_plus:
                    y_index = indexing.Y_plus_index_calc(self, self.CoordDF, y_vals_list)
                else:
                    y_index = indexing.coord_index_calc(self.CoordDF,'y',y_vals_list)
                budget_term = self.budgetDF[PhyTime,comp]

                y_vals_list = indexing.ycoords_from_coords(self,y_vals_list,mode='wall')[0]
                for i, y_val in enumerate(y_vals_list):
                    ax.cplot(budget_term[i],label=r"%s $y^+=%.2g$"% (comp,y_val))
                
                ncol = cplt.get_legend_ncols(len(budget_terms)*len(y_vals_list))
                ax.clegend(vertical=False,ncol=ncol, fontsize=16)
                
            else:
                budget_term = self.budgetDF[PhyTime,comp]
                budget_term = np.amax(budget_term,axis=0)
                ax.cplot(xaxis_vals,budget_term,label=r"maximum %s"%comp)
                
                ncol = cplt.get_legend_ncols(len(budget_terms))
                ax.clegend(vertical=False,ncol=ncol, fontsize=16)
        fig.tight_layout()
        return fig, ax

    def __str__(self):
        return self.budgetDF.__str__()
class CHAPSim_Reynolds_budget_base(CHAPSim_budget_base):
    def _budget_extract(self,PhyTime,comp1,comp2):
            
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

    @abstractmethod
    def _production_extract(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _advection_extract(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _turb_transport(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _pressure_diffusion(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _pressure_strain(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _viscous_diff(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _dissipation_extract(self,*args,**kwargs):
        raise NotImplementedError


class CHAPSim_budget_io(CHAPSim_Reynolds_budget_base):

    def _advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 

        uu = self.avg_data.UU_tensorDF[PhyTime,uu_comp]
        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        V_mean = self.avg_data.flow_AVGDF[PhyTime,'v']

        uu_dx = self.Domain.Grad_calc(self.avg_data.CoordDF,uu,'x')
        uu_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uu,'y')

        advection = -(U_mean*uu_dx + V_mean*uu_dy)
        return advection

    def _turb_transport(self,PhyTime,comp1,comp2):
        uu_comp = comp1+comp2
        uu_comp1 = uu_comp+'u'
        uu_comp2 = uu_comp+'v'

        
        if ord(uu_comp1[0]) > ord(uu_comp1[1]):
            uu_comp1 = uu_comp1[:2][::-1] + uu_comp1[2]
        if ord(uu_comp1[0]) > ord(uu_comp1[2]):
            uu_comp1 = uu_comp1[::-1]
        if ord(uu_comp1[1]) > ord(uu_comp1[2]):
            uu_comp1 = uu_comp1[0] + uu_comp1[1:][::-1]
            
        if ord(uu_comp2[0]) > ord(uu_comp2[1]):
            uu_comp2 = uu_comp2[:2][::-1] + uu_comp2[2]
        if ord(uu_comp2[0]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[::-1]
        if ord(uu_comp2[1]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[0] + uu_comp2[1:][::-1]

        u1u2u = self.avg_data.UUU_tensorDF[PhyTime,uu_comp1]
        u1u2v = self.avg_data.UUU_tensorDF[PhyTime,uu_comp2]

        u1u2u_dx = self.Domain.Grad_calc(self.avg_data.CoordDF,u1u2u,'x')
        u1u2v_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,u1u2v,'y')

        turb_transport = -(u1u2u_dx + u1u2v_dy)
        return turb_transport

    def _pressure_strain(self,PhyTime,comp1,comp2):
        u1u2 = comp1 + chr(ord(comp2)-ord('u')+ord('x'))
        u2u1 = comp2 + chr(ord(comp1)-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u1u2]
        pdu2dx1 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u2u1]\

        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain

    def _pressure_diffusion(self,PhyTime,comp1,comp2):
        if comp1 == 'u' and comp2 =='u':
            diff1 = diff2 = 'x'
        elif comp1 == 'v' and comp2 =='v':
            diff1 = diff2 = 'y'
        elif comp1 =='u' and comp2 =='v':
            diff1 = 'y'
            diff2 = 'x'
        elif comp1 == 'w' and comp2 == 'w':
            pressure_diff = np.zeros(self.avg_data.shape)
            return pressure_diff.flatten()
        else:
            raise ValueError

        pu1 = self.avg_data.PU_vectorDF[PhyTime,comp1]
        pu2 = self.avg_data.PU_vectorDF[PhyTime,comp2]

        rho_star = 1.0
        pu1_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu1,diff1)
        pu2_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu2,diff2)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff

    def _viscous_diff(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2
        u1u2 = self.avg_data.UU_tensorDF[PhyTime,uu_comp]

        REN = self.avg_data.metaDF['REN']
        viscous_diff = (1/REN)*self.Domain.Scalar_laplacian(self.avg_data.CoordDF,u1u2)
        return viscous_diff

    def _production_extract(self,PhyTime,comp1,comp2):
        U1U_comp = comp1 + 'u'
        U2U_comp = comp2 + 'u'
        U1V_comp = comp1 + 'v'
        U2V_comp = comp2 + 'v'
        
        uu_comp_list = [U1U_comp, U2U_comp,U1V_comp, U2V_comp]
        for i in range(len(uu_comp_list)):
            if ord(uu_comp_list[i][0]) > ord(uu_comp_list[i][1]):
                uu_comp_list[i] = uu_comp_list[i][::-1]
                
        U1U_comp, U2U_comp,U1V_comp, U2V_comp = itertools.chain(uu_comp_list)
        u1u = self.avg_data.UU_tensorDF[PhyTime,U1U_comp]
        u2u = self.avg_data.UU_tensorDF[PhyTime,U2U_comp]
        u1v = self.avg_data.UU_tensorDF[PhyTime,U1V_comp]
        u2v = self.avg_data.UU_tensorDF[PhyTime,U2V_comp]

        U1x_comp = comp1 + 'x'
        U2x_comp = comp2 + 'x'
        U1y_comp = comp1 + 'y'
        U2y_comp = comp2 + 'y'
        
        du1dx = self.avg_data.Velo_grad_tensorDF[PhyTime,U1x_comp]
        du2dx = self.avg_data.Velo_grad_tensorDF[PhyTime,U2x_comp]
        du1dy = self.avg_data.Velo_grad_tensorDF[PhyTime,U1y_comp]
        du2dy = self.avg_data.Velo_grad_tensorDF[PhyTime,U2y_comp]

        production = -(u1u*du2dx + u2u*du1dx + u1v*du2dy + u2v*du1dy)
        return production
    def _dissipation_extract(self,PhyTime,comp1,comp2):
        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        dU1dx_comp = comp1 + 'u'
        dU2dx_comp = comp2 + 'u'
        dU1dy_comp = comp1 + 'v'
        dU2dy_comp = comp2 + 'v'
        
        du1dxdu2dx = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dxdU2dx_comp]
        du1dydu2dy = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dydU2dy_comp]

        REN = self.avg_data.metaDF['REN']
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation

    def budget_plot(self, x_list,budget_terms=None,PhyTime=None,wall_units=True, fig=None, ax =None,**kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)

        fig, ax = super()._budget_plot(PhyTime, x_list,budget_terms,wall_units=wall_units, fig=fig, ax =ax,**kwargs)
        
        x_list = indexing.true_coords_from_coords(self.avg_data.CoordDF,'x',x_list)

        for a,x in zip(ax,x_list):
            title = self.Domain.in_to_out(r"$x/\delta=%.2g$"%x)
            a.set_title(title,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    def plot_integral_budget(self,budget_terms=None, PhyTime=None, wall_units=True, fig=None, ax=None, **kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)

        fig, ax = super()._plot_integral_budget(PhyTime=PhyTime, budget_terms=budget_terms,wall_units=wall_units, fig=fig, ax=ax, **kwargs)
        
        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        return fig, ax
    def plot_budget_x(self,budget_terms=None,y_vals_list='max',Y_plus=True,PhyTime=None,fig=None,ax=None,**kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)


        fig, ax = super()._plot_budget_x(budget_terms,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)

        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)
        
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_budget_tg(CHAPSim_Reynolds_budget_base):

    def __init__(self,*args,**kwargs):
        if 'PhyTime' in kwargs.keys():
            raise KeyError("PhyTime cannot be used in tg class\n")
        super().__init__(*args,PhyTime=None,**kwargs)

    def _advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 

        uu = self.avg_data.UU_tensorDF[PhyTime,uu_comp]
        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        V_mean = self.avg_data.flow_AVGDF[PhyTime,'v']

        uu_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uu,'y')

        advection = -V_mean*uu_dy
        return advection#.flatten()

    def _turb_transport(self,PhyTime,comp1,comp2):
        uu_comp = comp1+comp2
        uu_comp2 = uu_comp+'v'

        if ord(uu_comp2[0]) > ord(uu_comp2[1]):
            uu_comp2 = uu_comp2[:2][::-1] + uu_comp2[2]
        if ord(uu_comp2[0]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[::-1]
        if ord(uu_comp2[1]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[0] + uu_comp2[1:][::-1]

        u1u2v = self.avg_data.UUU_tensorDF[PhyTime,uu_comp2]

        u1u2v_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,u1u2v,'y')

        turb_transport = -u1u2v_dy
        return turb_transport#.flatten()

    def _pressure_strain(self,PhyTime,comp1,comp2):
        u1u2 = comp1 + chr(ord(comp2)-ord('u')+ord('x'))
        u2u1 = comp2 + chr(ord(comp1)-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u1u2]
        pdu2dx1 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u2u1]

        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain#.flatten()

    def _pressure_diffusion(self,PhyTime,comp1,comp2):

        diff1 = chr(ord(comp2)-ord('u')+ord('x'))
        diff2 = chr(ord(comp1)-ord('u')+ord('x'))
        comp_list = ['u','v','w']
        if comp1 not in comp_list:
            raise ValueError("comp1 must be %s, %s, or %s not %s"%(*comp_list,comp1))
        if comp2 not in comp_list:
            raise ValueError("comp2 must be %s, %s, or %s not %s"%(*comp_list,comp2))

        pu1 = self.avg_data.PU_vectorDF[PhyTime,comp1]
        pu2 = self.avg_data.PU_vectorDF[PhyTime,comp2]

        rho_star = 1.0
        if diff1 == 'y':
            pu1_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu1,'y')
        else:
            pu1_grad = np.zeros(self.avg_data.shape)

        if diff2 == 'y':
            pu2_grad = self.Domain.Grad_calc(self.avg_data.CoordDF,pu2,'y')
        else:
            pu2_grad = np.zeros(self.avg_data.shape)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff#.flatten()

    def _viscous_diff(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2
        u1u2 = self.avg_data.UU_tensorDF[PhyTime,uu_comp]

        REN = self.avg_data.metaDF['REN']
        viscous_diff = (1/REN)*self.Domain.Scalar_laplacian_tg(self.avg_data.CoordDF,u1u2)
        return viscous_diff#.flatten()

    def _production_extract(self,PhyTime,comp1,comp2):
        U1U_comp = comp1 + 'u'
        U2U_comp = comp2 + 'u'
        U1V_comp = comp1 + 'v'
        U2V_comp = comp2 + 'v'
        
        uu_comp_list = [U1U_comp, U2U_comp,U1V_comp, U2V_comp]
        for i in range(len(uu_comp_list)):
            if ord(uu_comp_list[i][0]) > ord(uu_comp_list[i][1]):
                uu_comp_list[i] = uu_comp_list[i][::-1]
                
        U1U_comp, U2U_comp,U1V_comp, U2V_comp = itertools.chain(uu_comp_list)
        u1u = self.avg_data.UU_tensorDF[PhyTime,U1U_comp]
        u2u = self.avg_data.UU_tensorDF[PhyTime,U2U_comp]
        u1v = self.avg_data.UU_tensorDF[PhyTime,U1V_comp]
        u2v = self.avg_data.UU_tensorDF[PhyTime,U2V_comp]

        U1x_comp = comp1 + 'x'
        U2x_comp = comp2 + 'x'
        U1y_comp = comp1 + 'y'
        U2y_comp = comp2 + 'y'
        
        du1dx = self.avg_data.Velo_grad_tensorDF[PhyTime,U1x_comp]
        du2dx = self.avg_data.Velo_grad_tensorDF[PhyTime,U2x_comp]
        du1dy = self.avg_data.Velo_grad_tensorDF[PhyTime,U1y_comp]
        du2dy = self.avg_data.Velo_grad_tensorDF[PhyTime,U2y_comp]

        production = -(u1u*du2dx + u2u*du1dx + u1v*du2dy + u2v*du1dy)
        return production#.flatten()
    
    def _dissipation_extract(self,PhyTime,comp1,comp2):
        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        du1dxdu2dx = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dxdU2dx_comp]
        du1dydu2dy = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dydU2dy_comp]


        REN = self.avg_data.metaDF['REN']
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation#.flatten()

    def budget_plot(self, times_list,wall_units=True, fig=None, ax =None,**kwargs):

        if not isinstance(times_list,(float,list)):
            times_list = [times_list]
        PhyTime = None
        fig, ax = super()._budget_plot(PhyTime, times_list,wall_units=wall_units, fig=fig, ax =ax,**kwargs)

        for a,t in zip(ax,times_list):
            a.set_title(r"$t^*=%s$"%t,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    def plot_integral_budget(self,budget_terms=None, wall_units=True, fig=None, ax=None, **kwargs):
        PhyTime = None
        fig, ax = super()._plot_integral_budget(budget_terms=budget_terms,PhyTime=PhyTime, wall_units=wall_units, fig=fig, ax=ax, **kwargs)
        ax.set_xlabel(r"$t^*$")

        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_budget_x(self,budget_terms=None,y_vals_list='max',Y_plus=True,fig=None,ax=None,**kwargs):
        PhyTime = None

        fig, ax = super()._plot_budget_x(budget_terms,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        
        ax.set_xlabel(r"$t^*$")
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_k_budget_base(ABC):
        def __init__(self,avg_data=None,PhyTime=None,*args,**kwargs):

            if avg_data is not None:
                self.avg_data = avg_data
            elif PhyTime is not None:
                self.avg_data = self._module._avg_class(PhyTime,*args,**kwargs)
            else:
                raise Exception

            if PhyTime is None:
                PhyTime = list(set([x[0] for x in self.avg_data.flow_AVGDF.index]))[0]
            
            uu_budgetDF = self._budget_extract(PhyTime,'u','u')
            vv_budgetDF = self._budget_extract(PhyTime,'v','v')
            ww_budgetDF = self._budget_extract(PhyTime,'w','w')

            self.budgetDF = 0.5*(uu_budgetDF + vv_budgetDF + ww_budgetDF)
            self.shape = self.avg_data.shape

class CHAPSim_k_budget_io(CHAPSim_k_budget_base,CHAPSim_budget_io):
    pass

class CHAPSim_k_budget_tg(CHAPSim_k_budget_base,CHAPSim_budget_tg):
    pass

class CHAPSim_Momentum_budget_base(CHAPSim_budget_base):
    def __init__(self,comp,avg_data=None,PhyTime=None,*args,**kwargs):

        if avg_data is not None:
            self.avg_data = avg_data
        elif PhyTime is not None:
            self.avg_data = self._module._avg_class(PhyTime,*args,**kwargs)
        else:
            raise Exception

        if PhyTime is None:
            PhyTime = list(set([x[0] for x in self.avg_data.flow_AVGDF.index]))[0]
        
        self.comp=comp
        self.budgetDF = self._budget_extract(PhyTime,comp)

    def _budget_extract(self,PhyTime,comp):
            
        advection = self._advection_extract(PhyTime,comp)
        pressure_grad = self._pressure_grad(PhyTime,comp)
        viscous = self._viscous_extract(PhyTime,comp)
        Reynolds_stress = self._turb_transport(PhyTime,comp)


        array_concat = [advection,pressure_grad,viscous,Reynolds_stress]

        budget_array = np.stack(array_concat,axis=0)
        
        budget_index = ['advection','pressure gradient','viscous stresses','reynolds stresses']  
        phystring_index = [PhyTime]*4

        budgetDF = cd.datastruct(budget_array,index =[phystring_index,budget_index])
        
        return budgetDF

    @abstractmethod
    def _advection_extract(self,PhyTime,comp):
        raise NotImplementedError
    
    @abstractmethod 
    def _pressure_grad(self,PhyTime,comp):
        raise NotImplementedError

    @abstractmethod
    def _viscous_extract(self,PhyTime,comp):
        raise NotImplementedError

    @abstractmethod
    def _turb_transport(self,PhyTime,comp):
        raise NotImplementedError

    def plot_integrated_budget(self, x_list,budget_terms=None,PhyTime=None, fig=None, ax =None,**kwargs):
        local_budgetDF = self.budgetDF.copy()
        middle_index = int(0.5*self.shape[0])
        for index, val in self.budgetDF:
            budget = np.zeros_like(val)
            for i in range(len(budget)//2):
                
                y = -1*self.avg_data.CoordDF['y'][middle_index:middle_index-i-1:-1]
                integrand = val[middle_index:middle_index-i-1:-1]
                budget[middle_index-i-1,:] = integrate_simps(integrand,y,axis=0)
                budget[middle_index+i,:] = integrate_simps(integrand,y,axis=0)
            self.budgetDF[index] = budget

        fig, ax = super()._budget_plot(PhyTime,x_list,budget_terms,False,fig,ax,**kwargs)

        self.budgetDF = local_budgetDF
        return fig, ax
class CHAPSim_Momentum_budget_io(CHAPSim_Momentum_budget_base):
    def _advection_extract(self,PhyTime,comp):
        
        
        UU = self.avg_data.flow_AVGDF[PhyTime,comp]*self.avg_data.flow_AVGDF[PhyTime,'u']
        UV = self.avg_data.flow_AVGDF[PhyTime,comp]*self.avg_data.flow_AVGDF[PhyTime,'v']

        advection_pre = np.stack([UU,UV],axis=0)

        return -1*self.Domain.Vector_div_io(self.avg_data.CoordDF,advection_pre)

    def _pressure_grad(self, PhyTime, comp):
        
        pressure = self.avg_data.flow_AVGDF[PhyTime,'P']
        dir = chr(ord(comp)-ord('u') + ord('x'))

        return -1.0*self.Domain.Grad_calc(self.avg_data.CoordDF,pressure,dir)

    def _viscous_extract(self, PhyTime, comp):
        dir = chr(ord(comp)-ord('u') + ord('x'))

        S_comp_1 = 0.5*(self.avg_data.Velo_grad_tensorDF[PhyTime,comp+'x'] +\
                        self.avg_data.Velo_grad_tensorDF[PhyTime,'u'+dir])

        S_comp_2 = 0.5*(self.avg_data.Velo_grad_tensorDF[PhyTime,comp+'y'] +\
                        self.avg_data.Velo_grad_tensorDF[PhyTime,'v'+dir])
        
        S_comp = np.stack([S_comp_1,S_comp_2],axis=0)
        REN = self.avg_data.metaDF['REN']
        mu_star = 1.0
        return (mu_star/REN)*self.Domain.Vector_div_io(self.avg_data.CoordDF,2*S_comp) 

    def _turb_transport(self, PhyTime, comp):

        comp_uu = comp + 'u'
        comp_uv = comp + 'v'

        if comp_uu[0] > comp_uu[1]:
            comp_uu = comp_uu[::-1]
        if comp_uv[0] > comp_uv[1]:
            comp_uv = comp_uv[::-1]
        
        uu = self.avg_data.UU_tensorDF[PhyTime,comp_uu]
        uv = self.avg_data.UU_tensorDF[PhyTime,comp_uv]

        advection_pre = np.stack([uu,uv],axis=0)

        return -1*self.Domain.Vector_div_io(self.avg_data.CoordDF,advection_pre)


    def budget_plot(self, x_list,budget_terms=None,PhyTime=None,wall_units=True, fig=None, ax =None,**kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)

        fig, ax = super()._budget_plot(PhyTime, x_list,budget_terms,wall_units=wall_units, fig=fig, ax =ax,**kwargs)
        
        x_list = indexing.true_coords_from_coords(self.avg_data.CoordDF,'x',x_list)

        for a,x in zip(ax,x_list):
            title = self.Domain.in_to_out(r"$x/\delta=%.2g$"%x)
            a.set_title(title,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    def plot_integral_budget(self,comp=None, PhyTime=None, wall_units=True, fig=None, ax=None, **kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)

        fig, ax = super()._plot_integral_budget(comp,PhyTime, wall_units=wall_units, fig=fig, ax=ax, **kwargs)
        
        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_budget_x(self,budget_terms=None,y_vals_list='max',Y_plus=True,PhyTime=None,fig=None,ax=None,**kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)


        fig, ax = super()._plot_budget_x(budget_terms,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)

        x_label = self.Domain.in_to_out(r"$x/\delta$")
        ax.set_xlabel(x_label)
        
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_integrated_budget(self, x_list,budget_terms=None,PhyTime=None, fig=None, ax =None,**kwargs):
        fig, ax = super().plot_integrated_budget(x_list,budget_terms,PhyTime, fig, ax ,**kwargs)

        x_list = indexing.true_coords_from_coords(self.avg_data.CoordDF,'x',x_list)

        for a,x in zip(ax,x_list):
            title = self.Domain.in_to_out(r"$x/\delta=%.2g$"%x)
            a.set_title(title,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    
class CHAPSim_Momentum_budget_tg(CHAPSim_Momentum_budget_base):
    
    
    def _advection_extract(self, PhyTime, comp):
        UV = self.avg_data.flow_AVGDF[PhyTime,comp]*self.avg_data.flow_AVGDF[PhyTime,'v']

        return self.Domain.Grad_calc(self.avg_data.CoordDF,UV,'y')

    def _viscous_extract(self, PhyTime, comp):
        dir = chr(ord(comp)-ord('u') + ord('x'))


        S_comp_2 = (self.avg_data.Velo_grad_tensorDF[PhyTime,comp+'y'] +\
                        self.avg_data.Velo_grad_tensorDF[PhyTime,'v'+dir])
        
        REN = self.avg_data.metaDF['REN']
        mu_star = 1.0
        
        return (mu_star/REN)*self.Domain.Grad_calc(self.avg_data.CoordDF,S_comp_2,'y')

    def _turb_transport(self, PhyTime, comp):
        comp_uv = comp + 'v'

        if comp_uv[0] > comp_uv[1]:
            comp_uv = comp_uv[::-1]

        uv = self.avg_data.UU_tensorDF[PhyTime,comp_uv]

        return -1*self.Domain.Grad_calc(self.avg_data.CoordDF,uv,'y')

    def budget_plot(self, times_list,wall_units=True, fig=None, ax =None,**kwargs):

        if not isinstance(times_list,(float,list)):
            times_list = [times_list]
        PhyTime = None
        fig, ax = super()._budget_plot(PhyTime, times_list,wall_units=wall_units, fig=fig, ax =ax,**kwargs)

        for a,t in zip(ax,times_list):
            a.set_title(r"$t^*=%s$"%t,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    def plot_integral_budget(self,budget_terms=None, wall_units=True, fig=None, ax=None, **kwargs):
        PhyTime = None
        fig, ax = super()._plot_integral_budget(budget_terms=budget_terms,PhyTime=PhyTime, wall_units=wall_units, fig=fig, ax=ax, **kwargs)
        ax.set_xlabel(r"$t^*$")

        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_budget_x(self,budget_terms=None,y_vals_list='max',Y_plus=True,fig=None,ax=None,**kwargs):
        PhyTime = None

        fig, ax = super()._plot_budget_x(budget_terms,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        
        ax.set_xlabel(r"$t^*$")
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_integrated_budget(self, times_list,budget_terms=None,PhyTime=None, fig=None, ax =None,**kwargs):
        fig, ax = super().plot_integrated_budget(times_list,budget_terms,PhyTime, fig, ax ,**kwargs)

        x_list = indexing.true_coords_from_coords(self.avg_data.CoordDF,'x',times_list)

        for a,t in zip(ax,times_list):
            a.set_title(r"$t^*=%s$"%t,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

class CHAPSim_FIK_base(CHAPSim_budget_base):
    def __init__(self,avg_data=None,PhyTime=None,*args,**kwargs):

        if avg_data is not None:
            self.avg_data = avg_data.copy()
        elif PhyTime is not None:
            avg_data = self._module._avg_class(PhyTime,*args,**kwargs)
        else:
            raise Exception

        if PhyTime is None:
            PhyTime = list(set([x[0] for x in self.avg_data.flow_AVGDF.index]))[0]
        
        self.budgetDF = self._budget_extract(PhyTime)

    def _budget_extract(self,PhyTime):
        laminar = self._laminar_extract(PhyTime)
        turbulent = self._turbulent_extract(PhyTime)
        inertia = self._inertia_extract(PhyTime)

        array_concat = [laminar,turbulent,inertia]

        budget_array = np.stack(array_concat,axis=0)
        budget_index = ['laminar', 'turbulent','non-homogeneous']
        phystring_index = [PhyTime]*3

        budgetDF = cd.datastruct(budget_array,index =[phystring_index,budget_index])

        return budgetDF

    @abstractmethod
    def _scale_vel(self,PhyTime):
        pass

    def _laminar_extract(self,PhyTime):

        bulk = self._scale_vel(PhyTime)
        REN = self.avg_data.metaDF['REN']
        const = 4.0 if self.Domain.is_cylind else 6.0
        return const/(REN*bulk)

    def _turbulent_extract(self,PhyTime):

        bulk = self._scale_vel(PhyTime)
        y_coords = self.avg_data.CoordDF['y']
        uv = self.avg_data.UU_tensorDF[PhyTime,'uv']

        turbulent = np.zeros_like(uv)
        for i,y in enumerate(y_coords):
            turbulent[i] =    6*y*uv[i,:]

        return self.Domain.Integrate_tot(self.CoordDF,turbulent)/bulk**2

    @abstractmethod
    def _inertia_extract(self,PhyTime):
        pass  
    
    @abstractmethod
    def plot(self,budget_terms=None,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        budget_terms = self._check_terms(budget_terms)
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= cplt.update_line_kw(line_kw)
        xaxis_vals = self.avg_data._return_xaxis()

        for comp in budget_terms:
            budget_term = self.budgetDF[PhyTime,comp].copy()
                
            label = r"%s"%comp.title()
            ax.cplot(xaxis_vals,budget_term,label=label,**line_kw)
        ax.cplot(xaxis_vals,np.sum(self.budgetDF.values,axis=0),label="Total",**line_kw)
        ncol = cplt.get_legend_ncols(len(budget_terms))
        ax.clegend(ncol=ncol,vertical=False)
        return fig, ax

    
class CHAPSim_FIK_io(CHAPSim_FIK_base):
    def _scale_vel(self, PhyTime):
        return self.avg_data.bulk_velo_calc(PhyTime)

    def _inertia_extract(self,PhyTime):
        y_coords = self.avg_data.CoordDF['y']

        bulk = self._scale_vel(PhyTime)


        pressure = self.avg_data.flow_AVGDF[PhyTime,'P']
        pressure_grad_x = self.Domain.Grad_calc(self.avg_data.CoordDF,pressure,'x')

        p_prime2 = pressure_grad_x - self.Domain.Integrate_tot(self.CoordDF,pressure_grad_x)

        u_mean2 = self.avg_data.flow_AVGDF[PhyTime,'u']**2
        uu = self.avg_data.UU_tensorDF[PhyTime,'uu']

        d_UU_dx = self.Domain.Grad_calc(self.avg_data.CoordDF,
                                        u_mean2+uu,'x')
        
        UV = self.avg_data.flow_AVGDF[PhyTime,'u']*self.avg_data.flow_AVGDF[PhyTime,'v']
        d_uv_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,UV,'y')

        REN = self.avg_data.metaDF['REN']
        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        d2u_dx2 = self.Domain.Grad_calc(self.avg_data.CoordDF,
                    self.Domain.Grad_calc(self.avg_data.CoordDF,U_mean,'x'),'x')

        I_x = d_UU_dx + d_uv_dy - (1/REN)*d2u_dx2

        I_x_prime  = I_x -  self.Domain.Integrate_tot(self.CoordDF,I_x)

        out = np.zeros_like(U_mean)
        for i,y in enumerate(y_coords):
            out[i] = (p_prime2 + I_x_prime)[i,:]*y**2


        return -3.0*self.Domain.Integrate_tot(self.CoordDF,out)/(bulk**2)

    def plot(self,*args,**kwargs):
        fig, ax = super().plot(*args,**kwargs)
        ax.set_xlabel(r"$x/\delta$")
        return fig, ax

class CHAPSim_FIK_tg(CHAPSim_FIK_base):

    def _scale_vel(self, PhyTime):
        return self.avg_data.bulk_velo_calc()
    def _laminar_extract(self,PhyTime):
        PhyTime=None
        return super()._laminar_extract(PhyTime)

    def _turbulent_extract(self,PhyTime):
        PhyTime=None
        return super()._turbulent_extract(PhyTime)

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

        
        out = np.zeros(U_mean.shape)
        for i,y in enumerate(y_coords):
            out[i] = (I_x_prime + dp_prime_dx + dudt)[i,:]*y**2

        return -3.0*self.Domain.Integrate_tot(self.CoordDF,out)/(bulk**2)

    def plot(self,budget_terms=None,*args,**kwargs):
        fig, ax = super().plot(budget_terms,PhyTime=None,*args,**kwargs)
        ax.set_xlabel(r"$t^*$")
        return fig, ax







