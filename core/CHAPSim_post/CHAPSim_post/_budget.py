"""
# _budget.py
File contains the implementation of the classes to process the 
Reynolds stress and turbulent kinetic energy budgets from the 
CHAPSim_AVG classes 
"""

import numpy as np
import matplotlib as mpl
from scipy import integrate

import sys
import os
import warnings
import gc
import itertools

import CHAPSim_post as cp
from .. import CHAPSim_plot as cplt
from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd

from ._average import CHAPSim_AVG
_avg_class = CHAPSim_AVG

class CHAPSim_budget_base():
    _module = sys.modules[__name__]

    def __init__(self,comp1,comp2,avg_data='',PhyTime=None,*args,**kwargs):

        if avg_data:
            self.avg_data = avg_data
        elif PhyTime:
            self.avg_data = self._module._avg_class(PhyTime,*args,**kwargs)
        else:
            raise Exception
        if PhyTime is None:
            PhyTime = list(set([x[0] for x in self.avg_data.flow_AVGDF.index]))[0]
        self.comp = comp1+comp2
        self.budgetDF = self._budget_extract(PhyTime,comp1,comp2)
        self.shape = self.avg_data.shape

    def _budget_extract(self,PhyTime,comp1,comp2):
        if type(PhyTime) == float and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
            
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

    def _budget_plot(self,PhyTime, x_list,wall_units=True, fig='', ax ='',**kwargs):
        if type(PhyTime) == float and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
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

        kwargs['squeeze'] = False
        lower_extent= 0.2
        kwargs['gridspec_kw'] = {'bottom': lower_extent}
        kwargs['constrained_layout'] = False
        # gridspec_kw={'bottom': lower_extent}
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize']=[7*ax_size[1],5*ax_size[0]+1]
            else:
                warnings.warn("Figure size calculator overidden: Figure quality may be degraded", stacklevel=2)
            fig, ax = cplt.subplots(*ax_size,**kwargs)
        elif not ax:
            ax=fig.subplots(*ax_size,**kwargs)
        else:
            ax =np.array([ax])
        ax=ax.flatten()
        comp_list = tuple([x[1] for x in self.budgetDF.index])
        #print(comp_list)
        i=0

        def ax_convert(j):
            return (int(j/2),j%2)
        
        
        if not hasattr(x_list,'__iter__'):
            x_list=[x_list]

        for i,x_loc in enumerate(x_list):
            for comp in comp_list:
                budget_values = self.budgetDF[PhyTime,comp].copy()
                x = self.avg_data._return_index(x_loc)
                if wall_units:
                    budget = budget_values[:Y_extent,x]/budget_scale[x]
                else:
                    budget = budget_values[:Y_extent,x]
                
                if self.comp == 'uv':
                    budget= budget * -1.0

                ax[i].cplot(Y_coords[x,:],budget,label=comp.title())
    
                
                if wall_units:
                    ax[i].set_xscale('log')
                    ax[i].set_xlim(left=1.0)
                    ax[i].set_xlabel(r"$y^+$")# ,fontsize=18)
                else:
                    ax[i].set_xlabel(r"$y/\delta$")# ,fontsize=18)
                
                # ax[i].set_title(r"$x/\delta=%.3g$" %x_coords[x],loc='right',pad=-20)
                # if i == 0:
                if mpl.rcParams['text.usetex'] == True:
                    ax[i].set_ylabel("Loss\ \ \ \ \ \ \ \ Gain")# ,fontsize=18)
                else:
                    ax[i].set_ylabel("Loss        Gain")# ,fontsize=18)
                
                #ax[i].grid()

                # handles, labels = ax[0,0].get_legend_handles_labels()
                # handles = cplt.flip_leg_col(handles,4)
                # labels = cplt.flip_leg_col(labels,4)

                # fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)# ,fontsize=17)
                # #ax[0,0].clegend(vertical=False,loc = 'lower left',ncol=4, bbox_to_anchor=(0,1.02))# ,fontsize=13)
                
            
        # if type(ax_size)==tuple:
        #     while k <ax_size[0]*ax_size[1]:
        #         i=ax_convert(k)
        #         ax[i].remove()exit
        #         k+=1   
        # gs = ax[0,0].get_gridspec()
        # gs.ax[0,0].get_gridspec()tight_layout(fig,rect=(0,0.1,1,1))
        return fig, ax
    def _plot_integral_budget(self,comp=None,PhyTime='',wall_units=True,fig='',ax='',**kwargs):
        if type(PhyTime) == float and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        budget_terms = tuple([x[1] for x in self.budgetDF.index])
        y_coords = self.avg_data.CoordDF['y']

        if comp is None:
            comp_list = tuple([x[1] for x in self.budgetDF.index])
        elif hasattr(comp,"__iter__") and not isinstance(comp,str) :
            comp_list = comp
        elif isinstance(comp,str):
            comp_list = [comp]
        else:
            raise TypeError("incorrect time")
        # print(budget_terms)
        if not all([comp in budget_terms for comp in comp_list]):
            raise KeyError("Invalid budget term provided")

        
        u_tau_star, delta_v_star = self.avg_data._wall_unit_calc(PhyTime)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplots(1,1,1)
        xaxis_vals = self.avg_data._return_xaxis()

        for comp in comp_list:
            integral_budget = np.zeros(self.avg_data.shape[1])
            budget_term = self.budgetDF[PhyTime,comp].copy()
            for i in range(self.avg_data.shape[1]):
                integral_budget[i] = 0.5*integrate.simps(budget_term[:,i],y_coords)
                if wall_units:
                    delta_star=1.0
                    integral_budget[i] /=(delta_star*u_tau_star[i]**3/delta_v_star[i])
            label = r"$\int^{\delta}_{-\delta}$ %s $dy$"%comp.title()
            ax.cplot(xaxis_vals,integral_budget,label=label)
        budget_symbol = {}

        # if wall_units:
        #     ax.set_ylabel(r"$\int_{-\delta}^{\delta} %s$ budget $dy\times u_{\tau}^4\delta/\nu $" %self.comp)# ,fontsize=16)
        # else:
        #     ax.set_ylabel(r"$\int_{-\delta}^{\delta} %s$ budget $dy\times 1/U_{b0}^3 $"%self.comp)# ,fontsize=16)
        ax.set_xlabel(r"$x/\delta$")# ,fontsize=18)
        ax.clegend(ncol=4,vertical=False)
        #ax.grid()
        return fig, ax
    def _plot_budget_x(self,comp,y_vals_list,Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        if PhyTime:
            if not isinstance(PhyTime,str) and PhyTime is not None:
                PhyTime = "{:.9g}".format(PhyTime)
        
        comp_index = [x[1] for x in self.budgetDF.index]
        if not comp.lower() in comp_index:
            msg = "comp must be a component of the"+\
                        " Reynolds stress budget: %s" %comp_index
            raise KeyError(msg) 
        if y_vals_list != 'max':
            if Y_plus:
                y_index = CT.Y_plus_index_calc(self, self.CoordDF, y_vals_list)
            else:
                y_index = CT.coord_index_calc(self.CoordDF,'y',y_vals_list)
            budget_term = self.budgetDF[PhyTime,comp]
            
        else:
            y_index = [None]
            budget_term = self.budgetDF[PhyTime,comp]
            budget_term = np.amax(budget_term,axis=0)

        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)
        xaxis_vals = self.avg_data._return_xaxis()
        if y_vals_list != 'max':
            for i in range(len(y_index)):
                ax.cplot(budget_term[i],label=r"$y^+=%.3g$"% y_vals_list[i])
            axes_items_num = len(ax.get_lines())
            ncol = 4 if axes_items_num>3 else axes_items_num
            ax.clegend(vertical=False,ncol=ncol, fontsize=16)
            # ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)$"%y_label)# ,fontsize=20)#)# ,fontsize=22)
            
        else:
            ax.cplot(xaxis_vals,budget_term,label=r"maximum %s"%comp)
            # ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"%y_label)# ,fontsize=20)#)# ,fontsize=22)

        fig.tight_layout()
        return fig, ax

    def __str__(self):
        return self.budgetDF.__str__()

class CHAPSim_budget_io(CHAPSim_budget_base):
    def _advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 

        uu = self.avg_data.UU_tensorDF[PhyTime,uu_comp]
        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        V_mean = self.avg_data.flow_AVGDF[PhyTime,'v']

        uu_dx = CT.Grad_calc(self.avg_data.CoordDF,uu,'x')
        uu_dy = CT.Grad_calc(self.avg_data.CoordDF,uu,'y')

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

        u1u2u_dx = CT.Grad_calc(self.avg_data.CoordDF,u1u2u,'x')
        u1u2v_dy = CT.Grad_calc(self.avg_data.CoordDF,u1u2v,'y')

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
        pu1_grad = CT.Grad_calc(self.avg_data.CoordDF,pu1,diff1)
        pu2_grad = CT.Grad_calc(self.avg_data.CoordDF,pu2,diff2)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff

    def _viscous_diff(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2
        u1u2 = self.avg_data.UU_tensorDF[PhyTime,uu_comp]

        REN = self.avg_data._metaDF['REN']
        viscous_diff = (1/REN)*CT.Scalar_laplacian_io(self.avg_data.CoordDF,u1u2)
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

        REN = self.avg_data._metaDF['REN']
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation

    def budget_plot(self, x_list,PhyTime='',wall_units=True, fig='', ax ='',**kwargs):
        if len(set([x[0] for x in self.avg_data.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.avg_data.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.avg_data.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super()._budget_plot(PhyTime, x_list,wall_units=wall_units, fig=fig, ax =ax,**kwargs)
        for a,x in zip(ax,x_list):
            a.set_title(r"$x^*=%.2f$"%x,loc='right')
            a.relim()
            a.autoscale_view()
        handles, labels = ax[0].get_legend_handles_labels()
        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
        ax[0].get_gridspec().tight_layout(fig,rect=(0,0.1,1,1))
        
        return fig, ax

    def plot_integral_budget(self,comp=None, PhyTime='', wall_units=True, fig='', ax='', **kwargs):
        if len(set([x[0] for x in self.avg_data.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.avg_data.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.avg_data.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super()._plot_integral_budget(comp,PhyTime, wall_units=wall_units, fig=fig, ax=ax, **kwargs)

        return fig, ax
    def plot_budget_x(self,comp=None,y_vals_list='max',Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.avg_data.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.avg_data.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.avg_data.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)

        
        if comp ==None:
            comp_list = [x[1] for x in self.budgetDF.index]
            comp_len = len(comp_list)
            for comp in comp_list:
                fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        else:
            comp_len = 1
            fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_budget_tg(CHAPSim_budget_base):
    def __init__(self,*args,**kwargs):
        if 'PhyTime' in kwargs.keys():
            raise KeyError("PhyTime cannot be used in tg class\n")
        super().__init__(*args,PhyTime=None,**kwargs)

    def _advection_extract(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2 

        uu = self.avg_data.UU_tensorDF[PhyTime,uu_comp]
        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        V_mean = self.avg_data.flow_AVGDF[PhyTime,'v']
        # uu_dx = Grad_calc(self.avg_data.CoordDF,uu,'x')
        uu_dy = CT.Grad_calc_tg(self.avg_data.CoordDF,uu)

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

        u1u2v_dy = CT.Grad_calc_tg(self.avg_data.CoordDF,u1u2v)

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
            pu1_grad = CT.Grad_calc_tg(self.avg_data.CoordDF,pu1)
        else:
            pu1_grad = np.zeros(self.avg_data.shape)

        if diff2 == 'y':
            pu2_grad = CT.Grad_calc_tg(self.avg_data.CoordDF,pu2)
        else:
            pu2_grad = np.zeros(self.avg_data.shape)

        pressure_diff = -(1/rho_star)*(pu1_grad + pu2_grad)
        return pressure_diff#.flatten()

    def _viscous_diff(self,PhyTime,comp1,comp2):
        uu_comp = comp1 + comp2
        u1u2 = self.avg_data.UU_tensorDF[PhyTime,uu_comp]

        REN = self.avg_data._metaDF['REN']
        viscous_diff = (1/REN)*CT.Scalar_laplacian_tg(self.avg_data.CoordDF,u1u2)
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


        REN = self.avg_data._metaDF['REN']
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation#.flatten()

    def budget_plot(self, times_list,wall_units=True, fig='', ax ='',**kwargs):

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

    def plot_integral_budget(self,comp=None, wall_units=True, fig='', ax='', **kwargs):
        PhyTime = None
        fig, ax = super()._plot_integral_budget(comp=comp,PhyTime=PhyTime, wall_units=wall_units, fig=fig, ax=ax, **kwargs)
        ax.set_xlabel(r"$t^*$")
        return fig, ax

    def plot_budget_x(self,comp=None,y_vals_list='max',Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        PhyTime = None
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)

        
        if comp ==None:
            comp_list = [x[1] for x in self.budgetDF.index]
            comp_len = len(comp_list)
            for comp in comp_list:
                fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        else:
            comp_len = 1
            fig, ax = super()._plot_budget_x(comp,y_vals_list,Y_plus,PhyTime,fig=fig, ax=ax)
        
        ax.set_xlabel(r"$t^*$")
        ax.get_gridspec().tight_layout(fig)
        return fig, ax