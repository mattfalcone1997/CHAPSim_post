import numpy as np
import matplotlib as mpl
import itertools
from abc import ABC, abstractmethod

from CHAPSim_post.utils import  misc_utils, indexing

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd
import CHAPSim_post as cp
from ._common import Common, postArray

class _budget_base(Common,ABC):

    def __init__(self,comp,avg_data,times = None):

        self.avg_data = avg_data

        if times is None:
            times = avg_data.times
        else:
            times = misc_utils.check_list_vals(times)
            times = [self.avg_data.check_PhyTime(time) for time in times]
        
        
        self.comp = comp
        for time in times:
            if time == times[0]:
                self.budgetDF = self._budget_extract(time,comp)
            else:
                self.budgetDF.concat(self._budget_extract(time,comp))

    @property
    def _meta_data(self):
        return self.avg_data._meta_data

    @property
    def shape(self):
        return self.avg_data.shape
    
    def _check_terms(self,comp):
        
        budget_terms = sorted(self.budgetDF.inner_index)

        if comp is None:
            comp_list = budget_terms
        elif isinstance(comp,(tuple,list)):
            comp_list = comp
        elif isinstance(comp,str):
            comp_list = [comp]
        else:
            raise TypeError("incorrect time")
        
        if not all([comp in budget_terms for comp in comp_list]):
            raise KeyError("Invalid budget term provided")

        return comp_list

    @property
    def Balance(self):
        times = list(self.avg_data.times)
        total_balance = []
        for time in times:
            balance = []
            for term in self.budgetDF.inner_index:
                balance.append(self.budgetDF[time,term])
            total_balance.append(np.array(balance).sum(axis=0))

        index = [times,['balance']*len(times)]
        return self._flowstruct_class(self._coorddata,
                                        np.array(total_balance),
                                        index=index)
            
    def __str__(self):
        return self.budgetDF.__str__()

    def _create_budget_axes(self,x_list,fig=None,ax=None,**kwargs):

        ax_size = len(x_list)
        ax_size=(int(np.ceil(ax_size/2)),2) if ax_size >2 else (ax_size,1)

        lower_extent= 0.2
        gridspec_kw = {'bottom': lower_extent}
        figsize= [7*ax_size[1],5*ax_size[0]+1]

        kwargs = cplt.update_subplots_kw(kwargs,gridspec_kw=gridspec_kw,figsize=figsize)
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(*ax_size,fig=fig,ax=ax,**kwargs)
        return fig, ax.flatten()

    @staticmethod
    def title_with_math(string):
        math_split = string.split('$')
        for i, split in enumerate(math_split):
            if i%2 == 0:
                math_split[i] = math_split[i].title()
        return "$".join(math_split)
class ReynoldsBudget_base(ABC):
    def _budget_extract(self,PhyTime,comp):
            
        production = self._production_extract(PhyTime,comp)
        advection = self._advection_extract(PhyTime,comp)
        turb_transport = self._turb_transport(PhyTime,comp)
        pressure_diffusion = self._pressure_diffusion(PhyTime,comp)
        pressure_strain = self._pressure_strain(PhyTime,comp)
        viscous_diff = self._viscous_diff(PhyTime,comp)
        dissipation = self._dissipation_extract(PhyTime,comp)

        array_concat = [production,advection,turb_transport,pressure_diffusion,\
                        pressure_strain,viscous_diff,dissipation]

        budget_array = np.stack(array_concat,axis=0)
        
        budget_index = ['production','advection','turbulent transport','pressure diffusion',\
                     'pressure strain','viscous diffusion','dissipation']  
        phystring_index = [PhyTime]*7

        budgetDF = self._flowstruct_class(self.avg_data._coorddata,
                                        budget_array,
                                        index =[phystring_index,budget_index])
        
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

class CHAPSim_budget_io(ReynoldsBudget_base,_budget_base):
    _flowstruct_class = cd.FlowStruct2D

    def _advection_extract(self,PhyTime,comp):

        uu = self.avg_data.UU_tensorDF[PhyTime,comp]
        U_mean = self.avg_data.flow_AVGDF[PhyTime,'u']
        V_mean = self.avg_data.flow_AVGDF[PhyTime,'v']

        uu_dx = self.Domain.Grad_calc(self.avg_data.CoordDF,uu,'x')
        uu_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uu,'y')

        advection = -(U_mean*uu_dx + V_mean*uu_dy)
        return advection

    def _turb_transport(self,PhyTime,comp):
        uu_comp1 = comp+'u'
        uu_comp2 = comp+'v'

        
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

    def _pressure_strain(self,PhyTime,comp):
        u1u2 = comp[0] + chr(ord(comp[1])-ord('u')+ord('x'))
        u2u1 = comp[1] + chr(ord(comp[0])-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u1u2]
        pdu2dx1 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u2u1]\

        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain

    def _pressure_diffusion(self,PhyTime,comp):
        comp1, comp2 = tuple(comp)
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

    def _viscous_diff(self,PhyTime,comp):
        u1u2 = self.avg_data.UU_tensorDF[PhyTime,comp]

        REN = self.avg_data.metaDF['REN']
        viscous_diff = (1/REN)*self.Domain.Scalar_laplacian(self.avg_data.CoordDF,u1u2)
        return viscous_diff

    def _production_extract(self,PhyTime,comp):
        comp1, comp2 = tuple(comp)

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

    def _dissipation_extract(self,PhyTime,comp):
        comp1, comp2 = tuple(comp)

        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        du1dxdu2dx = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dxdU2dx_comp]
        du1dydu2dy = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dydU2dy_comp]

        REN = self.avg_data.metaDF['REN']
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation

    def _wallunit_generator(self,x_index,PhyTime,wall_units):

        if wall_units:
            u_tau, delta_v = self.avg_data.wall_unit_calc(PhyTime)
            budget_scale = u_tau**3/delta_v

        def _x_Transform(data):
            if self.Domain.is_cylind:
                return (-1.*data.copy() -1.)/delta_v[x_index]
            else:
                return (data.copy() + 1.)/delta_v[x_index]

        def _y_Transform(data):
            return data/budget_scale[x_index]

        if wall_units:
            return _x_Transform, _y_Transform
        else:
            return None, None

    def plot_budget(self, x_list,PhyTime=None,budget_terms=None, wall_units=True,fig=None, ax =None,line_kw=None,**kwargs):
        
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        x_list = misc_utils.check_list_vals(x_list)
        x_list = indexing.true_coords_from_coords(self.CoordDF,'x',x_list)

        budget_terms = self._check_terms(budget_terms)

        fig, ax = self._create_budget_axes(x_list,fig=fig,ax=ax,**kwargs)
        line_kw= cplt.update_line_kw(line_kw)


        for i,x_loc in enumerate(x_list):
            x = self.CoordDF.index_calc('x',x_loc)[0]
            y_plus, budget_scale = self._wallunit_generator(x,PhyTime,wall_units)
            for comp in budget_terms:
                
                line_kw['label'] = self.title_with_math(comp)
                fig, ax[i] = self.budgetDF.plot_line(comp,'y',x_loc,time=PhyTime,channel_half=True,
                                                    transform_xdata=y_plus,
                                                    transform_ydata=budget_scale,
                                                    fig=fig,ax=ax[i],line_kw=line_kw)
            
            title = self.Domain.create_label(r"$x = %.2g$"%x_loc)
            ax[i].set_title(title,loc='right')

            if mpl.rcParams['text.usetex'] == True:
                ax[i].set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
            else:
                ax[i].set_ylabel(r"Loss        Gain")

            if wall_units:
                ax[i].set_xscale('log')
                ax[i].set_xlabel(r"$y^+$")

            else:
                x_label = self.Domain.create_label(r"$y$")
                ax[i].set_xlabel(x_label)

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
            
        return fig, ax

    # def plot_budget(self, x_list,PhyTime=None,budget_terms=None,wall_units=True, fig=None, ax =None,line_kw=None,**kwargs):
        

    #     fig, ax = super().plot_budget(x_list,PhyTime=PhyTime,budget_terms=budget_terms,
    #                                     fig=fig,
    #                                 ax=ax,line_kw=line_kw,**kwargs)
        
    #     x_indices = self.CoordDF.index_calc('x',x_list)
    #     u_tau, delta_v = self.avg_data.wall_unit_calc(PhyTime)
    #     budget_scale = u_tau**3/delta_v

    #     for i, a in enumerate(ax):
    #         if self.Domain.is_cylind:
    #             a.apply_func('x',lambda xdata: -1.*xdata -1.)
    #         else:
    #             a.shift_xaxis(1.)
                
    #         if wall_units:
    #             a.normalise('x',delta_v[x_indices[i]])
    #             a.normalise('y',budget_scale[x_indices[i]])

    #             a.set_xscale('log')
    #             a.set_xlabel(r"$y^+$")

    #         else:
    #             x_label = self.Domain.create_label(r"$y$")
    #             a.set_xlabel(x_label)

    #     return fig, ax

    def plot_integral_budget(self, budget_terms, PhyTime=None, fig=None, ax=None, line_kw=None, **kwargs):
        budget_terms = self._check_terms(budget_terms)
    
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= cplt.update_line_kw(line_kw)

        x_coords = self.CoordDF['x']

        for comp in budget_terms:
            budget_term = self.budgetDF[PhyTime,comp]
            int_budget = 0.5*self.Domain.Integrate_tot(self.CoordDF,budget_term)
            label = r"$\int^{\delta}_{-\delta}$ %s $dy$"%comp.title()
            ax.cplot(x_coords,int_budget,label=label,**line_kw)

        ax.set_xlabel(r"$x/\delta$")
        ncol = cplt.get_legend_ncols(len(budget_terms))
        ax.clegend(ncol=ncol,vertical=False)
        

        return fig, ax

class CHAPSim_budget_tg(ReynoldsBudget_base):
    _flowstruct_class = cd.FlowStruct1D
    def _advection_extract(self,PhyTime,comp):

        uu = self.avg_data.UU_tensorDF[PhyTime,comp]
        V_mean = self.avg_data.flow_AVGDF[PhyTime,'v']

        uu_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uu,'y')

        advection = -V_mean*uu_dy
        return advection

    def _turb_transport(self,PhyTime,comp):
        uu_comp2 = comp+'v'

        if ord(uu_comp2[0]) > ord(uu_comp2[1]):
            uu_comp2 = uu_comp2[:2][::-1] + uu_comp2[2]
        if ord(uu_comp2[0]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[::-1]
        if ord(uu_comp2[1]) > ord(uu_comp2[2]):
            uu_comp2 = uu_comp2[0] + uu_comp2[1:][::-1]

        u1u2v = self.avg_data.UUU_tensorDF[PhyTime,uu_comp2]

        u1u2v_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,u1u2v,'y')

        turb_transport = -u1u2v_dy
        return turb_transport

    def _pressure_strain(self,PhyTime,comp):
        u1u2 = comp[0] + chr(ord(comp[1])-ord('u')+ord('x'))
        u2u1 = comp[1] + chr(ord(comp[0])-ord('u')+ord('x'))

        rho_star = 1.0
        pdu1dx2 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u1u2]
        pdu2dx1 = self.avg_data.PR_Velo_grad_tensorDF[PhyTime,u2u1]

        pressure_strain = (1/rho_star)*(pdu1dx2 + pdu2dx1) 
        return pressure_strain

    def _pressure_diffusion(self,PhyTime,comp):
        comp1, comp2 = tuple(comp)

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
        return pressure_diff

    def _viscous_diff(self,PhyTime,comp):
        u1u2 = self.avg_data.UU_tensorDF[PhyTime,comp]

        REN = self.avg_data.metaDF['REN']
        viscous_diff = (1/REN)*self.Domain.Scalar_laplacian_tg(self.avg_data.CoordDF,u1u2)
        return viscous_diff

    def _production_extract(self,PhyTime,comp):
        comp1, comp2 = tuple(comp)

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
    
    def _dissipation_extract(self,PhyTime,comp):
        comp1, comp2 = tuple(comp)

        dU1dxdU2dx_comp = 'd'+ comp1 + 'dx' + 'd' + comp2 + 'dx'
        dU1dydU2dy_comp = 'd'+ comp1 + 'dy' + 'd' + comp2 + 'dy'
        
        du1dxdu2dx = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dxdU2dx_comp]
        du1dydu2dy = self.avg_data.DUDX2_tensorDF[PhyTime,dU1dydU2dy_comp]


        REN = self.avg_data.metaDF['REN']
        dissipation = -(2/REN)*(du1dxdu2dx + du1dydu2dy)
        return dissipation
    def _wallunit_generator(self,PhyTime,wall_units):

        if wall_units:
            u_tau, delta_v = self.avg_data.wall_unit_calc(PhyTime)
            budget_scale = u_tau**3/delta_v

        def _x_Transform(data):
            if self.Domain.is_cylind:
                return (-1.*data.copy() -1.)/delta_v
            else:
                return (data.copy() + 1.)/delta_v

        def _y_Transform(data):
            return data/budget_scale

        if wall_units:
            return _x_Transform, _y_Transform
        else:
            return None, None

    def plot_budget(self, PhyTime=None,budget_terms=None, wall_units=False,fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        budget_terms = self._check_terms(budget_terms)
        line_kw= cplt.update_line_kw(line_kw)

        y_plus, budget_scale = self._wallunit_generator(PhyTime,wall_units)

        for comp in budget_terms:
                
            line_kw['label'] = self.title_with_math(comp)
            fig, ax = self.budgetDF.plot_line(comp,time=PhyTime,
                                                transform_xdata=y_plus,
                                                transform_ydata=budget_scale,
                                                fig=fig,ax=ax,line_kw=line_kw)

        if mpl.rcParams['text.usetex'] == True:
            ax.set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
        else:
            ax.set_ylabel(r"Loss        Gain")

        if wall_units:
            ax.set_xscale('log')
            ax.set_xlabel(r"$y^+$")

        else:
            x_label = self.Domain.create_label(r"$y$")
            ax.set_xlabel(x_label)

        return fig, ax

class Budget_tg_array(postArray,_budget_base):
    def plot_budget(self,budget_terms=None, wall_units=False,fig=None, ax =None,line_kw=None,**kwargs):

        fig, ax = self._create_budget_axes(self._data_dict.keys(),fig,ax,**kwargs)
        
        for i, (label, data) in enumerate(self._data_dict.items()):
            fig, ax[i] = data.plot_budget(PhyTime = None, budget_terms=budget_terms,wall_units=wall_units,fig=fig,ax=ax[i],line_kw=line_kw)
            ax[i].get_legend().remove()
            ax[i].set_title(label,loc='right')

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax


class CHAPSim_budget_temp(CHAPSim_budget_tg,_budget_base):
    _flowstruct_class = cd.FlowStruct1D_time
    def plot_budget(self, time_list,budget_terms=None,wall_units=True, fig=None, ax =None,line_kw=None,**kwargs):
        
        budget_terms = self._check_terms(budget_terms)
        line_kw= cplt.update_line_kw(line_kw)
        fig, ax = self._create_budget_axes(time_list,fig,ax,**kwargs)

        for i,time in enumerate(time_list):
            fig, ax[i] = super().plot_budget(time,budget_terms=budget_terms,fig=fig,ax=ax[i],line_kw=line_kw)
            time_label = cp.styleParams.timeStyle
            ax[i].set_title(r"$%s = %.3g$"%(time_label,time),loc='right')
            
        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax

    def plot_integral_budget(self, budget_terms, fig=None, ax=None, line_kw=None, **kwargs):
        budget_terms = self._check_terms(budget_terms)
    
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= cplt.update_line_kw(line_kw)

        times = self.avg_data.times

        for comp in budget_terms:
            budget_term = self.budgetDF[None,comp]
            int_budget = 0.5*self.Domain.Integrate_tot(self.CoordDF,budget_term)
            label = r"$\int^{\delta}_{-\delta}$ %s $dy$"%comp.title()
            ax.cplot(times,int_budget,label=label,**line_kw)

        ncol = cplt.get_legend_ncols(len(budget_terms))
        ax.clegend(ncol=ncol,vertical=False)

        return fig, ax

class _k_budget(_budget_base,ABC):

    def __init__(self,comp,avg_data,times = None):

        self.avg_data = avg_data

        if times is None:
            times = avg_data.times
        else:
            times = misc_utils.check_list_vals(times)
            times = [self.avg_data.check_PhyTime(time) for time in times]
        
        
        self.comp = comp
        for time in times:
            if time == times[0]:
                self.budgetDF = self._k_budget_extract(time)
            else:
                self.budgetDF.concat(self._k_budget_extract(time))


    def _k_budget_extract(self,PhyTime):
        uu_budgetDF = self._budget_extract(PhyTime,'uu')
        vv_budgetDF = self._budget_extract(PhyTime,'vv')
        ww_budgetDF = self._budget_extract(PhyTime,'ww')

        self.budgetDF = 0.5*(uu_budgetDF + vv_budgetDF + ww_budgetDF)

class CHAPSim_k_budget_io(_k_budget,CHAPSim_budget_io):
    pass

class CHAPSim_k_budget_tg(_k_budget,CHAPSim_budget_tg):
    pass

class CHAPSim_k_budget_temp(_k_budget,CHAPSim_budget_temp):
    pass


class _momentum_budget_base(_budget_base):
    def _budget_extract(self,PhyTime,comp):
            
        advection = self._advection_extract(PhyTime,comp)
        pressure_grad = self._pressure_grad(PhyTime,comp)
        viscous = self._viscous_extract(PhyTime,comp)
        Reynolds_stress = self._turb_transport(PhyTime,comp)


        array_concat = [advection,pressure_grad,viscous,Reynolds_stress]

        budget_array = np.stack(array_concat,axis=0)
        
        budget_index = ['advection','pressure gradient','viscous stresses','reynolds stresses']  
        phystring_index = [PhyTime]*4

        budgetDF = self._flowstruct_class(self.avg_data._coorddata,budget_array,index =[phystring_index,budget_index])
        
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

class CHAPSim_momentum_budget_io(_momentum_budget_base,_budget_base):
    _flowstruct_class = cd.FlowStruct2D
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

    
    def plot_budget(self, x_list,PhyTime=None,budget_terms=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        x_list = misc_utils.check_list_vals(x_list)
        x_list = indexing.true_coords_from_coords(self.CoordDF,'x',x_list)

        budget_terms = self._check_terms(budget_terms)

        fig, ax = self._create_budget_axes(x_list,fig=fig,ax=ax,**kwargs)
        line_kw= cplt.update_line_kw(line_kw)


        for i,x_loc in enumerate(x_list):
            for comp in budget_terms:
                line_kw['label'] = self.title_with_math(comp)
                fig, ax[i] = self.budgetDF.plot_line(comp,'y',x_loc,time=PhyTime,channel_half=True,
                                                    fig=fig,ax=ax[i],line_kw=line_kw)
            
            title = self.Domain.create_label(r"$x = %.2g$"%x_loc)
            ax[i].set_title(title,loc='right')

            if mpl.rcParams['text.usetex'] == True:
                ax[i].set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
            else:
                ax[i].set_ylabel(r"Loss        Gain")


            x_label = self.Domain.create_label(r"$y$")
            ax[i].set_xlabel(x_label)

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
            
        return fig, ax

    def plot_integrated_budget(self,x_list,budget_terms=None,PhyTime=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        x_list = misc_utils.check_list_vals(x_list)
        budget_terms = self._check_terms(budget_terms)

        fig, ax = self._create_budget_axes(x_list,fig,ax,**kwargs)
        line_kw= cplt.update_line_kw(line_kw)

        x_indices = self.CoordDF.index_calc('x',x_list)
        y_coords = self.CoordDF['y']
        for i,x_loc in enumerate(x_indices):
            for comp in budget_terms:
                
                line_kw['label'] = self.title_with_math(comp)
                budget = self.budgetDF[PhyTime,comp]

                int_budget = self.Domain.Integrate_cumult(self.CoordDF,budget)
                ax[i].cplot(y_coords,int_budget[:,x_loc],**line_kw)


            if mpl.rcParams['text.usetex'] == True:
                ax[i].set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
            else:
                ax[i].set_ylabel(r"Loss        Gain")

            x_label = self.Domain.create_label(r"$y$")
            ax[i].set_xlabel(x_label)

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)
            
        return fig, ax

class CHAPSim_momentum_budget_tg(_momentum_budget_base,_budget_base):
    _flowstruct_class = cd.FlowStruct1D
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

    def _pressure_grad(self, PhyTime, comp):
        U_mean = self.avg_data.flow_AVGDF[PhyTime,comp]

        REN = self.avg_data.metaDF['REN']
        d2u_dy2 = self.Domain.Grad_calc(self.avg_data.CoordDF,
                    self.Domain.Grad_calc(self.avg_data.CoordDF,U_mean,'y'),'y')
        
        uv = self.avg_data.UU_tensorDF[PhyTime,comp + 'v']
        duv_dy = self.Domain.Grad_calc(self.avg_data.CoordDF,uv,'y')

        return -( (1/REN)*d2u_dy2 - duv_dy )

    def plot_budget(self, PhyTime=None,budget_terms=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        budget_terms = self._check_terms(budget_terms)
        line_kw= cplt.update_line_kw(line_kw)

        for comp in budget_terms:
                
            line_kw['label'] = self.title_with_math(comp)
            fig, ax = self.budgetDF.plot_line(comp,time=PhyTime,channel_half=True,
                                            fig=fig,ax=ax,line_kw=line_kw)

        if mpl.rcParams['text.usetex'] == True:
            ax.set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
        else:
            ax.set_ylabel(r"Loss        Gain")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)

        return fig, ax

    def plot_integrated_budget(self,budget_terms=None,PhyTime=None, fig=None, ax =None,line_kw=None,**kwargs):
        PhyTime = self.avg_data.check_PhyTime(PhyTime)
        budget_terms = self._check_terms(budget_terms)
        line_kw= cplt.update_line_kw(line_kw)

        fig, ax = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)
        y_coords = self.CoordDF['y']

        for comp in budget_terms:
            line_kw['label'] = self.title_with_math(comp)
            budget = self.budgetDF[PhyTime,comp]

            int_budget = self.Domain.Integrate_cumult(self.CoordDF,budget)
            ax.cplot(y_coords,int_budget,**line_kw)

        if mpl.rcParams['text.usetex'] == True:
            ax.set_ylabel(r"Loss\ \ \ \ \ \ \ \ Gain")
        else:
            ax.set_ylabel(r"Loss        Gain")

        x_label = self.Domain.create_label(r"$y$")
        ax.set_xlabel(x_label)


        ax.legend()

        return fig, ax

class Mom_budget_array(Budget_tg_array):
    def plot_integrated_budget(self,budget_terms=None, wall_units=False,fig=None, ax =None,line_kw=None,**kwargs):

        fig, ax = self._create_budget_axes(self._data_dict.keys(),fig,ax,**kwargs)
        
        for i, (label, data) in enumerate(self._data_dict.items()):
            fig, ax[i] = data.plot_budget(PhyTime = None, budget_terms=budget_terms,wall_units=wall_units,fig=fig,ax=ax[i],line_kw=line_kw)
            ax[i].get_legend().remove()
            ax[i].set_title(label,loc='right')

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax


class CHAPSim_momentum_budget_temp(CHAPSim_momentum_budget_tg,_budget_base):
    _flowstruct_class = cd.FlowStruct1D_time

    def plot_budget(self,times_list, budget_terms=None,fig=None, ax =None,line_kw=None,**kwargs):
        
        fig, ax = self._create_budget_axes(times_list,fig,ax,**kwargs)
        for i,time in enumerate(times_list):
            fig, ax[i] = super().plot_budget(PhyTime=time,budget_terms=budget_terms,fig=fig,ax=ax[i],line_kw=line_kw)

            time_label = cp.styleParams.timeStyle
            ax[i].set_title(r"$%s = %.3g$"%(time_label,time),loc='right')

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax

    def plot_integrated_budget(self,times_list, budget_terms=None,fig=None, ax =None,line_kw=None,**kwargs):
        
        fig, ax = self._create_budget_axes(times_list,fig,ax,**kwargs)
        for i,time in enumerate(times_list):
            fig, ax[i] = super().plot_integrated_budget(PhyTime=time,budget_terms=budget_terms,fig=fig,ax=ax[i],line_kw=line_kw)
            ax[i].get_legend().remove()

            time_label = cp.styleParams.timeStyle
            ax[i].set_title(r"$%s = %.3g$"%(time_label,time),loc='right')

        handles = ax[0].get_lines()
        labels = [line.get_label() for line in handles]

        handles = cplt.flip_leg_col(handles,4)
        labels = cplt.flip_leg_col(labels,4)

        fig.clegend(handles,labels,loc='upper center',bbox_to_anchor=(0.5,0.1),ncol=4)

        return fig, ax

class _FIK_developing_base(_budget_base):

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
    def plot(self,budget_terms=None,plot_total=True,PhyTime=None,fig=None,ax=None,line_kw=None,**kwargs):

        budget_terms = self._check_terms(budget_terms)
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5])
        fig, ax  = cplt.create_fig_ax_with_squeeze(fig,ax,**kwargs)

        line_kw= cplt.update_line_kw(line_kw)
        xaxis_vals = self.avg_data._return_xaxis()

        for comp in budget_terms:
            budget_term = self.budgetDF[PhyTime,comp].copy()
                
            label = self.title_with_math(comp)
            ax.cplot(xaxis_vals,budget_term,label=label,**line_kw)
        if plot_total:
            ax.cplot(xaxis_vals,np.sum(self.budgetDF.values,axis=0),label="Total",**line_kw)
        ncol = cplt.get_legend_ncols(len(budget_terms))
        ax.clegend(ncol=ncol,vertical=False)
        return fig, ax

class CHAPSim_FIK_io(_FIK_developing_base):
    def __init__(self,avg_data,times = None):
        self.avg_data = avg_data
        if times is None:
            times = avg_data.times
        else:
            times = misc_utils.check_list_vals(times)
            times = [self.avg_data.check_PhyTime(time) for time in times]

        for time in times:
            if time == times[0]:
                self.budgetDF = self._budget_extract(time)
            else:
                self.budgetDF.concat(self._budget_extract(time))

    def _scale_vel(self,PhyTime):
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

class CHAPSim_FIK_temp(_FIK_developing_base):
    def __init__(self,avg_data):
        self.avg_data = avg_data
        self.budgetDF = self._budget_extract(None)
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
