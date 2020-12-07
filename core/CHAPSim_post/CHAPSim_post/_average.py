"""
# _averaged.py
File contains the implementation of the classes to process the 
averaged results from the CHAPSim DNS solver 
"""

import numpy as np
import matplotlib as mpl
import h5py
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


from ._meta import CHAPSim_meta
_meta_class = CHAPSim_meta

class CHAPSim_AVG_base():
    _module = sys.modules[__name__]
    def __init__(self,*args,**kwargs):
        fromfile= kwargs.pop('fromfile',False)
        copy = kwargs.pop('copy',False)
        if fromfile:
            self._meta_data, self.CoordDF,self._metaDF,\
            self.NCL,self.shape,self._times,self.flow_AVGDF,self.PU_vectorDF,\
            self.UU_tensorDF,self.UUU_tensorDF,\
            self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
            self.DUDX2_tensorDF = self._hdf_extract(*args,**kwargs)
        elif copy:
            self._meta_data, self.CoordDF,self._metaDF,\
            self.NCL,self.shape,self._times,self.flow_AVGDF,self.PU_vectorDF,\
            self.UU_tensorDF,self.UUU_tensorDF,\
            self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
            self.DUDX2_tensorDF = self._copy_extract(*args,**kwargs)
        else:
            self._meta_data, self.CoordDF,self._metaDF,\
            self.NCL,self.shape,self._times,self.flow_AVGDF,self.PU_vectorDF,\
            self.UU_tensorDF,self.UUU_tensorDF,\
            self.Velo_grad_tensorDF, self.PR_Velo_grad_tensorDF,\
            self.DUDX2_tensorDF = self._extract_avg(*args,**kwargs)

    def _extract_avg(self,*args,**kwargs):
        raise NotImplementedError
    
    @classmethod
    def copy(cls,avg_data):
        return cls(avg_data,copy=True)

    def _copy_extract(self, avg_data):
        try:
            meta_data = self._module._meta_class.copy(avg_data._meta_data)
        except AttributeError:
            meta_data = avg_data._meta_data
        CoordDF = avg_data.CoordDF
        metaDF = avg_data._metaDF
        NCL = avg_data.NCL
        shape = avg_data.shape
        times = avg_data._times
        flow_AVGDF = avg_data.flow_AVGDF
        PU_vectorDF = avg_data.PU_vectorDF
        UU_tensorDF = avg_data.UU_tensorDF
        UUU_tensorDF = avg_data.UUU_tensorDF
        Velo_grad_tensorDF = avg_data.Velo_grad_tensorDF
        PR_Velo_grad_tensorDF = avg_data.PR_Velo_grad_tensorDF
        DUDX2_tensorDF = avg_data.DUDX2_tensorDF

        return_list = [meta_data, CoordDF, metaDF, NCL,shape,times,flow_AVGDF,PU_vectorDF,\
                        UU_tensorDF,UUU_tensorDF, Velo_grad_tensorDF, PR_Velo_grad_tensorDF,\
                        DUDX2_tensorDF]
        return itertools.chain(return_list)
    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)
    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_AVG'
        hdf_file = h5py.File(file_name,write_mode)
        group = hdf_file.create_group(base_name)
        group.create_dataset("NCL",data=self.NCL)
        hdf_file.close()
        self._meta_data.save_hdf(file_name,'a',group_name=base_name+'/meta_data')
        self.flow_AVGDF.to_hdf(file_name,key=base_name+'/flow_AVGDF',mode='a')#,format='fixed',data_columns=True)
        self.PU_vectorDF.to_hdf(file_name,key=base_name+'/PU_vectorDF',mode='a')#,format='fixed',data_columns=True)
        self.UU_tensorDF.to_hdf(file_name,key=base_name+'/UU_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.UUU_tensorDF.to_hdf(file_name,key=base_name+'/UUU_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.Velo_grad_tensorDF.to_hdf(file_name,key=base_name+'/Velo_grad_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.PR_Velo_grad_tensorDF.to_hdf(file_name,key=base_name+'/PR_Velo_grad_tensorDF',mode='a')#,format='fixed',data_columns=True)
        self.DUDX2_tensorDF.to_hdf(file_name,key=base_name+'/DUDX2_tensorDF',mode='a')#,format='fixed',data_columns=True)

    def _hdf_extract(self,file_name,shape,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_AVG'
        flow_AVGDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/flow_AVGDF')#pd.read_hdf(file_name,key=base_name+'/flow_AVGDF')
        PU_vectorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/PU_vectorDF')
        UU_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/UU_tensorDF')
        UUU_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/UUU_tensorDF')
        Velo_grad_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/Velo_grad_tensorDF')
        PR_Velo_grad_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/PR_Velo_grad_tensorDF')
        DUDX2_tensorDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/DUDX2_tensorDF')

        return_list = [flow_AVGDF,PU_vectorDF,UU_tensorDF,
                        UUU_tensorDF,Velo_grad_tensorDF,
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF]
        return itertools.chain(return_list)
    def _Reverse_decomp(self,flow_AVGDF,PU_vectorDF,UU_tensorDF,
                        UUU_tensorDF,Velo_grad_tensorDF,
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF):

        pu_vector_index = PU_vectorDF.index
        for index in pu_vector_index:
            P_mean = flow_AVGDF[index[0],'P']
            u_mean = flow_AVGDF[index]
            PU_vectorDF[index] -= P_mean*u_mean

        uu_tensor_index = UU_tensorDF.index
        for index in uu_tensor_index:
            u1_mean = flow_AVGDF[index[0],index[1][0]]
            u2_mean = flow_AVGDF[index[0],index[1][1]]
            UU_tensorDF[index] -= u1_mean*u2_mean


        uuu_tensor_index = UUU_tensorDF.index
        for index in uuu_tensor_index:
            u1u2 = UU_tensorDF[index[0],index[1][:2]]
            u2u3 = UU_tensorDF[index[0],index[1][1:]]
            comp13 = index[1][0] + index[1][2]
            u1u3 = UU_tensorDF[index[0],comp13]
            u1_mean = flow_AVGDF[index[0],index[1][0]]
            u2_mean = flow_AVGDF[index[0],index[1][1]]
            u3_mean = flow_AVGDF[index[0],index[1][2]]
            UUU_tensorDF[index] -= (u1_mean*u2_mean*u3_mean + u1_mean*u2u3 \
                        + u2_mean*u1u3 + u3_mean*u1u2)

        PR_velo_grad_index = Velo_grad_tensorDF.index
        for index in PR_velo_grad_index:
            p_mean = flow_AVGDF[index[0],'P']
            u_grad = Velo_grad_tensorDF[index]
            PR_Velo_grad_tensorDF[index] -= p_mean*u_grad

        dudx2_tensor_index = DUDX2_tensorDF.index
        for index in dudx2_tensor_index:
            comp1 = index[1][1] + index[1][3]
            comp2 = index[1][5] + index[1][7]
            u1x_grad = Velo_grad_tensorDF[index[0],comp1]
            u2x_grad = Velo_grad_tensorDF[index[0],comp2]
            DUDX2_tensorDF[index] -= u1x_grad*u2x_grad

        return flow_AVGDF,PU_vectorDF,UU_tensorDF,\
                        UUU_tensorDF,Velo_grad_tensorDF,\
                        PR_Velo_grad_tensorDF,DUDX2_tensorDF

    def _AVG_extract(self, *args,**kwargs):
        raise NotImplementedError

    def _return_index(self,*args,**kwargs):
        raise NotImplementedError
    
    def get_times(self):
        return self._times

    def _wall_unit_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        
        mu_star = 1.0
        rho_star = 1.0
        nu_star = mu_star/rho_star

        REN = self._metaDF['REN']
        
        tau_w = self._tau_calc(PhyTime)
    
        u_tau_star = np.sqrt(tau_w/rho_star)/np.sqrt(REN)
        delta_v_star = (nu_star/u_tau_star)/REN
        return u_tau_star, delta_v_star

    def _y_plus_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)

        u_tau_star, delta_v_star = self._wall_unit_calc(PhyTime)
        y_plus_shape=(self.shape[1],int(self.NCL[1]*0.5))
        y_plus = np.zeros(y_plus_shape)
        y_coord = self.CoordDF['y'][:int(self.NCL[1]*0.5)]
        for i in range(len(delta_v_star)):
            y_plus[i] = (1-abs(y_coord))/delta_v_star[i]
        return y_plus

    def _int_thickness_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        U0_index = int(np.floor(self.NCL[1]*0.5))
        U_mean = self.flow_AVGDF[PhyTime,'u'].copy()
        y_coords = self.CoordDF['y']

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
    def _plot_shape_factor(self,*arg,fig='',ax='',**kwargs):
        _, _, shape_factor = self._int_thickness_calc(*arg)
        # x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)
        ax.cplot(shape_factor,label=r"$H$")
        # ax.set_xlabel(r"$x/\delta$ ")# ,fontsize=18)
        ax.set_ylabel(r"$H$")# ,fontsize=18)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

    def plot_mom_thickness(self,*arg,fig='',ax='',**kwargs):
        _, theta, _ = self.int_thickness_calc(*arg)
        # x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)
        ax.cplot(theta,label=r"$\theta$")
        # ax.set_xlabel(r"$x/\delta$ ")# ,fontsize=18)
        ax.set_ylabel(r"$\theta$")# ,fontsize=18)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

    def plot_disp_thickness(self,*arg,fig='',ax='',**kwargs):
        delta, _, _ = self.int_thickness_calc(*arg)
        # x_coords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)
        ax.cplot(delta,label=r"$\delta^*$")
        # ax.set_xlabel(r"$x/\delta$ ")# ,fontsize=18)
        ax.set_ylabel(r"$\delta^*$")# ,fontsize=18)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

    def _avg_line_plot(self,x_vals, PhyTime,comp,fig='',ax='',**kwargs):
        # if not isinstance(PhyTime,str) and not np.isnan(PhyTime):
        #     PhyTime = "{:.9g}".format(PhyTime)

        if not fig:
            if "figsize" not in kwargs.keys():
                kwargs['figsize'] = [8,6]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.subplots(**kwargs)
        # y_size = int(self.shape[0]*0.5)
        for x in x_vals:
            try:
                index = self._return_index(x)
            except ValueError:
                warnings.warn("Wrong time input", stacklevel=2)
                continue

            velo_mean=self.flow_AVGDF[PhyTime,comp][:,index]
            ax.cplot(velo_mean)
        return fig, ax        

    def avg_line_plot(self,x_vals, *args,**kwargs):

        fig, ax = self._avg_line_plot(x_vals,*args,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        y_coords = self.CoordDF['y']
        for line in lines:
            line.set_xdata(y_coords)
        ax.set_xlabel(r"$y^*$")
        ax.set_ylabel(r"$\bar{u}$")
        ax.relim()
        ax.autoscale_view()

        return fig, ax
    
    def plot_near_wall(self,x_vals,PhyTime,fig='',ax='',**kwargs):
        fig, ax = self._avg_line_plot(x_vals,PhyTime,'u',fig=fig,ax=ax,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        u_tau_star, _ = self.wall_unit_calc(PhyTime)
        y_plus = self._y_plus_calc(PhyTime)
        Y_extent = int(self.shape[0]/2)

        for line,x_val in zip(lines,x_vals):
            x_loc = self._return_index(x_val)
            y_data = line.get_ydata()
            y_data = y_data[:Y_extent]/u_tau_star[x_loc]
            line.set_ydata(y_data)
            line.set_xdata(y_plus[x_loc])
            ylim = ax.get_ylim()[1]
            ax.set_ylim(top=1.1*max(ylim,np.amax(y_data)))

        ax.cplot(y_plus[x_loc],y_plus[x_loc],color='red',linestyle='--',
                            label=r"$\bar{u}^+=y^+$")
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel(r"$\bar{u}^+$")
        ax.set_xscale('log')
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_Reynolds(self,comp1,comp2,x_val,PhyTime,norm=None,Y_plus=True,fig='',ax='',**kwargs):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
        comp_uu =comp1 + comp2
        if comp1 == 'w' and (comp2=='v' or comp2 =='u'):
            comp_uu = comp_uu[::-1]
        elif comp1 == 'v' and comp2 =='u':
            comp_uu = comp_uu[::-1]     
        
        x_loc = [self._return_index(x) for x in x_val]
        uu = self.UU_tensorDF[PhyTime,comp_uu].copy()
        
        if comp_uu == 'uv':
            uu *= -1.0

        uu = uu[:int(self.NCL[1]/2)]
        y_coord = self.CoordDF['y'][:int(self.NCL[1]/2)]
        #Reynolds_wall_units = np.zeros_like(uu)
        u_tau_star, delta_v_star = self._wall_unit_calc(PhyTime)
        if norm=='wall':
            for i in range(self.shape[1]):
                uu[:,i] = uu[:,i]/(u_tau_star[i]*u_tau_star[i])

        elif norm=='local-bulk':
            velo_bulk=self._bulk_velo_calc(PhyTime)
            for i in range(self.shape[1]):
                uu[:,i] = uu[:,i]/(velo_bulk[i]*velo_bulk[i])

        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.c_add_subplot(1,1,1)

        if isinstance(x_loc,int):
            if Y_plus:
                y_coord_local = (1-np.abs(y_coord))/delta_v_star[x_loc]
            else:
                y_coord_local = y_coord
            ax.cplot(y_coord_local,uu[:,x_loc])

        elif isinstance(x_loc,list):
            for j,x in enumerate(x_loc):
                if Y_plus:
                    y_coord_local = (1-np.abs(y_coord))/delta_v_star[x]
                else:
                    y_coord_local = y_coord
                label=r"$x/\delta =%.3g$" % self.CoordDF['x'][x]
                ax.cplot(y_coord_local,uu[:,x],label=label)

            # axes_items_num = len(ax.get_lines())
            # ncol = 4 if axes_items_num>3 else axes_items_num
            # ax.clegend(vertical=False,ncol=ncol)
        else:
            raise TypeError("\033[1;32 x_loc must be of type list or int")

        y_label = comp_uu[0] +'\'' + comp_uu[1] +'\''
        if norm=='wall':
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/u_\tau^2$"% y_label)# ,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/u_\tau^2$"% y_label)# ,fontsize=20)
        elif norm=='local-bulk':
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/U_b^2$"% y_label)# ,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/U_b^2$"% y_label)# ,fontsize=20)
        else:
            if comp_uu == 'uv':
                ax.set_ylabel(r"$-\langle %s\rangle/U_{b0}^2$"% y_label)# ,fontsize=20)
            else:
                ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"% y_label)# ,fontsize=20)
        
        if Y_plus:
            ax.set_xlabel(r"$y^+$")# ,fontsize=20)
        else:
            ax.set_xlabel(r"$y/\delta$")# ,fontsize=20)
        #ax.grid()
        
        fig.tight_layout()
        
        return fig, ax

    def plot_Reynolds_x(self,comp1,comp2,y_vals_list,Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        if PhyTime:
            if not isinstance(PhyTime,str) and PhyTime is not None:
                PhyTime = "{:.9g}".format(PhyTime)
        
        # PUT IN IO CLASS
        # if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
        #     avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
        #     if PhyTime and PhyTime != avg_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
        #     PhyTime = avg_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        
        comp_uu =comp1 + comp2
        if comp1 == 'w' and (comp2=='v' or comp2 =='u'):
            comp_uu = comp_uu[::-1]
        elif comp1 == 'v' and comp2 =='u':
            comp_uu = comp_uu[::-1]  
        
        if y_vals_list != 'max':
            if Y_plus:
                y_index = CT.Y_plus_index_calc(self, self.CoordDF, y_vals_list)
            else:
                y_index = CT.coord_index_calc(self.CoordDF,'y',y_vals_list)
            rms_vals = self.UU_tensorDF[PhyTime,comp_uu].copy()[y_index]
            # U1_mean = self.flow_AVGDF.loc[PhyTime,comp1].copy().values.reshape(self.shape)[y_index]
            # U2_mean = self.flow_AVGDF.loc[PhyTime,comp2].copy().values.reshape(self.shape)[y_index]
            # rms_vals = UU-U1_mean*U2_mean
            
        else:
            y_index = [None]
            rms_vals = self.UU_tensorDF[PhyTime,comp_uu].copy()
            # U1_mean = self.flow_AVGDF.loc[PhyTime,comp1].copy().values.reshape(self.shape)
            # U2_mean = self.flow_AVGDF.loc[PhyTime,comp2].copy().values.reshape(self.shape)
            # rms_vals = UU-U1_mean*U2_mean
            rms_vals = np.amax(rms_vals,axis=0)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax=fig.c_add_subplot(1,1,1)

        # x_coords = self.CoordDF['x'].dropna().values            
        y_label = comp_uu[0] +'\'' + comp_uu[1] +'\''
        if y_vals_list != 'max':
        
            for i in range(len(y_index)):
                ax.cplot(rms_vals[i],label=r"$y^+=%.3g$"% y_vals_list[i])
            axes_items_num = len(ax.get_lines())
            ncol = 4 if axes_items_num>3 else axes_items_num
            ax.clegend(vertical=False,ncol=ncol, fontsize=16)
            # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
            ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)$"%y_label)# ,fontsize=20)#)# ,fontsize=22)
            
        else:
            ax.cplot(rms_vals,label=r"$(\langle %s\rangle/U_{b0}^2)_{max}$"%y_label)
            # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
            ax.set_ylabel(r"$\langle %s\rangle/U_{b0}^2$"%y_label)# ,fontsize=20)#)# ,fontsize=22)
        
        # ax.set_xlabel(r"$x/\delta$",text_kwargs)
        # ax.set_ylabel(r"$(\langle %s\rangle/U_{b0}^2)_{max}$"%comp_uu,text_kwargs)#)# ,fontsize=22)
        #ax.grid()
        fig.tight_layout()
        return fig, ax

        # def _avg_line_plot(self,x_vals,PhyTime,comp,)

    def _bulk_velo_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
            
        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        # wall_velo = self._meta_data.moving_wall_calc()
        
        bulk_velo=np.zeros(self.shape[1])
        # if relative:
        #     for i in range(self.NCL[1]):
        #         u_velo[i,:]=u_velo[i,:] - wall_velo
        for i in range(self.shape[1]):
            bulk_velo[i] = 0.5*integrate.simps(u_velo[:,i],ycoords)
            
        return bulk_velo

    def plot_bulk_velocity(self,PhyTime,fig='',ax='',**kwargs):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
    
        bulk_velo = self._bulk_velo_calc(PhyTime)
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig, ax=cplt.subplots(**kwargs)
        elif not ax:
            ax =fig.add_subplot(1,1,1)
        ax.cplot(bulk_velo)
        ax.set_ylabel(r"$U_b^*$")# ,fontsize=20)
        # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
        #ax.grid()
        return fig, ax

    def _tau_calc(self,PhyTime):
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)

        u_velo = self.flow_AVGDF[PhyTime,'u']
        ycoords = self.CoordDF['y']
        
        # wall_velo = self._meta_data.moving_wall_calc()
        
        tau_star = np.zeros_like(u_velo[1])
        mu_star = 1.0
        #time0=time.time()
        #sol, h_list = CT.Stencil_calc([0,1,2,3], 1)
        #print(time.time()-time0)
        #a,b,c,d = CT.Stencil_coeffs_eval(sol,h_list,[ycoords[0]--1.0,ycoords[1]-ycoords[0],ycoords[2]-ycoords[1]])
        for i in range(self.shape[1]):
            #tau_star[i] = mu_star*(a*wall_velo[i] + b*u_velo[0,i] + c*u_velo[1,i] + d*u_velo[2,i])
            tau_star[i] = mu_star*(u_velo[0,i]-0.0)/(ycoords[0]--1.0)#*(-1*u_velo[1,i] + 4*u_velo[0,i] - 3*wall_velo[i])/(0.5*ycoords[1]-1.5*(-1.0)+y_coords[0])
    
        return tau_star

    def plot_skin_friction(self,PhyTime,fig='',ax='',**kwargs):
        rho_star = 1.0
        REN = self._metaDF['REN']
        tau_star = self._tau_calc(PhyTime)
        bulk_velo = self._bulk_velo_calc(PhyTime)
        
        skin_friction = (2.0/(rho_star*bulk_velo*bulk_velo))*(1/REN)*tau_star
        # xcoords = self.CoordDF['x'].dropna().values
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [7,5]
            fig,ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        
        ax.cplot(skin_friction,label=r"$C_f$")
        # ax.set_xlabel(r"$x/\delta$")# ,fontsize=20)
        ax.set_ylabel(r"$C_f$")# ,fontsize=20)
        fig.tight_layout()
        #ax.grid()
        return fig, ax

    def plot_eddy_visc(self,x_val,PhyTime,Y_plus=True,Y_plus_max=100,fig='',ax='',**kwargs):
        
        if not isinstance(PhyTime,str) and PhyTime is not None:
            PhyTime = "{:.9g}".format(PhyTime)
                
        x_loc = [self._return_index(x) for x in x_val]
    
        # PUT IN IO CLASS
        # if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
        #     avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
        #     if PhyTime and PhyTime != avg_time:
        #         warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time))
        #     PhyTime = avg_time
        # else:
        #     assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        # if type(PhyTime) == float:
        #     PhyTime = "{:.9g}".format(PhyTime)
        if isinstance(x_loc,int):
            x_loc_local = [x_loc]
        elif isinstance(x_loc,list):
            x_loc_local=x_loc
        else:
            raise TypeError("\033[1;32 variable x_loc must be of type int of list of int")
        uv = self.UU_tensorDF[PhyTime,'uv']
        # U = self.flow_AVGDF.loc[PhyTime,'u'].values.reshape(self.shape)
        # V = self.flow_AVGDF.loc[PhyTime,'v'].values.reshape(self.shape)
        # uv = UV-U*V
        dUdy = self.Velo_grad_tensorDF[PhyTime,'uy']
        dVdx = self.Velo_grad_tensorDF[PhyTime,'vx']
        REN = self._metaDF['REN']
        mu_t = -uv*REN/(dUdy + dVdx)
        mu_t = mu_t[:,x_loc_local]
        y_coord = self._meta_data.CoordDF['y']
        
        
        if not fig:
            if 'figsize' not in kwargs.keys():
                kwargs['figsize'] = [10,5]
            fig, ax = cplt.subplots(**kwargs)
        elif not ax:
            ax = fig.add_subplot(1,1,1)
        linestyle_list=['-','--','-.']
        # x_coord = self._meta_data.CoordDF['x'].dropna().values
        
        for i in range(len(x_loc_local)):
            if Y_plus:
                avg_time = self.flow_AVGDF.index[0][0]
                #wall_params = self._meta_data.metaDF.loc['moving_wallflg':'VeloWall']
                # u_tau_star, delta_v_star = self._wall_unit_calc(avg_time)
                y_coord_local = self._y_plus_calc(PhyTime)[i]
                #y_coord_local = y_coord_local[y_coord_local<Y_plus_max]
                
                mu_t_local = mu_t[:int(y_coord.size/2),i]
                
            else:
                y_coord_local = y_coord
                mu_t_local = mu_t[:,i]
            
            # label = r"$x/\delta = %.3g$" %x_coord[x_loc_local[i]]
            
            ax.cplot(y_coord_local,mu_t_local)#label=label)
            if Y_plus:
                ax.set_xlabel(r"$y^+$")# ,fontsize=18)
                ax.set_xlim([0,Y_plus_max])
                ax.set_ylim([-0.5,max(mu_t_local)*1.2])
            else:
                ax.set_xlabel(r"$y/\delta$")# ,fontsize=18)
                ax.set_xlim([-1,-0.1])
                ax.set_ylim([-0.5,max(mu_t_local)*1.2])
            ax.set_ylabel(r"$\mu_t/\mu$")# ,fontsize=16)

        return fig, ax
    def __iadd__(self,other_avg):
        pass

class CHAPSim_AVG_io(CHAPSim_AVG_base):
    tgpost = False
    def _extract_avg(self,time,meta_data='',path_to_folder='',time0='',abs_path=True):
        
        if not meta_data:
            meta_data = self._module._meta_class(path_to_folder,abs_path,False)
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL = meta_data.NCL
       
        if isinstance(time,float):
            DF_list = self._AVG_extract(time,time0,path_to_folder,abs_path)
        elif hasattr(time,'__iter__'):
            for PhyTime in time:
                if 'DF_list' not in locals():
                    DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                else:
                    DF_temp=[]
                    local_DF_list = self._AVG_extract(PhyTime,time0,path_to_folder,abs_path)
                    for i, local_DF in enumerate(DF_list):
                        DF_list[i].concat(local_DF_list[i])
                    # DF_list=DF_temp
                    
        else:
            raise TypeError("\033[1;32 `time' can only be a float or a list")
        
        DF_list=self._Reverse_decomp(*DF_list)

        times = list(set([x[0] for x in DF_list[0].index]))
        shape = (NCL[1],NCL[0])

        return_list = [meta_data, CoordDF, metaDF, NCL,shape,times, *DF_list]
        return itertools.chain(return_list)
    
    def _hdf_extract(self,file_name,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_io'
        
        
        meta_data = self._module._meta_class.from_hdf(file_name,group_name+'/meta_data')
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL=meta_data.NCL
        shape = (NCL[1],NCL[0])
        parent_list = list(super()._hdf_extract(file_name,shape,group_name))
        # for DF in parent_list:
        #     DF.data(shape)

        hdf_file = h5py.File(file_name,'r')
        times = list(set([x[0] for x in parent_list[0].index]))
    
        hdf_file.close()

        return_list = [meta_data,CoordDF,metaDF,NCL,shape,times]

        return itertools.chain(return_list + parent_list)
    # @profile(stream=open("mem_check_avg.txt",'w'))    
    def _AVG_extract(self,Time_input,time0,path_to_folder,abs_path):

        AVG_info, NSTATIS1, PhyTime, NCL = self._extract_file(Time_input,path_to_folder,abs_path)

        if time0:
            AVG_info0, NSTATIS0,_,_ = self._extract_file(time0,path_to_folder,abs_path)
            AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)
            del AVG_info0

        (NCL1, NCL2) = NCL
        AVG_info = AVG_info.reshape(21,50,NCL2,NCL1)
            
        #Velo_AVG = np.zeros((3,NCL2,NCL1))
        Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        Pr_Velo_grad_tensor = np.zeros((9,NCL2,NCL1))
        DUDX2_tensor = np.zeros((81,NCL2,NCL1))
        
        for i in range(3):
            for j in range(3):
                Velo_grad_tensor[i*3+j,:,:] = AVG_info[6+j,i,:,:]
                Pr_Velo_grad_tensor[i*3+j,:,:] = AVG_info[9+j,i,:,:]
        for i in range(9):
            for j in range(9):
                DUDX2_tensor[i*9+j] = AVG_info[12+j,i,:,:] 
        
        flow_AVG = AVG_info[0,:4,:,:].copy()
        PU_vector = AVG_info[2,:3,:,:].copy()
        UU_tensor = AVG_info[3,:6,:,:].copy()
        UUU_tensor = AVG_info[5,:10,:,:].copy()

        del AVG_info; gc.collect()
        #======================================================================
        # flow_AVG = flow_AVG.reshape((4,NCL2*NCL1))
        
        # PU_vector = PU_vector.reshape((3,NCL1*NCL2))
        # UU_tensor = UU_tensor.reshape((6,NCL1*NCL2))
        # UUU_tensor = UUU_tensor.reshape((10,NCL1*NCL2))
        # Velo_grad_tensor = Velo_grad_tensor.reshape((9,NCL1*NCL2))
        # Pr_Velo_grad_tensor = Pr_Velo_grad_tensor.reshape((9,NCL1*NCL2))
        # DUDX2_tensor = DUDX2_tensor.reshape((81,NCL1*NCL2))
        #======================================================================
        #Set up of pandas dataframes
        Phy_string = '%.9g' % PhyTime
        flow_index = [[Phy_string]*4,['u','v','w','P']]
        vector_index = [[Phy_string]*3,['u','v','w']]
        sym_2_tensor_index = [[Phy_string]*6,['uu','uv','uw','vv','vw','ww']]
        sym_3_tensor_index = [[Phy_string]*10,['uuu','uuv','uuw','uvv',\
                                'uvw','uww','vvv','vvw','vww','www']]
        tensor_2_index = [[Phy_string]*9,['ux','uy','uz','vx','vy','vz',\
                                         'wx','wy','wz']]
        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']

        dudx_list = list(itertools.product(du_list,dx_list))
        dudx_list = ["".join(dudx) for dudx in dudx_list]
        comp_string_list = list(itertools.product(dudx_list,dudx_list))
        comp_string_list = ["".join(comp_string) for comp_string in comp_string_list]

        # OR:
        # comp_string_list =[]
        # for du1 in du_list:
        #     for dx1 in dx_list: 
        #         for du2 in du_list:
        #             for dx2 in dx_list:
        #                 comp_string_list.append(du1+dx1+du2+dx2)

        shape = [NCL2,NCL1]
        tensor_4_index = [[Phy_string]*81,comp_string_list]

        flow_AVGDF = cd.datastruct(flow_AVG,index=flow_index,copy=False) 
        PU_vectorDF = cd.datastruct(PU_vector,index=vector_index,copy=False) 
        UU_tensorDF = cd.datastruct(UU_tensor,index=sym_2_tensor_index,copy=False) 
        UUU_tensorDF = cd.datastruct(UUU_tensor,index=sym_3_tensor_index,copy=False) 
        Velo_grad_tensorDF = cd.datastruct(Velo_grad_tensor,index=tensor_2_index,copy=False) 
        PR_Velo_grad_tensorDF = cd.datastruct(Pr_Velo_grad_tensor,index=tensor_2_index,copy=False) 
        DUDX2_tensorDF = cd.datastruct(DUDX2_tensor,index=tensor_4_index,copy=False) 

        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]
    
    def _extract_file(self,PhyTime,path_to_folder,abs_path):
        instant = "%0.9E" % PhyTime
        
        file_string = "DNS_perioz_AVERAGD_T" + instant + "_FLOW.D"

        file_folder = "2_averagd_D"
        if not abs_path:
            file_path = os.path.abspath(os.path.join(path_to_folder, \
                                     file_folder, file_string))
        else:
            file_path = os.path.join(path_to_folder, \
                                     file_folder, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)
        int_info = np.fromfile(file,dtype='int32',count=4)    
        
        NCL1 = int_info[0]
        NCL2 = int_info[1]
        NSTATIS = int_info[3]
        
        dummy_size = NCL1*NCL2*50*21
        r_info = np.fromfile(file,dtype='float64',count=3)
        
        PhyTime = r_info[0]
        #REN = r_info[1]
        #DT = r_info[2]
        AVG_info = np.zeros(dummy_size)
        AVG_info = np.fromfile(file,dtype='float64',count=dummy_size)

        file.close()
        return AVG_info, NSTATIS, PhyTime, [NCL1,NCL2]

    def save_hdf(self,file_name,write_mode,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_io'
        super().save_hdf(file_name,write_mode,group_name)

    # def check_times(self,PhyTime):
    #     if isinstance(PhyTime,float) or isinstance(PhyTime,int):
    #         PhyTime = "%.9g"%PhyTime
    #     elif not isinstance(PhyTime,str):
    #         raise TypeError("PhyTime is the wrong type")

    #     if len(self.get_times()) == 1:
    #         avg_time = self.get_times()[0]
    #         if PhyTime != avg_time:
    #             warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
    #         PhyTime = avg_time
    #     else:
    #         if not PhyTime in self.get_times():
    #             raise ValueError("The time given is not in this present in class")
    #     return PhyTime

    def _return_index(self,x_val):
        return CT.coord_index_calc(self.CoordDF,'x',x_val)

    def _return_xaxis(self):
        return self.CoordDF['x']

    def int_thickness_calc(self, PhyTime=''):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        return super()._int_thickness_calc(PhyTime)

    def wall_unit_calc(self,PhyTime=''):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        return self._wall_unit_calc(PhyTime)

    def plot_shape_factor(self, PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = self._plot_shape_factor(PhyTime,fig=fig,ax=ax,**kwargs)
        x_coords = self.CoordDF['x']
        line = ax.get_lines()[-1]
        line.set_xdata(x_coords)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_mom_thickness(self, PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        fig, ax = super().plot_mom_thickness(PhyTime,fig=fig,ax=ax,**kwargs)
        x_coords = self.CoordDF['x']
        line = ax.get_lines()[-1]
        line.set_xdata(x_coords)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_disp_thickness(self, PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_disp_thickness(PhyTime,fig=fig,ax=ax,**kwargs)
        x_coords = self.CoordDF['x']
        line = ax.get_lines()[-1]
        line.set_xdata(x_coords)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_Reynolds(self,comp1,comp2,x_vals,PhyTime='',norm=None,Y_plus=True,fig='',ax='',**kwargs):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        

        fig, ax = super().plot_Reynolds(comp1,comp2,x_vals,PhyTime,
                                        norm=norm,Y_plus=Y_plus,
                                        fig=fig,ax=ax,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        for line,x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$"%float(x))

        axes_items_num = len(lines)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.relim()
        ax.autoscale_view()

        return fig, ax        

    def plot_Reynolds_x(self,comp1,comp2,y_vals_list,Y_plus=True,PhyTime='',fig='',ax='',**kwargs):
        
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_Reynolds_x(comp1,comp2,y_vals_list,Y_plus=True,
                                            PhyTime=PhyTime,fig=fig,ax=ax,**kwargs)
        
        line_no = 1 if y_vals_list == 'max' else len(y_vals_list)
        lines = ax.get_lines()[-line_no:]
        x_coord = self.CoordDF['x']
        for line in lines:
            line.set_xdata(x_coord)
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax
    def bulk_velo_calc(self,PhyTime=''):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        return self._bulk_velo_calc(PhyTime)

    def plot_bulk_velocity(self,PhyTime='',fig='',ax='',**kwargs):
        
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_bulk_velocity(PhyTime,fig,ax,**kwargs)
        line = ax.get_lines()[-1]
        x_coord = self.CoordDF['x']
        line.set_xdata(np.array(x_coord))
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def tau_calc(self,PhyTime=''):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
            
        return self._tau_calc(PhyTime)

    def plot_skin_friction(self,PhyTime='',fig='',ax='',**kwargs):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        

        fig, ax = super().plot_skin_friction(PhyTime,fig,ax,**kwargs)
        line = ax.get_lines()[-1]
        x_coord = self.CoordDF['x']
        line.set_xdata(np.array(x_coord))
        ax.set_xlabel(r"$x^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self,x_vals,PhyTime='',Y_plus=True,Y_plus_max=100,fig='',ax='',**kwargs):
        
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"
        
        fig, ax = super().plot_eddy_visc(x_vals,PhyTime,Y_plus,Y_plus_max,fig,ax,**kwargs)
        lines = ax.get_lines()[-len(x_vals):]
        for line, x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$" % float(x))

        axes_items_num = len(x_vals)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def avg_line_plot(self,x_vals,comp,PhyTime='',fig='',ax='',*args,**kwargs):

        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super().avg_line_plot(x_vals,PhyTime,comp,fig='',ax='',**kwargs)
        line_no=len(x_vals)
        lines = ax.get_lines()[-line_no:]
        for line, x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$"% x)

        axes_items_num = len(x_vals)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

    def plot_near_wall(self,x_vals,PhyTime='',fig='',ax='',**kwargs):
        if len(set([x[0] for x in self.UU_tensorDF.index])) == 1:
            avg_time = list(set([x[0] for x in self.UU_tensorDF.index]))[0]
            if PhyTime and PhyTime != avg_time:
                warnings.warn("\033[1;33PhyTime being set to variable present (%g) in CHAPSim_AVG class" %float(avg_time), stacklevel=2)
            PhyTime = avg_time
        else:
            assert PhyTime in set([x[0] for x in self.UU_tensorDF.index]), "PhyTime must be present in CHAPSim_AVG class"

        fig, ax = super().plot_near_wall(x_vals,PhyTime,fig=fig,ax=ax,**kwargs)
        line_no=len(x_vals)
        lines = ax.get_lines()[-line_no-1:-1]
        for line, x in zip(lines,x_vals):
            line.set_label(r"$x^*=%.3g$"% x)

        axes_items_num = len(x_vals)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax

class CHAPSim_AVG_tg_base(CHAPSim_AVG_base):
    module = sys.modules[__name__]
    tgpost = True
    def _extract_avg(self,PhyTimes,*args,meta_data='',path_to_folder='',time0='',abs_path=True,permissive=False):

        if isinstance(path_to_folder,list):
            folder_path=path_to_folder[0]
            if not permissive:
                for path in path_to_folder:
                    PhyTimes = list(set(PhyTimes).intersection(CT.time_extract(path,abs_path)))
        else:
            folder_path=path_to_folder
            PhyTimes = list(set(PhyTimes).intersection(CT.time_extract(path_to_folder,abs_path)))
        PhyTimes=sorted(PhyTimes)
        if cp.TEST:
            PhyTimes=PhyTimes[-3:]

        if not meta_data:
            meta_data = self._module._meta_class(folder_path,abs_path,tgpost=True)
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL = meta_data.NCL

        if isinstance(PhyTimes,float):
            times = ['%.9g' % PhyTimes]
            DF_list = self._AVG_extract(PhyTimes,folder_path,abs_path,metaDF,time0)
        elif hasattr(PhyTimes,'__iter__'):
            times = ['%.9g' % time for time in PhyTimes]
            for PhyTime in PhyTimes:
                if 'DF_list' not in locals():
                    DF_list = self._AVG_extract(PhyTime,folder_path,abs_path,metaDF,time0)
                else:
                    DF_temp=[]
                    local_DF_list = self._AVG_extract(PhyTime,folder_path,abs_path,metaDF,time0)
                    for i, DF in enumerate(DF_list):
                        DF_list[i].append(local_DF_list[i],axis=0)
                    # DF_list=DF_temp
            

        else:
            raise TypeError("\033[1;32 `PhyTimes' can only be a float or a list") 
        
        old_shape =[NCL[1],len(PhyTimes)]
        for i,DF in enumerate(DF_list):
            for key, val in DF_list[i]:
                DF_list[i][key] = DF_list[i][key].T
            # DF.columns = range(flat_shape)
            # for index, row in DF.iterrows():
            #     DF.loc[index] = pd.Series(row.values.reshape(old_shape).T.reshape(flat_shape))
            # DF.data(old_shape[::-1])

        DF_list=self._Reverse_decomp(*DF_list)

        return_list = [meta_data, CoordDF, metaDF, NCL,old_shape,times,*DF_list]
        if isinstance(path_to_folder,list):
            i=2
            for path in path_to_folder[1:]:                
                AVG_list = list(CHAPSim_AVG_tg_base._extract_avg(self,PhyTimes,meta_data=meta_data,
                                path_to_folder=path,time0=time0,abs_path=abs_path,permissive=False))
                return_list = self._ensemble_average(return_list,AVG_list,i,permissive)
                i+=1

        # self._Reverse_decomp(*DF_list)
        return itertools.chain(return_list)

    def _ensemble_average(self,return_list, AVG_list,number,permissive=False):
        if not permissive:
            coe2 = (number-1)/number ; coe3 = 1/number
            assert return_list[1].equals(AVG_list[1]), "Coordinates are not the same"
            assert return_list[3] == AVG_list[3], "Mesh is not the same"
            assert return_list[5] == AVG_list[5], "Times must be same for non permissive ensemble averaging"
            for i in range(6,13):
                index = return_list[i].index
                return_list[i] = coe2*return_list[i] + coe3*AVG_list[i]
                # return_list[i] = cd.datastruct(array,index = index).data(AVG_list[i].data.FrameShape)
        else:
            raise NotImplementedError

        return return_list

    def Perform_ensemble(self):
        raise NotImplementedError
    def _extract_file(self,PhyTime,path_to_folder,abs_path):
        instant = "%0.9E" % PhyTime
        
        file_string = "DNS_perixz_AVERAGD_T" + instant + "_FLOW.D"
        
        file_folder = "2_averagd_D"
        if not abs_path:
            file_path = os.path.abspath(os.path.join(path_to_folder, \
                                        file_folder, file_string))
        else:
            file_path = os.path.join(path_to_folder, \
                                        file_folder, file_string)
                
        file = open(file_path,'rb')
        
        int_info = np.zeros(4)
        r_info = np.zeros(3)
        int_info = np.fromfile(file,dtype='int32',count=4)    
        
        NCL2 = int_info[0]
        NSZ = int_info[1]
        ITERG = int_info[2]
        NSTATIS = int_info[3]
        dummy_size = NCL2*NSZ
        r_info = np.fromfile(file,dtype='float64',count=3)
        
        PhyTime = r_info[0]
        AVG_info = np.zeros(dummy_size)
        AVG_info = np.fromfile(file,dtype='float64',count=dummy_size)

        AVG_info = AVG_info.reshape(NSZ,NCL2)

        file.close()
        
        return AVG_info, NSTATIS
    def _AVG_extract(self,PhyTime,path_to_folder,abs_path,metaDF,time0):

        factor = metaDF['NCL1_tg_io'][0]*metaDF['NCL3'] if self._module.dissipation_correction else 1.0
        AVG_info, NSTATIS1 = self._extract_file(PhyTime,path_to_folder,abs_path)
        ioflowflg = True if metaDF['NCL1_tg_io'][1]>2 else False
        if ioflowflg and time0:
            AVG_info0, NSTATIS0 = self._extract_file(time0,path_to_folder,abs_path)
            AVG_info = (AVG_info*NSTATIS1 - AVG_info0*NSTATIS0)/(NSTATIS1-NSTATIS0)
        # print(AVG_info[51])
        flow_AVG = AVG_info[:4]
        PU_vector = AVG_info[4:7]
        UU_tensor = AVG_info[7:13]
        UUU_tensor = AVG_info[13:23]
        Velo_grad_tensor = AVG_info[23:32]
        Pr_Velo_grad_tensor = AVG_info[32:41]
        DUDX2_tensor = AVG_info[41:]*factor

        Phy_string = '%.9g' % PhyTime
        flow_index = [[None]*4,['u','v','w','P']]
        vector_index = [[None]*3,['u','v','w']]
        sym_2_tensor_index = [[None]*6,['uu','uv','uw','vv','vw','ww']]
        sym_3_tensor_index = [[None]*10,['uuu','uuv','uuw','uvv',\
                                'uvw','uww','vvv','vvw','vww','www']]
        tensor_2_index = [[None]*9,['ux','uy','uz','vx','vy','vz',\
                                         'wx','wy','wz']]
        du_list = ['du','dv','dw']
        dx_list = ['dx','dy','dz']

        dudx_list = list(itertools.product(du_list,dx_list))
        dudx_list = ["".join(dudx) for dudx in dudx_list]
        comp_string_list = list(itertools.product(dudx_list,dudx_list))
        comp_string_list = ["".join(comp_string) for comp_string in comp_string_list]
        
        tensor_4_index=[[None]*81,comp_string_list]

        flow_AVGDF = cd.datastruct(flow_AVG,index=flow_index)
        PU_vectorDF = cd.datastruct(PU_vector,index=vector_index)
        UU_tensorDF = cd.datastruct(UU_tensor,index=sym_2_tensor_index)
        UUU_tensorDF = cd.datastruct(UUU_tensor,index=sym_3_tensor_index)
        Velo_grad_tensorDF = cd.datastruct(Velo_grad_tensor,index=tensor_2_index)
        PR_Velo_grad_tensorDF = cd.datastruct(Pr_Velo_grad_tensor,index=tensor_2_index)
        DUDX2_tensorDF = cd.datastruct(DUDX2_tensor,index=tensor_4_index)

        return [flow_AVGDF, PU_vectorDF, UU_tensorDF, UUU_tensorDF,\
                    Velo_grad_tensorDF, PR_Velo_grad_tensorDF,DUDX2_tensorDF]
    
    def _hdf_extract(self,file_name,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_tg'
        base_name = group_name
        
        hdf_file = h5py.File(file_name,'r')
        shape = tuple(hdf_file[base_name].attrs["shape"][:])
        times = list(np.char.decode(hdf_file[base_name].attrs["times"][:]))
        hdf_file.close()

        parent_list = list(super()._hdf_extract(file_name,shape,group_name=base_name))
        meta_data = self._module._meta_class.from_hdf(file_name,base_name+'/meta_data')
        CoordDF = meta_data.CoordDF
        metaDF = meta_data.metaDF
        NCL=meta_data.NCL

        # for DF in parent_list:
        #     DF.data(shape)

        return_list = [meta_data,CoordDF,metaDF,NCL,shape,times]

        return itertools.chain(return_list + parent_list)

    def save_hdf(self,file_name,write_mode,group_name=''):
        if not group_name:
            group_name = 'CHAPSim_AVG_tg'
        super().save_hdf(file_name,write_mode,group_name)
        base_name=group_name if group_name else 'CHAPSim_AVG'

        hdf_file = h5py.File(file_name,'a')
        group = hdf_file[base_name]
        group.attrs['shape'] = np.array(self.shape)
        group.attrs['times'] = np.array([np.string_(x) for x in self.get_times()])
        hdf_file.close()


    def _return_index(self,PhyTime):
        if not isinstance(PhyTime,str):
            PhyTime = "{:.9g}".format(PhyTime)

        if PhyTime not in self.get_times():
            raise ValueError("time %s must be in times"% PhyTime)
        for i in range(len(self.get_times())):
            if PhyTime==self.get_times()[i]:
                return i

    def _return_xaxis(self):
        return np.array([float(time) for time in self.get_times()])

    def wall_unit_calc(self):
        return self._wall_unit_calc(None)

    def int_thickness_calc(self):
        PhyTime = None
        return super()._int_thickness_calc(PhyTime)

    def plot_shape_factor(self,fig='',ax='',**kwargs):
        PhyTime = None
        fig, ax = self._plot_shape_factor(PhyTime,fig=fig,ax=ax,**kwargs)
        times = np.array([float(x) for x in self.get_times()])
        line=ax.get_lines()[-1]
        line.set_xdata(times)
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_Reynolds(self,comp1,comp2,PhyTime,norm=None,Y_plus=True,fig='',ax='',**kwargs):

        fig, ax = super().plot_Reynolds(comp1,comp2,PhyTime,None,
                                        norm=norm,Y_plus=Y_plus,
                                        fig=fig,ax=ax,**kwargs)
        lines = ax.get_lines()[-len(PhyTime):]
        for line,time in zip(lines,PhyTime):
            line.set_label(r"$t^*=%.3g$"%float(time))
        axes_items_num = len(lines)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.relim()
        ax.autoscale_view()

        fig.tight_layout()
        return fig, ax

    def plot_Reynolds_x(self,comp1,comp2,y_vals_list,Y_plus=True,fig='',ax='',**kwargs):
        fig, ax = super().plot_Reynolds_x(comp1,comp2,y_vals_list,Y_plus=True,
                                            PhyTime=None,fig=fig,ax=ax,**kwargs)
        
        line_no = 1 if y_vals_list == 'max' else len(y_vals_list)
        lines = ax.get_lines()[-line_no:]
        times = np.array([float(x) for x in self.get_times()])
        for line in lines:
            line.set_xdata(times)
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        ax.get_gridspec().tight_layout(fig)
        return fig, ax    

    def bulk_velo_calc(self):
        return super()._bulk_velo_calc(None)

    def plot_bulk_velocity(self,fig='',ax='',**kwargs):
        fig, ax = super().plot_bulk_velocity(None,fig,ax,**kwargs)
        line = ax.get_lines()[-1]
        times = [float(x) for x in self.get_times()]
        line.set_xdata(np.array(times))
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def tau_calc(self):
        return self._tau_calc(None)

    def plot_skin_friction(self,fig='',ax='',**kwargs):
        fig, ax = super().plot_skin_friction(None,fig=fig,ax=ax,**kwargs)
        line = ax.get_lines()[-1]
        times = [float(x) for x in self.get_times()]
        line.set_xdata(np.array(times))
        ax.set_xlabel(r"$t^*$")
        ax.relim()
        ax.autoscale_view()
        return fig, ax

    def plot_eddy_visc(self,times,Y_plus=True,Y_plus_max=100,fig='',ax='',**kwargs):
        fig, ax = super().plot_eddy_visc(times,None,Y_plus,Y_plus_max,fig,ax,**kwargs)
        lines = ax.get_lines()[-len(times):]
        try:
            for line, time in zip(lines,times):
                line.set_label(r"$t^*=%.3g$" % float(time))
        except TypeError:
            lines.set_label(r"$t^*=%.3g$" % float(times))

        axes_items_num = len(times)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)

        return fig, ax

    def avg_line_plot(self,times,*args,fig='',ax='',**kwargs):
        
        fig, ax = super().avg_line_plot(times,None,*args,fig=fig,ax=ax,**kwargs)

        lines = ax.get_lines()[-len(times):]
        for line, time in zip(lines,times):
            if not isinstance(time,float):
                time = float(time)
            line.set_label(r"$t^*=%g$"% time)
            
        axes_items_num = len(times)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        
        return fig, ax

    def plot_near_wall(self,times,fig='',ax='',**kwargs):
        fig, ax = super().plot_near_wall(times,None,fig=fig,ax=ax,**kwargs)
        line_no=len(times)
        lines = ax.get_lines()[-line_no:]
        for line, time in zip(lines,times):
            line.set_label(r"$t^*=%.3g$"% time)

        axes_items_num = len(times)
        ncol = 4 if axes_items_num>3 else axes_items_num
        ax.clegend(vertical=False,ncol=ncol)
        ax.get_gridspec().tight_layout(fig)
        return fig, ax
        
class CHAPSim_AVG_tg(CHAPSim_AVG_tg_base):
    def _extract_avg(self,path_to_folder='',time0='',abs_path=True,*args,**kwargs):
        if isinstance(path_to_folder,list):
            times = CT.time_extract(path_to_folder[0],abs_path)
        else:
            times = CT.time_extract(path_to_folder,abs_path)
        if time0:
            times = list(filter(lambda x: x > time0, times))
        return super()._extract_avg(times,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path,*args,**kwargs)

class CHAPSim_AVG():
    def __new__(cls,*args,tgpost=False,**kwargs):
        if tgpost:
            return CHAPSim_AVG_tg(*args,**kwargs)
        else:
            return CHAPSim_AVG_io(*args,**kwargs)

    @classmethod
    def from_hdf(cls,*args,tgpost=False,**kwargs):
        if tgpost:
            return cls(tgpost=tgpost,fromfile=True,*args,**kwargs)
        else:
            return cls(fromfile=True,*args,**kwargs)
