"""
# _quadrant_a.py
Module for quadrant analysis from
instantaneous results from CHAPSim DNS solver
"""

import numpy as np

import sys
import warnings
import time
import gc
import itertools
from abc import ABC, abstractmethod

import CHAPSim_post as cp
from CHAPSim_post.utils import docstring, gradient, indexing, misc_utils

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd

from ._average import CHAPSim_AVG_io, CHAPSim_AVG_tg, CHAPSim_AVG_temp
_avg_io_class = CHAPSim_AVG_io
_avg_tg_class = CHAPSim_AVG_tg
_avg_temp_class = CHAPSim_AVG_temp


from ._fluct import CHAPSim_fluct_io, CHAPSim_fluct_tg, CHAPSim_fluct_temp
_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg
_fluct_temp_class = CHAPSim_fluct_temp

from ._meta import CHAPSim_meta
_meta_class = CHAPSim_meta

from ._common import Common

class _Quad_Anl_base(Common,ABC):
    def __init__(self,*args,from_hdf=False,**kwargs):
        if not from_hdf:
            self._quad_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)

    @property
    def shape(self):
        return self._avg_data.shape

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        self._meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',key+'/avg_data')
        self.QuadAnalDF.to_hdf(file_name,key=key+'/QuadAnalDF',mode='a')
        self.QuadNumDF.to_hdf(file_name,key=key+'/QuadNumDF',mode='a')
        self.QuadDTDF.to_hdf(file_name,key=key+'/QuadDTDF',mode='a')
        self.QuadDurDF.to_hdf(file_name,key=key+'/QuadDurDF',mode='a')

    def _comp_calc(self,h,quadrant):
        return "h=%g Q%d"%(h,quadrant)

    def _check_quadrant(self,Quadrants):
        if Quadrants is None:
            Quadrants = [1,2,3,4]

        else:
            Quadrants = misc_utils.check_list_vals(Quadrants)
            if not all(quad in [1,2,3,4] for quad in Quadrants):
                msg = "The quadrants provided must be in 1 2 3 4"
                raise ValueError(msg)
        return Quadrants

    @abstractmethod
    def _quad_extract(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def _hdf_extract(self,*args,**kwargs):
        raise NotImplementedError

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(from_hdf=True,*args,**kwargs)

    def _quadrant_extract(self,fluctDF,PhyTime):
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)

        u_array=fluctDF[PhyTime,'u']
        v_array=fluctDF[PhyTime,'v']

        u_array_isneg=u_array<0
        v_array_isneg=v_array<0

        quadrant_array = np.zeros_like(v_array_isneg,dtype='i4')

        for i in range(1,5): #determining quadrant
            if i ==1:
                quadrant_array_temp = np.logical_and(~u_array_isneg,~v_array_isneg)#not fluct_u_isneg and not fluct_v_isneg
                quadrant_array += quadrant_array_temp*1
            elif i==2:
                quadrant_array_temp = np.logical_and(u_array_isneg,~v_array_isneg)#not fluct_u_isneg and fluct_v_isneg
                quadrant_array += quadrant_array_temp*2
            elif i==3:
                quadrant_array_temp =  np.logical_and(u_array_isneg,v_array_isneg)
                quadrant_array += quadrant_array_temp*3
            elif i==4:
                quadrant_array_temp =  np.logical_and(~u_array_isneg,v_array_isneg)#fluct_u_isneg and not fluct_v_isneg
                quadrant_array += quadrant_array_temp*4

        assert(quadrant_array.all()<=4 and quadrant_array.all()>=1)  
        fluct_uv = np.ma.array(u_array*v_array) 

        return fluct_uv, quadrant_array 

    def _create_symmetry(self,h_list,quad_anal_array,num_array,mean_dt, mean_dur):
        if not cp.rcParams['SymmetryAVG'] or  not self.metaDF['iCase'] ==1:
            return

        old_quad_array = quad_anal_array.copy()
        old_num_array = num_array.copy()
        old_dt_array = mean_dt.copy()
        old_dur_array = mean_dur.copy()

        middle_index = old_dur_array.shape[1] // 2
        symmetry_map = {1: 4,2:3,4:1,3:2}

        for j,h in enumerate(h_list):
            for i in range(4):
                i_symm = symmetry_map[i+1]-1
                quad_anal_array[4*j+i,:middle_index] = 0.5*(old_quad_array[4*j+i,:middle_index] - old_quad_array[4*j+i_symm,-middle_index:][::-1])
                quad_anal_array[4*j+i,-middle_index:] = quad_anal_array[4*j+i,:middle_index][::-1]

                num_array[4*j+i,:middle_index] = 0.5*(old_num_array[4*j+i,:middle_index] + old_num_array[4*j+i_symm,-middle_index:][::-1])
                num_array[4*j+i,-middle_index:] = num_array[4*j+i,:middle_index][::-1]
                
                mean_dt[4*j+i,:middle_index] = 0.5*(old_dt_array[4*j+i,:middle_index] + old_dt_array[4*j+i_symm,-middle_index:][::-1])
                mean_dt[4*j+i,-middle_index:] = mean_dt[4*j+i,:middle_index][::-1]
                
                mean_dur[4*j+i,:middle_index] = 0.5*(old_dur_array[4*j+i,:middle_index] + old_dur_array[4*j+i_symm,-middle_index:][::-1])
                mean_dur[4*j+i,-middle_index:] = mean_dur[4*j+i,:middle_index][::-1]


class CHAPSim_Quad_Anl_io(_Quad_Anl_base):
    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__#'CHAPSim_Quad_Anal'

        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)

        
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module.CHAPSim_meta.from_hdf(file_name,key+'/meta_data')
        self._avg_data = self._module._avg_io_class.from_hdf(file_name,key+'/avg_data')
        
        self.QuadAnalDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/QuadAnalDF')
        self.QuadNumDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/QuadNumDF')
        self.QuadDTDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/QuadDTDF')
        self.QuadDurDF = cd.FlowStruct2D.from_hdf(file_name,key=key+'/QuadDurDF')

    def _quad_extract(self,h_list,path_to_folder='.',time0=None,abs_path=True):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        
        if cp.rcParams['TEST']:
            times.sort(); times= times[-3:]

        self._meta_data = self._module._meta_class(path_to_folder,abs_path)
        self._avg_data = self._module._avg_io_class(max(times),  path_to_folder,
                                                     time0, abs_path)

        dt_array = np.array([np.diff(times)[0],*np.diff(times)])

        

        for i, (timing, dt) in enumerate(zip(times,dt_array)):
            time1 = time.time()
            fluct_data = self._module._fluct_io_class(timing, self._avg_data,
                                                        time0=time0, 
                                                        path_to_folder=path_to_folder,
                                                        abs_path=abs_path)
            time2 = time.time()
            fluct_uv, quadrant_array = self._quadrant_extract(fluct_data.fluctDF,
                                                                timing)
            
            
            coe3 = i/(i + 1)
            coe2 = 1/(i + 1)

            if i ==0:
                self.num_array = np.zeros((len(h_list)*4, *fluct_uv.shape),dtype=np.int32)
                self.prev_array = np.full((len(h_list)*4, *fluct_uv.shape),False)
                self.total_event_times = np.zeros_like(self.num_array,dtype=np.float64)

                quad_anal_array  = self._quad_calc(dt,fluct_uv,quadrant_array,h_list)
            else:
                local_quad_anal_array = self._quad_calc(dt,fluct_uv,quadrant_array,h_list)

                quad_anal_array = quad_anal_array*coe3 + local_quad_anal_array*coe2

            gc.collect()
            print("Time %g: %.3g %.3g"%(timing,time2 - time1, time.time()- time2))

        del self.prev_array; del local_quad_anal_array
        gc.collect()

        total_time = times[-1] - times[0]
        total_num_array = np.mean(self.num_array,axis=1)
        self.num_array[self.num_array==0] = 1

        total_mean_dt = np.mean(total_time/self.num_array,axis=1)
        total_mean_dur = np.mean(self.total_event_times/self.num_array,axis=1)

        del self.num_array; del self.total_event_times

        self._create_symmetry(h_list, 
                            quad_anal_array,
                            total_num_array, 
                            total_mean_dt, 
                            total_mean_dur)
        

        comp = [self._comp_calc(*x) for x in itertools.product(h_list,range(1,5))]
        PhyTime = [None]*len(comp)


        index = list(zip(PhyTime,comp))

        self.QuadAnalDF =cd.FlowStruct2D(self._coorddata,quad_anal_array,index=index)
        self.QuadNumDF =cd.FlowStruct2D(self._coorddata,total_num_array,index=index)
        self.QuadDTDF = cd.FlowStruct2D(self._coorddata,total_mean_dt,index=index)
        self.QuadDurDF = cd.FlowStruct2D(self._coorddata,total_mean_dur,index=index)
    

    def _quad_calc(self,dt,fluct_uv,quadrant_array,h_list):

        avg_time = max(self._avg_data.times)
        uu=self._avg_data.UU_tensorDF[avg_time,'uu']
        vv=self._avg_data.UU_tensorDF[avg_time,'vv']

        u_rms = np.sqrt(uu)
        v_rms = np.sqrt(vv)

        quad_anal_array=np.empty((len(h_list)*4,*self.shape))

        for j,h in enumerate(h_list):
            for i in range(4):
                num_q = self.num_array[4*j+i]
                prev_q = self.prev_array[4*j+i]
                tot_event_q = self.total_event_times[4*j+i]

                no_mask = np.logical_and(quadrant_array == (i+1),
                                         abs(fluct_uv) > h*u_rms*v_rms)
                fluct_uv.mask = ~no_mask
                uv_q = fluct_uv.filled(0.).mean(axis=0)

                new_event = np.logical_and(~prev_q,no_mask)
                num_q += new_event.astype(int) 
                tot_event_q[no_mask] += dt
                prev_q = no_mask.copy()

                quad_anal_array[4*j+i]=uv_q

        return quad_anal_array


    def plot_line(self,h_list,coord_list,prop_dir,Quadrants=None,x_vals=0,y_mode='half_channel',norm=False,fig=None,ax=None,line_kw=None,**kwargs):
        x_vals = misc_utils.check_list_vals(x_vals) 

        if prop_dir == 'x':
            coord_list = self._avg_data.ycoords_from_coords(coord_list,x_vals=x_vals,mode=y_mode)
            if x_vals ==0:
                coord_list = coord_list[0]
            else:
                coord_list = np.diag(coord_list)
            coord_int = self._avg_data.ycoords_from_norm_coords(coord_list,x_vals=x_vals,mode=y_mode)
        else:
            coord_int = coord_list = indexing.true_coords_from_coords(self.CoordDF,'x',coord_list)
            
        if norm:
            time = max(self._avg_data.times)
            norm_Quadrant = self.QuadAnalDF/self._avg_data.UU_tensorDF[time,'uv']
        else:
            norm_Quadrant = self.QuadAnalDF

        Quadrants = self._check_quadrant(Quadrants)

        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5*len(coord_list),12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,len(coord_list),fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num,len(coord_list)))

        symbol = 'x' if prop_dir == 'y' else 'x'
        unit = misc_utils.get_title_symbol(symbol,y_mode,False)
        
        # time = norm_Quadrant.times[0]
        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                for j,coord in enumerate(coord_int):
                    fig, ax[i,j] = norm_Quadrant.plot_line(comp, prop_dir, coord,
                                                            time = None,
                                                            labels=[f'$h = {h}$'],
                                                            channel_half = True,
                                                            fig= fig, ax= ax[i,j],
                                                            line_kw=line_kw)
                    
                    ax[0,j].set_title(r"$%s=%.5g$"%(unit,coord_list[j]),loc='left')
                    x_label = f"${prop_dir}$"
                    ax[-1,j].set_xlabel(x_label)

            ax[i,0].set_ylabel(r"$Q%d$"%quad)
            

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0,0].clegend(vertical=False,ncol=ncol)

        return fig, ax

    def plot_events(self,h_list,coord_list,prop_dir,Quadrants=None,x_vals=0,y_mode='half_channel',fig=None,ax=None,line_kw=None,**kwargs):
        x_vals = misc_utils.check_list_vals(x_vals) 

        if prop_dir == 'x':
            coord_list = self._avg_data.ycoords_from_coords(coord_list,x_vals=x_vals,mode=y_mode)
            coord_int = self._avg_data.ycoords_from_norm_coords(coord_list,x_vals=x_vals,mode=y_mode)
        else:
            coord_int = coord_list = indexing.true_coords_from_coords(self.CoordDF,'x',coord_list)
            

        Quadrants = self._check_quadrant(Quadrants)

        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5*len(coord_list),12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,len(coord_list),fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num,len(coord_list)))

        symbol = 'x' if prop_dir == 'y' else 'x'
        unit = misc_utils.get_title_symbol(symbol,y_mode,False)

        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                for j,coord in enumerate(coord_int):
                    fig, ax[i,j] = self.QuadNumDF.plot_line(comp, prop_dir, coord,
                                                            time = None,
                                                            labels=[f'$h = {h}$'],
                                                            channel_half = True,
                                                            fig= fig, ax= ax[i,j],
                                                            line_kw=line_kw)
                x_label = f"${prop_dir}$"

                ax[i,j].set_xlabel(x_label)
                ax[i,j].set_ylabel(r"$Q%d$(Number of events)"%quad)
                ax[i,j].set_title(r"$%s=%.5g$"%(unit,coord_list[j]),loc='left')

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0,0].clegend(vertical=False,ncol=ncol)

        return fig, ax

    def plot_event_interval(self,h_list,coord_list,prop_dir,Quadrants=None,x_vals=0,y_mode='half_channel',fig=None,ax=None,line_kw=None,**kwargs):
        x_vals = misc_utils.check_list_vals(x_vals) 

        if prop_dir == 'x':
            coord_list = self._avg_data.ycoords_from_coords(coord_list,x_vals=x_vals,mode=y_mode)
            coord_int = self._avg_data.ycoords_from_norm_coords(coord_list,x_vals=x_vals,mode=y_mode)
        else:
            coord_int = coord_list = indexing.true_coords_from_coords(self.CoordDF,'x',coord_list)
            

        Quadrants = self._check_quadrant(Quadrants)

        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5*len(coord_list),12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,len(coord_list),fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num,len(coord_list)))

        symbol = 'x' if prop_dir == 'y' else 'x'
        unit = misc_utils.get_title_symbol(symbol,y_mode,False)

        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                for j,coord in enumerate(coord_int):
                    fig, ax[i,j] = self.QuadDTDF.plot_line(comp, prop_dir, coord,
                                                            time = None,
                                                            labels=[f'$h = {h}$'],
                                                            channel_half = True,
                                                            fig= fig, ax= ax[i,j],
                                                            line_kw=line_kw)
                x_label = f"${prop_dir}$"

                ax[i,j].set_xlabel(x_label)
                ax[i,j].set_ylabel(r"$T_{Q%d}$"%quad)
                ax[i,j].set_title(r"$%s=%.5g$"%(unit,coord_list[j]),loc='left')

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0,0].clegend(vertical=False,ncol=ncol)

        return fig, ax

    
    def plot_event_duration(self,h_list,coord_list,prop_dir,Quadrants=None,x_vals=0,y_mode='half_channel',fig=None,ax=None,line_kw=None,**kwargs):
        x_vals = misc_utils.check_list_vals(x_vals) 

        if prop_dir == 'x':
            coord_list = self._avg_data.ycoords_from_coords(coord_list,x_vals=x_vals,mode=y_mode)
            coord_int = self._avg_data.ycoords_from_norm_coords(coord_list,x_vals=x_vals,mode=y_mode)
        else:
            coord_int = coord_list = indexing.true_coords_from_coords(self.CoordDF,'x',coord_list)
            

        Quadrants = self._check_quadrant(Quadrants)

        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5*len(coord_list),12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,len(coord_list),fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num,len(coord_list)))

        symbol = 'x' if prop_dir == 'y' else 'x'
        unit = misc_utils.get_title_symbol(symbol,y_mode,False)

        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                for j,coord in enumerate(coord_int):
                    fig, ax[i,j] = self.QuadDurDF.plot_line(comp, prop_dir, coord,
                                                            time = None,
                                                            labels=[f'$h = {h}$'],
                                                            channel_half = True,
                                                            fig= fig, ax= ax[i,j],
                                                            line_kw=line_kw)
                x_label = f"${prop_dir}$"

                ax[i,j].set_xlabel(x_label)
                ax[i,j].set_ylabel(r"$\Delta T_{Q%d}$"%quad)
                ax[i,j].set_title(r"$%s=%.5g$"%(unit,coord_list[j]),loc='left')

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0,0].clegend(vertical=False,ncol=ncol)

        return fig, ax

class CHAPSim_Quad_Anl_tg(_Quad_Anl_base):
    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__#'CHAPSim_Quad_Anal'

        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)

        
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module.CHAPSim_meta.from_hdf(file_name,key+'/meta_data')
        self._avg_data = self._module._avg_tg_class.from_hdf(file_name,key+'/avg_data')
        
        self.QuadAnalDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/QuadAnalDF')
        self.QuadNumDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/QuadNumDF')
        self.QuadDTDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/QuadDTDF')
        self.QuadDurDF = cd.FlowStruct1D.from_hdf(file_name,key=key+'/QuadDurDF')

    def _quad_extract(self,h_list,path_to_folder='.',ntimes=None,time0=None,abs_path=True):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        
        if ntimes is not None:
            times = times[-ntimes:]

        if cp.rcParams['TEST']:
            times.sort(); times= times[-3:]

        self._meta_data = self._module._meta_class(path_to_folder,abs_path)
        self._avg_data = self._module._avg_tg_class(max(times), 
                                                    path_to_folder, 
                                                    time0, abs_path)

        dt_array = np.array([np.diff(times)[0],*np.diff(times)])

        

        for i, (timing, dt) in enumerate(zip(times,dt_array)):
            time1 = time.time()
            fluct_data = self._module._fluct_tg_class(timing, self._avg_data,
                                                        time0=time0, 
                                                        path_to_folder=path_to_folder,
                                                        abs_path=abs_path)
            time2 = time.time()

            fluct_uv, quadrant_array = self._quadrant_extract(fluct_data.fluctDF,
                                                                timing)
            
            
            coe3 = i/(i + 1)
            coe2 = 1/(i + 1)

            if i ==0:
                self.num_array = np.zeros((len(h_list)*4, *fluct_uv.shape),dtype=np.int32)
                self.prev_array = np.full((len(h_list)*4, *fluct_uv.shape),False)
                self.total_event_times = np.zeros_like(self.num_array,dtype=np.float64)

                quad_anal_array  = self._quad_calc(dt,fluct_uv,quadrant_array,h_list)
            else:
                local_quad_anal_array = self._quad_calc(dt,fluct_uv,quadrant_array,h_list)

                quad_anal_array = quad_anal_array*coe3 + local_quad_anal_array*coe2

            gc.collect()
            print("Time %g: %.3g %.3g"%(timing,time2 - time1, time.time()- time2))
        del self.prev_array; del local_quad_anal_array
        gc.collect()

        total_time = times[-1] - times[0]
        total_num_array = np.mean(self.num_array,axis=(1,3))
        self.num_array[self.num_array==0] = 1

        total_mean_dt = np.mean(total_time/self.num_array,axis=(1,3))
        total_mean_dur = np.mean(self.total_event_times/self.num_array,axis=(1,3))

        del self.num_array; del self.total_event_times

        self._create_symmetry(h_list, 
                            quad_anal_array,
                            total_num_array, 
                            total_mean_dt, 
                            total_mean_dur)
        
        comp = [self._comp_calc(*x) for x in itertools.product(h_list,range(1,5))]
        PhyTime = [None]*len(comp)


        index = list(zip(PhyTime,comp))

        self.QuadAnalDF =cd.FlowStruct1D(self._coorddata,quad_anal_array,index=index)
        self.QuadNumDF =cd.FlowStruct1D(self._coorddata,total_num_array,index=index)
        self.QuadDTDF = cd.FlowStruct1D(self._coorddata,total_mean_dt,index=index)
        self.QuadDurDF = cd.FlowStruct1D(self._coorddata,total_mean_dur,index=index)

    def _quad_calc(self,dt,fluct_uv,quadrant_array,h_list):

        avg_time = max(self._avg_data.times)
        uu=self._avg_data.UU_tensorDF[avg_time,'uu']
        vv=self._avg_data.UU_tensorDF[avg_time,'vv']

        u_rms = np.sqrt(uu)
        v_rms = np.sqrt(vv)

        quad_anal_array=np.empty((len(h_list)*4,*self.shape))

        for j,h in enumerate(h_list):
            for i in range(4):
                num_q = self.num_array[4*j+i]
                prev_q = self.prev_array[4*j+i]
                tot_event_q = self.total_event_times[4*j+i]

                no_mask = np.logical_and(quadrant_array == (i+1),
                                     abs(fluct_uv) > h*(u_rms*v_rms)[:,np.newaxis])
                fluct_uv.mask = ~no_mask
                uv_q = fluct_uv.filled(0.).mean(axis=(0,2))

                new_event = np.logical_and(~prev_q,no_mask)
                num_q += new_event.astype(int) 
                tot_event_q[no_mask] += dt
                prev_q = no_mask.copy()

                quad_anal_array[4*j+i]=uv_q

        return quad_anal_array

    def plot_line(self,h_list,Quadrants=None,norm=False,fig=None,ax=None,line_kw=None,**kwargs):
                
        if norm:
            time = max(self._avg_data.times)
            norm_Quadrant = self.QuadAnalDF/self._avg_data.UU_tensorDF[time,'uv']
        else:
            norm_Quadrant = self.QuadAnalDF

        Quadrants = self._check_quadrant(Quadrants)
        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5,12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,fig=fig,ax=ax,**kwargs)

        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                fig, ax[i] = norm_Quadrant.plot_line(comp,
                                                    time = None,
                                                    label=f'$h = {h}$',
                                                    channel_half = True,
                                                    fig= fig, ax= ax[i],
                                                    line_kw=line_kw)
            ax[i].set_ylabel(r"$Q%d$"%quad)

        x_label = self.Domain.create_label(f"$y$")
        ax[-1].set_xlabel(x_label)

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0].clegend(vertical=False,ncol=ncol)

        return fig, ax

    def plot_events(self,h_list,Quadrants=None,fig=None,ax=None,line_kw=None,**kwargs):

            

        Quadrants = self._check_quadrant(Quadrants)

        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5,12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num))


        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                fig, ax[i] = self.QuadNumDF.plot_line(comp,
                                                    time = None,
                                                    label=f'$h = {h}$',
                                                    channel_half = True,
                                                    fig= fig, ax= ax[i],
                                                    line_kw=line_kw)
                
                ax[i].set_ylabel(r"$Q%d$(Number of events)"%quad)
        
        x_label = self.Domain.create_label(f"$y$")
        ax[-1].set_xlabel(x_label)

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0].clegend(vertical=False,ncol=ncol)

        return fig, ax

    def plot_event_interval(self,h_list,Quadrants=None,fig=None,ax=None,line_kw=None,**kwargs):    

        Quadrants = self._check_quadrant(Quadrants)

        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5,12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num))


        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                fig, ax[i] = self.QuadDTDF.plot_line(comp,
                                                    time = None,
                                                    label=f'$h = {h}$',
                                                    channel_half = True,
                                                    fig= fig, ax= ax[i],
                                                    line_kw=line_kw)
                
                ax[i].set_ylabel(r"$T_{Q%d}$"%quad)
        
        x_label = self.Domain.create_label(f"$y$")
        ax[-1].set_xlabel(x_label)

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0].clegend(vertical=False,ncol=ncol)

        return fig, ax

    def plot_event_duration(self,h_list,Quadrants=None,fig=None,ax=None,line_kw=None,**kwargs):
        Quadrants = self._check_quadrant(Quadrants)

        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[5,12])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num))


        for i, quad  in enumerate(Quadrants):
            for h in h_list:
                comp = self._comp_calc(h,quad)
                fig, ax[i] = self.QuadDurDF.plot_line(comp,
                                                    time = None,
                                                    label=f'$h = {h}$',
                                                    channel_half = True,
                                                    fig= fig, ax= ax[i],
                                                    line_kw=line_kw)
                
                ax[i].set_ylabel(r"$\Delta T_{Q%d}$"%quad)
        
        x_label = self.Domain.create_label(f"$y$")
        ax[-1].set_xlabel(x_label)

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0].clegend(vertical=False,ncol=ncol)

        return fig, ax


class CHAPSim_Quad_Anl_temp(CHAPSim_Quad_Anl_tg):
    def _hdf_extract(self,file_name,key=None):
        if key is None:
            key = self.__class__.__name__#'CHAPSim_Quad_Anal'

        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)

        
        hdf_obj.check_type_id(self.__class__)

        self._meta_data = self._module.CHAPSim_meta.from_hdf(file_name,key+'/meta_data')
        self._avg_data = self._module._avg_tg_class.from_hdf(file_name,key+'/avg_data')
        
        self.QuadAnalDF = cd.FlowStruct1D_time.from_hdf(file_name,key=key+'/QuadAnalDF')
    
    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key = self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        self._meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',key+'/avg_data')
        self.QuadAnalDF.to_hdf(file_name,key=key+'/QuadAnalDF',mode='a')

    def _quad_extract(self,h_list,path_to_folder='.',time0=None,abs_path=True):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
            
        if cp.rcParams['TEST']:
            times.sort(); times= times[-3:]

        self._meta_data = self._module._meta_class(path_to_folder,abs_path,tgpost=True)
        self._avg_data = self._module._avg_temp_class(path_to_folder, 
                                                    time0, abs_path)
        
        dt_array = np.array([np.diff(times)[0],*np.diff(times)])

        quad_array = []
        quad_index = []
        comp = [self._comp_calc(*x) for x in itertools.product(h_list,range(1,5))]

        for i, (timing, dt) in enumerate(zip(times,dt_array)):
            time1 = time.time()
            fluct_data = self._module._fluct_temp_class(timing, self._avg_data,
                                                        time0=time0, 
                                                        path_to_folder=path_to_folder,
                                                        abs_path=abs_path)
                                                    
            time2 = time.time()

            fluct_uv, quadrant_array = self._quadrant_extract(fluct_data.fluctDF,
                                                                timing)

            quad_anal_array = self._quad_calc(fluct_uv,quadrant_array,h_list)
            quad_array.append(quad_anal_array)

            
            PhyTime = [timing]*len(comp)
            index = list(zip(PhyTime,comp))

            quad_index.extend(index)
            QuadAnal = np.concatenate(quad_array)
            print("Time %g: %.3g %.3g"%(timing,time2 - time1, time.time()- time2))

        self._create_symmetry(h_list,quad_anal_array)
        self.QuadAnalDF = cd.FlowStruct1D_time(self._coorddata,QuadAnal,index=quad_index)

    def _quad_calc(self,fluct_uv,quadrant_array,h_list):

        avg_time = max(self._avg_data.times)
        uu=self._avg_data.UU_tensorDF[avg_time,'uu']
        vv=self._avg_data.UU_tensorDF[avg_time,'vv']

        u_rms = np.sqrt(uu)
        v_rms = np.sqrt(vv)

        quad_anal_array=np.empty((len(h_list)*4,*self.shape))

        for j,h in enumerate(h_list):
            for i in range(4):

                no_mask = np.logical_and(quadrant_array == (i+1),
                                     abs(fluct_uv) > h*(u_rms*v_rms)[:,np.newaxis])
                fluct_uv.mask = ~no_mask
                quad_anal_array[4*j+i] = fluct_uv.filled(0.).mean(axis=(0,2))

        return quad_anal_array


    def _create_symmetry(self,h_list,quad_anal_array):
        if not cp.rcParams['SymmetryAVG'] or  not self.metaDF['iCase'] ==1:
            return

        old_quad_array = quad_anal_array.copy()

        middle_index = old_quad_array.shape[1] // 2
        size = quad_anal_array.shape[0]
        symmetry_map = {1: 3,2:1 ,4:-3,3:-1}

        for j in range(size):
            i = j%4
            i_symm = symmetry_map[i+1]
            quad_anal_array[j,:middle_index] = 0.5*(old_quad_array[j,:middle_index] - old_quad_array[j+i_symm,-middle_index:][::-1])
            quad_anal_array[j,-middle_index:] = quad_anal_array[j,:middle_index][::-1]

                
    def plot_line(self,h_list,locs,axis,Quadrants=None,x_vals=0,y_mode='half_channel',norm=False,fig=None,ax=None,line_kw=None,**kwargs):
                
        if norm:
            time = max(self._avg_data.times)
            norm_Quadrant = self.QuadAnalDF/self._avg_data.UU_tensorDF[time,'uv']
        else:
            norm_Quadrant = self.QuadAnalDF

        Quadrants = self._check_quadrant(Quadrants)
        quad_num = len(Quadrants)

        kwargs = cplt.update_subplots_kw(kwargs,figsize=[12,5*len(locs)])
        fig, ax, _ = cplt.create_fig_ax_without_squeeze(quad_num,len(locs),fig=fig,ax=ax,**kwargs)
        ax = ax.reshape((quad_num,len(locs)))

        if axis == 't':
            unit = misc_utils.get_title_symbol('y',y_mode,False)
            coord_list = self._avg_data.ycoords_from_coords(locs,x_vals=x_vals,mode=y_mode)[0]
            coord_int = self._avg_data.ycoords_from_norm_coords(coord_list,x_vals=x_vals,mode=y_mode)[0]
            for i, quad  in enumerate(Quadrants):
                for h in h_list:
                    comp = self._comp_calc(h,quad)
                    for j, y in enumerate(coord_int):
                        fig, ax[i,j] = norm_Quadrant.plot_line_time(comp, y,
                                                            labels=[f'$h = {h}$'],
                                                            channel_half = True,
                                                            fig= fig, ax= ax[i,j],
                                                            line_kw=line_kw)
                        ax[0,j].set_title(r"$%s=%.3g$"%(unit,coord_list[j]),loc='left')
                        x_label = self.Domain.create_label(f"$t$")
                        ax[-1,j].set_xlabel(x_label)
                
                ax[i,0].set_ylabel(r"$Q%d$"%quad)

        else:   
            for i, quad  in enumerate(Quadrants):
                for j, time in enumerate(locs):
                    for h in h_list:
                        comp = self._comp_calc(h,quad)
                        fig, ax[i,j] = norm_Quadrant.plot_line(comp,
                                                            time = time,
                                                            label=f'$h = {h}$',
                                                            channel_half = True,
                                                            fig= fig, ax= ax[i,j],
                                                            line_kw=line_kw)
                    x_label = self.Domain.create_label(f"$y$")
                    ax[-1,j].set_xlabel(x_label)
                    title = self.Domain.create_label(r"$t=%.3g$"%time)
                    ax[0,j].set_title(title,loc='left')
                
                ax[i,0].set_ylabel(r"$Q%d$"%quad)

        ncol = cplt.get_legend_ncols(len(h_list))
        ax[0,0].clegend(vertical=False,ncol=ncol)

        return fig, ax


            

                        