"""
# _quadrant_a.py
Module for quadrant analysis from
instantaneous results from CHAPSim DNS solver
"""

import numpy as np

import sys
import warnings
import gc
import itertools

import CHAPSim_post as cp
from .. import CHAPSim_plot as cplt
from .. import CHAPSim_Tools as CT
from .. import CHAPSim_dtypes as cd

from ._average import CHAPSim_AVG_io, CHAPSim_AVG_tg_base
_avg_io_class = CHAPSim_AVG_io
_avg_tg_base_class = CHAPSim_AVG_tg_base

from ._fluct import CHAPSim_fluct_io, CHAPSim_fluct_tg
_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg

from ._meta import CHAPSim_meta
_meta_class = CHAPSim_meta

class CHAPSim_Quad_Anl_base():
    _module = sys.modules[__name__]
    def __init__(self,*args,**kwargs):
        fromfile=kwargs.pop('fromfile',False)
        if not fromfile:
            self._meta_data, self.NCL, self._avg_data,\
            self.QuadAnalDF,self.shape = self._quad_extract(*args,**kwargs)
        else:
            self._meta_data, self.NCL, self._avg_data,\
            self.QuadAnalDF,self.shape = self._hdf_extract(*args,**kwargs)
    def save_hdf(self,file_name,write_mode,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Quad_Anal'

        self._meta_data.save_hdf(file_name,write_mode,base_name+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',base_name+'/avg_data')
        self.QuadAnalDF.to_hdf(file_name,key=base_name+'/QuadAnalDF',mode='a')

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(fromfile=True,*args,**kwargs)

    @staticmethod
    def _quadrant_extract(fluctDF,PhyTime,NCL):
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)
        if len(set([x[0] for x in fluctDF.index])) == 1:
            fluct_time = list(set([x[0] for x in fluctDF.index]))[0]

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
        fluct_uv=u_array*v_array 

        return fluct_uv, quadrant_array 

    def line_plot(self,h_list,coord_list,prop_dir,x_vals=0,y_mode='half_channel',norm=False,fig=None,ax=None,line_kw=None,**kwargs):
        assert x_vals is None or not hasattr(x_vals,'__iter__')

        if fig is None:
            kwargs = cplt.update_subplots_kw(kwargs,squeeze=False,figsize=[12,5*len(coord_list)])
            fig, ax = cplt.subplots(4,len(coord_list),**kwargs)
        elif ax is None:
            kwargs = cplt.update_subplots_kw(kwargs,squeeze=False)
            ax = fig.subplots(4,len(coord_list),**kwargs)

        if prop_dir =='y':
            index = [self._avg_data._return_index(x) for x in coord_list]

        elif prop_dir == 'x':
            index = CT.y_coord_index_norm(self._avg_data,self._meta_data.CoordDF,
                                                        coord_list,x_vals,y_mode)
            if x_vals is not None:
                index=list(itertools.chain(*index))
        else:
            raise ValueError("The propagation direction of the quadrant analysis must be `x' or `y'")
        if norm: 
            avg_time = list(set([x[0] for x in self._avg_data.UU_tensorDF.index]))[0]
            uv=self._avg_data.UU_tensorDF[avg_time,'uv']

        coords = self._meta_data.CoordDF[prop_dir]

        unit=r"x/\delta"if prop_dir =='y' else r"y/\delta" if y_mode=='half_channel' \
                else r"\delta_u" if y_mode=='disp_thickness' \
                else r"\theta" if y_mode=='mom_thickness' else r"y^+" \
                if x_vals is None or x_vals!=0 else r"y^{+0}"
        
        line_kw=cplt.update_line_kw(line_kw)
        for i in range(1,5):
            for h in h_list:
                quad_anal = self.QuadAnalDF[h,i]
                if norm:
                    quad_anal/=uv
                for j in range(len(coord_list)):
                    if x_vals is None and prop_dir=='x':
                        quad_anal_index= np.zeros(self.shape[1])
                        for k in range(self.shape[1]):
                            quad_anal_index[k]=quad_anal[index[k][j],k]
                    else:
                        quad_anal_index=quad_anal[index[j],:] if prop_dir == 'x' else quad_anal[:,index[j]].T
                    ax[i-1,j].cplot(coords,quad_anal_index,label=r"$h=%.5g$"%h,**line_kw)
                    ax[i-1,j].set_xlabel(r"$%s/\delta$"%prop_dir)# ,fontsize=20)
                    ax[i-1,j].set_ylabel(r"$Q%d$"%i)# ,fontsize=20)
                    ax[i-1,j].set_title(r"$%s=%.5g$"%(unit,coord_list[j]),loc='left')# ,fontsize=16)
                    ax[i-1,j].toggle_default_line_markers()

        ncol = 4 if len(h_list)>3 else len(h_list)
        ax[0,0].clegend(vertical=False,ncol=ncol)
        ax[0,0].get_gridspec().tight_layout(fig)

        fig.tight_layout()
        return fig, ax

class CHAPSim_Quad_Anl_io(CHAPSim_Quad_Anl_base):
    def _quad_extract(self,h_list,path_to_folder='',time0=None,abs_path=True):
        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if cp.Params['TEST']:
            times.sort(); times= times[-3:]
        meta_data = self._module._meta_class(path_to_folder,abs_path)
        NCL = meta_data.NCL
        try:
            avg_data = self._module._avg_io_class(max(times),meta_data,path_to_folder,time0,abs_path)
        except Exception:
            times.remove(max(times))
            avg_data = self._module._avg_io_class(max(times),meta_data,path_to_folder,time0)
        i=1
        for timing in times:
            fluct_data = self._module._fluct_io_class(timing,avg_data,time0=time0,path_to_folder=path_to_folder,
                                        abs_path=abs_path)
            fluct_uv, quadrant_array = self._quadrant_extract(fluct_data.fluctDF,timing,NCL)
            coe3 = (i-1)/i
            coe2 = 1/i
            if i==1:
                quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
            else:
                local_quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
                assert local_quad_anal_array.shape == quad_anal_array.shape, "shape of previous array (%d,%d) " % quad_anal_array.shape\
                    + " and current array (%d,%d) must be the same" % local_quad_anal_array.shape
                quad_anal_array = quad_anal_array*coe3 + local_quad_anal_array*coe2
            gc.collect()
            i += 1
        index=[[],[]]
        for h in h_list:
            index[0].extend([h]*4)
        index[1]=[1,2,3,4]*len(h_list)
        shape = avg_data.shape        
        QuadAnalDF=cd.datastruct(quad_anal_array,index=index)

        return meta_data, NCL, avg_data, QuadAnalDF, shape

    def _hdf_extract(self,file_name,group_name=''):
        base_name=group_name if group_name else 'CHAPSim_Quad_Anal'
        meta_data = self._module.CHAPSim_meta.from_hdf(file_name,base_name+'/meta_data')
        NCL= meta_data.NCL
        avg_data = self._module._avg_io_class.from_hdf(file_name,base_name+'/avg_data')
        shape = (NCL[1],NCL[0])
        
        QuadAnalDF = cd.datastruct.from_hdf(file_name,shapes=shape,key=base_name+'/QuadAnalDF')#pd.read_hdf(file_name,key=base_name+'/autocorrDF').data([shape_x,shape_z])
                
        return meta_data, NCL, avg_data, QuadAnalDF, shape

    @staticmethod
    def _quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,PhyTime):
        if type(PhyTime) == float: #Convert float to string to be compatible with dataframe
            PhyTime = "{:.10g}".format(PhyTime)

        avg_time = list(set([x[0] for x in avg_data.UU_tensorDF.index]))[0]
        uv_q=np.zeros((4,*NCL[::-1][1:]))

    
        uu=avg_data.UU_tensorDF[avg_time,'uu']
        vv=avg_data.UU_tensorDF[avg_time,'vv']
        u_rms = np.sqrt(uu)
        v_rms = np.sqrt(vv)

        quad_anal_array=np.empty((len(h_list)*4,NCL[1],NCL[0]))
        for h,j in zip(h_list,range(len(h_list))):
            for i in range(1,5):
                quad_array=quadrant_array == i
                fluct_array = np.abs(quad_array*fluct_uv) > h*u_rms*v_rms
                uv_q[i-1]=np.mean(fluct_uv*fluct_array,axis=0)
            quad_anal_array[j*4:j*4+4]=uv_q
        return quad_anal_array

class CHAPSim_Quad_Anl_tg(CHAPSim_Quad_Anl_base):
    _module = sys.modules[__name__]
    def _quad_extract(self,h_list,path_to_folder='',time0=None,abs_path=True):
        times = CT.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))
        if cp.Params['TEST']:
            times.sort(); times= times[-3:]
        meta_data = self._module._meta_class(path_to_folder,abs_path)
        NCL = meta_data.NCL

        avg_data = self._module._avg_tg_base_class(times,meta_data=meta_data,path_to_folder=path_to_folder,time0=time0,abs_path=abs_path)
        
        for timing in times:
            fluct_data = self._module._fluct_tg_class(timing,avg_data,path_to_folder=path_to_folder,abs_path=abs_path)
            fluct_uv, quadrant_array = self._quadrant_extract(fluct_data.fluctDF,timing,NCL)
            
            if 'quad_anal_array' not in locals():
                quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
            else:
                local_quad_anal_array = self._quad_calc(avg_data,fluct_uv,quadrant_array,NCL,h_list,timing)
                quad_anal_array = np.vstack([quad_anal_array,local_quad_anal_array])
            gc.collect()
        index=[[],[]]
        for h in h_list:
            index[0].extend([h]*4)
        index[1]=[1,2,3,4]*len(h_list)
        shape = avg_data.shape        
        QuadAnalDF=cd.datastruct(quad_anal_array,index=index)

        return meta_data, NCL, avg_data, QuadAnalDF, shape