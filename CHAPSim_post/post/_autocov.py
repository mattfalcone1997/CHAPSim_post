import numpy as np

import time
from abc import ABC, abstractmethod

import CHAPSim_post as cp
from CHAPSim_post.utils import parallel, misc_utils

import CHAPSim_post.plot as cplt
import CHAPSim_post.dtypes as cd
from scipy.signal import correlate
# from ._f90_ext_base import autocov_calc_z, autocov_calc_x

from ._average import CHAPSim_AVG_io, CHAPSim_AVG_tg, CHAPSim_AVG_temp
_avg_io_class = CHAPSim_AVG_io
_avg_tg_class = CHAPSim_AVG_tg
_avg_temp_class = CHAPSim_AVG_temp

from ._fluct import CHAPSim_fluct_io,CHAPSim_fluct_tg, CHAPSim_fluct_temp
_fluct_io_class = CHAPSim_fluct_io
_fluct_tg_class = CHAPSim_fluct_tg
_fluct_temp_class = CHAPSim_fluct_temp

from ._common import Common


from ._meta import CHAPSim_meta
_meta_class = CHAPSim_meta

from CHAPSim_post._libs import autocorr_parallel
class _autocov_base(Common,ABC):
    def __init__(self,*args,from_hdf=False,**kwargs):

        if not from_hdf:
            self._autocov_extract(*args,**kwargs)
        else:
            self._hdf_extract(*args,**kwargs)

    @property
    def shape(self):
        raise NotImplementedError

    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(from_hdf=True,*args,**kwargs)
    

    def save_hdf(self,file_name,write_mode,key=None):
        if key is None:
            key =  self.__class__.__name__

        hdf_obj = cd.hdfHandler(file_name,write_mode,key=key)
        hdf_obj.set_type_id(self.__class__)

        # hdf_obj.attrs['shape_x'] = np.array(self.shape_x)
        # hdf_obj.attrs['shape_z'] = np.array(self.shape_z)
        hdf_obj.attrs['comp'] = np.array([np.string_(x) for x in self.comp])

        self._meta_data.save_hdf(file_name,'a',key+'/meta_data')
        self._avg_data.save_hdf(file_name,'a',key+'/avg_data')
        self.autocorrDF.to_hdf(file_name,key=key+'/autocorrDF',mode='a')#,format='fixed',data_columns=True)

    @abstractmethod
    def _autocov_extract(self,*args,**kwargs):
        raise NotImplementedError

# class _autocov_developing(_autocov_base):
#     pass

class CHAPSim_autocov_io(_autocov_base):
    def _autocov_extract(self,comp,path_to_folder=".",time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if cp.rcParams['TEST']:
            times.sort(); times= times[-5:]
            
        self._meta_data = self._module._meta_class(path_to_folder)
        self.comp=tuple(comp)

        self._avg_data = self._module._avg_io_class(max(times),path_to_folder,time0,abs_path)

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]*0.5)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("Variable max_z_sep must be less than half NCL3 in readdata file\n")
        
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]*0.5)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("Variable max_x_sep must be less than half NCL3 in readdata file\n")

        for i,timing in enumerate(times):
            time1 = time.time()
            fluct_data = self._module._fluct_io_class(timing,self._avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)

            coe3 = i/(i+1)
            coe2 = 1/(i+1)

            if i==0:
                R_x, R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)
            else:
                local_R_x, local_R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)
                if R_x.shape != local_R_x.shape or R_z.shape != local_R_z.shape:
                    msg = "There is a problem. the shapes of the local and averaged array are different"
                    raise ValueError(msg)
                R_x = R_x*coe3 + local_R_x*coe2
                R_z = R_z*coe3 + local_R_z*coe2
            print(time.time() - time1,flush=True)

        if cp.rcParams['SymmetryAVG'] and self.metaDF['iCase'] ==1:
            vy_count = comp.count('v')

            R_x = 0.5*(R_x + R_x[:,::-1]*(-1)**vy_count )
            R_z = 0.5*(R_z + R_z[:,::-1]*(-1)**vy_count )

        self.autocorrDF = cd.datastruct({'x':R_x,'z':R_z})

    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self.comp = tuple(np.char.decode(hdf_obj.attrs["comp"][:]))

        self.autocorrDF = cd.datastruct.from_hdf(file_name,key=key+'/autocorrDF')
        self._meta_data = self._module._meta_class.from_hdf(file_name,key=key+'/meta_data')
        self._avg_data = self._module._avg_io_class.from_hdf(file_name,key=key+'/avg_data')

    def _autocov_calc(self,fluct_data,comp,PhyTime,max_x_sep,max_z_sep):
        PhyTime = fluct_data.check_PhyTime(PhyTime)

        fluct_vals1=fluct_data.fluctDF[PhyTime,comp[0]]
        fluct_vals2=fluct_data.fluctDF[PhyTime,comp[1]]

        R_x = autocorr_parallel.autocov_calc_io_x(fluct_vals1,fluct_vals2,max_x_sep)#.mean(axis=1)
        R_z = autocorr_parallel.autocov_calc_io_z(fluct_vals1,fluct_vals2,max_z_sep)
        
        return R_x, R_z

class CHAPSim_autocov_tg(_autocov_base):
    _tgpost = True
    def _autocov_extract(self,comp,path_to_folder=".",time0=None,ntimes=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if ntimes is not None:
            times = sorted(times)[-ntimes:]

        if cp.rcParams['TEST']:
            times= sorted(times)[-5:]
            
        self._meta_data = self._module._meta_class(path_to_folder,tgpost=True)
        self.comp=tuple(comp)

        self._avg_data = self._module._avg_tg_class(max(times),path_to_folder,time0,abs_path)

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]*0.5)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("Variable max_z_sep must be less than half NCL3 in readdata file\n")
        
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]*0.5)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("Variable max_x_sep must be less than half NCL3 in readdata file\n")

        for i,timing in enumerate(times):
            time1 = time.time()
            fluct_data = self._module._fluct_tg_class(timing,self._avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)

            coe3 = i/(i+1)
            coe2 = 1/(i+1)
            time2 = time.time()
            if i==0:
                R_x, R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)
            else:
                local_R_x, local_R_z = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)

                R_x = R_x*coe3 + local_R_x*coe2
                R_z = R_z*coe3 + local_R_z*coe2

            print(i, timing, time.time() - time2,time2-time1,flush=True)

        if cp.rcParams['SymmetryAVG'] and self.metaDF['iCase'] ==1:
            vy_count = comp.count('v')

            R_x = 0.5*(R_x + R_x[:,::-1]*pow(-1,vy_count))
            R_z = 0.5*(R_z + R_z[:,::-1]*pow(-1,vy_count))

        self.autocorrDF = cd.datastruct({'x':R_x,'z':R_z})

    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self.comp = tuple(np.char.decode(hdf_obj.attrs["comp"][:]))

        self.autocorrDF = cd.datastruct.from_hdf(file_name,key=key+'/autocorrDF')
        self._meta_data = self._module._meta_class.from_hdf(file_name,key=key+'/meta_data')
        self._avg_data = self._module._avg_tg_class.from_hdf(file_name,key=key+'/avg_data')

    def _autocov_calc(self,fluct_data,comp,PhyTime,max_x_sep,max_z_sep):
        PhyTime = fluct_data.check_PhyTime(PhyTime)

        fluct_vals1=fluct_data.fluctDF[PhyTime,comp[0]]
        fluct_vals2=fluct_data.fluctDF[PhyTime,comp[1]]

        # R_x = self.autocov_calc_tg_x_np(fluct_vals1,fluct_vals2)#.mean(axis=1)
        # R_z = self.autocov_calc_tg_z_np(fluct_vals1,fluct_vals2)

        R_x = autocorr_parallel.autocov_calc_tg_x(fluct_vals1,fluct_vals2,max_x_sep)#.mean(axis=1)
        R_z = autocorr_parallel.autocov_calc_tg_z(fluct_vals1,fluct_vals2,max_z_sep)
        
        return R_x, R_z
    def autocov_calc_tg_x_np(self,fluct_vals1,fluct_vals2):
        NCL1 = fluct_vals1.shape[2]
        NCL2 = fluct_vals1.shape[1]
        NCL3 = fluct_vals1.shape[0]

        R_x_temp = np.zeros((NCL1,NCL2,NCL3))
        for j in range(NCL3):
            for i in range(NCL2):
                R_x_temp[:,i,j] = np.correlate(fluct_vals1[j,i,:],fluct_vals2[j,i,:],mode='same')/NCL1
        
        mid = int(NCL1*0.5)
        R_x = R_x_temp.mean(axis=-1)
        return R_x

    def autocov_calc_tg_z_np(self,fluct_vals1,fluct_vals2):
        NCL1 = fluct_vals1.shape[2]
        NCL2 = fluct_vals1.shape[1]
        NCL3 = fluct_vals1.shape[0]

        R_z_temp = np.zeros((NCL3,NCL2,NCL1))
        for i in range(NCL2):
            for j in range(NCL1):
                R_z_temp[:,i,j] = np.correlate(fluct_vals1[:,i,j],fluct_vals2[:,i,j],mode='same')/NCL3
        
        mid = int(NCL3*0.5)
        R_z = R_z_temp.mean(axis=-1)
        return 0.5*(R_z[mid:] + R_z[:mid][::-1])

    def plot_line(self,comp,y_vals,y_mode='half_channel',norm=True,line_kw=None,fig=None,ax=None,**kwargs):
        if not comp in ('x','z'):
            msg = f"comp must be the string 'x' or 'z' not {comp} "
            raise ValueError(msg)

        y_vals = misc_utils.check_list_vals(y_vals)
        Ruu = self.autocorrDF[comp].copy()

        line_kw = cplt.update_line_kw(line_kw)

        if norm:
            Ruu_0=Ruu[0].copy()
            for i in range(Ruu.shape[0]):
                Ruu[i]/=Ruu_0
        
        kwargs = cplt.update_subplots_kw(kwargs,figsize=[10,5*len(y_vals)])
        fig, ax,single_output = cplt.create_fig_ax_without_squeeze(len(y_vals),fig=fig,ax=ax,**kwargs)
        
        y_vals = self._avg_data.ycoords_from_coords(y_vals,mode=y_mode)[0]
        y_index_axis_vals = self._avg_data.y_coord_index_norm(y_vals,mode=y_mode)[0]
        
        coord = self._meta_data.CoordDF[comp][:Ruu.shape[0]]

        title_symbol = misc_utils.get_title_symbol('y',y_mode,local=False)
        for j,_ in enumerate(y_vals):
            ax[j].cplot(coord,Ruu[:,y_index_axis_vals[j]],**line_kw)
            ax[j].set_ylabel(r"$R_{%s%s}$"%self.comp)
            ax[j].set_xlabel(r"$\Delta %s/\delta$"%comp)
            ax[j].set_title(r"$%s=%.3g$"%(title_symbol,y_vals[j]),loc='right')

        if single_output:
            return fig, ax[0]
        else:
            return fig, ax

class CHAPSim_autocov_temp(CHAPSim_autocov_tg):
    def _autocov_extract(self,comp,path_to_folder=".",time0=None,abs_path=True,max_x_sep=None,max_z_sep=None):
        times = misc_utils.time_extract(path_to_folder,abs_path)
        if time0 is not None:
            times = list(filter(lambda x: x > time0, times))

        if cp.rcParams['TEST']:
            times= sorted(times)[-5:]
            
        self._meta_data = self._module._meta_class(path_to_folder,tgpost=True)
        self.comp=tuple(comp)

        self._avg_data = self._module._avg_temp_class(path_to_folder,time0,abs_path,PhyTimes=max(times))

        if max_z_sep is None:
            max_z_sep=int(self.NCL[2]*0.5)
        elif max_z_sep>self.NCL[2]:
            raise ValueError("Variable max_z_sep must be less than half NCL3 in readdata file\n")
        
        if max_x_sep is None:
            max_x_sep=int(self.NCL[0]*0.5)
        elif max_x_sep>self.NCL[0]:
            raise ValueError("Variable max_x_sep must be less than half NCL3 in readdata file\n")
        R_x = np.array((max_x_sep,self.NCL[1],len(times)))
        R_z = np.array((max_z_sep,self.NCL[1],len(times)))

        for i,timing in enumerate(times):
            time1 = time.time()
            fluct_data = self._module._fluct_temp_class(timing,self._avg_data,time0=time0,path_to_folder=path_to_folder,abs_path=abs_path)

            R_x_single_t, R_z_single_t = self._autocov_calc(fluct_data,comp,timing,max_x_sep,max_z_sep)
            
            R_x[:,:,i] = R_x_single_t
            R_z[:,:,i] = R_z_single_t

            print(time.time() - time1,flush=True)

        if cp.rcParams['SymmetryAVG'] and self.metaDF['iCase'] ==1:
            vy_count = comp.count('v')

            R_x = 0.5*(R_x + R_x[:,::-1]*(-1)**vy_count )
            R_z = 0.5*(R_z + R_z[:,::-1]*(-1)**vy_count )

        self.autocorrDF = cd.datastruct({'x':R_x,'z':R_z})

    def _hdf_extract(self,file_name, key=None):
        if key is None:
            key =  self.__class__.__name__
        
        hdf_obj = cd.hdfHandler(file_name,'r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self.comp = tuple(np.char.decode(hdf_obj.attrs["comp"][:]))

        self.autocorrDF = cd.datastruct.from_hdf(file_name,key=key+'/autocorrDF')
        self._meta_data = self._module._meta_class.from_hdf(file_name,key=key+'/meta_data')
        self._avg_data = self._module._avg_temp_class.from_hdf(file_name,key=key+'/avg_data')